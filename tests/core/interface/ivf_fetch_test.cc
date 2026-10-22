// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <memory>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/core/interface/index_factory.h>
#include "algorithm/ivf/ivf_searcher_context.h"
#include "tests/test_util.h"

using namespace zvec::core_interface;

namespace {
class InspectableIVFIndex : public IVFIndex {
 public:
  int initialize(const BaseIndexParam &param) {
    return init(param);
  }
  std::weak_ptr<zvec::core::IndexProvider> fetch_provider() {
    return static_cast<zvec::core::IVFSearcherContext *>(
               acquire_context().get())
        ->fetch_provider();
  }
};
}  // namespace

TEST(IndexInterface, DenseFetchScratchPreservesOwnedOutputs) {
  auto &pool = zvec::ailego::MemoryLimitPool::get_instance();
  const auto capacity = pool.capacity();
  ASSERT_EQ(0, pool.init(64UL * 1024 * 1024));
  auto cleanup = zvec::ailego::ScopeGuard::Make([&]() {
    zvec::test_util::RemoveTestFiles("flat_fetch_scratch_*.index");
    EXPECT_EQ(0, pool.init(capacity));
  });
  VectorDataBuffer reused;
  for (const auto type : {QuantizerType::kNone, QuantizerType::kFP16}) {
    for (const auto storage : {StorageOptions::StorageType::kMMAP,
                               StorageOptions::StorageType::kBufferPool}) {
      for (const uint32_t dim : {4, 16, 24, 16392}) {
        const std::string path = "flat_fetch_scratch_0.index";
        zvec::test_util::RemoveTestFiles(path);
        auto param = FlatIndexParamBuilder()
                         .with_metric_type(MetricType::kL2sq)
                         .with_data_type(DataType::DT_FP32)
                         .with_dimension(dim)
                         .with_quantizer_param(QuantizerParam(type))
                         .build();
        auto index = IndexFactory::CreateAndInitIndex(*param);
        ASSERT_NE(nullptr, index);
        ASSERT_EQ(0, index->open(path, {storage, true}));
        for (uint32_t id = 0; id < 4; ++id) {
          const std::vector<float> value(dim, float(id));
          ASSERT_EQ(0, index->add(VectorData{DenseVector{value.data()}}, id));
        }
        VectorDataBuffer kept;
        ASSERT_EQ(0, index->fetch(0, &kept));
        const auto snapshot = std::get<DenseVectorBuffer>(kept.vector_buffer).data;
        for (uint32_t i = 0; i < 12; ++i) {
          const std::vector<float> value(dim, float(i % 4));
          ASSERT_EQ(0, index->fetch(i % 4, &reused));
          EXPECT_EQ(std::string(reinterpret_cast<const char *>(value.data()),
                                value.size() * sizeof(float)),
                    std::get<DenseVectorBuffer>(reused.vector_buffer).data);
        }
        const auto previous = std::get<DenseVectorBuffer>(reused.vector_buffer).data;
        EXPECT_EQ(zvec::core::IndexError_Runtime, index->fetch(4, &reused));
        EXPECT_EQ(previous, std::get<DenseVectorBuffer>(reused.vector_buffer).data);
        EXPECT_EQ(snapshot, std::get<DenseVectorBuffer>(kept.vector_buffer).data);
        ASSERT_EQ(0, index->close());
      }
    }
  }
}

TEST(IndexInterface, IvfFetchCacheTracksOwnersAndKeepsOutputsIndependent) {
  auto &pool = zvec::ailego::MemoryLimitPool::get_instance();
  const auto capacity = pool.capacity();
  ASSERT_EQ(0, pool.init(64UL * 1024 * 1024));
  auto cleanup = zvec::ailego::ScopeGuard::Make([&]() {
    zvec::test_util::RemoveTestFiles("ivf_fetch_cache_*.index");
    EXPECT_EQ(0, pool.init(capacity));
  });
  for (const auto type : {QuantizerType::kNone, QuantizerType::kFP16}) {
    for (const auto storage : {StorageOptions::StorageType::kMMAP,
                               StorageOptions::StorageType::kBufferPool}) {
      std::vector<std::shared_ptr<InspectableIVFIndex>> indexes;
      for (uint32_t owner = 0; owner < 2; ++owner) {
        const auto path = "ivf_fetch_cache_" + std::to_string(owner) + ".index";
        zvec::test_util::RemoveTestFiles(path);
        auto param = IVFIndexParamBuilder()
                         .with_metric_type(MetricType::kL2sq)
                         .with_data_type(DataType::DT_FP32)
                         .with_dimension(16)
                         .with_n_list(1)
                         .with_quantizer_param(QuantizerParam(type))
                         .build();
        auto index = std::make_shared<InspectableIVFIndex>();
        ASSERT_EQ(0, index->initialize(*param));
        Index::Pointer base = index;
        ASSERT_EQ(0, base->open(path, {storage, true}));
        for (uint32_t id = 0; id < 64; ++id) {
          const std::vector<float> value(16, float(owner * 100 + id));
          ASSERT_EQ(0, base->add(VectorData{DenseVector{value.data()}}, id));
        }
        ASSERT_EQ(0, base->train());
        indexes.push_back(std::move(index));
      }
      Index::Pointer first = indexes[0], second = indexes[1];
      const std::vector<float> query(16, 7.0f);
      auto query_param = std::make_shared<IVFQueryParam>();
      query_param->topk = query_param->nprobe = 1;
      query_param->fetch_vector = true;
      SearchResult result;
      ASSERT_EQ(0, first->search(VectorData{DenseVector{query.data()}},
                                 query_param, &result));
      ASSERT_EQ(1U, result.doc_list_.size());
      ASSERT_NE(nullptr, result.doc_list_[0].vector());
      const std::string borrowed(
          static_cast<const char *>(result.doc_list_[0].vector()), 16 * 4);
      VectorDataBuffer kept, next;
      ASSERT_EQ(0, first->fetch(7, &kept));
      const auto expected =
          std::get<DenseVectorBuffer>(kept.vector_buffer).data;
      auto provider = indexes[0]->fetch_provider();
      ASSERT_FALSE(provider.expired());
      ASSERT_EQ(0, first->fetch(9, &next));
      auto reused = indexes[0]->fetch_provider();
      EXPECT_FALSE(provider.owner_before(reused) ||
                   reused.owner_before(provider));
      EXPECT_EQ(
          borrowed,
          std::string(static_cast<const char *>(result.doc_list_[0].vector()),
                      16 * 4));
      EXPECT_EQ(expected, std::get<DenseVectorBuffer>(kept.vector_buffer).data);
      ASSERT_EQ(0, second->fetch(7, &next));
      EXPECT_TRUE(provider.expired());
      EXPECT_FLOAT_EQ(
          107.0f,
          *reinterpret_cast<const float *>(
              std::get<DenseVectorBuffer>(next.vector_buffer).data.data()));
      ASSERT_EQ(0, first->fetch(7, &next));
      EXPECT_EQ(expected, std::get<DenseVectorBuffer>(next.vector_buffer).data);
      EXPECT_EQ(zvec::core::IndexError_NoExist, first->fetch(64, &next));
      EXPECT_EQ(expected, std::get<DenseVectorBuffer>(next.vector_buffer).data);
      std::vector<std::thread> readers;
      for (uint32_t thread = 0; thread < 2; ++thread) {
        readers.emplace_back([&, thread]() {
          VectorDataBuffer value;
          for (uint32_t i = 0; i < 16; ++i) {
            ASSERT_EQ(0, first->fetch(i + thread, &value));
            EXPECT_FLOAT_EQ(float(i + thread),
                            *reinterpret_cast<const float *>(
                                std::get<DenseVectorBuffer>(value.vector_buffer)
                                    .data.data()));
          }
        });
      }
      for (auto &reader : readers) reader.join();
      provider = indexes[0]->fetch_provider();
      ASSERT_EQ(0, first->close());
      EXPECT_TRUE(provider.expired());
      ASSERT_EQ(0, first->open("ivf_fetch_cache_0.index", {storage, false, true}));
      ASSERT_EQ(0, first->fetch(7, &next));
      EXPECT_EQ(expected, std::get<DenseVectorBuffer>(next.vector_buffer).data);
      ASSERT_EQ(0, first->close());
      ASSERT_EQ(0, second->close());
    }
  }
}
