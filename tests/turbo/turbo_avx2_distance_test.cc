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

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include "distance/avx2/split_distance.h"
#include "distance/avx512/split_distance.h"
#include "quantizer/distance.h"

namespace zvec::turbo {
namespace {

template <typename T>
void CheckSplits(DataType type, QuantizeType quantize,
                 const std::array<SplitDistanceQuantizer::Distance, 3> &splits,
                 CpuArchType arch = CpuArchType::kAVX2, size_t count = 9) {
  constexpr size_t kMaxRows = 32;
  std::mt19937 generator(7391);
  std::uniform_real_distribution<float> random(-3.1f, 3.1f);
  const auto metrics = {MetricType::kSquaredEuclidean,
                        MetricType::kInnerProduct, MetricType::kCosine};
  const int rounding = std::fegetround();
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(0, std::fesetround(mode));
    for (size_t dim :
         {8U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, 96U, 256U, 768U}) {
      std::vector<T> finite_query(dim), finite_rows(kMaxRows * dim);
      for (auto &value : finite_query)
        value = static_cast<T>(random(generator));
      for (auto &value : finite_rows) value = static_cast<T>(random(generator));
      // Multi-NaN payloads use the frozen whole-symbol oracle in exact_bits.py;
      // existing kernel widths can select different payloads.
      for (int special : {0, 1, 2}) {
        auto query = finite_query;
        auto rows = finite_rows;
        if (special == 1 || special == 2) {
          rows[0] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
          rows[dim] = static_cast<T>(std::numeric_limits<float>::infinity());
          query[dim - 1] = static_cast<T>(-0.0f);
          if (special == 2) {
            const uint64_t qnan = sizeof(T) == 4 ? 0x7fc0002a : 0x7e2a;
            const uint64_t dnan = sizeof(T) == 4 ? 0x7fc00139 : 0x7e39;
            std::memcpy(static_cast<void *>(query.data()), &qnan, sizeof(T));
            std::memcpy(static_cast<void *>(rows.data()), &dnan, sizeof(T));
          }
        }
        size_t metric_index = 0;
        for (auto metric : metrics) {
          if (quantize == QuantizeType::kRaw &&
              metric != MetricType::kSquaredEuclidean) {
            ++metric_index;
            continue;
          }
          SCOPED_TRACE(::testing::Message()
                       << "dim=" << dim << " metric=" << metric_index
                       << " special=" << special << " rounding=" << mode
                       << " quantize=" << static_cast<int>(quantize)
                       << " count=" << count);
          auto kernels = get_distance_kernels(metric, type, quantize, arch);
          if (!kernels.batch) {
            std::fesetround(rounding);
            GTEST_SKIP();
          }
          std::array<float, kMaxRows> expected{}, actual{}, batched{};
          std::array<const void *, kMaxRows> pointers;
          for (size_t row = 0; row < count; ++row)
            pointers[row] = rows.data() + row * dim;
          for (size_t row = 0; row < count;) {
            const size_t batch =
                arch == CpuArchType::kAVX512
                    ? (row + 1 < count ? 2 : 1)
                    : (type == DataType::kFp32 && row + 4 <= count ? 4 : 1);
            kernels.batch(pointers.data() + row, query.data(), batch, dim,
                          expected.data() + row, nullptr);
            row += batch;
          }
          kernels.batch(pointers.data(), query.data(), count, dim,
                        batched.data(), nullptr);
          ASSERT_EQ(0, std::memcmp(expected.data(), batched.data(),
                                   sizeof(expected)));
          const size_t alignment =
              arch == CpuArchType::kAVX2
                  ? 8
                  : (quantize == QuantizeType::kRaw ? 32 : 16);
          if (!splits[metric_index] || dim % alignment) {
            ++metric_index;
            continue;
          }
          for (size_t cut : {0UL, 1UL, 7UL, 8UL, dim / 2 + 1, dim - 1, dim}) {
            std::array<SplitVector, kMaxRows> vectors;
            std::vector<T> suffix(kMaxRows * (dim - cut));
            for (size_t row = 0; row < count; ++row) {
              const T *source = rows.data() + row * dim;
              T *second = suffix.data();
              if (cut != dim) {
                second += row * (dim - cut);
                std::memcpy(second, source + cut, (dim - cut) * sizeof(T));
              }
              vectors[row] = {source, second, cut * sizeof(T)};
            }
            ASSERT_TRUE(splits[metric_index](vectors.data(), count,
                                             query.data(), dim, actual.data(),
                                             INFINITY));
            ASSERT_EQ(0, std::memcmp(expected.data(), actual.data(),
                                     sizeof(expected)))
                << "dim=" << dim << " cut=" << cut
                << " metric=" << metric_index;
            if (metric_index == 0) {
              for (float cutoff : {expected[4], .01f}) {
                ASSERT_TRUE(splits[0](vectors.data(), count, query.data(), dim,
                                      actual.data(), cutoff));
                for (size_t row = 0; row < count; ++row) {
                  if (!(std::isinf(actual[row]) && expected[row] >= cutoff)) {
                    ASSERT_EQ(0, std::memcmp(&expected[row], &actual[row],
                                             sizeof(float)));
                  }
                }
                if (cutoff == .01f && dim > 64 && special < 2 && count >= 8) {
                  EXPECT_TRUE(std::isinf(actual[4]));
                }
              }
            }
          }
          actual = expected;
          std::array<SplitVector, kMaxRows> invalid;
          for (size_t row = 0; row < count; ++row)
            invalid[row] = {pointers[row], nullptr, dim * sizeof(T)};
          ++invalid[count - 1].first_bytes;
          EXPECT_FALSE(splits[metric_index](invalid.data(), count, query.data(),
                                            dim, actual.data(), INFINITY));
          ASSERT_EQ(
              0, std::memcmp(expected.data(), actual.data(), sizeof(expected)));
          --invalid[count - 1].first_bytes;
          EXPECT_FALSE(splits[metric_index](invalid.data(), count, query.data(),
                                            dim + 1, actual.data(), INFINITY));
          ASSERT_EQ(
              0, std::memcmp(expected.data(), actual.data(), sizeof(expected)));
          ++metric_index;
        }
      }
    }
  }
  ASSERT_EQ(0, std::fesetround(rounding));
}

template <typename T>
void CheckSelectiveRejection(DataType type, QuantizeType quantize,
                             SplitDistanceQuantizer::Distance split) {
  constexpr size_t kDim = 96;
  constexpr float kCutoff = 128.0f;
  auto kernels = get_distance_kernels(MetricType::kSquaredEuclidean, type,
                                      quantize, CpuArchType::kAVX512);
  if (!kernels.batch) GTEST_SKIP() << "AVX-512 kernels are unavailable";
  std::array<T, kDim> query{};
  std::vector<T> rows(32 * kDim);
  std::array<const void *, 32> pointers;
  for (size_t row = 0; row < pointers.size(); ++row)
    pointers[row] = rows.data() + row * kDim;
  for (size_t count :
       {8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U, 17U, 25U, 32U}) {
    for (unsigned rejected = 0; rejected < 4; ++rejected) {
      for (size_t row = 0; row < count; ++row) {
        const bool reject = rejected & (1U << ((row / 4) % 2));
        for (size_t d = 0; d < kDim; ++d)
          rows[row * kDim + d] = static_cast<T>(d < 64 ? (reject ? 2.0f : .5f)
                                                       : 1.0f + row * .125f);
      }
      std::array<float, 32> expected{}, actual{};
      kernels.batch(pointers.data(), query.data(), count, kDim, expected.data(),
                    nullptr);
      for (size_t cut : {size_t{1}, size_t{63}, size_t{65}, kDim}) {
        SCOPED_TRACE(::testing::Message()
                     << "count=" << count << " cut=" << cut << " rejected="
                     << rejected << " quantize=" << static_cast<int>(quantize));
        std::array<SplitVector, 32> vectors;
        std::vector<T> suffix(count * (kDim - cut));
        for (size_t row = 0; row < count; ++row) {
          const T *source = rows.data() + row * kDim;
          T *second = nullptr;
          if (cut != kDim) {
            second = suffix.data() + row * (kDim - cut);
            std::memcpy(second, source + cut, (kDim - cut) * sizeof(T));
          }
          vectors[row] = {source, second, cut * sizeof(T)};
        }
        ASSERT_TRUE(split(vectors.data(), count, query.data(), kDim,
                          actual.data(), kCutoff));
        for (size_t row = 0; row < count; ++row) {
          const bool complete_group = (row / 4 + 1) * 4 <= count;
          if (complete_group && (rejected & (1U << ((row / 4) % 2)))) {
            EXPECT_EQ(INFINITY, actual[row]) << "row=" << row;
          } else {
            ASSERT_EQ(0,
                      std::memcmp(&expected[row], &actual[row], sizeof(float)))
                << "row=" << row;
          }
        }
      }
    }
  }
}

TEST(Avx2Distance, SplitAndWiderBatchesPreserveExistingBatchBits) {
#if defined(__x86_64__) || defined(_M_X64)
  for (size_t count = 1; count <= 32; ++count) {
    if (count > 25 && count != 32) continue;
    CheckSplits<float>(
        DataType::kFp32, QuantizeType::kFp32,
        {avx2::squared_euclidean_fp32_split_batch,
         avx2::inner_product_fp32_split_batch, avx2::cosine_fp32_split_batch},
        CpuArchType::kAVX2, count);
    CheckSplits<ailego::Float16>(
        DataType::kFp16, QuantizeType::kFp16,
        {avx2::squared_euclidean_fp16_split_batch,
         avx2::inner_product_fp16_split_batch, avx2::cosine_fp16_split_batch},
        CpuArchType::kAVX2, count);
  }
#else
  GTEST_SKIP() << "AVX2 kernels require x86-64";
#endif
}

TEST(TurboAvx512Distance, Fp32SplitAndWiderBatchPreserveBits) {
  for (size_t count = 1; count <= 32; ++count) {
    if (count > 25 && count != 32) continue;
    CheckSplits<float>(
        DataType::kFp32, QuantizeType::kFp32,
        {avx512::squared_euclidean_fp32_split_batch, nullptr, nullptr},
        CpuArchType::kAVX512, count);
  }
}

TEST(TurboAvx512Distance, Fp16SplitAndWiderBatchPreserveBits) {
  for (size_t count = 1; count <= 32; ++count) {
    if (count > 25 && count != 32) continue;
    CheckSplits<ailego::Float16>(
        DataType::kFp16, QuantizeType::kFp16,
        {avx512::squared_euclidean_fp16_split_batch, nullptr, nullptr},
        CpuArchType::kAVX512, count);
    CheckSplits<ailego::Float16>(
        DataType::kFp16, QuantizeType::kRaw,
        {avx512::squared_euclidean_raw_fp16_split_batch, nullptr, nullptr},
        CpuArchType::kAVX512, count);
  }
}

TEST(TurboAvx512Distance, FourRowGroupsRejectIndependently) {
  CheckSelectiveRejection<float>(DataType::kFp32, QuantizeType::kFp32,
                                 avx512::squared_euclidean_fp32_split_batch);
  CheckSelectiveRejection<ailego::Float16>(
      DataType::kFp16, QuantizeType::kFp16,
      avx512::squared_euclidean_fp16_split_batch);
  CheckSelectiveRejection<ailego::Float16>(
      DataType::kFp16, QuantizeType::kRaw,
      avx512::squared_euclidean_raw_fp16_split_batch);
}

}  // namespace
}  // namespace zvec::turbo
