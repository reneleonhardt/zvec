#pragma once
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "ailego/internal/cpu_features.h"

namespace ci = zvec::core_interface;
static void require(bool ok) {
  if (!ok) throw std::runtime_error("benchmark operation failed");
}
struct Fixture {
  uint32_t count, dim, queries;
  std::vector<float> vectors, query;
  explicit Fixture(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    uint32_t header[4];
    f.read(reinterpret_cast<char *>(header), sizeof(header));
    require(f.good() && header[0] == 0x56454331);
    count = header[1];
    dim = header[2];
    queries = header[3];
    vectors.resize(size_t(count) * dim);
    query.resize(size_t(queries) * dim);
    f.read(reinterpret_cast<char *>(vectors.data()), vectors.size() * 4);
    f.read(reinterpret_cast<char *>(query.data()), query.size() * 4);
    require(f.good());
  }
};
struct Alloc {
  uint64_t calls, bytes;
};
using Begin = void (*)();
using End = Alloc (*)();
static ci::BaseIndexParam::Pointer params(const Fixture &f,
                                          const std::string &kind) {
  if (kind == "ivf")
    return ci::IVFIndexParamBuilder()
        .with_metric_type(ci::MetricType::kL2sq)
        .with_data_type(ci::DataType::DT_FP32)
        .with_dimension(f.dim)
        .with_n_list(64)
        .with_n_iters(8)
        .build();
  if (kind == "physical" || kind == "raw16" || kind == "cos32" || kind == "ip32")
    return ci::FlatIndexParamBuilder()
        .with_metric_type(kind == "raw16" ? ci::MetricType::kL2sq
                          : kind == "ip32" ? ci::MetricType::kInnerProduct
                                           : ci::MetricType::kCosine)
        .with_data_type(ci::DataType::DT_FP32)
        .with_dimension(f.dim)
        .with_use_contiguous_memory(false)
        .with_storage_data_type(kind == "physical" || kind == "raw16"
                                    ? ci::DataType::DT_FP16
                                    : ci::DataType::DT_UNDEFINED)
        .build();
  auto p = ci::FlatIndexParamBuilder()
               .with_metric_type(ci::MetricType::kL2sq)
               .with_data_type(ci::DataType::DT_FP32)
               .with_dimension(f.dim)
               .with_use_contiguous_memory(false)
               .build();
  if (kind == "fp16")
    p->quantizer_param =
        std::make_shared<ci::QuantizerParam>(ci::QuantizerType::kFP16);
  return p;
}
static void prepare(const Fixture &f, const std::string &root) {
  for (const std::string kind :
       {"flat", "fp16", "ivf", "physical", "raw16", "cos32", "ip32"}) {
    auto index = ci::IndexFactory::CreateAndInitIndex(*params(f, kind));
    require(bool(index));
    require(index->open(root + "/" + kind + ".index",
                        {ci::StorageOptions::StorageType::kMMAP, true}) == 0);
    for (uint32_t id = 0; id < f.count; ++id)
      require(index->add(ci::VectorData{ci::DenseVector{f.vectors.data() +
                                                        size_t(id) * f.dim}},
                         id) == 0);
    require(index->train() == 0);
    require(index->close() == 0);
  }
}

static void configure_arch() {
  const char *arch = std::getenv("VECTOR_PERF_ARCH");
  auto &flags = zvec::ailego::internal::CpuFeatures::static_flags_;
  if (arch && std::string(arch) == "avx2") {
    require(flags.AVX2 && flags.F16C);
    flags.AVX512F = false;
    flags.AVX512_VNNI = false;
    flags.AVX512_FP16 = false;
  }
  std::fprintf(stderr, "arch=%s avx2=%d f16c=%d avx512=%d\n",
               arch ? arch : "auto", flags.AVX2, flags.F16C, flags.AVX512F);
}
