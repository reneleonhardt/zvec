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
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>
#include "distance/neon/fp16/cosine.h"
#include "distance/neon/fp16/inner_product.h"
#include "distance/neon/fp16/squared_euclidean.h"
#include "distance/neon/fp32/cosine.h"
#include "distance/neon/fp32/inner_product.h"
#include "distance/neon/fp32/squared_euclidean.h"
#include "distance/neon_fp16/fp16/squared_euclidean.h"
#include "distance/neon/record_quantized_int4/common.h"
#include "distance/neon/record_quantized_int8/common.h"
#include "distance/scalar/fp16/cosine.h"
#include "distance/scalar/fp16/inner_product.h"
#include "distance/scalar/fp16/squared_euclidean.h"
#include "distance/scalar/fp32/cosine.h"
#include "distance/scalar/fp32/inner_product.h"
#include "distance/scalar/fp32/squared_euclidean.h"
#include "distance/scalar/record_quantized_int4/common.h"
#include "distance/scalar/record_quantized_int8/common.h"

namespace zvec::turbo {
namespace {

using RawDistanceFn = void (*)(const void *, const void *, size_t, float *);

// Only used by the AArch64 branches of the tests below.
[[maybe_unused]] void CompareDistance(RawDistanceFn expected_fn,
                                      RawDistanceFn actual_fn, const void *a,
                                      const void *b, size_t dim,
                                      float tolerance) {
  float expected = 0.0f;
  float actual = 0.0f;
  expected_fn(a, b, dim, &expected);
  actual_fn(a, b, dim, &actual);
  EXPECT_NEAR(expected, actual, tolerance) << "dim=" << dim;
}

TEST(NeonDistance, Fp32MatchesScalarForRemainders) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  std::vector<float> a(33);
  std::vector<float> b(33);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(scalar::inner_product_fp32_distance,
                    neon::inner_product_fp32_distance, a.data(), b.data(), dim,
                    1e-5f);
    CompareDistance(scalar::squared_euclidean_fp32_distance,
                    neon::squared_euclidean_fp32_distance, a.data(), b.data(),
                    dim, 1e-5f);
    CompareDistance(scalar::cosine_fp32_distance, neon::cosine_fp32_distance,
                    a.data(), b.data(), dim, 1e-5f);
  }
#endif
}

TEST(NeonDistance, Fp32BatchPreservesScalarBits) {
  const int rounding = std::fegetround();
  auto restore = ailego::ScopeGuard::Make([&]() { std::fesetround(rounding); });
  const RawDistanceFn points[] = {neon::squared_euclidean_fp32_distance,
                                  neon::inner_product_fp32_distance,
                                  neon::cosine_fp32_distance};
  const decltype(&neon::squared_euclidean_fp32_batch_distance) batches[] = {
      neon::squared_euclidean_fp32_batch_distance,
      neon::inner_product_fp32_batch_distance,
      neon::cosine_fp32_batch_distance};
  std::mt19937 generator(1927);
  std::uniform_real_distribution<float> random(-3.2f, 3.2f);
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(0, std::fesetround(mode));
    for (float scale : {1.f, 1e10f, 1e-20f, 1e20f}) {
      std::vector<float> query(770);
      for (auto &v : query) v = random(generator) * scale;
      std::vector<std::vector<float>> rows(9, std::vector<float>(770));
      std::vector<const void *> pointers;
      for (auto &row : rows) {
        for (auto &v : row) v = random(generator) * scale;
        pointers.push_back(row.data() + 1);
      }
      rows[0] = query;
      pointers[0] = rows[0].data() + 1;
      rows[1][3] = -0.f;
      rows[2][2] = std::numeric_limits<float>::infinity();
      rows[3][1] = std::numeric_limits<float>::quiet_NaN();
      for (size_t dim : {0U, 1U, 7U, 8U, 9U, 16U, 17U, 31U, 32U, 768U, 769U}) {
        for (size_t count : {0U, 1U, 3U, 4U, 5U, 7U, 8U, 9U}) {
          std::vector<float> expected(count + 1, 12345.f), actual = expected;
          for (size_t metric = 0; metric < 3; ++metric) {
            for (size_t i = 0; i < count; ++i)
              points[metric](pointers[i], query.data() + 1, dim, &expected[i]);
            batches[metric](pointers.data(), query.data() + 1, count, dim,
                            actual.data(), nullptr);
            EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(),
                                     actual.size() * sizeof(float)))
                << "metric=" << metric << " dim=" << dim << " count=" << count
                << " scale=" << scale;
          }
        }
      }
    }
  }
}

TEST(NeonDistance, Fp16DotBatchPreservesScalarBits) {
#if !defined(AILEGO_ARM64_GNU_LIKE)
  GTEST_SKIP() << "NEON FP16 conversion requires AArch64";
#else
  const int rounding = std::fegetround();
  auto restore = ailego::ScopeGuard::Make([&]() { std::fesetround(rounding); });
  std::mt19937 generator(6211);
  std::vector<uint16_t> query(1538);
  std::vector<std::vector<uint16_t>> rows(9, query);
  for (auto &value : query) value = generator() % 0x7c00;
  std::vector<const void *> pointers;
  for (auto &row : rows) {
    for (auto &value : row)
      value = (generator() % 0x7c00) | (generator() & 0x8000);
    pointers.push_back(row.data() + 1);
  }
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(0, std::fesetround(mode));
    for (uint16_t special : {0U, 0x8000U, 0x7c00U, 0x7e11U, 0x7c01U}) {
      rows[0][1] = special;
      for (size_t dim : {0U, 1U, 3U, 4U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U,
                         33U, 768U, 769U, 1536U}) {
        for (size_t count : {0U, 1U, 3U, 4U, 5U, 7U, 8U, 9U}) {
          std::vector<float> expected(count + 1, 12345.f), actual = expected;
          for (bool cosine : {false, true}) {
            auto point = cosine ? neon::cosine_fp16_distance
                                : neon::inner_product_fp16_distance;
            auto batch = cosine ? neon::cosine_fp16_batch_distance
                                : neon::inner_product_fp16_batch_distance;
            for (size_t i = 0; i < count; ++i)
              point(pointers[i], query.data() + 1, dim, &expected[i]);
            batch(pointers.data(), query.data() + 1, count, dim, actual.data(),
                  nullptr);
            EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(),
                                     actual.size() * sizeof(float)))
                << "dim=" << dim << " count=" << count << " half=" << special;
          }
        }
      }
    }
  }
#endif
}

TEST(NeonDistance, Fp32SplitDotPreservesPointBits) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  const int rounding = std::fegetround();
  auto restore = ailego::ScopeGuard::Make([&]() { std::fesetround(rounding); });
  std::mt19937 generator(2087);
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(0, std::fesetround(mode));
    for (size_t dim : {8U, 16U, 32U, 768U}) {
      std::vector<float> query(dim + 1);
      for (auto &value : query)
        value = (static_cast<int>(generator() % 10001) - 5000) * .001f;
      std::vector<std::vector<float>> rows(9, std::vector<float>(dim)),
          first(9, std::vector<float>(dim + 2)),
          second(9, std::vector<float>(dim + 1));
      for (auto &row : rows)
        for (auto &value : row)
          value = (static_cast<int>(generator() % 10001) - 5000) * .001f;
      rows[0][0] = -0.f;
      rows[1][1] = std::numeric_limits<float>::infinity();
      rows[2][2] = std::numeric_limits<float>::quiet_NaN();
      const uint32_t signaling_nan = 0x7fa12345, negative_nan = 0xffc12345;
      std::memcpy(&rows[3][3], &signaling_nan, sizeof(float));
      std::memcpy(&rows[4][4], &negative_nan, sizeof(float));
      std::fill(rows[5].begin(), rows[5].end(),
                std::numeric_limits<float>::denorm_min());
      std::fill(rows[6].begin(), rows[6].end(),
                std::numeric_limits<float>::max() / 4);
      std::vector<SplitVector> spans(9);
      for (size_t prefix = 0; prefix <= dim; ++prefix) {
        for (size_t row = 0; row < rows.size(); ++row) {
          size_t cut = (prefix + row * 3) % (dim + 1);
          std::fill(first[row].begin(), first[row].end(), 99.f);
          std::copy_n(rows[row].data(), cut, first[row].data() + 1);
          std::copy(rows[row].begin() + cut, rows[row].end(),
                    second[row].begin() + 1);
          spans[row] = {first[row].data() + 1, second[row].data() + 1,
                        (cut == dim ? cut + 1 : cut) * sizeof(float)};
        }
        for (size_t count : {0U, 1U, 3U, 4U, 5U, 7U, 8U, 9U}) {
          for (bool cosine : {false, true}) {
            auto point = cosine ? neon::cosine_fp32_distance
                                : neon::inner_product_fp32_distance;
            auto split = cosine ? neon::cosine_fp32_split_batch_distance
                                : neon::inner_product_fp32_split_batch_distance;
            std::vector<float> expected(count + 1, 12345.f), actual = expected;
            for (size_t row = 0; row < count; ++row)
              point(rows[row].data(), query.data() + 1, dim, &expected[row]);
            ASSERT_TRUE(split(spans.data(), count, query.data() + 1, dim,
                              actual.data(), 0.f));
            EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(),
                                     actual.size() * sizeof(float)))
                << "dim=" << dim << " prefix=" << prefix << " count=" << count;
          }
        }
      }
      SplitVector invalid[5];
      std::fill_n(invalid, 5,
                  SplitVector{rows[0].data(), nullptr, dim * sizeof(float)});
      invalid[4].first_bytes = sizeof(float);
      float untouched[5] = {12345.f, 12345.f, 12345.f, 12345.f, 12345.f};
      EXPECT_FALSE(neon::inner_product_fp32_split_batch_distance(
          invalid, 5, query.data(), dim, untouched, 0.f));
      for (float value : untouched) EXPECT_EQ(12345.f, value);
      invalid[4].second = rows[0].data();
      invalid[4].first_bytes = 1;
      EXPECT_FALSE(neon::cosine_fp32_split_batch_distance(
          invalid, 5, query.data(), dim, untouched, 0.f));
      EXPECT_FALSE(neon::cosine_fp32_split_batch_distance(
          spans.data(), 5, query.data(), 7, untouched, 0.f));
      for (float value : untouched) EXPECT_EQ(12345.f, value);
    }
  }
#endif
}

TEST(NeonDistance, Fp32SplitPreservesScalarBits) {
  std::mt19937 generator(637);
  std::uniform_real_distribution<float> random(-3.f, 3.f);
  for (size_t dim : {0U, 8U, 9U, 16U, 768U, 769U}) {
    std::vector<std::vector<float>> rows(9, std::vector<float>(dim + 1));
    std::vector<float> query(dim + 1);
    for (auto &row : rows)
      for (auto &v : row) v = random(generator);
    for (auto &v : query) v = random(generator);
    for (size_t prefix = 0; prefix <= dim; prefix += 4) {
      std::vector<std::vector<float>> first(9), second(9);
      std::vector<SplitVector> vectors;
      for (size_t i = 0; i < rows.size(); ++i) {
        const size_t cut = std::min(dim, (prefix + i * 8) % (dim + 8));
        first[i].assign(rows[i].begin(), rows[i].begin() + cut + 1);
        second[i].assign(rows[i].begin() + cut, rows[i].end());
        vectors.push_back({first[i].data() + 1, second[i].data() + 1,
                           cut * sizeof(float)});
      }
      for (size_t count : {0U, 1U, 3U, 4U, 5U, 7U, 8U, 9U}) {
        bool supported = false;
#if defined(AILEGO_ARM64_NEON)
        supported = dim % 8 == 0 && (count == 0 || prefix % 8 == 0);
#endif
        std::vector<float> expected(count + 1, 12345.f), actual = expected;
        if (supported)
          for (size_t i = 0; i < count; ++i)
            neon::squared_euclidean_fp32_distance(
                rows[i].data() + 1, query.data() + 1, dim, &expected[i]);
        EXPECT_EQ(supported,
                  neon::squared_euclidean_fp32_bounded_split_batch_distance(
                      vectors.data(), count, query.data() + 1, dim, actual.data(),
                      std::numeric_limits<float>::infinity()));
        EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(),
                                actual.size() * sizeof(float)))
            << "dim=" << dim << " prefix=" << prefix << " count=" << count;
      }
    }
    float untouched[2] = {12345.f, 12345.f};
    SplitVector invalid[] = {
        {rows[0].data(), nullptr, dim * sizeof(float)},
        {rows[0].data(), rows[0].data(), (dim + 8) * sizeof(float)}};
    EXPECT_FALSE(neon::squared_euclidean_fp32_bounded_split_batch_distance(
        invalid, 2, query.data(), dim, untouched,
        std::numeric_limits<float>::infinity()));
    EXPECT_EQ(12345.f, untouched[0]);
    EXPECT_EQ(12345.f, untouched[1]);
  }
}

template <typename T>
void check_bounded_distances(
    size_t stride, SplitDistanceQuantizer::Distance bounded,
    void (*ordinary)(const void *, const void *, size_t, float *)) {
  auto encode = [](float value) {
    if constexpr (std::is_same_v<T, float>) return value;
    else {
      uint16_t half;
      ailego::FloatHelper::ToFP16(&value, 1, &half);
      return half;
    }
  };
  for (size_t dim : {stride, size_t(64), 64 + stride, size_t(128), size_t(256), size_t(768)}) {
    std::vector<T> query(dim, encode(.25f));
    std::vector<std::vector<T>> rows(49, std::vector<T>(dim, encode(2.f)));
    rows[4] = query;
    rows[5] = query;
    rows[5].back() = encode(.2501f);
    rows[2].back() = encode(std::numeric_limits<float>::quiet_NaN());
    rows[3].back() = encode(std::numeric_limits<float>::infinity());
    for (size_t row : {32U, 37U, 42U, 47U}) rows[row] = query;
    rows[37].back() = encode(.273f);
    rows[42].back() = encode(std::numeric_limits<float>::quiet_NaN());
    for (size_t d = 0; d < dim; ++d)
      rows[47][d] = encode(.25f + .005f * (int(d % 7) - 3));
    std::vector<SplitVector> views;
    std::vector<float> expected(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      const size_t prefix = (i * stride) % (dim + stride);
      views.push_back({rows[i].data(), rows[i].data() + prefix,
                        prefix * sizeof(T)});
      ordinary(rows[i].data(), query.data(), dim, &expected[i]);
    }
    for (float cutoff : {0.f, .00001f, 1.f, -1.f,
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN()}) {
      for (size_t count : {0U, 1U, 3U, 4U, 5U, 8U, 9U, 32U, 33U, 37U, 41U, 49U}) {
        std::vector<float> actual(count + 1, 12345.f);
        ASSERT_TRUE(bounded(
            views.data(), count, query.data(), dim, actual.data(), cutoff));
        for (size_t i = 0; i < count; ++i) {
          if (!std::isfinite(cutoff) || expected[i] < cutoff)
            EXPECT_EQ(0, std::memcmp(&expected[i], &actual[i], sizeof(float)));
          else
            EXPECT_TRUE(std::isnan(actual[i]) || actual[i] >= cutoff);
        }
        EXPECT_EQ(12345.f, actual[count]);
        if (count >= 4 && dim > 64 && cutoff == 1.f)
          EXPECT_EQ(std::numeric_limits<float>::infinity(), actual[0]);
        if (count >= 37 && dim >= 256 && cutoff == 1.f)
          EXPECT_EQ(std::numeric_limits<float>::infinity(), actual[33]);
      }
    }
  }
}

TEST(NeonDistance, BoundedFp32KeepsEveryPossibleWinnerExact) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP();
#endif
  check_bounded_distances<float>(
      8, neon::squared_euclidean_fp32_bounded_split_batch_distance,
      neon::squared_euclidean_fp32_distance);
}

TEST(NeonDistance, BoundedFp16KeepsEveryPossibleWinnerExact) {
  const auto kernels = get_distance_kernels(
      MetricType::kSquaredEuclidean, DataType::kFp16, QuantizeType::kFp16,
      CpuArchType::kAuto);
  const auto *kernel = kernels.dist.target<
      decltype(&neon_fp16::squared_euclidean_fp16_distance_neon_fp16)>();
  if (!kernel ||
      *kernel != neon_fp16::squared_euclidean_fp16_distance_neon_fp16)
    GTEST_SKIP();
  check_bounded_distances<uint16_t>(
      16, neon_fp16::squared_euclidean_fp16_split_batch_distance,
      neon_fp16::squared_euclidean_fp16_distance_neon_fp16);
}

TEST(NeonDistance, Fp16MatchesScalarForRemainders) {
#if !defined(AILEGO_ARM64_GNU_LIKE)
  GTEST_SKIP() << "FP16 NEON distance kernels require GNU-like AArch64";
#else
  std::vector<float> a_fp32(33);
  std::vector<float> b_fp32(33);
  for (size_t i = 0; i < a_fp32.size(); ++i) {
    a_fp32[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b_fp32[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }
  std::vector<uint16_t> a(a_fp32.size());
  std::vector<uint16_t> b(b_fp32.size());
  ailego::FloatHelper::ToFP16(a_fp32.data(), a_fp32.size(), a.data());
  ailego::FloatHelper::ToFP16(b_fp32.data(), b_fp32.size(), b.data());

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(scalar::inner_product_fp16_distance,
                    neon::inner_product_fp16_distance, a.data(), b.data(), dim,
                    2e-3f);
    CompareDistance(scalar::squared_euclidean_fp16_distance,
                    neon::squared_euclidean_fp16_distance, a.data(), b.data(),
                    dim, 2e-3f);
    CompareDistance(scalar::cosine_fp16_distance, neon::cosine_fp16_distance,
                    a.data(), b.data(), dim, 2e-3f);
  }
#endif
}

TEST(NeonDistance, Fp16AccumulatesInFp32) {
#if !defined(AILEGO_ARM64_GNU_LIKE)
  GTEST_SKIP() << "FP16 NEON distance kernels require GNU-like AArch64";
#else
  constexpr size_t dim = 8;
  std::vector<float> a_fp32(dim, 256.0f);
  std::vector<float> b_fp32(dim, 256.0f);
  std::vector<float> zero_fp32(dim, 0.0f);
  std::vector<uint16_t> a(dim);
  std::vector<uint16_t> b(dim);
  std::vector<uint16_t> zero(dim);
  ailego::FloatHelper::ToFP16(a_fp32.data(), dim, a.data());
  ailego::FloatHelper::ToFP16(b_fp32.data(), dim, b.data());
  ailego::FloatHelper::ToFP16(zero_fp32.data(), dim, zero.data());

  CompareDistance(scalar::inner_product_fp16_distance,
                  neon::inner_product_fp16_distance, a.data(), b.data(), dim,
                  0.0f);
  CompareDistance(scalar::squared_euclidean_fp16_distance,
                  neon::squared_euclidean_fp16_distance, a.data(), zero.data(),
                  dim, 0.0f);
#endif
}

TEST(NeonDistance, RecordIntegerProductsMatchScalarForRemainders) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  std::vector<int8_t> int8_a(97);
  std::vector<int8_t> int8_b(97);
  for (size_t i = 0; i < int8_a.size(); ++i) {
    int8_a[i] = static_cast<int8_t>((i * 37) % 255 - 127);
    int8_b[i] = static_cast<int8_t>((i * 53) % 255 - 127);
  }
  for (size_t dim = 0; dim <= int8_a.size(); ++dim) {
    EXPECT_FLOAT_EQ(
        scalar::internal::ip_int8_scalar(int8_a.data(), int8_b.data(), dim),
        neon::internal::ip_int8_neon(int8_a.data(), int8_b.data(), dim));
  }

  std::vector<uint8_t> int4_a(49);
  std::vector<uint8_t> int4_b(49);
  for (size_t i = 0; i < int4_a.size(); ++i) {
    int4_a[i] = static_cast<uint8_t>((i * 67) & 0xff);
    int4_b[i] = static_cast<uint8_t>((i * 89) & 0xff);
  }
  for (size_t dim = 0; dim <= int4_a.size() * 2; dim += 2) {
    EXPECT_FLOAT_EQ(
        scalar::internal::ip_int4_scalar(int4_a.data(), int4_b.data(), dim),
        neon::internal::ip_int4_neon(int4_a.data(), int4_b.data(), dim));
  }
#endif
}

TEST(NeonDistance, DispatchProvidesAllKernels) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  constexpr std::array<MetricType, 3> metrics = {MetricType::kSquaredEuclidean,
                                                 MetricType::kCosine,
                                                 MetricType::kInnerProduct};
  for (MetricType metric : metrics) {
    const auto fp32 = get_distance_kernels(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kNEON);
    EXPECT_TRUE(fp32.dist);
    EXPECT_TRUE(fp32.batch);

    const auto fp16 = get_distance_kernels(
        metric, DataType::kFp16, QuantizeType::kFp16, CpuArchType::kNEON);
    EXPECT_TRUE(fp16.dist);
    EXPECT_TRUE(fp16.batch);

    const auto int8 = get_distance_kernels(
        metric, DataType::kInt8, QuantizeType::kRecord, CpuArchType::kNEON);
    EXPECT_TRUE(int8.dist);
    EXPECT_TRUE(int8.batch);

    const auto int4 = get_distance_kernels(
        metric, DataType::kInt4, QuantizeType::kRecord, CpuArchType::kNEON);
    EXPECT_TRUE(int4.dist);
    EXPECT_TRUE(int4.batch);
  }
#endif
}

}  // namespace
}  // namespace zvec::turbo
