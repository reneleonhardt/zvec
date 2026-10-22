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

#include "neon_fp16/fp16/squared_euclidean.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_NEON
#include <arm_neon.h>
#include "neon_fp16/fp16/fp32_batch.h"
#else
#include "scalar/fp16/squared_euclidean.h"
#endif

namespace zvec::turbo::neon_fp16 {

#if ZVEC_TURBO_FP16_NEON
namespace {

template <bool Bounded>
bool split_batch_distance(const SplitVector *vectors, size_t count,
                           const void *query, size_t dim, float *distances,
                           float cutoff) {
  if (dim % 16) return false;
  for (size_t i = 0; i < count; ++i)
    if (vectors[i].first_bytes % (16 * sizeof(float16_t)) ||
        vectors[i].first_bytes / sizeof(float16_t) > dim)
      return false;
  const auto *q = static_cast<const float16_t *>(query);
  using Accumulator = detail::Fp32DistanceAccumulator<true>;
  size_t i = 0;
  for (; count - i >= 4;) {
    const float16_t *data[4];
    size_t cuts[4];
    Accumulator sums[4];
    for (size_t row = 0; row < 4; ++row) {
      data[row] = static_cast<const float16_t *>(vectors[i + row].first);
      cuts[row] = vectors[i + row].first_bytes / sizeof(float16_t);
    }
    size_t d = 0;
    bool rejected = false;
    bool single_survivor = false;
    while (d < dim) {
      size_t end = dim;
      for (size_t row = 0; row < 4; ++row) {
        if (cuts[row] == d) {
          data[row] = static_cast<const float16_t *>(vectors[i + row].second);
          cuts[row] = dim;
        }
        end = std::min(end, cuts[row]);
      }
      if constexpr (Bounded)
        if (d < 64) end = std::min(end, size_t(64));
      for (; d < end; d += 16) {
        const auto query0 = vld1q_f16(q + d), query1 = vld1q_f16(q + d + 8);
        const auto q0 = vcvt_f32_f16(vget_low_f16(query0));
        const auto q1 = vcvt_high_f32_f16(query0);
        const auto q2 = vcvt_f32_f16(vget_low_f16(query1));
        const auto q3 = vcvt_high_f32_f16(query1);
        for (size_t row = 0; row < 4; ++row) {
          sums[row].add16(data[row], q0, q1, q2, q3);
          data[row] += 16;
        }
      }
      if constexpr (Bounded) {
        if (d == 64 && d < dim) {
          unsigned survivors = 0;
          for (size_t row = 0; row < 4; ++row)
            if (!(sums[row].reduce() >= cutoff)) survivors |= 1U << row;
          rejected = survivors == 0;
          if (rejected) break;
          if (dim >= 256 && (survivors & (survivors - 1)) == 0) {
            // For long rows, recompute the winner with the ordinary reduction.
            const size_t row = __builtin_ctz(survivors);
            float distance;
            split_batch_distance<false>(vectors + i + row, 1, query, dim,
                                        &distance, 0);
            for (size_t r = 0; r < 4; ++r)
              distances[i + r] = std::numeric_limits<float>::infinity();
            distances[i + row] = distance;
            single_survivor = true;
            break;
          }
        }
      }
    }
    if (!single_survivor)
      for (size_t row = 0; row < 4; ++row)
        distances[i + row] = rejected ? std::numeric_limits<float>::infinity()
                                      : sums[row].reduce();
    i += 4;
    if constexpr (Bounded) {
      if (!rejected && !single_survivor) {
        const size_t rest = std::min(count - i, (32 - i % 32) % 32);
        split_batch_distance<false>(vectors + i, rest, query, dim, distances + i, 0);
        i += rest;
      }
    }
  }
  for (; i < count; ++i) {
    Accumulator sum;
    const auto *rhs = q;
    auto accumulate = [&](const void *span, size_t length) {
      const auto *lhs = static_cast<const float16_t *>(span);
      for (size_t d = 0; d < length; d += 16, rhs += 16) {
        const auto q0 = vld1q_f16(rhs), q1 = vld1q_f16(rhs + 8);
        sum.add16(lhs + d, vcvt_f32_f16(vget_low_f16(q0)),
                  vcvt_high_f32_f16(q0), vcvt_f32_f16(vget_low_f16(q1)),
                  vcvt_high_f32_f16(q1));
      }
    };
    const size_t first = vectors[i].first_bytes / sizeof(float16_t);
    accumulate(vectors[i].first, first);
    accumulate(vectors[i].second, dim - first);
    distances[i] = sum.reduce();
  }
  return true;
}

// Widen before subtraction as well as multiplication and accumulation:
// even the difference between two finite FP16 inputs may exceed FP16 range.
// FP32 arithmetic also preserves small squared differences in long sums.
inline float squared_euclidean_fp16_accum(const float16_t *lhs,
                                          const float16_t *rhs, size_t dim) {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const float16x8_t lhs0 = vld1q_f16(lhs + i);
    const float16x8_t rhs0 = vld1q_f16(rhs + i);
    const float16x8_t lhs1 = vld1q_f16(lhs + i + 8);
    const float16x8_t rhs1 = vld1q_f16(rhs + i + 8);
    const float32x4_t d0 = vsubq_f32(vcvt_f32_f16(vget_low_f16(lhs0)),
                                     vcvt_f32_f16(vget_low_f16(rhs0)));
    const float32x4_t d1 =
        vsubq_f32(vcvt_high_f32_f16(lhs0), vcvt_high_f32_f16(rhs0));
    const float32x4_t d2 = vsubq_f32(vcvt_f32_f16(vget_low_f16(lhs1)),
                                     vcvt_f32_f16(vget_low_f16(rhs1)));
    const float32x4_t d3 =
        vsubq_f32(vcvt_high_f32_f16(lhs1), vcvt_high_f32_f16(rhs1));
    sum0 = vfmaq_f32(sum0, d0, d0);
    sum1 = vfmaq_f32(sum1, d1, d1);
    sum2 = vfmaq_f32(sum2, d2, d2);
    sum3 = vfmaq_f32(sum3, d3, d3);
  }
  if (i + 8 <= dim) {
    const float16x8_t lhs0 = vld1q_f16(lhs + i);
    const float16x8_t rhs0 = vld1q_f16(rhs + i);
    const float32x4_t d0 = vsubq_f32(vcvt_f32_f16(vget_low_f16(lhs0)),
                                     vcvt_f32_f16(vget_low_f16(rhs0)));
    const float32x4_t d1 =
        vsubq_f32(vcvt_high_f32_f16(lhs0), vcvt_high_f32_f16(rhs0));
    sum0 = vfmaq_f32(sum0, d0, d0);
    sum1 = vfmaq_f32(sum1, d1, d1);
    i += 8;
  }
  float total =
      vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  for (; i < dim; ++i) {
    float diff = static_cast<float>(lhs[i]) - static_cast<float>(rhs[i]);
    total += diff * diff;
  }
  return total;
}

}  // namespace
#endif

bool squared_euclidean_fp16_split_batch_distance(
    const SplitVector *vectors, size_t count, const void *query, size_t dim,
    float *distances, float cutoff) {
#if ZVEC_TURBO_FP16_NEON
  if (dim <= 64 || !std::isfinite(cutoff))
    return split_batch_distance<false>(vectors, count, query, dim, distances, 0);
  return split_batch_distance<true>(vectors, count, query, dim, distances, cutoff);
#else
  (void)vectors;
  (void)count;
  (void)query;
  (void)dim;
  (void)distances;
  (void)cutoff;
  return false;
#endif
}

// Compute squared euclidean distance between a single FP16 vector pair.
void squared_euclidean_fp16_distance_neon_fp16(const void *a, const void *b,
                                               size_t dim, float *distance) {
#if ZVEC_TURBO_FP16_NEON
  *distance =
      squared_euclidean_fp16_accum(reinterpret_cast<const float16_t *>(a),
                                   reinterpret_cast<const float16_t *>(b), dim);
#else
  // Compiled without FEAT_FP16 (e.g. no -march=armv8.2-a+fp16), so delegate
  // to the scalar kernel. Never leave `distance` unwritten: turbo.cc selects
  // these entry points from CpuFeatures flags, and a no-op here would
  // silently return whatever the caller's buffer already held.
  scalar::squared_euclidean_fp16_distance(a, b, dim, distance);
#endif
}

// Batch version of squared_euclidean_fp16_distance_neon_fp16.
void squared_euclidean_fp16_batch_distance_neon_fp16(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
#if ZVEC_TURBO_FP16_NEON
  (void)extra_values;
  const float16_t *typed_query = reinterpret_cast<const float16_t *>(query);
  size_t i = 0;
  for (; n - i >= 4; i += 4) {
    detail::fp32_distance_batch4<true>(vectors + i, typed_query, dim,
                                       distances + i);
  }
  for (; i < n; ++i) {
    distances[i] = squared_euclidean_fp16_accum(
        reinterpret_cast<const float16_t *>(vectors[i]), typed_query, dim);
  }
#else
  scalar::squared_euclidean_fp16_batch_distance(vectors, query, n, dim,
                                                distances, extra_values);
#endif
}

}  // namespace zvec::turbo::neon_fp16
