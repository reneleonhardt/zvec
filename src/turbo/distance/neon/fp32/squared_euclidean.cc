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

#include "neon/fp32/squared_euclidean.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <zvec/ailego/internal/platform.h>

#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#else
#include "scalar/fp32/squared_euclidean.h"
#endif

namespace zvec::turbo::neon {

void squared_euclidean_fp32_distance(const void *a, const void *b, size_t dim,
                                     float *distance) {
#if defined(AILEGO_ARM64_NEON)
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);
  const float *last = lhs + dim;
  const float *last_aligned = lhs + ((dim >> 3) << 3);

  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    const float32x4_t diff0 = vsubq_f32(vld1q_f32(lhs), vld1q_f32(rhs));
    const float32x4_t diff1 = vsubq_f32(vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
    sum0 = vfmaq_f32(sum0, diff0, diff0);
    sum1 = vfmaq_f32(sum1, diff1, diff1);
  }
  if (last - lhs >= 4) {
    const float32x4_t diff = vsubq_f32(vld1q_f32(lhs), vld1q_f32(rhs));
    sum0 = vfmaq_f32(sum0, diff, diff);
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(sum0, sum1));
  for (; lhs != last; ++lhs, ++rhs) {
    const float diff = *lhs - *rhs;
    result += diff * diff;
  }
  *distance = result;
#else
  scalar::squared_euclidean_fp32_distance(a, b, dim, distance);
#endif
}

namespace {

template <bool Bounded>
bool split_batch_distance(
    const SplitVector *vectors, size_t count, const void *query, size_t dim,
    float *distances, float cutoff) {
#if defined(AILEGO_ARM64_NEON)
  if (dim % 8) return false;
  for (size_t i = 0; i < count; ++i)
    if (vectors[i].first_bytes % (8 * sizeof(float)) ||
        vectors[i].first_bytes / sizeof(float) > dim)
      return false;
  const auto *q = static_cast<const float *>(query);
  size_t i = 0;
  for (; count - i >= 4;) {
    const float *data[4];
    size_t cuts[4];
    float32x4_t sum0[4], sum1[4];
    for (size_t row = 0; row < 4; ++row) {
      data[row] = static_cast<const float *>(vectors[i + row].first);
      cuts[row] = vectors[i + row].first_bytes / sizeof(float);
      sum0[row] = sum1[row] = vdupq_n_f32(0);
    }
    size_t d = 0;
    bool rejected = false;
    bool single_survivor = false;
    // Switch page spans without restarting any row's accumulator sequence.
    while (d < dim) {
      size_t end = dim;
      for (size_t row = 0; row < 4; ++row) {
        if (cuts[row] == d) {
          data[row] = static_cast<const float *>(vectors[i + row].second);
          cuts[row] = dim;
        }
        end = std::min(end, cuts[row]);
      }
      if constexpr (Bounded)
        if (d < 64) end = std::min(end, size_t(64));
      for (; d < end; d += 8) {
        const auto q0 = vld1q_f32(q + d);
        const auto q1 = vld1q_f32(q + d + 4);
        for (size_t row = 0; row < 4; ++row) {
          const auto diff0 = vsubq_f32(vld1q_f32(data[row]), q0);
          const auto diff1 = vsubq_f32(vld1q_f32(data[row] + 4), q1);
          sum0[row] = vfmaq_f32(sum0[row], diff0, diff0);
          sum1[row] = vfmaq_f32(sum1[row], diff1, diff1);
          data[row] += 8;
        }
      }
      if constexpr (Bounded) {
        if (d == 64 && d < dim) {
          unsigned survivors = 0;
          for (size_t row = 0; row < 4; ++row)
            if (!(vaddvq_f32(vaddq_f32(sum0[row], sum1[row])) >= cutoff))
              survivors |= 1U << row;
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
                                      : vaddvq_f32(vaddq_f32(sum0[row], sum1[row]));
    i += 4;
    if constexpr (Bounded) {
      // Retry at the next 32-row chunk when one group could not be rejected.
      if (!rejected && !single_survivor) {
        const size_t rest = std::min(count - i, (32 - i % 32) % 32);
        split_batch_distance<false>(vectors + i, rest, query, dim, distances + i, 0);
        i += rest;
      }
    }
  }
  for (; i < count; ++i) {
    float32x4_t sum0 = vdupq_n_f32(0), sum1 = vdupq_n_f32(0);
    const auto *rhs = q;
    auto accumulate = [&](const void *span, size_t length) {
      const auto *lhs = static_cast<const float *>(span);
      for (size_t d = 0; d < length; d += 8, rhs += 8) {
        const auto diff0 = vsubq_f32(vld1q_f32(lhs + d), vld1q_f32(rhs));
        const auto diff1 = vsubq_f32(vld1q_f32(lhs + d + 4), vld1q_f32(rhs + 4));
        sum0 = vfmaq_f32(sum0, diff0, diff0);
        sum1 = vfmaq_f32(sum1, diff1, diff1);
      }
    };
    const size_t first_dim = vectors[i].first_bytes / sizeof(float);
    accumulate(vectors[i].first, first_dim);
    accumulate(vectors[i].second, dim - first_dim);
    distances[i] = vaddvq_f32(vaddq_f32(sum0, sum1));
  }
  return true;
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

}  // namespace

bool squared_euclidean_fp32_split_batch_distance(
    const SplitVector *vectors, size_t count, const void *query, size_t dim,
    float *distances) {
  return split_batch_distance<false>(vectors, count, query, dim, distances, 0);
}

bool squared_euclidean_fp32_bounded_split_batch_distance(
    const SplitVector *vectors, size_t count, const void *query, size_t dim,
    float *distances, float cutoff) {
  if (dim <= 64 || !std::isfinite(cutoff))
    return squared_euclidean_fp32_split_batch_distance(vectors, count, query, dim,
                                                      distances);
  return split_batch_distance<true>(vectors, count, query, dim, distances, cutoff);
}

void squared_euclidean_fp32_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const * /*extra_values*/) {
  size_t i = 0;
#if defined(AILEGO_ARM64_NEON)
  // Share query loads while retaining the scalar kernel's two accumulators.
  if (dim % 8 == 0) {
    const auto *q = static_cast<const float *>(query);
    for (; i + 4 <= n; i += 4) {
      float32x4_t sum0[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0),
                             vdupq_n_f32(0)};
      float32x4_t sum1[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0),
                             vdupq_n_f32(0)};
      for (size_t d = 0; d < dim; d += 8) {
        const auto q0 = vld1q_f32(q + d);
        const auto q1 = vld1q_f32(q + d + 4);
        for (size_t row = 0; row < 4; ++row) {
          const auto *data = static_cast<const float *>(vectors[i + row]) + d;
          const auto diff0 = vsubq_f32(vld1q_f32(data), q0);
          const auto diff1 = vsubq_f32(vld1q_f32(data + 4), q1);
          sum0[row] = vfmaq_f32(sum0[row], diff0, diff0);
          sum1[row] = vfmaq_f32(sum1[row], diff1, diff1);
        }
      }
      for (size_t row = 0; row < 4; ++row)
        distances[i + row] = vaddvq_f32(vaddq_f32(sum0[row], sum1[row]));
    }
  }
#endif
  for (; i < n; ++i) {
    squared_euclidean_fp32_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::neon
