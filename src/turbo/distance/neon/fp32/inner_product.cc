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

#include "neon/fp32/inner_product.h"
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <zvec/ailego/internal/platform.h>

#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#else
#include "scalar/fp32/inner_product.h"
#endif

namespace zvec::turbo::neon {

void inner_product_fp32_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(AILEGO_ARM64_NEON)
  const float *lhs = reinterpret_cast<const float *>(a);
  const float *rhs = reinterpret_cast<const float *>(b);
  const float *last = lhs + dim;
  const float *last_aligned = lhs + ((dim >> 3) << 3);

  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    sum0 = vfmaq_f32(sum0, vld1q_f32(lhs), vld1q_f32(rhs));
    sum1 = vfmaq_f32(sum1, vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
  }
  if (last - lhs >= 4) {
    sum0 = vfmaq_f32(sum0, vld1q_f32(lhs), vld1q_f32(rhs));
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(sum0, sum1));
  for (; lhs != last; ++lhs, ++rhs) {
    result += *lhs * *rhs;
  }
  *distance = -result;
#else
  scalar::inner_product_fp32_distance(a, b, dim, distance);
#endif
}

void inner_product_fp32_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const * /*extra_values*/) {
  size_t i = 0;
#if defined(AILEGO_ARM64_NEON)
  const auto *q = static_cast<const float *>(query);
  for (; n - i >= 4; i += 4) {
    const float *rows[4];
    float32x4_t sum0[4] = {}, sum1[4] = {};
    for (size_t row = 0; row < 4; ++row)
      rows[row] = static_cast<const float *>(vectors[i + row]);
    size_t d = 0;
    // Interleave rows without changing either accumulator's operation order.
    for (; d + 8 <= dim; d += 8) {
      const auto q0 = vld1q_f32(q + d), q1 = vld1q_f32(q + d + 4);
      for (size_t row = 0; row < 4; ++row) {
        sum0[row] = vfmaq_f32(sum0[row], vld1q_f32(rows[row] + d), q0);
        sum1[row] = vfmaq_f32(sum1[row], vld1q_f32(rows[row] + d + 4), q1);
      }
    }
    if (d + 4 <= dim) {
      const auto q0 = vld1q_f32(q + d);
      for (size_t row = 0; row < 4; ++row)
        sum0[row] = vfmaq_f32(sum0[row], vld1q_f32(rows[row] + d), q0);
      d += 4;
    }
    float totals[4];
    for (size_t row = 0; row < 4; ++row)
      totals[row] = vaddvq_f32(vaddq_f32(sum0[row], sum1[row]));
    for (; d < dim; ++d)
      for (size_t row = 0; row < 4; ++row)
        totals[row] += rows[row][d] * q[d];
    for (size_t row = 0; row < 4; ++row)
      distances[i + row] = -totals[row];
  }
#endif
  for (; i < n; ++i) {
    inner_product_fp32_distance(vectors[i], query, dim, &distances[i]);
  }
}

bool inner_product_fp32_split_batch_distance(const SplitVector *vectors,
                                             size_t count, const void *query,
                                             size_t dim, float *distances,
                                             float /*cutoff*/) {
#if defined(AILEGO_ARM64_NEON)
  if (dim % 8) return false;
  for (size_t i = 0; i < count; ++i)
    if (vectors[i].first_bytes % sizeof(float) ||
        (vectors[i].first_bytes / sizeof(float) < dim && !vectors[i].second))
      return false;
  const auto *q = static_cast<const float *>(query);
  auto compute = [&](auto row_count, size_t first) {
    constexpr size_t rows = decltype(row_count)::value;
    const float *data[rows];
    size_t cuts[rows];
    float32x4_t sum0[rows] = {}, sum1[rows] = {};
    for (size_t row = 0; row < rows; ++row) {
      data[row] = static_cast<const float *>(vectors[first + row].first);
      cuts[row] =
          std::min(dim, vectors[first + row].first_bytes / sizeof(float));
    }
    size_t d = 0;
    while (d < dim) {
      size_t end = dim;
      for (size_t row = 0; row < rows; ++row) {
        if (cuts[row] == d) {
          data[row] = static_cast<const float *>(vectors[first + row].second);
          cuts[row] = dim;
        }
        end = std::min(end, cuts[row] & ~size_t(7));
      }
      for (; d < end; d += 8) {
        const auto q0 = vld1q_f32(q + d), q1 = vld1q_f32(q + d + 4);
        for (size_t row = 0; row < rows; ++row) {
          sum0[row] = vfmaq_f32(sum0[row], vld1q_f32(data[row]), q0);
          sum1[row] = vfmaq_f32(sum1[row], vld1q_f32(data[row] + 4), q1);
          data[row] += 8;
        }
      }
      if (d == dim) break;
      const auto q0 = vld1q_f32(q + d), q1 = vld1q_f32(q + d + 4);
      for (size_t row = 0; row < rows; ++row) {
        float boundary[8];
        const float *values = data[row];
        // Gather only the SIMD chunk crossing the page, preserving both chains.
        if (cuts[row] < d + 8) {
          const size_t prefix = cuts[row] - d;
          const auto *second =
              static_cast<const float *>(vectors[first + row].second);
          std::memcpy(boundary, data[row], prefix * sizeof(float));
          std::memcpy(boundary + prefix, second, (8 - prefix) * sizeof(float));
          values = boundary;
          data[row] = second + 8 - prefix;
          cuts[row] = dim;
        } else {
          data[row] += 8;
        }
        sum0[row] = vfmaq_f32(sum0[row], vld1q_f32(values), q0);
        sum1[row] = vfmaq_f32(sum1[row], vld1q_f32(values + 4), q1);
      }
      d += 8;
    }
    for (size_t row = 0; row < rows; ++row)
      distances[first + row] = -vaddvq_f32(vaddq_f32(sum0[row], sum1[row]));
  };
  size_t i = 0;
  for (; count - i >= 4; i += 4)
    compute(std::integral_constant<size_t, 4>{}, i);
  for (; i < count; ++i) compute(std::integral_constant<size_t, 1>{}, i);
  return true;
#else
  (void)vectors;
  (void)count;
  (void)query;
  (void)dim;
  (void)distances;
  return false;
#endif
}

}  // namespace zvec::turbo::neon
