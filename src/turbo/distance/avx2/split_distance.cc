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

#include "avx2/split_distance.h"

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace zvec::turbo::avx2 {
namespace {

template <typename T>
__m256 load(const T *data) {
  if constexpr (std::is_same_v<T, float>)
    return _mm256_loadu_ps(data);
  else
    return _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(data)));
}

template <typename T>
__m256 load(const SplitVector &vector, size_t d) {
  const size_t cut = vector.first_bytes / sizeof(T);
  if (d + 8 <= cut) return load(static_cast<const T *>(vector.first) + d);
  if (d >= cut) return load(static_cast<const T *>(vector.second) + d - cut);
  T joined[8];
  const size_t first = cut - d;
  std::memcpy(joined, static_cast<const T *>(vector.first) + d,
              first * sizeof(T));
  std::memcpy(joined + first, vector.second, (8 - first) * sizeof(T));
  return load(joined);
}

template <typename T, bool Dot>
float reduce(__m256 value) {
  if constexpr (Dot && !std::is_same_v<T, float>) {
    // FP16 dot products add within each 128-bit half before combining halves.
    value = _mm256_hadd_ps(value, value);
    value = _mm256_hadd_ps(value, value);
    return _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(value),
                                    _mm256_extractf128_ps(value, 1)));
  } else {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                            _mm256_extractf128_ps(value, 1));
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
  }
}

template <typename T, bool Dot>
bool split_batch(const SplitVector *vectors, size_t count, const void *query,
                 size_t dim, float *distances, float cutoff) {
  if (dim % 8) return false;
  for (size_t i = 0; i < count; ++i) {
    const size_t cut = vectors[i].first_bytes / sizeof(T);
    if (vectors[i].first_bytes % sizeof(T) || (cut && !vectors[i].first) ||
        (cut < dim && !vectors[i].second))
      return false;
  }
  const auto *q = static_cast<const T *>(query);
  for (size_t i = 0; i < count;) {
    const size_t rows = std::min<size_t>(4, count - i);
    __m256 sums[4];
    for (size_t row = 0; row < rows; ++row) sums[row] = _mm256_setzero_ps();
    bool rejected = false;
    for (size_t d = 0; d < dim; d += 8) {
      const __m256 rhs = load(q + d);
      for (size_t row = 0; row < rows; ++row) {
        const __m256 lhs = load<T>(vectors[i + row], d);
        if constexpr (Dot) {
          sums[row] = _mm256_fmadd_ps(rhs, lhs, sums[row]);
        } else {
          const __m256 diff = _mm256_sub_ps(rhs, lhs);
          sums[row] = _mm256_fmadd_ps(diff, diff, sums[row]);
        }
      }
      if constexpr (!Dot) {
        if (d == 56 && dim > 64 && rows == 4 && std::isfinite(cutoff)) {
          rejected = true;
          for (size_t row = 0; row < rows; ++row)
            rejected &= reduce<T, false>(sums[row]) >= cutoff;
          if (rejected) break;
        }
      }
    }
    for (size_t row = 0; row < rows; ++row) {
      const float sum = reduce<T, Dot>(sums[row]);
      distances[i + row] = rejected ? std::numeric_limits<float>::infinity()
                                    : (Dot ? -sum : sum);
      uint32_t bits;
      std::memcpy(&bits, &distances[i + row], sizeof(bits));
      if ((bits & 0x7fffffffU) > 0x7f800000U) return false;
    }
    i += rows;
  }
  return true;
}

}  // namespace
}  // namespace zvec::turbo::avx2
#endif

namespace zvec::turbo::avx2 {

bool squared_euclidean_fp32_split_batch(const SplitVector *vectors,
                                        size_t count, const void *query,
                                        size_t dim, float *distances,
                                        float cutoff) {
#if defined(__AVX2__) && defined(__F16C__)
  return split_batch<float, false>(vectors, count, query, dim, distances,
                                   cutoff);
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

bool inner_product_fp32_split_batch(const SplitVector *vectors, size_t count,
                                    const void *query, size_t dim,
                                    float *distances, float cutoff) {
#if defined(__AVX2__) && defined(__F16C__)
  return split_batch<float, true>(vectors, count, query, dim, distances,
                                  cutoff);
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

bool cosine_fp32_split_batch(const SplitVector *vectors, size_t count,
                             const void *query, size_t dim, float *distances,
                             float cutoff) {
#if defined(__AVX2__) && defined(__F16C__)
  if (!split_batch<float, true>(vectors, count, query, dim, distances, cutoff))
    return false;
  for (size_t i = 0; i < count; ++i) distances[i] += 1.0f;
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

bool squared_euclidean_fp16_split_batch(const SplitVector *vectors,
                                        size_t count, const void *query,
                                        size_t dim, float *distances,
                                        float cutoff) {
#if defined(__AVX2__) && defined(__F16C__)
  return split_batch<uint16_t, false>(vectors, count, query, dim, distances,
                                      cutoff);
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

bool inner_product_fp16_split_batch(const SplitVector *vectors, size_t count,
                                    const void *query, size_t dim,
                                    float *distances, float cutoff) {
#if defined(__AVX2__) && defined(__F16C__)
  return split_batch<uint16_t, true>(vectors, count, query, dim, distances,
                                     cutoff);
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

bool cosine_fp16_split_batch(const SplitVector *vectors, size_t count,
                             const void *query, size_t dim, float *distances,
                             float cutoff) {
#if defined(__AVX2__) && defined(__F16C__)
  if (!split_batch<uint16_t, true>(vectors, count, query, dim, distances,
                                   cutoff))
    return false;
  for (size_t i = 0; i < count; ++i) distances[i] += 1.0f;
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

}  // namespace zvec::turbo::avx2
