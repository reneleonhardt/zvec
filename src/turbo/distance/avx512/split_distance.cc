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

#include "avx512/split_distance.h"

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace zvec::turbo::avx512 {
namespace {

template <typename T>
__m512 load(const T *data) {
  if constexpr (std::is_same_v<T, float>)
    return _mm512_loadu_ps(data);
  else
    return _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data)));
}

template <typename T>
__m512 load(const SplitVector &vector, size_t d) {
  const size_t cut = vector.first_bytes / sizeof(T);
  if (d + 16 <= cut) return load(static_cast<const T *>(vector.first) + d);
  if (d >= cut) return load(static_cast<const T *>(vector.second) + d - cut);
  T joined[16];
  const size_t first = cut - d;
  std::memcpy(joined, static_cast<const T *>(vector.first) + d,
              first * sizeof(T));
  std::memcpy(joined + first, vector.second, (16 - first) * sizeof(T));
  return load(joined);
}

template <bool Raw>
float reduce(__m512 first, __m512 second) {
  if constexpr (Raw) {
    // Raw FP16 has two accumulators and a different reduction tree.
    const __m512 value = _mm512_add_ps(first, second);
    const __m256 halves = _mm256_add_ps(_mm512_castps512_ps256(value),
                                        _mm512_extractf32x8_ps(value, 1));
    const __m128 quarters = _mm_add_ps(_mm256_castps256_ps128(halves),
                                       _mm256_extractf128_ps(halves, 1));
    const __m128 pairs = _mm_hadd_ps(quarters, quarters);
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 0x55)));
  } else {
    return _mm512_reduce_add_ps(first);
  }
}

template <typename T, bool Raw = false>
bool split_batch(const SplitVector *vectors, size_t count, const void *query,
                 size_t dim, float *distances, float cutoff) {
  if (dim % (Raw ? 32 : 16)) return false;
  for (size_t i = 0; i < count; ++i) {
    const size_t cut = vectors[i].first_bytes / sizeof(T);
    if (vectors[i].first_bytes % sizeof(T) || (cut && !vectors[i].first) ||
        (cut < dim && !vectors[i].second))
      return false;
  }
  const auto *q = static_cast<const T *>(query);
  for (size_t i = 0; i < count;) {
    const size_t rows = std::min<size_t>(4, count - i);
    __m512 first[4], second[4];
    for (size_t row = 0; row < rows; ++row) {
      first[row] = _mm512_setzero_ps();
      second[row] = _mm512_setzero_ps();
    }
    bool rejected = false;
    for (size_t d = 0; d < dim; d += 16) {
      const __m512 rhs = load(q + d);
      for (size_t row = 0; row < rows; ++row) {
        const __m512 diff = _mm512_sub_ps(rhs, load<T>(vectors[i + row], d));
        __m512 &sum = Raw && (d % 32) ? second[row] : first[row];
        sum = _mm512_fmadd_ps(diff, diff, sum);
      }
      if (d == 48 && dim > 64 && rows == 4 && std::isfinite(cutoff)) {
        rejected = true;
        for (size_t row = 0; row < rows; ++row)
          rejected &= reduce<Raw>(first[row], second[row]) >= cutoff;
        if (rejected) break;
      }
    }
    for (size_t row = 0; row < rows; ++row) {
      distances[i + row] = rejected ? std::numeric_limits<float>::infinity()
                                    : reduce<Raw>(first[row], second[row]);
      uint32_t bits;
      std::memcpy(&bits, &distances[i + row], sizeof(bits));
      if ((bits & 0x7fffffffU) > 0x7f800000U) return false;
    }
    i += rows;
  }
  return true;
}

}  // namespace
}  // namespace zvec::turbo::avx512
#endif

namespace zvec::turbo::avx512 {

bool squared_euclidean_fp32_split_batch(const SplitVector *vectors,
                                        size_t count, const void *query,
                                        size_t dim, float *distances,
                                        float cutoff) {
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)
  return split_batch<float>(vectors, count, query, dim, distances, cutoff);
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
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)
  return split_batch<uint16_t>(vectors, count, query, dim, distances, cutoff);
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

bool squared_euclidean_raw_fp16_split_batch(const SplitVector *vectors,
                                            size_t count, const void *query,
                                            size_t dim, float *distances,
                                            float cutoff) {
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)
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

}  // namespace zvec::turbo::avx512
