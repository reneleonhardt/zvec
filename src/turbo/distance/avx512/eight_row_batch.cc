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

#include "avx512/fp16/squared_euclidean.h"
#include "avx512/fp32/inner_product.h"
#include "avx512/fp32/squared_euclidean.h"
#include "avx512_vnni/fp16/squared_euclidean.h"
#include "common/fp16_common.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#include <array>
#include <cstdint>
#include <type_traits>
#include <zvec/ailego/internal/platform.h>
#endif

namespace zvec::turbo {
#if defined(__AVX512F__)
namespace {

using Batch = void (*)(const void *const *, const void *, size_t, size_t,
                       float *, const void *const *);

template <typename T>
__m512 load(const T *values) {
  if constexpr (std::is_same_v<T, float>)
    return _mm512_loadu_ps(values);
  else
    return _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values)));
}

template <bool Raw>
float reduce(__m512 first, __m512 second) {
  if constexpr (Raw) {
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

template <typename T, bool Inner, bool Raw>
bool score_eight(const void *const *vectors, const T *query, size_t dim,
                 const std::array<const T *, 8> &prefetch, float *distances) {
  __m512 first[8], second[8];
  for (size_t row = 0; row < 8; ++row) {
    first[row] = _mm512_setzero_ps();
    if constexpr (Raw) second[row] = _mm512_setzero_ps();
  }
  size_t d = 0;
  for (; d + (Raw ? 32 : 16) <= dim; d += Raw ? 32 : 16) {
    const __m512 q = load(query + d);
    const __m512 q2 = Raw ? load(query + d + 16) : _mm512_setzero_ps();
    for (size_t row = 0; row < 8; ++row) {
      const auto *data = static_cast<const T *>(vectors[row]);
      if constexpr (!Raw) {
        if (prefetch[row]) ailego_prefetch(prefetch[row] + d);
      }
      const __m512 value = load(data + d);
      if constexpr (Inner) {
        first[row] = _mm512_fmadd_ps(q, value, first[row]);
      } else {
        const __m512 diff = _mm512_sub_ps(q, value);
        first[row] = _mm512_fmadd_ps(diff, diff, first[row]);
        if constexpr (Raw) {
          const __m512 diff2 = _mm512_sub_ps(q2, load(data + d + 16));
          second[row] = _mm512_fmadd_ps(diff2, diff2, second[row]);
        }
      }
    }
  }
  if constexpr (Raw) {
    if (d + 16 <= dim) {
      const __m512 q = load(query + d);
      for (size_t row = 0; row < 8; ++row) {
        const __m512 diff =
            _mm512_sub_ps(q, load(static_cast<const T *>(vectors[row]) + d));
        first[row] = _mm512_fmadd_ps(diff, diff, first[row]);
      }
    }
  }
  for (size_t row = 0; row < 8; ++row) {
    float distance = reduce<Raw>(first[row], Raw ? second[row] : first[row]);
    if constexpr (Inner) distance = -distance;
    distances[row] = distance;
  }
  const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(distances));
  const __m256i magnitude =
      _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  return _mm256_movemask_epi8(
             _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000))) != 0;
}

template <typename T, bool Inner = false, bool Raw = false>
void eight_row_batches(const void *const *vectors, const void *query, size_t n,
                       size_t dim, float *distances,
                       const void *const *extra_values, Batch legacy) {
  if constexpr (Raw) {
    if (n >= 8)
      for (size_t row = 0; row < 8; ++row) ailego_prefetch(vectors[row]);
  }
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    std::array<const T *, 8> prefetch{};
    for (size_t row = 0; row < 8; ++row) {
      const size_t next = i + row + (Raw ? 8 : 16);
      if (next < n) prefetch[row] = static_cast<const T *>(vectors[next]);
    }
    if constexpr (Raw) {
      for (const T *row : prefetch)
        if (row) ailego_prefetch(row);
    }
    if (score_eight<T, Inner, Raw>(vectors + i, static_cast<const T *>(query),
                                   dim, prefetch, distances + i)) {
      // Keep NaN payload selection in the original, separately compiled kernel.
      legacy(vectors, query, n, dim, distances, extra_values);
      return;
    }
  }
  if (i < n)
    legacy(vectors + i, query, n - i, dim, distances + i, extra_values);
}

}  // namespace
#endif

namespace avx512 {

void inner_product_fp32_batch_distance_avx512_legacy(const void *const *,
                                                     const void *, size_t,
                                                     size_t, float *,
                                                     const void *const *);
void squared_euclidean_fp32_batch_distance_avx512_legacy(const void *const *,
                                                         const void *, size_t,
                                                         size_t, float *,
                                                         const void *const *);
void squared_euclidean_fp16_batch_distance_avx512_legacy(const void *const *,
                                                         const void *, size_t,
                                                         size_t, float *,
                                                         const void *const *);

void inner_product_fp32_batch_distance_avx512(const void *const *vectors,
                                              const void *query, size_t n,
                                              size_t dim, float *distances,
                                              const void *const *extra_values) {
#if defined(__AVX512F__)
  if (dim % 16 == 0) {
    eight_row_batches<float, true>(
        vectors, query, n, dim, distances, extra_values,
        inner_product_fp32_batch_distance_avx512_legacy);
    return;
  }
#endif
  inner_product_fp32_batch_distance_avx512_legacy(vectors, query, n, dim,
                                                  distances, extra_values);
}

void squared_euclidean_fp32_batch_distance_avx512(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
#if defined(__AVX512F__)
  if (dim % 16 == 0) {
    eight_row_batches<float>(
        vectors, query, n, dim, distances, extra_values,
        squared_euclidean_fp32_batch_distance_avx512_legacy);
    return;
  }
#endif
  squared_euclidean_fp32_batch_distance_avx512_legacy(vectors, query, n, dim,
                                                      distances, extra_values);
}

void squared_euclidean_fp16_batch_distance_avx512(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
#if ZVEC_TURBO_FP16_AVX512
  if (dim % 16 == 0) {
    eight_row_batches<uint16_t>(
        vectors, query, n, dim, distances, extra_values,
        squared_euclidean_fp16_batch_distance_avx512_legacy);
    return;
  }
#endif
  squared_euclidean_fp16_batch_distance_avx512_legacy(vectors, query, n, dim,
                                                      distances, extra_values);
}

}  // namespace avx512

namespace avx512_vnni {

void squared_euclidean_fp16_batch_distance_legacy(const void *const *,
                                                  const void *, size_t, size_t,
                                                  float *, const void *const *);

void squared_euclidean_fp16_batch_distance(const void *const *vectors,
                                           const void *query, size_t n,
                                           size_t dim, float *distances,
                                           const void *const *extra_values) {
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)) || \
    (defined(_MSC_VER) && defined(__AVX512F__))
  if (dim < 512 && dim % 16 == 0) {
    eight_row_batches<uint16_t, false, true>(
        vectors, query, n, dim, distances, extra_values,
        squared_euclidean_fp16_batch_distance_legacy);
    return;
  }
#endif
  squared_euclidean_fp16_batch_distance_legacy(vectors, query, n, dim,
                                               distances, extra_values);
}

}  // namespace avx512_vnni
}  // namespace zvec::turbo
