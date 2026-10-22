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

#include "avx2/fp32/inner_product.h"
#if defined(__AVX2__)
#include <immintrin.h>
#include <array>
#include <zvec/ailego/internal/platform.h>
#endif

namespace zvec::turbo::avx2 {

// Keep the original kernel in its own translation unit to retain NaN payloads.
void inner_product_fp32_batch_distance_avx2_legacy(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values);

void inner_product_fp32_batch_distance_avx2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances,
                                            const void *const *extra_values) {
#if defined(__AVX2__)
  if (dim % 8 || n < 8) {
    inner_product_fp32_batch_distance_avx2_legacy(vectors, query, n, dim,
                                                  distances, extra_values);
    return;
  }
  const auto *typed_query = static_cast<const float *>(query);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    std::array<const float *, 8> prefetch{};
    __m256 sums[8];
    for (size_t row = 0; row < 8; ++row) {
      sums[row] = _mm256_setzero_ps();
      if (i + row + 8 < n)
        prefetch[row] = static_cast<const float *>(vectors[i + row + 8]);
    }
    for (size_t d = 0; d < dim; d += 8) {
      const __m256 q = _mm256_loadu_ps(typed_query + d);
      for (size_t row = 0; row < 8; ++row) {
        const auto *data = static_cast<const float *>(vectors[i + row]);
        sums[row] = _mm256_fmadd_ps(q, _mm256_loadu_ps(data + d), sums[row]);
        if (prefetch[row]) ailego_prefetch(prefetch[row] + d);
      }
    }
    for (size_t row = 0; row < 8; ++row) {
      __m128 sum = _mm_add_ps(_mm256_castps256_ps128(sums[row]),
                              _mm256_extractf128_ps(sums[row], 1));
      sum = _mm_hadd_ps(sum, sum);
      sum = _mm_hadd_ps(sum, sum);
      distances[i + row] = -_mm_cvtss_f32(sum);
    }
    const __m256i bits = _mm256_castps_si256(_mm256_loadu_ps(distances + i));
    const __m256i magnitude =
        _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
    if (_mm256_movemask_epi8(
            _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000)))) {
      inner_product_fp32_batch_distance_avx2_legacy(vectors, query, n, dim,
                                                    distances, extra_values);
      return;
    }
  }
  if (i < n)
    inner_product_fp32_batch_distance_avx2_legacy(
        vectors + i, query, n - i, dim, distances + i, extra_values);
#else
  inner_product_fp32_batch_distance_avx2_legacy(vectors, query, n, dim,
                                                distances, extra_values);
#endif
}

}  // namespace zvec::turbo::avx2
