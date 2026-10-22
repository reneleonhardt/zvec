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

#include "conversion/avx512/cosine_decode.h"
#include "conversion/avx2/cosine_decode.h"

#if defined(__AVX512F__) && defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512 {

bool fp16_cosine_decode(const uint16_t *input, size_t dim, float norm,
                        float *output) {
#if defined(__AVX512F__) && defined(__AVX2__) && defined(__F16C__)
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256i half =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
    const __m256i special =
        _mm256_cmpeq_epi16(_mm256_and_si256(half, _mm256_set1_epi16(0x7c00)),
                           _mm256_set1_epi16(0x7c00));
    if (!_mm256_movemask_epi8(special)) {
      const __m512 scaled =
          _mm512_mul_ps(_mm512_cvtph_ps(half), _mm512_set1_ps(norm));
      const __m512i bits = _mm512_castps_si512(scaled);
      const __m512i magnitude =
          _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff));
      const __mmask16 outside =
          _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x38800000)) |
          _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x477fefff));
      if (!outside) {
        // Round ties away before the exact conversion to avoid extra FP flags.
        const __m512i rounded =
            _mm512_and_si512(_mm512_add_epi32(bits, _mm512_set1_epi32(0x1000)),
                             _mm512_set1_epi32(-8192));
        const __m256i restored = _mm512_cvtps_ph(_mm512_castsi512_ps(rounded),
                                                 _MM_FROUND_TO_NEAREST_INT);
        _mm512_storeu_ps(output + i, _mm512_cvtph_ps(restored));
        continue;
      }
    }
    if (!avx2::fp16_cosine_decode(input + i, 16, norm, output + i)) return false;
  }
  return i == dim ||
         avx2::fp16_cosine_decode(input + i, dim - i, norm, output + i);
#else
  (void)input;
  (void)dim;
  (void)norm;
  (void)output;
  return false;
#endif
}

}  // namespace zvec::turbo::avx512
