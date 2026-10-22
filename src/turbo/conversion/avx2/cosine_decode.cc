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

#include "conversion/avx2/cosine_decode.h"

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#include <zvec/ailego/utility/float_helper.h>
#endif

namespace zvec::turbo::avx2 {

bool fp16_cosine_decode(const uint16_t *input, size_t dim, float norm,
                        float *output) {
#if defined(__AVX2__) && defined(__F16C__)
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m128i half =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + i));
    const __m128i special = _mm_cmpeq_epi16(
        _mm_and_si128(half, _mm_set1_epi16(0x7c00)), _mm_set1_epi16(0x7c00));
    if (!_mm_movemask_epi8(special)) {
      const __m256 scaled =
          _mm256_mul_ps(_mm256_cvtph_ps(half), _mm256_set1_ps(norm));
      const __m256i bits = _mm256_castps_si256(scaled);
      const __m256i magnitude =
          _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
      const __m256i outside = _mm256_or_si256(
          _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), magnitude),
          _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x477fefff)));
      if (!_mm256_movemask_epi8(outside)) {
        // Round ties away in integer bits, then convert an exact finite half.
        const __m256i rounded =
            _mm256_and_si256(_mm256_add_epi32(bits, _mm256_set1_epi32(0x1000)),
                             _mm256_set1_epi32(-8192));
        const __m128i restored = _mm256_cvtps_ph(_mm256_castsi256_ps(rounded),
                                                 _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_ps(output + i, _mm256_cvtph_ps(restored));
        continue;
      }
    }
    // Preserve scalar NaN payloads, subnormal rounding and signed zero.
    for (size_t j = i; j < i + 8; ++j) {
      const float scaled = ailego::FloatHelper::ToFP32(input[j]) * norm;
      output[j] =
          ailego::FloatHelper::ToFP32(ailego::FloatHelper::ToFP16(scaled));
    }
  }
  for (; i < dim; ++i) {
    const float scaled = ailego::FloatHelper::ToFP32(input[i]) * norm;
    output[i] =
        ailego::FloatHelper::ToFP32(ailego::FloatHelper::ToFP16(scaled));
  }
  return true;
#else
  (void)input;
  (void)dim;
  (void)norm;
  (void)output;
  return false;
#endif
}

}  // namespace zvec::turbo::avx2
