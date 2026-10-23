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

#pragma once

#include "quantizer/split_distance_quantizer.h"

namespace zvec::turbo::avx512 {

bool inner_product_fp32_split_batch(const SplitVector *, size_t,
                                    const void *, size_t, float *, float);
bool cosine_fp32_split_batch(const SplitVector *, size_t, const void *, size_t,
                             float *, float);
bool squared_euclidean_fp32_split_batch(const SplitVector *, size_t,
                                        const void *, size_t, float *, float);
bool squared_euclidean_fp16_split_batch(const SplitVector *, size_t,
                                        const void *, size_t, float *, float);
bool squared_euclidean_raw_fp16_split_batch(const SplitVector *, size_t,
                                            const void *, size_t, float *,
                                            float);

}  // namespace zvec::turbo::avx512
