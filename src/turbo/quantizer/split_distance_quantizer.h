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

#include <cstddef>

namespace zvec::turbo {

struct SplitVector {
  const void *first;
  const void *second;
  size_t first_bytes;
};

//! Optional distance kernel for codes split across pinned pages.
class SplitDistanceQuantizer {
 public:
  //! The second span starts at first_bytes in each original code. On false,
  //! callers must discard all distances and recompute through the ordinary
  //! path. Scores below cutoff remain exact; proven losers may be marked
  //! infinite. A nonfinite cutoff requests every ordinary distance without
  //! pruning.
  using Distance = bool (*)(const SplitVector *vectors, size_t count,
                            const void *query, size_t dim, float *distances,
                            float cutoff);

  virtual ~SplitDistanceQuantizer() = default;

  virtual Distance split_distance() const = 0;
};

}  // namespace zvec::turbo
