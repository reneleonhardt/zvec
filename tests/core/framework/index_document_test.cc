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

#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_document.h>

TEST(IndexDocumentHeap, RejectBeforeConstructionPreservesOrdering) {
  using namespace zvec::core;
  for (bool indexed : {false, true}) {
    for (size_t limit : {1U, 3U}) {
      IndexDocumentHeap actual(limit, 4.5f);
      zvec::ailego::Heap<IndexDocument> reference(limit);
      uint32_t id = 0;
      for (float score : {6.f, 4.f, 4.f, 5.f, 3.f, 3.f,
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(), -0.f, 0.f,
                          -std::numeric_limits<float>::infinity()}) {
        if (score <= 4.5f) {
          if (indexed)
            reference.emplace(id, score, id + 7);
          else
            reference.emplace(id, score);
        }
        if (indexed)
          actual.emplace(id, score, id + 7);
        else
          actual.emplace(id, score);
        ASSERT_EQ(reference.size(), actual.size());
        for (size_t i = 0; i < actual.size(); ++i) {
          const auto &want = *(reference.begin() + i);
          const auto &got = *(actual.begin() + i);
          EXPECT_EQ(want.key(), got.key());
          EXPECT_FLOAT_EQ(want.score(), got.score());
          EXPECT_EQ(want.index(), got.index());
        }
        ++id;
      }
    }
  }
}
