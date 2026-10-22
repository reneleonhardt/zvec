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

#include <algorithm>
#include <cstring>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

//! Cursor over a scatter read. IVF blocks and vectors are logical byte
//! ranges, so neither needs to be aligned with an underlying storage page.
class ScatterCursor {
 public:
  explicit ScatterCursor(const IndexStorage::Segment::ScatterBlock &block)
      : spans_(block.spans) {}

  const uint8_t *data() {
    normalize();
    return span_index_ < spans_.size() ? spans_[span_index_].data + span_offset_
                                       : nullptr;
  }

  size_t contiguous_size() {
    normalize();
    return span_index_ < spans_.size() ? spans_[span_index_].size - span_offset_
                                       : 0;
  }

  bool skip(size_t size) {
    while (size != 0) {
      normalize();
      if (span_index_ == spans_.size()) {
        return false;
      }
      const size_t available = spans_[span_index_].size - span_offset_;
      const size_t consumed = std::min(size, available);
      span_offset_ += consumed;
      size -= consumed;
    }
    return true;
  }

  bool copy(void *output, size_t size) {
    auto *dst = static_cast<uint8_t *>(output);
    while (size != 0) {
      normalize();
      if (span_index_ == spans_.size()) {
        return false;
      }
      const auto &span = spans_[span_index_];
      const size_t available = span.size - span_offset_;
      const size_t copied = std::min(size, available);
      std::memcpy(dst, span.data + span_offset_, copied);
      dst += copied;
      span_offset_ += copied;
      size -= copied;
    }
    return true;
  }

 private:
  void normalize() {
    while (span_index_ < spans_.size() &&
           span_offset_ == spans_[span_index_].size) {
      ++span_index_;
      span_offset_ = 0;
    }
  }

  const std::vector<IndexStorage::Segment::ReadSpan> &spans_;
  size_t span_index_{0};
  size_t span_offset_{0};
};

}  // namespace zvec::core
