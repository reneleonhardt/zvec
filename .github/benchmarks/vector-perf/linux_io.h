#pragma once
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include "fixture.h"

inline uint64_t io_counter(const std::string &name) {
  std::ifstream in("/proc/self/io");
  std::string key;
  uint64_t value;
  while (in >> key >> value)
    if (key == name + ":") return value;
  throw std::runtime_error("missing process I/O counter");
}
inline uint64_t io_bytes() {
  return io_counter("read_bytes");
}
inline void prepare_cold_index(const char *source, const char *dest) {
  // Every reader opens a fresh pool. Reuse immutable bytes without generating
  // a file's worth of writeback immediately before measuring cold reads.
  require(link(source, dest) == 0);
  const int fd = open(dest, O_RDONLY);
  require(fd >= 0);
  require(fsync(fd) == 0);
  struct stat info;
  require(fstat(fd, &info) == 0);
  require(posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0);
  void *mapping = mmap(nullptr, info.st_size, PROT_READ, MAP_SHARED, fd, 0);
  require(mapping != MAP_FAILED);
  const size_t page_size = sysconf(_SC_PAGESIZE);
  std::vector<unsigned char> resident((info.st_size + page_size - 1) /
                                      page_size);
  require(mincore(mapping, info.st_size, resident.data()) == 0);
  for (auto state : resident) require((state & 1) == 0);
  munmap(mapping, info.st_size);
  close(fd);
}
