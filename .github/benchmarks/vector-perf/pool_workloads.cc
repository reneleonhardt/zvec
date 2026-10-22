#include <sys/resource.h>
#include "fixture.h"
#include "linux_io.h"

using Clock = std::chrono::steady_clock;
int main(int argc, char **argv) {
  try {
    configure_arch();
    require(argc == 8 && getpriority(PRIO_PROCESS, 0) == 0);
    const std::string mode = argv[1], root = argv[2], kind = argv[3],
                      storage = argv[4];
    const size_t count = std::stoul(argv[5]), trials = std::stoul(argv[6]),
                 pool_mb = std::stoul(argv[7]);
    Fixture f(root + "/fixture.bin");
    require(count <= f.count && (kind == "flat" || kind == "ivf"));
    zvec::ailego::MemoryLimitPool::get_instance().init(pool_mb << 20);
    Begin begin = nullptr;
    End end = nullptr;
    if (mode == "alloc") {
      begin = reinterpret_cast<Begin>(
          dlsym(RTLD_DEFAULT, "vector_probe_begin_all"));
      end = reinterpret_cast<End>(dlsym(RTLD_DEFAULT, "vector_probe_end_all"));
      require(begin && end);
      void *(*volatile allocate)(size_t) = std::malloc;
      begin();
      void *probe = allocate(97);
      std::free(probe);
      auto sanity = end();
      require(sanity.calls == 1 && sanity.bytes == 97);
    }
    auto type = storage == "bufferpool"
                    ? ci::StorageOptions::StorageType::kBufferPool
                    : ci::StorageOptions::StorageType::kMMAP;
    std::puts(
        "trial,stage,operations,wall_ns,cpu_ns,allocations,allocated_bytes,"
        "read_bytes,write_bytes");
    for (size_t trial = 0; trial < trials; ++trial) {
      const std::string path = root + "/write-" + std::to_string(getpid()) +
                               "-" + std::to_string(trial) + ".index";
      auto index = ci::IndexFactory::CreateAndInitIndex(*params(f, kind));
      require(bool(index));
      auto measure = [&](const char *stage, size_t operations, auto operation) {
        auto start = Clock::now();
        auto cpu = std::clock();
        uint64_t io0 = io_bytes(), wo0 = io_counter("write_bytes");
        if (begin) begin();
        operation();
        Alloc allocated{};
        if (end) allocated = end();
        double wall =
            std::chrono::duration<double, std::nano>(Clock::now() - start)
                .count();
        double cpu_ns = double(std::clock() - cpu) / CLOCKS_PER_SEC * 1e9;
        std::printf("%zu,%s,%zu,%.3f,%.3f,%.3f,%.3f,%llu,%llu\n", trial, stage,
                    operations, wall / operations, cpu_ns / operations,
                    double(allocated.calls) / operations,
                    double(allocated.bytes) / operations,
                    (unsigned long long)(io_bytes() - io0),
                    (unsigned long long)(io_counter("write_bytes") - wo0));
        std::fflush(stdout);
      };
      measure("open", 1,
              [&] { require(index->open(path, {type, true, false}) == 0); });
      measure("add", count, [&] {
        for (uint32_t id = 0; id < count; ++id)
          require(index->add(ci::VectorData{ci::DenseVector{
                                 f.vectors.data() + size_t(id) * f.dim}},
                             id) == 0);
      });
      measure("train", 1, [&] { require(index->train() == 0); });
      measure("flush", 1, [&] { require(index->flush() == 0); });
      measure("close", 1, [&] { require(index->close() == 0); });
      index.reset();
      index = ci::IndexFactory::CreateAndInitIndex(*params(f, kind));
      require(index->open(path, {type, false, true}) == 0);
      ci::VectorDataBuffer output;
      for (uint32_t id = 0; id < count; ++id) {
        require(index->fetch(id, &output) == 0);
        const auto &data =
            std::get<ci::DenseVectorBuffer>(output.vector_buffer).data;
        require(data.size() == f.dim * 4 &&
                std::memcmp(data.data(), f.vectors.data() + size_t(id) * f.dim,
                            data.size()) == 0);
      }
      require(index->close() == 0);
      index.reset();
      require(std::remove(path.c_str()) == 0);
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
