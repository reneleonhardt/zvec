#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>
#include "fixture.h"
#include "linux_io.h"

using Clock = std::chrono::steady_clock;
struct Doc {
  uint64_t key;
  uint32_t score, index;
};
using Expected = std::vector<std::array<Doc, 10>>;
static ci::BaseIndexQueryParam::Pointer query_param(const std::string &kind,
                                                    bool filtered) {
  ci::BaseIndexQueryParam::Pointer q;
  if (kind == "ivf")
    q = ci::IVFQueryParamBuilder().with_topk(10).with_nprobe(8).build();
  else
    q = ci::FlatQueryParamBuilder().with_topk(10).build();
  q->filter = std::make_shared<ci::IndexFilter>();
  if (filtered) q->filter->set([](uint64_t id) { return id % 7 < 2; });
  return q;
}
static ci::Index::Pointer open_index(const Fixture &f, const std::string &file,
                                     const std::string &kind,
                                     const std::string &storage) {
  auto definition = params(f, kind);
  if (storage == "contiguous")
    std::static_pointer_cast<ci::FlatIndexParam>(definition)
        ->use_contiguous_memory = true;
  auto index = ci::IndexFactory::CreateAndInitIndex(*definition);
  require(bool(index));
  require(index->open(file, {storage != "bufferpool"
                                 ? ci::StorageOptions::StorageType::kMMAP
                                 : ci::StorageOptions::StorageType::kBufferPool,
                             false, true}) == 0);
  return index;
}
static void search(ci::Index &index, const Fixture &f, size_t id,
                   const ci::BaseIndexQueryParam::Pointer &q, bool filtered,
                   ci::SearchResult &out, const Expected *gold = nullptr) {
  // The interface moves the callback into its context on each search.
  if (filtered) q->filter->set([](uint64_t key) { return key % 7 < 2; });
  require(
      index.search(ci::VectorData{ci::DenseVector{f.query.data() + id * f.dim}},
                   q, &out) == 0 &&
      out.doc_list_.size() == 10);
  if (filtered)
    for (const auto &doc : out.doc_list_) require(doc.key() % 7 >= 2);
  if (gold)
    for (size_t j = 0; j < 10; ++j) {
      uint32_t b;
      float s = out.doc_list_[j].score();
      std::memcpy(&b, &s, 4);
      const auto &g = (*gold)[id][j];
      if (g.key != out.doc_list_[j].key() || b != g.score ||
          g.index != out.doc_list_[j].index()) {
        std::fprintf(stderr,
                     "oracle mismatch query=%zu row=%zu expected=(%llu,%08x,%u) "
                     "actual=(%llu,%08x,%u)\n",
                     id, j, (unsigned long long)g.key, g.score, g.index,
                     (unsigned long long)out.doc_list_[j].key(), b,
                     out.doc_list_[j].index());
        throw std::runtime_error("search oracle mismatch");
      }
    }
}
struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  size_t pending, total, generation = 0;
  explicit Gate(size_t n) : pending(n), total(n) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    size_t g = generation;
    if (--pending == 0) {
      pending = total;
      ++generation;
      cv.notify_all();
    } else
      cv.wait(lock, [&] { return generation != g; });
  }
};
int main(int argc, char **argv) {
  try {
    configure_arch();
    require(argc >= 6);
    std::string mode = argv[1], root = argv[2], kind = argv[3],
                storage = argv[4];
    bool filtered = std::stoi(argv[5]);
    Fixture f(root + "/fixture.bin");
    zvec::ailego::MemoryLimitPool::get_instance().init(128UL << 20);
    std::fprintf(stderr, "pid=%d nice=%d\n", getpid(),
                 getpriority(PRIO_PROCESS, 0));
    if (mode == "warm" || mode == "alloc" || mode == "cold" ||
        mode == "fetchwarm" || mode == "fetchalloc" || mode == "fetchcold" ||
        mode == "coldalloc" || mode == "fetchcoldalloc")
      require(getpriority(PRIO_PROCESS, 0) == 0);
    if (mode == "prepare") {
      prepare(f, root);
      return 0;
    }
    if (mode == "clonecheck") {
      const auto path = root + "/clone-test.index";
      for (size_t trial = 0; trial < 3; ++trial) {
        prepare_cold_index((root + "/flat.index").c_str(), path.c_str());
        {
          auto index = open_index(f, path, "flat", "bufferpool");
          {
            auto q = query_param("flat", false);
            ci::SearchResult result;
            search(*index, f, trial, q, false, result);
          }
          require(index->close() == 0);
        }
        require(std::remove(path.c_str()) == 0);
      }
      return 0;
    }
    const bool fetching = mode.rfind("fetch", 0) == 0;
    std::string fetched_gold;
    const auto fetch_oracle = root + "/fetch-" + kind + ".bin";
    const auto oracle =
        root + "/oracle-" + kind + "-" + storage + "-" + argv[5] + ".bin";
    Expected expected(f.queries);
    if (mode == "fetchoracle") {
      auto index = open_index(f, root + "/" + kind + ".index", kind, storage);
      std::ofstream out(fetch_oracle, std::ios::binary);
      ci::VectorDataBuffer output;
      for (size_t id = 0; id < f.count; ++id) {
        require(index->fetch(id, &output) == 0);
        const auto &data =
            std::get<ci::DenseVectorBuffer>(output.vector_buffer).data;
        require(data.size() == f.dim * 4);
        out.write(data.data(), data.size());
      }
      require(out.good());
      require(index->close() == 0);
      return 0;
    }
    if (fetching) {
      fetched_gold.resize(size_t(f.count) * f.dim * 4);
      std::ifstream in(fetch_oracle, std::ios::binary);
      in.read(fetched_gold.data(), fetched_gold.size());
      require(in.good());
    }
    if (mode == "oracle") {
      auto index = open_index(f, root + "/" + kind + ".index", kind, storage);
      auto q = query_param(kind, filtered);
      ci::SearchResult result;
      for (size_t i = 0; i < f.queries; ++i) {
        search(*index, f, i, q, filtered, result);
        for (size_t j = 0; j < 10; ++j) {
          auto &d = expected[i][j];
          d.key = result.doc_list_[j].key();
          float s = result.doc_list_[j].score();
          std::memcpy(&d.score, &s, 4);
          d.index = result.doc_list_[j].index();
        }
      }
      std::ofstream out(oracle, std::ios::binary);
      out.write(reinterpret_cast<const char *>(expected.data()),
                expected.size() * sizeof(expected[0]));
      require(out.good());
      require(index->close() == 0);
      return 0;
    }
    if (!fetching) {
      std::ifstream in(oracle, std::ios::binary);
      in.read(reinterpret_cast<char *>(expected.data()),
              expected.size() * sizeof(expected[0]));
      require(in.good());
    }
    auto fetch = [&](ci::Index &index, size_t id,
                     ci::VectorDataBuffer &output) {
      id = (id * 2683 + 17) % f.count;
      require(index.fetch(id, &output) == 0);
      const auto &data =
          std::get<ci::DenseVectorBuffer>(output.vector_buffer).data;
      require(data.size() == f.dim * 4 &&
              std::memcmp(data.data(), fetched_gold.data() + id * f.dim * 4,
                          data.size()) == 0);
    };
    if (mode == "check" || mode == "fetchcheck") {
      auto index = open_index(f, root + "/" + kind + ".index", kind, storage);
      auto q = query_param(kind, filtered);
      ci::SearchResult result;
      ci::VectorDataBuffer output;
      for (size_t i = 0; i < 2 * f.queries; ++i) {
        if (fetching)
          fetch(*index, i, output);
        else
          search(*index, f, i % f.queries, q, filtered, result, &expected);
      }
      require(index->close() == 0);
      return 0;
    }
    if (mode == "cold" || mode == "fetchcold" || mode == "coldalloc" ||
        mode == "fetchcoldalloc") {
      Begin begin = nullptr;
      End end = nullptr;
      if (mode.find("alloc") != std::string::npos) {
        begin = reinterpret_cast<Begin>(
            dlsym(RTLD_DEFAULT, "vector_probe_begin_all"));
        end =
            reinterpret_cast<End>(dlsym(RTLD_DEFAULT, "vector_probe_end_all"));
        require(begin && end);
      }
      require(argc == 8);
      size_t batch = std::stoul(argv[6]), trials = std::stoul(argv[7]);
      std::puts(
          "trial,queries,open_ns,open_cpu_ns,wall_ns,cpu_ns,total_wall_ns,"
          "total_cpu_ns,open_disk_bytes,search_disk_bytes,open_allocations,"
          "open_allocated_bytes,allocations,allocated_bytes,total_allocations,"
          "total_allocated_bytes");
      for (size_t trial = 0; trial < trials; ++trial) {
        const auto clone = root + "/cold-" + std::to_string(getpid()) + "-" +
                           std::to_string(trial) + ".index";
        prepare_cold_index((root + "/" + kind + ".index").c_str(), clone.c_str());
        uint64_t io0 = io_bytes();
        auto opened = Clock::now();
        auto open_cpu = std::clock();
        if (begin) begin();
        auto index = open_index(f, clone, kind, storage);
        double open_ns =
            std::chrono::duration<double, std::nano>(Clock::now() - opened)
                .count();
        const double open_cpu_ns =
            double(std::clock() - open_cpu) / CLOCKS_PER_SEC * 1e9;
        Alloc open_allocated{};
        if (end) open_allocated = end();
        uint64_t io1 = io_bytes();
        auto q = query_param(kind, filtered);
        ci::SearchResult result;
        ci::VectorDataBuffer output;
        auto cpu = std::clock();
        auto start = Clock::now();
        if (begin) begin();
        for (size_t i = 0; i < batch; ++i) {
          if (fetching)
            fetch(*index, trial * 7 + i, output);
          else
            search(*index, f, (trial * 7 + i) % f.queries, q, filtered, result,
                   &expected);
        }
        double wall =
            std::chrono::duration<double, std::nano>(Clock::now() - start)
                .count() /
            batch;
        double cpu_ns =
            double(std::clock() - cpu) / CLOCKS_PER_SEC * 1e9 / batch;
        Alloc allocated{};
        if (end) allocated = end();
        uint64_t io2 = io_bytes();
        std::printf(
            "%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%llu,%llu,%llu,%.3f,%."
            "3f,%.3f,%.3f\n",
            trial, batch, open_ns, open_cpu_ns, wall, cpu_ns,
            open_ns / batch + wall, open_cpu_ns / batch + cpu_ns,
            (unsigned long long)(io1 - io0), (unsigned long long)(io2 - io1),
            (unsigned long long)open_allocated.calls,
            (unsigned long long)open_allocated.bytes,
            double(allocated.calls) / batch, double(allocated.bytes) / batch,
            double(open_allocated.calls + allocated.calls) / batch,
            double(open_allocated.bytes + allocated.bytes) / batch);
        std::fflush(stdout);
        require(index->close() == 0);
        index.reset();
        require(std::remove(clone.c_str()) == 0);
      }
      return 0;
    }
    require((mode == "warm" || mode == "alloc" || mode == "fetchwarm" ||
             mode == "fetchalloc") &&
            argc == 9);
    size_t threads = std::stoul(argv[6]), trials = std::stoul(argv[8]);
    double duration = std::stod(argv[7]);
    auto index = open_index(f, root + "/" + kind + ".index", kind, storage);
    Gate gate(threads + 1);
    Clock::time_point deadline;
    std::vector<std::vector<double>> latency(threads);
    std::vector<size_t> operations(threads);
    std::vector<Alloc> allocated(threads);
    Begin begin = nullptr;
    End end = nullptr;
    if (mode == "alloc" || mode == "fetchalloc") {
      begin =
          reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT, "vector_probe_begin"));
      end = reinterpret_cast<End>(dlsym(RTLD_DEFAULT, "vector_probe_end"));
      require(begin && end);
      void *(*volatile allocate)(size_t) = std::malloc;
      begin();
      void *probe = allocate(97);
      std::free(probe);
      auto sanity = end();
      require(sanity.calls == 1 && sanity.bytes == 97);
    }
    std::vector<std::thread> workers;
    for (size_t t = 0; t < threads; ++t)
      workers.emplace_back([&, t] {
        try {
          auto q = query_param(kind, filtered);
          ci::SearchResult result;
          ci::VectorDataBuffer output;
          latency[t].reserve(100000);
          for (size_t i = 0; i < f.queries; ++i) {
            if (fetching)
              fetch(*index, i + t * 7, output);
            else
              search(*index, f, (i + t * 7) % f.queries, q, filtered, result,
                     &expected);
          }
          gate.wait();
          for (size_t trial = 0; trial < trials; ++trial) {
            latency[t].clear();
            size_t n = 0;
            gate.wait();
            if (begin) begin();
            do {
              auto start = Clock::now();
              if (fetching)
                fetch(*index, n + t * 7, output);
              else
                search(*index, f, (n + t * 7) % f.queries, q, filtered, result,
                       &expected);
              if (!begin)
                latency[t].push_back(std::chrono::duration<double, std::nano>(
                                         Clock::now() - start)
                                         .count());
              ++n;
            } while (begin ? n < 64 : Clock::now() < deadline);
            if (end) allocated[t] = end();
            operations[t] = n;
            gate.wait();
            gate.wait();
          }
        } catch (const std::exception &e) {
          std::fprintf(stderr, "worker %zu: %s\n", t, e.what());
          std::abort();
        }
      });
    gate.wait();
    std::fprintf(stderr, "ready\n");
    std::puts(
        "trial,threads,queries,wall_ns,cpu_ns,p50_ns,p95_ns,disk_bytes,"
        "allocations,allocated_bytes");
    for (size_t trial = 0; trial < trials; ++trial) {
      deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(duration));
      auto cpu = std::clock();
      uint64_t io0 = io_bytes();
      auto start = Clock::now();
      gate.wait();
      gate.wait();
      double wall =
          std::chrono::duration<double, std::nano>(Clock::now() - start)
              .count();
      double cpu_ns = double(std::clock() - cpu) / CLOCKS_PER_SEC * 1e9;
      uint64_t disk = io_bytes() - io0;
      size_t n = 0;
      uint64_t calls = 0, bytes_alloc = 0;
      std::vector<double> all;
      for (size_t t = 0; t < threads; ++t) {
        n += operations[t];
        calls += allocated[t].calls;
        bytes_alloc += allocated[t].bytes;
        all.insert(all.end(), latency[t].begin(), latency[t].end());
      }
      std::sort(all.begin(), all.end());
      double p50 = all.empty() ? 0 : all[all.size() / 2],
             p95 = all.empty() ? 0 : all[all.size() * 95 / 100];
      std::printf("%zu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%llu,%.4f,%.4f\n", trial,
                  threads, n, wall / n, cpu_ns / n, p50, p95,
                  (unsigned long long)disk, double(calls) / n,
                  double(bytes_alloc) / n);
      std::fflush(stdout);
      gate.wait();
    }
    for (auto &worker : workers) worker.join();
    require(index->close() == 0);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
