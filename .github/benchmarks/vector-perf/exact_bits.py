#!/usr/bin/env python3
"""Compare linked distance and decode kernels with frozen originals."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


KERNELS = [
    ("avx2", "fp32/inner_product", "inner_product_fp32_batch_distance_avx2", False),
    ("avx2", "fp32/squared_euclidean", "squared_euclidean_fp32_batch_distance_avx2", False),
    ("avx2", "fp16/inner_product", "inner_product_fp16_batch_distance_avx2", True),
    ("avx2", "fp16/squared_euclidean", "squared_euclidean_fp16_batch_distance_avx2", True),
    ("avx512", "fp32/inner_product", "inner_product_fp32_batch_distance_avx512", False),
    ("avx512", "fp32/squared_euclidean", "squared_euclidean_fp32_batch_distance_avx512", False),
    ("avx512", "fp16/inner_product", "inner_product_fp16_batch_distance_avx512", True),
    ("avx512", "fp16/squared_euclidean", "squared_euclidean_fp16_batch_distance_avx512", True),
    ("avx512_vnni", "fp16/squared_euclidean", "squared_euclidean_fp16_batch_distance", True),
]

DRIVER = r"""
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <type_traits>
#include <vector>
@DECLARATIONS@
using Batch = void (*)(const void *const *, const void *, size_t, size_t,
                      float *, const void *const *);
template <class T> T value(uint32_t bits) {
  T result;
  if constexpr (std::is_same_v<T, float>) std::memcpy(&result, &bits, sizeof(T));
  else result = static_cast<T>(bits);
  return result;
}
uint32_t bits(float v) { uint32_t b; std::memcpy(&b, &v, sizeof(b)); return b; }
template <class T> size_t check(const char *name, Batch current, Batch original) {
  size_t failures = 0, comparisons = 0;
  const int rounding = std::fegetround();
  const unsigned original_csr = _mm_getcsr();
  std::mt19937 random(7391);
  constexpr bool half = !std::is_same_v<T, float>;
  const uint32_t sign = half ? 0x8000U : 0x80000000U;
  const uint32_t infinity = half ? 0x7c00U : 0x7f800000U;
  const uint32_t nan = half ? 0x7e00U : 0x7fc00000U;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(mode);
    for (size_t dim : {8U, 15U, 16U, 17U, 31U, 32U, 33U, 64U, 96U, 256U, 768U}) {
      for (int special = 0; special < 10; ++special) {
        std::vector<T> query(dim), rows(32 * dim);
        for (auto *values : {&query, &rows}) {
          for (T &x : *values) {
            uint32_t b = half ? 0x2800U + random() % 0x2400U
                              : 0x3c000000U + (random() & 0x07ffffffU);
            if (random() & 1) b |= sign;
            x = value<T>(b);
          }
        }
        if (special == 1) {
          for (size_t d = 0; d < dim; ++d) query[d] = value<T>((d & 1) ? sign : 0);
          for (size_t d = 0; d < rows.size(); ++d) rows[d] = value<T>((d & 1) ? 0 : sign);
        } else if (special == 2) {
          query[0] = value<T>(infinity);
          for (size_t row = 0; row < 32; ++row) {
            rows[row * dim] = value<T>(infinity | ((row & 1) ? sign : 0));
            rows[row * dim + 1] = value<T>((row & 1) ? sign : 0);
          }
        } else if (special == 3) {
          query[0] = value<T>(nan | 0x2aU);
          for (size_t row = 0; row < 32; ++row)
            rows[row * dim] = value<T>(nan | (0x139U + row));
        } else if (special == 4) {
          for (size_t row = 0; row < 32; ++row)
            for (size_t d = 0; d < dim; d += 7)
              rows[row * dim + d] = value<T>(nan | (1 + (row * 17 + d) % 511)
                                              | ((d & 1) ? sign : 0));
        } else if (special == 5) {
          for (size_t d = 0; d < dim; ++d)
            query[d] = value<T>((1 + d % 127) | ((d & 1) ? sign : 0));
          for (size_t d = 0; d < rows.size(); ++d)
            rows[d] = value<T>((1 + d % 251) | ((d & 1) ? 0 : sign));
        } else if (special == 6) {
          query[0] = value<T>(infinity | 0x2aU);
          for (size_t row = 0; row < 32; ++row)
            rows[row * dim] = value<T>(infinity | (0x139U + row));
        } else if (special == 7) {
          const uint32_t largest = half ? 0x7bffU : 0x7f7fffffU;
          for (T &x : query) x = value<T>(largest);
          for (size_t row = 0; row < 32; ++row)
            for (size_t d = 0; d < dim; ++d)
              rows[row * dim + d] = value<T>(largest | ((row & 1) ? sign : 0));
        } else if (special == 8) {
          if constexpr (!half) {
            for (T &x : query) x = value<T>(0x1c800000U);
            for (T &x : rows) x = value<T>(0x1c800000U);
          }
        } else if (special == 9) {
          if constexpr (!half) {
            for (T &x : query) x = value<T>(0);
            for (T &x : rows) x = value<T>(0);
            const size_t width = std::strstr(name, "avx2/") == name ? 8 : 16;
            if (dim >= width) {
              const size_t start = dim / width * width - width;
              const size_t other = start + (width == 8 ? 2 : 1);
              query[start] = value<T>(0x1f800000U);
              query[other] = value<T>(0x1f800001U);
              for (size_t row = 0; row < 32; ++row) {
                rows[row * dim + start] = value<T>(0x20800000U);
                rows[row * dim + other] = value<T>(0xa0800000U);
              }
            }
          }
        }
        std::array<const void *, 32> pointers;
        for (size_t row = 0; row < pointers.size(); ++row) pointers[row] = rows.data() + row * dim;
        for (size_t count = 1; count <= 32; ++count) {
          if (count > 25 && count != 32) continue;
          std::array<float, 34> expected, actual;
          expected.fill(value<float>(0xdeadbeefU));
          actual = expected;
          std::feclearexcept(FE_ALL_EXCEPT);
          _mm_setcsr(_mm_getcsr() & ~0x3fu);
          original(pointers.data(), query.data(), count, dim, expected.data() + 1, nullptr);
          const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
          const unsigned expected_csr = _mm_getcsr() & 0x3fu;
          std::feclearexcept(FE_ALL_EXCEPT);
          _mm_setcsr(_mm_getcsr() & ~0x3fu);
          current(pointers.data(), query.data(), count, dim, actual.data() + 1, nullptr);
          const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
          const unsigned actual_csr = _mm_getcsr() & 0x3fu;
          ++comparisons;
          if ((expected_flags != actual_flags || expected_csr != actual_csr) && failures++ < 16)
            std::fprintf(stderr, "%s dim=%zu count=%zu mode=%d case=%d expected-flags=%x actual-flags=%x expected-MXCSR=%x actual-MXCSR=%x\n",
                         name, dim, count, mode, special, expected_flags, actual_flags, expected_csr, actual_csr);
          for (size_t row = 0; row < actual.size(); ++row) {
            const uint32_t wanted = row && row <= count ? bits(expected[row]) : 0xdeadbeefU;
            if (bits(actual[row]) == wanted && bits(expected[row]) == wanted) continue;
            if (failures++ < 16)
              std::fprintf(stderr, "%s dim=%zu count=%zu mode=%d case=%d slot=%zu expected=%08x actual=%08x\n",
                           name, dim, count, mode, special, row, wanted, bits(actual[row]));
          }
        }
      }
    }
  }
  std::fesetround(rounding);
  _mm_setcsr(original_csr);
  std::printf("%s: %zu comparisons, %zu output/flag differences\n", name, comparisons, failures);
  return failures;
}
using Decode = bool (*)(const uint16_t *, size_t, float, float *);
size_t check_decode(const char *name, Decode current, Decode original) {
  size_t failures = 0, comparisons = 0;
  const unsigned original_csr = _mm_getcsr();
  const int rounding = std::fegetround();
  auto compare = [&](const uint16_t *input, size_t dim, float norm) {
    std::vector<float> expected(dim + 2, 42.25f), actual(expected);
    const unsigned csr = _mm_getcsr() & ~0x3fu;
    _mm_setcsr(csr);
    const bool expected_ok = original(input, dim, norm, expected.data() + 1);
    const unsigned expected_flags = _mm_getcsr() & 0x3fu;
    _mm_setcsr(csr);
    const bool actual_ok = current(input, dim, norm, actual.data() + 1);
    const unsigned actual_flags = _mm_getcsr() & 0x3fu;
    ++comparisons;
    if (!expected_ok || !actual_ok || expected_flags != actual_flags ||
        std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float))) {
      if (failures++ < 16)
        std::fprintf(stderr, "%s dim=%zu norm=%a mode=%d expected-MXCSR=%x actual-MXCSR=%x\n",
                     name, dim, norm, std::fegetround(), expected_flags, actual_flags);
    }
    if (dim == 8 && norm == 0.f && std::fegetround() == FE_TONEAREST)
      std::printf("%s half-subnormal zero product: frozen-MXCSR=%x current-MXCSR=%x\n",
                  name, expected_flags, actual_flags);
  };
  std::vector<uint16_t> all_half(65537);
  for (size_t i = 0; i < 65536; ++i) all_half[i + 1] = static_cast<uint16_t>(i);
  const std::array<uint16_t, 16> edges = {
      0x3c00, 0xbc00, 0x3555, 0xb555, 0x0400, 0x8400, 0x7bff, 0xfbff,
      0x0001, 0x8001, 0x0000, 0x8000, 0x7e11, 0x7d02, 0xfe23, 0x7c00};
  std::array<uint16_t, 35> short_input;
  std::array<uint16_t, 33> normal_input;
  for (size_t i = 0; i < short_input.size(); ++i) short_input[i] = edges[i % edges.size()];
  for (size_t i = 0; i < normal_input.size(); ++i) normal_input[i] = edges[i % 8];
  const std::array<float, 15> norms = {0.f, -0.f, 1.f, -1.f, .003f, 2.3f, 1000.f,
      1.00048828125f, 0x1p-25f, 0x1p-126f, 0x1p-149f, 0x1.fffffep-15f,
      0x1p-14f, 0x1.ffdffep+15f, 0x1.ffep+15f};
  std::mt19937 random(8193);
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(mode);
    for (float norm : norms) {
      compare(all_half.data() + 1, 65536, norm);
      for (size_t dim = 0; dim <= 33; ++dim)
        compare(short_input.data() + 1, dim, norm);
      for (size_t dim : {16U, 17U, 32U, 33U})
        compare(normal_input.data(), dim, norm);
    }
    for (size_t trial = 0; trial < 4096; ++trial) {
      std::array<uint16_t, 33> input;
      for (auto &x : input) x = static_cast<uint16_t>(random());
      uint32_t norm_bits = random();
      if (trial & 1) norm_bits = (norm_bits & 0x807fffffU) | ((112 + trial % 30) << 23);
      if ((norm_bits & 0x7f800000U) == 0x7f800000U) norm_bits &= ~0x00800000U;
      compare(input.data(), trial % 34, value<float>(norm_bits));
    }
  }
  std::fesetround(rounding);
  _mm_setcsr(original_csr);
  std::printf("%s: %zu comparisons, %zu output/MXCSR differences\n", name, comparisons, failures);
  return failures;
}
int main() {
  __builtin_cpu_init();
  const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
                    && __builtin_cpu_supports("f16c");
  const bool avx512 = avx2 && __builtin_cpu_supports("avx512f")
                     && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw")
                     && __builtin_cpu_supports("avx512vl");
  size_t failures = 0, executed = 0;
@CHECKS@
  if (!executed) { std::fprintf(stderr, "No supported kernel was checked\n"); return 1; }
  return failures ? 1 : 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-ref", required=True, help="Frozen source commit (not the current checkout)")
    libraries = parser.add_mutually_exclusive_group(required=True)
    libraries.add_argument("--current-lib-dir", type=Path)
    libraries.add_argument("--build-root", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--avx2-march", default="core-avx2")
    parser.add_argument("--avx512-march", default="icelake-server")
    parser.add_argument("--output", type=Path, help="Keep the compiled proof binary at this path")
    parser.add_argument("--build-only", action="store_true", help="Compile without executing (requires --output)")
    parser.add_argument("--distance-only", action="store_true", help="Skip decoder proof for references predating the decoder")
    args = parser.parse_args()
    if args.build_only and args.output is None:
        parser.error("--build-only requires --output")
    root = args.source_root.resolve()
    source_ref = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{args.source_ref}^{{commit}}"],
        text=True,
    ).strip()
    libdir = args.current_lib_dir
    if libdir is None:
        candidates = [args.build_root / "lib", args.build_root / "zvec/lib"]
        libdir = next((p for p in candidates if (p / "libzvec_turbo.a").is_file()), candidates[0])
    archives = [libdir.resolve() / f"libzvec_{name}.a" for name in ("turbo", "ailego")]
    for archive in archives:
        if not archive.is_file():
            parser.error(f"missing current library: {archive}")
    compiler = shlex.split(args.cxx)
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-fno-strict-aliasing", "-pthread"]
    includes = [f"-I{root / p}" for p in ("src/include", "src", "src/turbo", "src/turbo/distance")]
    declarations, checks = [], []
    with tempfile.TemporaryDirectory(prefix="zvec-exact-bits-") as directory:
        temporary = Path(directory)
        # FP16 IP's batch width is defined in this header, not its .cc file.
        header = "avx2/fp16/inner_product_common.h"
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{source_ref}:src/turbo/distance/{header}"], text=True)
        namespace = "namespace zvec::turbo::avx2 {"
        if source.count(namespace) != 1:
            raise RuntimeError(f"unexpected namespace in frozen header: {header}")
        frozen_header = temporary / header
        frozen_header.parent.mkdir(parents=True)
        frozen_header.write_text(source.replace(namespace, "namespace zvec::turbo::reference::avx2 {", 1))
        includes.insert(0, f"-I{temporary}")
        objects = []
        for index, (isa, path, symbol, half) in enumerate(KERNELS):
            filename = f"src/turbo/distance/{isa}/{path}.cc"
            source = subprocess.check_output(["git", "-C", str(root), "show", f"{source_ref}:{filename}"], text=True)
            namespace = f"namespace zvec::turbo::{isa} {{"
            if source.count(namespace) != 1:
                raise RuntimeError(f"unexpected namespace in frozen source: {filename}")
            source = source.replace(namespace, f"namespace zvec::turbo::reference::{isa} {{", 1)
            reference, obj = temporary / f"reference{index}.cc", temporary / f"reference{index}.o"
            reference.write_text(source)
            march = args.avx2_march if isa == "avx2" else args.avx512_march
            subprocess.run(compiler + flags + includes + [f"-march={march}", "-c", str(reference), "-o", str(obj)], check=True)
            objects.append(str(obj))
            for prefix in ("", "reference::"):
                declarations.append(f"namespace zvec::turbo::{prefix}{isa} {{ void {symbol}(const void *const *, const void *, size_t, size_t, float *, const void *const *); }}")
            checks.append(f'  if ({"avx2" if isa == "avx2" else "avx512"}) {{ ++executed; failures += check<{"uint16_t" if half else "float"}>("{isa}/{path}", zvec::turbo::{isa}::{symbol}, zvec::turbo::reference::{isa}::{symbol}); }} else std::puts("SKIP {isa}/{path}: unavailable CPU features");')
        if not args.distance_only:
            filename = "src/turbo/conversion/avx2/cosine_decode.cc"
            source = subprocess.check_output(["git", "-C", str(root), "show", f"{source_ref}:{filename}"], text=True)
            namespace = "namespace zvec::turbo::avx2 {"
            if source.count(namespace) != 1:
                raise RuntimeError(f"unexpected namespace in frozen source: {filename}")
            reference, obj = temporary / "reference_decode.cc", temporary / "reference_decode.o"
            reference.write_text(source.replace(namespace, "namespace zvec::turbo::reference::avx2 {", 1))
            subprocess.run(compiler + flags + includes + [f"-march={args.avx2_march}", "-c", str(reference), "-o", str(obj)], check=True)
            objects.append(str(obj))
            for isa in ("avx2", "avx512", "reference::avx2"):
                declarations.append(f"namespace zvec::turbo::{isa} {{ bool fp16_cosine_decode(const uint16_t *, size_t, float, float *); }}")
            for isa in ("avx2", "avx512"):
                checks.append(f'  if ({isa}) {{ ++executed; failures += check_decode("{isa}/cosine_decode", zvec::turbo::{isa}::fp16_cosine_decode, zvec::turbo::reference::avx2::fp16_cosine_decode); }} else std::puts("SKIP {isa}/cosine_decode: unavailable CPU features");')
        driver = temporary / "driver.cc"
        driver.write_text(DRIVER.replace("@DECLARATIONS@", "\n".join(declarations)).replace("@CHECKS@", "\n".join(checks)))
        executable = args.output.resolve() if args.output else temporary / "exact-bits"
        executable.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(compiler + flags + ["-march=x86-64", str(driver), *objects,
                       "-Wl,--start-group", *map(str, archives), "-Wl,--end-group", "-ldl", "-o", str(executable)], check=True)
        print(f"Frozen source: {source_ref}; current libraries: {libdir.resolve()}", flush=True)
        if not args.build_only:
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
