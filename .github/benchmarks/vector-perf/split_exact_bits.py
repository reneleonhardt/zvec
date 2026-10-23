#!/usr/bin/env python3
"""Compare current x86 split/fallback calls with frozen-main contiguous batches."""

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

KERNELS = [
    ("avx2", f"{dtype}/{metric}", f"{metric}_{dtype}_batch_distance_avx2",
     f"{metric}_{dtype}_split_batch", dtype == "fp16", 8)
    for dtype in ("fp32", "fp16")
    for metric in ("inner_product", "cosine", "squared_euclidean")
] + [
    ("avx512", f"{dtype}/{metric}", f"{metric}_{dtype}_batch_distance_avx512",
     f"{metric}_{dtype}_split_batch", dtype == "fp16", 16)
    for dtype in ("fp32", "fp16")
    for metric in ("squared_euclidean", "inner_product", "cosine")
    if dtype == "fp32" or metric == "squared_euclidean"
] + [
    ("avx512_vnni", "fp16/squared_euclidean", "squared_euclidean_fp16_batch_distance",
     "squared_euclidean_raw_fp16_split_batch", True, 32),
]

ADAPTER = r"""
#include <algorithm>
#include "quantizer/split_distance_quantizer.h"
static size_t boundary_case = 0, rejected_calls = 0;
template <typename T, auto Split, Batch Fallback>
void split_batch(const void *const *vectors, const void *query, size_t count,
                 size_t dim, float *distances, const void *const *) {
  std::array<zvec::turbo::SplitVector, 32> spans;
  for (size_t row = 0; row < count; ++row) {
    const size_t cut = boundary_case == 0 ? 0 : boundary_case == 1 ? dim :
                       boundary_case == 2 ? (row * 7 + 1) % (dim + 1) : dim - 1;
    spans[row] = {cut ? vectors[row] : nullptr,
                  cut == dim ? nullptr : static_cast<const T *>(vectors[row]) + cut,
                  cut * sizeof(T)};
  }
  if (!Split(spans.data(), count, query, dim, distances, __builtin_inff())) {
    ++rejected_calls;
    Fallback(vectors, query, count, dim, distances, nullptr);
  }
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    libraries = parser.add_mutually_exclusive_group(required=True)
    libraries.add_argument("--current-lib-dir", type=Path)
    libraries.add_argument("--build-root", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--avx2-march", default="core-avx2")
    parser.add_argument("--avx512-march", default="icelake-server")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.build_only and args.output is None:
        parser.error("--build-only requires --output")
    root = args.source_root.resolve()
    source_ref = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{args.source_ref}^{{commit}}"], text=True).strip()
    libdir = args.current_lib_dir
    if libdir is None:
        candidates = [args.build_root / "lib", args.build_root / "zvec/lib"]
        libdir = next((p for p in candidates if (p / "libzvec_turbo.a").is_file()), candidates[0])
    archives = [libdir.resolve() / f"libzvec_{name}.a" for name in ("turbo", "ailego")]
    for archive in archives:
        if not archive.is_file():
            parser.error(f"missing current library: {archive}")
    spec = importlib.util.spec_from_file_location("exact_bits", root / ".github/benchmarks/vector-perf/exact_bits.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    driver = oracle.DRIVER.split("using Decode =", 1)[0]
    driver = driver.replace("Batch current, Batch original) {", "Batch current, Batch original, size_t multiple) {")
    driver = driver.replace("      for (int special =", "      if (dim % multiple) continue;\n      for (int special =")
    driver += ADAPTER
    compiler = shlex.split(args.cxx)
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-fno-strict-aliasing", "-pthread"]
    includes = [f"-I{root / p}" for p in ("src/include", "src", "src/turbo", "src/turbo/distance")]

    def frozen(path, isa):
        source = subprocess.check_output(["git", "-C", str(root), "show", f"{source_ref}:{path}"], text=True)
        namespace = f"namespace zvec::turbo::{isa} {{"
        if source.count(namespace) != 1:
            raise RuntimeError(f"unexpected frozen namespace in {path}")
        return source.replace(namespace, f"namespace zvec::turbo::reference::{isa} {{", 1)

    declarations = ['#include "quantizer/split_distance_quantizer.h"']
    checks = []
    with tempfile.TemporaryDirectory(prefix="zvec-x86-split-exact-") as directory:
        temporary = Path(directory)
        # Cosine must call frozen IP, and FP16 IP width lives in its header.
        headers = [(isa, path + ".h") for isa, path, *_ in KERNELS]
        headers.append(("avx2", "fp16/inner_product_common.h"))
        for isa, path in headers:
            header = temporary / isa / path
            header.parent.mkdir(parents=True, exist_ok=True)
            header.write_text(frozen(f"src/turbo/distance/{isa}/{path}", isa))
        objects = []
        for index, (isa, path, batch, split, half, multiple) in enumerate(KERNELS):
            source = temporary / f"reference{index}.cc"
            source.write_text(frozen(f"src/turbo/distance/{isa}/{path}.cc", isa))
            obj = temporary / f"reference{index}.o"
            march = args.avx2_march if isa == "avx2" else args.avx512_march
            subprocess.run(compiler + flags + [f"-I{temporary}", *includes, f"-march={march}",
                           "-c", str(source), "-o", str(obj)], check=True)
            objects.append(str(obj))
            for prefix in ("", "reference::"):
                declarations.append(f"namespace zvec::turbo::{prefix}{isa} {{ void {batch}(const void *const *, const void *, size_t, size_t, float *, const void *const *); }}")
            split_isa = "avx2" if isa == "avx2" else "avx512"
            declarations.append(f"namespace zvec::turbo::{split_isa} {{ bool {split}(const SplitVector *, size_t, const void *, size_t, float *, float); }}")
            scalar = "uint16_t" if half else "float"
            checks.append(f'''  if ({split_isa}) {{
    ++executed;
    rejected_calls = 0;
    for (boundary_case = 0; boundary_case < 4; ++boundary_case) {{
      std::printf("Boundary case %zu (empty/full/staggered/last-element)\\n", boundary_case);
      failures += check<{scalar}>("{isa}/{path}_split", split_batch<{scalar}, zvec::turbo::{split_isa}::{split}, zvec::turbo::{isa}::{batch}>, zvec::turbo::reference::{isa}::{batch}, {multiple});
    }}
    std::printf("{isa}/{path}: %zu rejected calls used current contiguous fallback\\n", rejected_calls);
  }} else std::puts("SKIP {isa}/{path}: unavailable CPU features");''')
        driver += r"""
int main() {
  __builtin_cpu_init();
  const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
                    && __builtin_cpu_supports("f16c");
  const bool avx512 = avx2 && __builtin_cpu_supports("avx512f")
                     && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw")
                     && __builtin_cpu_supports("avx512vl");
  size_t failures = 0, executed = 0;
@CHECKS@
  if (!executed) { std::fprintf(stderr, "No supported split kernel was checked\n"); return 1; }
  return failures ? 1 : 0;
}
"""
        source = temporary / "driver.cc"
        source.write_text(driver.replace("@DECLARATIONS@", "\n".join(declarations)).replace("@CHECKS@", "\n".join(checks)))
        executable = args.output.resolve() if args.output else temporary / "x86-split-exact"
        executable.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(compiler + flags + includes + ["-march=x86-64", str(source), *objects,
                       "-Wl,--start-group", *map(str, archives), "-Wl,--end-group", "-ldl", "-o", str(executable)], check=True)
        print(f"Frozen source: {source_ref}; current libraries: {libdir.resolve()}", flush=True)
        print("Split plus current contiguous fallback; all rounding modes, output bits, FE flags and six MXCSR flags", flush=True)
        if not args.build_only:
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
