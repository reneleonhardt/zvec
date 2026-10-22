#!/usr/bin/env python3
"""Check AVX2 FP32 source parity before building any project libraries."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from exact_bits import DRIVER, KERNELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-ref", required=True, help="Frozen main commit to compare")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    args = parser.parse_args()
    root = args.source_root.resolve()
    reference = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", f"{args.source_ref}^{{commit}}"],
        text=True,
    ).strip()
    if reference != "8d4996924bce78f8b072a353803c305101bcc807":
        parser.error("--source-ref must resolve to frozen main 8d4996924bce78f8b072a353803c305101bcc807")
    compiler = shlex.split(args.cxx)
    # Match the Release benchmark kernel build; keep its default FP contraction.
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-fno-strict-aliasing", "-pthread"]
    includes = [f"-I{root / path}" for path in
                ("src/include", "src", "src/turbo", "src/turbo/distance")]
    kernels = [entry for entry in KERNELS if entry[0] == "avx2" and entry[1].startswith("fp32/")]
    if len(kernels) != 2:
        raise RuntimeError("expected exactly two AVX2 FP32 kernel families")
    print(f"Frozen source: {reference}; current sources: {root}", flush=True)
    print("Kernel flags: " + shlex.join(flags + ["-march=core-avx2"]), flush=True)
    with tempfile.TemporaryDirectory(prefix="zvec-fp32-preflight-") as directory:
        temporary = Path(directory)
        objects, declarations, checks = [], [], []
        for index, (isa, path, symbol, _) in enumerate(kernels):
            filename = f"src/turbo/distance/{isa}/{path}.cc"
            source = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{reference}:{filename}"], text=True)
            namespace = f"namespace zvec::turbo::{isa} {{"
            if source.count(namespace) != 1:
                raise RuntimeError(f"unexpected frozen namespace: {filename}")
            frozen = temporary / f"reference{index}.cc"
            frozen.write_text(source.replace(namespace, f"namespace zvec::turbo::reference::{isa} {{", 1))
            for label, translation_unit in (("reference", frozen), ("current", root / filename)):
                obj = temporary / f"{label}{index}.o"
                subprocess.run(compiler + flags + includes + ["-march=core-avx2", "-c",
                               str(translation_unit), "-o", str(obj)], check=True)
                objects.append(str(obj))
            for prefix in ("", "reference::"):
                declarations.append(f"namespace zvec::turbo::{prefix}{isa} {{ void {symbol}(const void *const *, const void *, size_t, size_t, float *, const void *const *); }}")
            checks.append(f'  if (avx2) {{ ++executed; failures += check<float>("{isa}/{path}", zvec::turbo::{isa}::{symbol}, zvec::turbo::reference::{isa}::{symbol}); }} else std::puts("SKIP {isa}/{path}: unavailable CPU features");')
        driver = temporary / "driver.cc"
        driver.write_text(DRIVER.replace("@DECLARATIONS@", "\n".join(declarations)).replace("@CHECKS@", "\n".join(checks)))
        executable = temporary / "fp32-preflight"
        subprocess.run(compiler + flags + ["-march=x86-64", str(driver), *objects,
                       "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
