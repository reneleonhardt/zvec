#!/usr/bin/env python3
"""Check restored distance kernels before building any project libraries."""

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
    restored = {("avx2", "fp32/inner_product"),
                ("avx2", "fp32/squared_euclidean"),
                ("avx512", "fp16/squared_euclidean")}
    kernels = [entry for entry in KERNELS if entry[:2] in restored]
    if len(kernels) != len(restored):
        raise RuntimeError("missing restored distance kernel family")
    print(f"Frozen source: {reference}; current sources: {root}", flush=True)
    print("Kernel flags: " + shlex.join(flags), flush=True)
    with tempfile.TemporaryDirectory(prefix="zvec-fp32-preflight-") as directory:
        temporary = Path(directory)
        objects, declarations, checks = [], [], []
        helper = temporary / "float_helper.o"
        subprocess.run(compiler + flags + includes + ["-march=x86-64",
                       "-ffunction-sections", "-fdata-sections", "-c",
                       str(root / "src/ailego/utility/float_helper.cc"),
                       "-o", str(helper)], check=True)
        objects.append(str(helper))
        for index, (isa, path, symbol, half) in enumerate(kernels):
            filename = f"src/turbo/distance/{isa}/{path}.cc"
            source = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{reference}:{filename}"], text=True)
            if (root / filename).read_bytes() != source.encode():
                raise RuntimeError(
                    f"{filename} must remain byte-identical to frozen main; "
                    "sampled inputs cannot prove every NaN payload")
            namespace = f"namespace zvec::turbo::{isa} {{"
            if source.count(namespace) != 1:
                raise RuntimeError(f"unexpected frozen namespace: {filename}")
            frozen = temporary / f"reference{index}.cc"
            frozen.write_text(source.replace(namespace, f"namespace zvec::turbo::reference::{isa} {{", 1))
            march = "core-avx2" if isa == "avx2" else "icelake-server"
            for label, translation_unit in (("reference", frozen), ("current", root / filename)):
                obj = temporary / f"{label}{index}.o"
                subprocess.run(compiler + flags + includes + [f"-march={march}", "-c",
                               str(translation_unit), "-o", str(obj)], check=True)
                objects.append(str(obj))
            for prefix in ("", "reference::"):
                declarations.append(f"namespace zvec::turbo::{prefix}{isa} {{ void {symbol}(const void *const *, const void *, size_t, size_t, float *, const void *const *); }}")
            checks.append(f'  if ({isa}) {{ ++executed; failures += check<{"uint16_t" if half else "float"}>("{isa}/{path}", zvec::turbo::{isa}::{symbol}, zvec::turbo::reference::{isa}::{symbol}); }} else std::puts("SKIP {isa}/{path}: unavailable CPU features");')
        driver = temporary / "driver.cc"
        driver.write_text(DRIVER.replace("@DECLARATIONS@", "\n".join(declarations)).replace("@CHECKS@", "\n".join(checks)))
        executable = temporary / "fp32-preflight"
        subprocess.run(compiler + flags + ["-march=x86-64", str(driver), *objects,
                       "-Wl,--gc-sections", "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
