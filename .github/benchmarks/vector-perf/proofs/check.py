#!/usr/bin/env python3
"""Check arithmetic proof obligations with Z3; no project build is required."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import accumulation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def bv(value, width=32):
    return f"(_ bv{value % (1 << width)} {width})"


def source(root, path):
    return (root / path).read_text()


def tokens(text):
    """Ignore comments and formatting, but never ignore changed source tokens."""
    lexemes = re.findall(r'''//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\w+|[^\s]''',
                         text, flags=re.S)
    return [item for item in lexemes if not item.startswith(("//", "/*"))]


def fingerprint(text):
    return hashlib.sha256(json.dumps(tokens(text)).encode()).hexdigest()


def check_sources(root, manifest):
    for path, expected in manifest.items():
        if fingerprint(source(root, path)) != expected:
            raise ValueError(f"source/model review required: {path}")


def accumulation_sources(root):
    paths = set()
    distance = root / "src/turbo/distance"
    for family in distance.iterdir():
        for dtype in ("fp16", "fp32"):
            for path in (family / dtype).glob("*"):
                if path.suffix in (".cc", ".h") and any(
                        name in path.stem for name in ("inner_product", "squared_euclidean", "cosine", "common", "distance", "fp32_batch")):
                    paths.add(path.relative_to(root).as_posix())
    paths.add("src/turbo/distance/sse2/simd.h")
    paths.update(f"src/turbo/distance/{family}/split_distance.cc" for family in ("avx2", "avx512"))
    return paths


def required_sources(root):
    return accumulation_sources(root) | {
        "src/ailego/utility/float_helper.cc",
        "src/turbo/conversion/avx2/cosine_decode.cc",
        "src/turbo/conversion/avx512/cosine_decode.cc",
    }


class Proofs:
    def __init__(self):
        self.cases = []

    def add(self, name, declarations, counterexample, expected="unsat"):
        self.cases.append((name, declarations, counterexample, expected))

    def run(self, solver="z3", output=None):
        script = ["(set-option :timeout 10000)"]
        for name, declarations, counterexample, _ in self.cases:
            script += [f'(echo "{name}")', "(push)", declarations,
                       f"(assert {counterexample})", "(check-sat)", "(pop)"]
        script.append("(exit)")
        script = "\n".join(script) + "\n"
        if output:
            output.write_text(script)
        result = subprocess.run([solver, "-in", "-smt2"], input=script,
                                capture_output=True, text=True, timeout=180)
        expected = [item for case in self.cases for item in (case[0], case[3])]
        actual = result.stdout.splitlines()
        if result.returncode or actual != expected or result.stderr:
            raise RuntimeError(f"proof failed (unknown/timeouts are failures):\n"
                               f"{result.stdout}\n{result.stderr}")
        for name, _, _, status in self.cases:
            print(f"{name}: {status}")
        print(f"PASS {len(self.cases)} solver obligations")


def classifiers(proofs, root):
    distance = root / "src/turbo/distance"
    pattern = r"\(bits & (0x[\da-f]+)U\) > (0x[\da-f]+)U"
    sites = []
    for path in sorted(distance.rglob("*.cc")):
        for mask, threshold in re.findall(pattern, path.read_text()):
            sites.append((path.relative_to(root).as_posix(), int(mask, 16),
                          int(threshold, 16)))
    if len(sites) != 10:
        raise ValueError(f"review classifier coverage: expected 10 sites, got {len(sites)}")
    for index, (path, mask, threshold) in enumerate(sites):
        declarations = "(declare-const b (_ BitVec 32))"
        predicate = f"(bvugt (bvand b {bv(mask)}) {bv(threshold)})"
        definition = ("(and (= ((_ extract 30 23) b) #xff) "
                      "(not (= ((_ extract 22 0) b) (_ bv0 23))))")
        proofs.add(f"nan/{index}/{path}", declarations,
                   f"(not (= {predicate} {definition}))")
    proofs.add("negative-control/nan-infinity", declarations,
               "(and (= b #x7f800000) "
               "(bvuge (bvand b #x7fffffff) #x7f800000))", "sat")


def table(text, name, size=512):
    match = re.search(rf"{name}\[{size}\]\s*=\s*\{{(.*?)\}}", text, re.S)
    if not match:
        raise ValueError(f"missing scalar table {name}")
    values = [int(x.strip(), 0) for x in match[1].split(",") if x.strip()]
    if len(values) != size:
        raise ValueError(f"wrong scalar table size: {name}")
    return values


def decoder(proofs, root):
    helper = source(root, "src/ailego/utility/float_helper.cc")
    bases, shifts = table(helper, "base_table"), table(helper, "shift_table")
    mantissas = table(helper, "mantissa_table", 2048)
    exponents = table(helper, "exponent_table", 64)
    offsets = table(helper, "offset_table", 64)
    for fraction in range(1024):
        top = fraction.bit_length() - 1
        subnormal = 0 if not fraction else ((top + 103) << 23) | ((fraction << (23 - top)) & 0x7fffff)
        if mantissas[fraction] != subnormal or mantissas[1024 + fraction] != 0x38000000 + (fraction << 13):
            raise ValueError("scalar widening mantissa table no longer matches the model")
    # These are all table entries reachable by the symbolic fast interval.
    for sign in (0, 1):
        for exponent in range(113, 143):
            index = sign * 256 + exponent
            if bases[index] != (sign << 15) + ((exponent - 112) << 10) or shifts[index] != 13:
                raise ValueError("scalar normal-half table no longer matches the model")
        for exponent in range(31):
            index = sign * 32 + exponent
            if exponents[index] != (sign << 31) + (exponent << 23) or offsets[index] != (1024 if exponent else 0):
                raise ValueError("scalar widening exponent/offset table no longer matches the model")
    subnormal = "#x00000000"
    for top in range(10):
        value = f"(bvor {bv((top + 103) << 23)} (bvand (bvshl fraction {bv(23 - top)}) #x007fffff))"
        subnormal = f"(ite (bvuge fraction {bv(1 << top)}) {value} {subnormal})"
    proofs.add("f16c/finite-input-widening-exact-bits", f"""
(declare-const h (_ BitVec 16))
(define-fun exponent () (_ BitVec 32) ((_ zero_extend 27) ((_ extract 14 10) h)))
(define-fun fraction () (_ BitVec 32) ((_ zero_extend 22) ((_ extract 9 0) h)))
(define-fun sign () (_ BitVec 32) (bvshl ((_ zero_extend 16) (bvand h #x8000)) (_ bv16 32)))
(define-fun scalar () (_ BitVec 32)
  (bvor sign (ite (= exponent (_ bv0 32)) {subnormal}
    (bvor (bvshl (bvadd exponent (_ bv112 32)) (_ bv23 32))
           (bvshl fraction (_ bv13 32))))))
""", "(and (not (= exponent (_ bv31 32))) "
             "(not (= scalar (fp.to_ieee_bv ((_ to_fp 8 24) RNE ((_ to_fp 5 11) h))))))")
    for isa in ("avx2", "avx512"):
        text = source(root, f"src/turbo/conversion/{isa}/cosine_decode.cc")
        constants = set(re.findall(r"0x[\da-f]+", text))
        if constants != {"0x7c00", "0x7fffffff", "0x38800000", "0x477fefff", "0x1000"}:
            raise ValueError(f"review {isa} decoder constants")
        declarations = """(declare-const b (_ BitVec 32))
(define-fun m () (_ BitVec 32) (bvand b #x7fffffff))
(define-fun safe () Bool (and (bvuge m #x38800000) (bvule m #x477fefff)))
(define-fun rounded () (_ BitVec 32) (bvand (bvadd b #x00001000) #xffffe000))
(define-fun packed () (_ BitVec 16)
  ((_ extract 15 0)
    (bvadd (bvor (bvlshr (bvand b #x80000000) (_ bv16 32))
                 (bvshl (bvsub (bvlshr m (_ bv23 32)) (_ bv112 32)) (_ bv10 32)))
           (bvadd (bvlshr (bvand b #x007fffff) (_ bv13 32))
                  (bvand (bvlshr b (_ bv12 32)) (_ bv1 32))))))
(define-fun unpacked () (_ BitVec 32)
  (bvor (bvshl ((_ zero_extend 16) (bvand packed #x8000)) (_ bv16 32))
    (bvor (bvshl (bvadd ((_ zero_extend 27) ((_ extract 14 10) packed))
                       (_ bv112 32)) (_ bv23 32))
           (bvshl ((_ zero_extend 22) ((_ extract 9 0) packed)) (_ bv13 32)))))
"""
        proofs.add(f"{isa}/decoder-table-domain", declarations,
                   "(and safe (or (bvult ((_ extract 30 23) b) (_ bv113 8)) "
                   "(bvugt ((_ extract 30 23) b) (_ bv142 8))))")
        proofs.add(f"{isa}/decoder-signed-magnitude-comparisons", declarations,
                   "(not (= safe (and (not (bvslt m #x38800000)) "
                   "(not (bvsgt m #x477fefff)))))")
        proofs.add(f"{isa}/decoder-scalar-bits", declarations,
                   "(and safe (not (= rounded unpacked)))")
        proofs.add(f"{isa}/decoder-exact-finite-conversion", declarations,
                   "(and safe (or (= ((_ extract 14 10) packed) #b00000) "
                   "(= ((_ extract 14 10) packed) #b11111) "
                   "(not (= (bvand rounded #x00001fff) #x00000000))))")
        proofs.add(f"{isa}/decoder-ieee-roundtrip-all-rounding-modes", declarations + """
(declare-const rm RoundingMode)
(define-fun narrow () (_ FloatingPoint 5 11)
  ((_ to_fp 5 11) rm ((_ to_fp 8 24) rounded)))
(define-fun wide () (_ FloatingPoint 8 24) ((_ to_fp 8 24) rm narrow))
""", "(and safe (not (= (fp.to_ieee_bv wide) rounded)))")
        proofs.add(f"{isa}/decoder-input-special-class", """(declare-const h (_ BitVec 16))
""", "(not (= (= (bvand h #x7c00) #x7c00) "
             "(= ((_ extract 14 10) h) #b11111)))")
    proofs.add("negative-control/decoder-overflow-bound", declarations,
               "(and (= b #x477ff000) "
               "(= (bvand rounded #x7fffffff) #x47800000))", "sat")
    proofs.add("negative-control/decoder-rounding-increment", declarations,
               "(and safe (= b #x3f801000) "
               "(not (= unpacked (bvand (bvadd b #x00000800) #xffffe000))))", "sat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--solver", default="z3")
    parser.add_argument("--emit", type=Path, help="also save the exact SMT-LIB queries")
    args = parser.parse_args()
    if not shutil.which(args.solver):
        parser.error("Z3 is required (Homebrew z3 / Ubuntu apt install z3)")
    manifest = json.loads((HERE / "sources.json").read_text())
    missing = required_sources(args.source_root) - manifest.keys()
    if missing:
        raise ValueError(f"unreviewed accumulation source(s): {sorted(missing)}")
    check_sources(args.source_root, manifest)
    proofs = Proofs()
    classifiers(proofs, args.source_root)
    decoder(proofs, args.source_root)
    accumulation.obligations(proofs)
    proofs.run(args.solver, args.emit)


if __name__ == "__main__":
    main()
