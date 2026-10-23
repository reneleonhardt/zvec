#!/usr/bin/env python3
"""Audit actual kernel ASTs, optimized LLVM IR and assembly without linking."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess

import check
import register_check


FMA = re.compile(r"^(?:_mm(?:256|512)?_(?:mask3_)?fmadd_p[sh]|vfmaq_f(?:16|32))$")
ACCUMULATOR = re.compile(r"^(?:accumulator\w*|accs|sum\d*|sums\d*|totals?|res(?:ult)?)$")
FLOAT = re.compile(r"float|double|__m(?:128|256|512)(?:h)?(?:\[|$)|float(?:16|32)x\d")
WRAPPERS = {"ImplicitCastExpr", "ParenExpr", "ExprWithCleanups", "MaterializeTemporaryExpr"}


def walk(node):
    yield node
    for child in node.get("inner", []):
        yield from walk(child)


def unwrap(node):
    while (node.get("kind") in WRAPPERS or node.get("castKind") == "NoOp") and len(node.get("inner", [])) == 1:
        node = node["inner"][0]
    return node


def expression(node):
    node = unwrap(node)
    kind = node.get("kind")
    inner = node.get("inner", [])
    if kind == "DeclRefExpr":
        return node.get("referencedDecl", {}).get("name", "?")
    if kind == "ArraySubscriptExpr":
        return f"{expression(inner[0])}[{expression(inner[1])}]"
    if kind in ("IntegerLiteral", "FloatingLiteral"):
        return node["value"]
    if kind == "CallExpr":
        return f"{expression(inner[0])}({','.join(expression(x) for x in inner[1:])})"
    if kind == "UnresolvedLookupExpr":
        return node.get("name", "?")
    if kind == "MemberExpr":
        return (expression(inner[0]) + "." if inner else "") + node.get("name", "?")
    return f"{kind}:{node.get('opcode', '')}({','.join(expression(x) for x in inner)})"


def ast_roots(text):
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        root, offset = decoder.raw_decode(text, offset)
        yield root


def source_trace(text):
    """Extract and check feedback edges; never infer an edge from a file hash."""
    records = set()
    for root in ast_roots(text):
        for node in walk(root):
            if node.get("kind") not in ("BinaryOperator", "CompoundAssignOperator"):
                continue
            if node.get("opcode") not in ("=", "+="):
                continue
            lhs, rhs = node.get("inner", [None, None])
            name = expression(lhs)
            if not ACCUMULATOR.fullmatch(re.sub(r"\[[^]]*\]", "", name).split(".")[-1]):
                continue
            if not FLOAT.search(lhs.get("type", {}).get("qualType", "")):
                continue
            rhs = unwrap(rhs)
            if node["opcode"] == "+=":
                product = rhs.get("kind") == "BinaryOperator" and rhs.get("opcode") == "*"
                squared = rhs.get("kind") == "CallExpr" and expression(rhs["inner"][0]) == "SquaredDifference"
                if not product and not squared:
                    raise ValueError(f"unmodeled scalar accumulator update: {name} += {expression(rhs)}")
                operation = "scalar-mul-add" if product else "scalar-squared-difference-add"
            elif rhs.get("kind") == "CallExpr":
                args = rhs["inner"]
                operation = expression(args[0])
                if FMA.fullmatch(operation):
                    feedback = 1 if operation.startswith("vfmaq") else 3
                    if expression(args[feedback]) != name:
                        raise ValueError(f"wrong accumulation feedback: {name} = {expression(rhs)}")
                    if any(expression(arg) == name for i, arg in enumerate(args[1:], 1) if i != feedback):
                        raise ValueError(f"accumulator reused as a data operand: {name}")
                elif re.fullmatch(r"_mm(?:256|512)?_add_ps", operation):
                    if expression(args[1]) != name:
                        continue  # An ordered horizontal reduction, not an update.
                    product = unwrap(args[2])
                    if product.get("kind") != "CallExpr" or not re.fullmatch(
                            r"_mm(?:256|512)?_mul_ps", expression(product["inner"][0])):
                        continue
                    operation = "separate-vector-mul-add"
                else:
                    continue
            else:
                continue
            records.add((name, operation, lhs["type"]["qualType"], expression(rhs)))
    return sorted(records)


def decoder_trace(text):
    """Bind the decoder proof's integer expressions to compiler-parsed ASTs."""
    aliases = {"and_si128": "and", "and_si256": "and", "and_si512": "and",
               "or_si256": "or", "add_epi32": "add", "cmpgt_epi32": "gt",
               "cmpgt_epi32_mask": "gt", "cmplt_epi32_mask": "lt",
               "cmpeq_epi16": "eq", "castps_si256": "bits", "castps_si512": "bits",
               "movemask_epi8": "mask"}
    def canonical(node):
        node = unwrap(node)
        kind, inner = node.get("kind"), node.get("inner", [])
        if kind == "IntegerLiteral":
            return int(node["value"])
        if kind == "UnaryOperator" and node.get("opcode") == "-":
            return -canonical(inner[0])
        if kind == "UnaryOperator" and node.get("opcode") == "!":
            return ("not", canonical(inner[0]))
        if kind == "DeclRefExpr":
            return expression(node)
        if kind == "BinaryOperator" and node.get("opcode") == "|":
            return ("or", canonical(inner[0]), canonical(inner[1]))
        if kind == "CallExpr":
            name = re.sub(r"^_mm(?:256|512)?_", "", expression(inner[0]))
            args = [canonical(x) for x in inner[1:]]
            if name.startswith("set1_epi"):
                return args[0]
            if name == "__builtin_ia32_cmpd512_mask" and args[2:] in (["_MM_CMPINT_LT", -1], ["_MM_CMPINT_NLE", -1]):
                return ("lt" if args[2] == "_MM_CMPINT_LT" else "gt", args[0], args[1])
            return (aliases.get(name, name), *args)
        raise ValueError(f"unmodeled decoder AST: {kind}")
    definitions = {}
    guarded = False
    def visit(node, conditions=()):
        nonlocal guarded
        if node.get("kind") == "IfStmt":
            children = node["inner"]
            then = len(children) - (2 if node.get("hasElse") else 1)
            condition = children[then - 1]
            for child in children[:then]:
                visit(child, conditions)
            visit(children[then], conditions + ((condition, True),))
            if node.get("hasElse"):
                visit(children[-1], conditions + ((condition, False),))
            return
        if node.get("kind") == "VarDecl" and node.get("name") == "rounded":
            guards = [canonical(condition) if positive else ("not", canonical(condition))
                      for condition, positive in conditions]
            guarded = (("not", ("mask", "special")) in guards and any(
                guard in guards for guard in (("not", "outside"), ("not", ("mask", "outside")))))
        for child in node.get("inner", []):
            visit(child, conditions)
    for root in ast_roots(text):
        visit(root)
        for node in walk(root):
            if node.get("kind") == "VarDecl" and node.get("name") in ("magnitude", "rounded", "special", "outside"):
                definitions[node["name"]] = canonical(node["inner"][-1])
    expected = {
        "magnitude": ("and", "bits", 0x7fffffff),
        "rounded": ("and", ("add", "bits", 0x1000), -8192),
        "special": ("eq", ("and", "half", 0x7c00), 0x7c00),
    }
    for name, graph in expected.items():
        if definitions.get(name) != graph:
            raise ValueError(f"decoder source/model mismatch: {name}: {definitions.get(name)}")
    if not guarded:
        raise ValueError("decoder rounding is not dominated by both proved guards")
    if definitions.get("outside") not in (
            ("or", ("gt", 0x38800000, "magnitude"), ("gt", "magnitude", 0x477fefff)),
            ("or", ("lt", "magnitude", 0x38800000), ("gt", "magnitude", 0x477fefff))):
        raise ValueError(f"decoder source/model mismatch: outside: {definitions.get('outside')}")
    return definitions


def audit_ir(text, native_half):
    if not re.search(r"^define .*@", text, re.M):
        raise ValueError("no compiled function definitions")
    for line in text.splitlines():
        if line.startswith((";", "source_filename")):
            continue
        # Clang's explicit horizontal-reduction intrinsics themselves carry
        # reassoc. Never extend that permission to an accumulation instruction.
        if "@llvm.vector.reduce.fadd." in line:
            line = re.sub(r"\breassoc\s+", "", line)
        if re.search(r"\b(?:fast|reassoc|nnan|ninf|nsz|arcp|afn)\b", line):
            raise ValueError(f"unsafe LLVM floating-point assumptions: {line.strip()}")
    if re.search(r'"(?:no-nans-fp-math|no-infs-fp-math|unsafe-fp-math|no-signed-zeros-fp-math)"="true"', text):
        raise ValueError("unsafe function floating-point attributes")
    if not native_half and re.search(r"(?:fadd|fsub|fmul|fdiv)\s+(?:contract\s+)?<\d+ x half>|llvm\.(?:fma|fmuladd)\.(?:v\d+)?f16", text):
        raise ValueError("FP16 arithmetic in a widened FP32 kernel")
    if re.search(r"(?:@(?:feset|feclear|feupdate)|asm\s+sideeffect)", text):
        raise ValueError("unexpected FP environment mutation or inline assembly")


def audit_assembly(text, family, native_half=False):
    instructions = [line.strip() for line in text.splitlines()
                    if line.startswith("\t") and not line.lstrip().startswith((".", "#", "//"))]
    if not instructions or not any(re.search(r"\b(?:fadd|fmul|fmla|v?(?:add|mul)(?:ps|ss|ph|sh)|vfmadd)", line) for line in instructions):
        raise ValueError("no emitted distance arithmetic (stub or unsupported target)")
    if family == "sse2" and any(re.match(r"v\w+", line) for line in instructions):
        raise ValueError("SSE2 kernel acquired VEX/EVEX instructions")
    if family == "avx2" and re.search(r"\b(?:zmm\d+|k[0-7])\b", text):
        raise ValueError("AVX2 kernel acquired AVX-512 registers")
    if re.search(r"\b(?:ldmxcsr|vldmxcsr|fldcw|msr\s+(?:fpcr|fpsr))\b", text, re.I):
        raise ValueError("FP environment write in kernel assembly")
    if not native_half and re.search(r"\bv(?:add|sub|mul|div|fmadd\d+)(?:ph|sh)\b|\b(?:fadd|fsub|fmul|fmla|fmls|fdiv)\b[^\n]*(?:\.[48]h\b|\bh\d+\b)", text):
        raise ValueError("FP16 arithmetic in widened kernel assembly")


def audit_build_commands(path, root):
    root = root.resolve()
    bad = re.compile(r"^(?:-Ofast|-ffast-math|-funsafe-math-optimizations|-fassociative-math|-ffinite-math-only|-fno-signed-zeros|/fp:fast)$")
    count = 0
    for entry in json.loads(path.read_text()):
        filename = Path(entry["file"])
        if not filename.is_absolute():
            filename = Path(entry["directory"]) / filename
        if filename.resolve().is_relative_to(root / "src/turbo"):
            args = entry.get("arguments") or shlex.split(entry["command"])
            if any(bad.fullmatch(arg) for arg in args):
                raise ValueError(f"unsafe production compile options: {filename}")
            count += 1
    if not count:
        raise ValueError("compile database contains no Turbo sources")
    return count


def source_obligations(proofs, records):
    """Instantiate the scheduling theorem from AST-derived vector types/edges."""
    seen = set()
    for record in records:
        for index, (_, operation, dtype, _) in enumerate(record["recurrences"]):
            if (operation, dtype) in seen:
                continue
            seen.add((operation, dtype))
            if dtype.startswith("__m"):
                width = int(re.search(r"\d+", dtype)[0])
            elif re.search(r"float(16|32)x(\d+)", dtype):
                bits, lanes = map(int, re.search(r"float(16|32)x(\d+)", dtype).groups())
                width = bits * lanes
            elif dtype == "float":
                width = 32
            else:
                raise ValueError(f"unmodeled source accumulator type: {dtype}")
            sort = f"(_ BitVec {width})"
            declarations = ["(declare-const mode (_ BitVec 32))",
                            "(declare-const initial (_ BitVec 8))",
                            f"(declare-fun update ((_ BitVec 32) {sort} {sort} {sort}) {sort})",
                            f"(declare-fun flags ((_ BitVec 32) {sort} {sort} {sort}) (_ BitVec 8))",
                            f"(declare-const state (Array Int {sort}))",
                            "(declare-const r Int)", "(declare-const s Int)"]
            declarations += [f"(declare-const {name} {sort})" for name in ("q", "x", "y")]
            # Actual source feedback position was checked before creating this
            # edge. Its ordered arithmetic/conversion graph remains opaque.
            declarations += [f"""
(define-fun after_r () (Array Int {sort})
  (store state r (update mode q x (select state r))))
(define-fun after_s () (Array Int {sort})
  (store state s (update mode q y (select state s))))
(define-fun rs () (Array Int {sort})
  (store after_r s (update mode q y (select after_r s))))
(define-fun sr () (Array Int {sort})
  (store after_s r (update mode q x (select after_s r))))
(define-fun rs_flags () (_ BitVec 8)
  (bvor initial (bvor (flags mode q x (select state r)) (flags mode q y (select after_r s)))))
(define-fun sr_flags () (_ BitVec 8)
  (bvor initial (bvor (flags mode q y (select state s)) (flags mode q x (select after_s r)))))
"""]
            counterexample = "(and (distinct r s) (or (not (= rs sr)) (not (= rs_flags sr_flags))))"
            proofs.add(f"source/{record['path']}/{index}/{operation}", "\n".join(declarations), counterexample)


@lru_cache(maxsize=4)
def compiler_identity(compiler):
    identity = subprocess.check_output([*compiler, "--version"], text=True)
    programs = {Path(shutil.which(compiler[0])).resolve()}
    for name in ("clang", "cc1plus"):
        resolved = subprocess.check_output([*compiler, f"-print-prog-name={name}"], text=True).strip()
        if Path(resolved).is_file():
            programs.add(Path(resolved).resolve())
    return identity + "".join(hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(programs))


def cache_hit(directory, identity, artifacts):
    stamp = directory / "stamp.json"
    if not stamp.exists():
        return False
    saved = json.loads(stamp.read_text())
    if saved["compiler"] != identity or "artifacts" not in saved:
        return False
    if not all((directory / name).is_file() for name in artifacts):
        return False
    if not all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
               for p, digest in saved["dependencies"].items()):
        return False
    if any(hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
           for name, digest in saved["artifacts"].items()):
        raise ValueError("changed compiler-cache artifact")
    return True


def compile_one(root, cache, compiler, target, march, path, gnu=False):
    family = path.split("/")[3]
    native_half = family in ("neon_fp16", "avx512_fp16") and (family == "avx512_fp16" or path.endswith("cosine.cc"))
    command = [*compiler, "-std=c++17", "-O3", "-DNDEBUG", "-fno-strict-aliasing", "-pthread",
               f"-march={march}"]
    if gnu:
        command.append("-fchecking=2")
    if not gnu:
        command.append(f"--target={target}")
    command += [f"-I{root / p}" for p in ("src/include", "src", "src/turbo", "src/turbo/distance")]
    environment = {name: os.environ.get(name) for name in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH",
                   "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "COMPILER_PATH", "GCC_EXEC_PREFIX", "DEVELOPER_DIR")}
    key = hashlib.sha256(json.dumps([command, path, environment]).encode()).hexdigest()[:24]
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    identity = compiler_identity(tuple(compiler))
    artifacts = ("kernel.s",) if gnu else ("ast.json", "kernel.ll", "kernel.s")
    stamp = directory / "stamp.json"
    hit = cache_hit(directory, identity, artifacts)
    if not hit:
        def run(arguments, output=None):
            with (directory / "errors.txt").open("w") as errors:
                if output:
                    with output.open("w") as stream:
                        subprocess.run([*command, *arguments, str(root / path)], stdout=stream,
                                       stderr=errors, check=True, timeout=90)
                else:
                    subprocess.run([*command, *arguments, str(root / path)], stderr=errors,
                                   check=True, timeout=90)
        try:
            if not gnu:
                run(["-fsyntax-only", "-Xclang", "-ast-dump=json", "-Xclang",
                     "-ast-dump-filter=zvec::turbo"], directory / "ast.json")
                run(["-S", "-emit-llvm", "-o", str(directory / "kernel.ll")])
            run(["-S", "-MD", "-MF", str(directory / "dependencies.d"),
                 "-o", str(directory / "kernel.s")])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{path}: {error}\n{(directory / 'errors.txt').read_text()}") from error
        dependencies = (directory / "dependencies.d").read_text().replace("\\\n", " ")
        paths = shlex.split(dependencies.split(":", 1)[1])
        stamp.write_text(json.dumps({"compiler": identity, "dependencies": {
            str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "artifacts": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in artifacts}}))
    audit_assembly((directory / "kernel.s").read_text(), family, native_half)
    if gnu:
        return {"path": path, "target": target, "march": march, "cache_hit": hit,
                "compiler": "GCC assembly", "recurrences": []}
    # Keep the existing AST/IR/assembly cache warm. Verification has its own
    # compiler-keyed sidecar and must produce exactly the unaudited assembly.
    flags = register_check.VERIFY + register_check.DUMP
    verified = directory / hashlib.sha256(json.dumps(flags).encode()).hexdigest()[:16]
    verified.mkdir(exist_ok=True)
    verified_hit = cache_hit(verified, identity, ("kernel.s", "registers.txt"))
    if not verified_hit:
        with (verified / "registers.txt").open("w") as diagnostics:
            subprocess.run([*command, *flags, "-S", str(root / path), "-o", str(verified / "kernel.s")],
                           stderr=diagnostics, check=True, timeout=90)
        saved = json.loads(stamp.read_text())
        saved["artifacts"] = {name: hashlib.sha256((verified / name).read_bytes()).hexdigest()
                              for name in ("kernel.s", "registers.txt")}
        (verified / "stamp.json").write_text(json.dumps(saved))
    if (directory / "kernel.s").read_bytes() != (verified / "kernel.s").read_bytes():
        raise ValueError(f"verification changed emitted assembly: {path}")
    registers = register_check.audit((verified / "registers.txt").read_text())
    ast = (directory / "ast.json").read_text()
    trace = source_trace(ast)
    decode = decoder_trace(ast) if "/conversion/" in path else None
    audit_ir((directory / "kernel.ll").read_text(), native_half)
    if "/distance/" in path and not path.endswith("cosine.cc") and not trace:
        raise ValueError(f"{path}: no recognized accumulation edges")
    if "/distance/" in path and family != "scalar" and not path.endswith("cosine.cc") and not any(dtype != "float" for _, _, dtype, _ in trace):
        raise ValueError(f"{path}: vector accumulator edges were not extracted")
    return {"path": path, "target": target, "march": march, "cache_hit": hit,
            "register_cache_hit": verified_hit, "register_rewrite": registers,
            "recurrences": trace, "decoder": decode,
            "source": check.fingerprint((root / path).read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=check.ROOT)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-commands", type=Path, help="also audit the actual production build options")
    parser.add_argument("--gnu-cxx", help="also audit assembly from this native x86 GCC")
    parser.add_argument("--gnu-arm-cxx", help="ARM cross GCC, required together with --gnu-cxx")
    parser.add_argument("--build-options-only", action="store_true", help="audit the actual build options without compiling")
    args = parser.parse_args()
    if bool(args.gnu_cxx) != bool(args.gnu_arm_cxx):
        parser.error("provide both GNU compilers to check every CPU family")
    root = args.source_root.resolve()
    if args.compile_commands:
        print(f"PASS {audit_build_commands(args.compile_commands, root)} production compile commands")
    if args.build_options_only:
        if not args.compile_commands:
            parser.error("--build-options-only requires --compile-commands")
        return
    if args.cache is None or args.output is None:
        parser.error("--cache and --output are required for compiler checks")
    manifest = json.loads((check.HERE / "sources.json").read_text())
    if check.required_sources(root) - manifest.keys():
        raise ValueError("unreviewed source inventory")
    check.check_sources(root, manifest)
    darwin = platform.system() == "Darwin"
    targets = {"x86": "x86_64-apple-macos14" if darwin else "x86_64-linux-gnu",
               "arm": "arm64-apple-macos14" if darwin else "aarch64-linux-gnu"}
    marches = {"scalar": "x86-64", "sse2": "x86-64", "avx2": "core-avx2",
               "avx512": "icelake-server", "avx512_vnni": "icelake-server",
               "avx512_fp16": "sapphirerapids", "neon": "armv8-a", "neon_fp16": "armv8.2-a+fp16"}
    sources = sorted(p for p in check.accumulation_sources(root) if p.endswith(".cc"))
    sources += [f"src/turbo/conversion/{family}/cosine_decode.cc" for family in ("avx2", "avx512")]
    results = []
    def task(path):
        family = path.split("/")[3]
        return compile_one(root, args.cache.resolve(), shlex.split(args.cxx),
                           targets["arm" if family.startswith("neon") else "x86"], marches[family], path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(task, sources):
            results.append(record)
            print(f"PASS {record['path']}: {len(record['recurrences'])} source recurrence edges; cache={record['cache_hit']}", flush=True)
    proofs = check.Proofs()
    source_obligations(proofs, results)
    proofs.run()
    print(f"PASS {len(results)} cross-target translation units")
    if args.gnu_cxx:
        def gnu_task(path):
            family = path.split("/")[3]
            arm = family.startswith("neon")
            compiler = args.gnu_arm_cxx if arm else args.gnu_cxx
            return compile_one(root, args.cache.resolve(), shlex.split(compiler),
                               targets["arm" if arm else "x86"], marches[family], path, gnu=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for record in pool.map(gnu_task, sources):
                results.append(record)
                print(f"PASS GCC {record['path']}: cache={record['cache_hit']}", flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
