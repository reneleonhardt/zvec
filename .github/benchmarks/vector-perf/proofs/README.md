# Arithmetic proof checks

Run from the repository root, with Python 3 and Z3 on `PATH`:

```sh
nice -n 10 python3 .github/benchmarks/vector-perf/proofs/test_check.py
nice -n 10 python3 .github/benchmarks/vector-perf/proofs/check.py
```

## Source and compiler checks

With Clang installed, check the actual C++ AST, optimized LLVM IR and emitted
assembly for both x86-64 and ARM targets, without linking project libraries:

```sh
nice -n 10 python3 .github/benchmarks/vector-perf/proofs/test_compiler_check.py
nice -n 10 python3 .github/benchmarks/vector-perf/proofs/test_register_check.py
nice -n 10 python3 .github/benchmarks/vector-perf/proofs/compiler_check.py \
  --cache .local/compiler-proof-cache --output .local/compiler-proof-report.json
```

Linux needs ARM cross C++ headers (`libstdc++-15-dev-arm64-cross` on Ubuntu). macOS uses
Xcode's SDK for both targets. Linux can additionally check GCC assembly with
`--gnu-cxx g++ --gnu-arm-cxx aarch64-linux-gnu-g++` when both GCC toolchains are
installed. CI uses the smaller headers-only Clang setup. No CPU
feature is skipped based on the machine executing the compiler: all eight
source families, including native AVX-512 FP16, are cross-compiled. Compilation
does **not** imply execution on those CPUs.

The compiler check currently covers 37 translation units per compiler: scalar,
SSE2, AVX2/FMA/F16C, AVX-512, raw FP16/VNNI-namespace, native AVX-512 FP16,
NEON and NEON FP16 IP/L2/cosine paths, split kernels and both cosine decoders.

* AST checks extract the actual accumulator feedback edges, argument order and
  vector types. Wrong-chain FMA feedback is rejected. Distinct extracted graph
  signatures instantiate raw-vector-bit and sticky-flag scheduling obligations;
  duplicate edges reuse the same theorem rather than adding numerical fixtures.
* Decoder AST checks match the integer rounding, signed comparison and special
  classification expressions to the proved formulas. The rounding declaration
  must be inside both the special-input and range guards.
* Optimized LLVM IR rejects fast-math assumptions, reassociation of accumulator
  updates, and half arithmetic in widened FP32 kernels. Explicit horizontal
  reduction intrinsics are the narrow exception: Clang represents some with a
  `reassoc` reduction call even under ordinary compilation. This does not permit
  `reassoc` on a loop's FMA/add/multiply instructions.
* Assembly audits reject inactive stubs, VEX/EVEX leakage into SSE2, AVX-512
  registers in AVX2, half arithmetic in widened paths, and FP-environment writes.
* `--compile-commands build/compile_commands.json --build-options-only` rejects
  unsafe floating-point options in an actual Turbo build. The benchmark workflow
  runs this check after CMake configuration and before compiling a new candidate.

The artifact cache checks compiler/driver identity, command and target options,
relevant include/SDK environment, every compiler-reported source/header content,
and cached artifact checksums. Header/compiler changes invalidate affected
entries; artifact corruption fails closed. Only two low-priority compilations
run concurrently. Local cache/report files are not project deliverables.

Real-compiler negative controls deliberately compile wrong-chain feedback and
fast-math code, and mutate a parsed decoder rounding constant or remove guards.
Cheap unit controls cover ISA leakage, half arithmetic and cache invalidation.

## Register rewriting and FMA payload priority

Every Clang translation unit is also compiled with `-verify-regalloc`,
`-verify-coalescing`, `-verify-machineinstrs` and `-verify-misched`. These enable
LLVM's structural checks of live intervals, register interference, machine
instructions and scheduling. Available checks depend on the compiler build;
these options are not an independent proof of the allocator. Optional GCC runs
use `-fchecking=2` for its available internal checks.

`register_check.py` independently compares LLVM's machine instructions immediately
before and after `virtregrewriter`, the pass that replaces virtual registers
with physical registers. It requires the same blocks, stable instruction slots,
opcodes, ordered operands, immediate values, memory operands and FP-state uses.
A virtual register must map consistently to one physical storage location,
including its checked sub-register views. Surviving copies keep their original
slots; an eliminated copy must resolve to identical register views. Stack
addresses and spill/reload instructions are compared too. Missing, truncated
or unrecognized instruction dumps fail closed.

This checks FMA's **encoded operand roles**, including x86 132/213/231 forms and
ARM FP16/FP32 FMLA/FMADD. It never equates multiplication operands or substitutes
one FMA encoding for another. Identical instruction semantics with identical
ordered raw-bit operands preserve any hardware NaN-payload priority and exception
behavior at this pass boundary. This argument covers arbitrary payloads without
a payload fixture set or an assumed common Intel/AMD/ARM priority table. It
does not derive a vendor's priority rule or prove the hardware implements it.
Scalar and SSE2 paths receive the same register checks even where no FMA exists.

The verifier-enabled assembly must be byte-identical to the ordinary compile.
Verification artifacts reuse a separate dependency-checked sidecar cache, so
the existing AST/IR cache stays warm. Tests deliberately corrupt FMA encodings,
operand order, register assignments/views, copy slots, FP-state uses and stack
addresses; one control mutates an actual compiler dump.

`check.py --emit path.smt2` saves the exact solver input for independent inspection or
execution with `z3 path.smt2`. Each ordinary query asks for a counterexample;
`unsat` is the required result. Deliberately broken negative controls require
`sat`. Solver errors, missing answers, `unknown`, and timeouts fail the check.
These tools are never linked into the library and add no runtime comparisons,
allocations, or CPU-feature dependencies.

## Classifiers and decoder

* Every current integer FP32 NaN classifier in the distance sources is extracted
  and checked against exponent-all-ones plus nonzero fraction, for arbitrary
  32-bit inputs. This includes both signs and every quiet/signaling payload.
* Both x86 cosine decoders exclude all special FP16 input encodings before
  conversion. The symbolic exponent check covers every 16-bit encoding.
  The checker validates the scalar widening tables and symbolically proves
  widening every finite half, including subnormals and signed zero, has the
  same bits as IEEE FP16 -> FP32 conversion.
* For either sign and every FP32 result whose magnitude is in
  `0x38800000..0x477fefff`, the scalar helper's table indices are in its normal
  half domain. The checker reads and validates every reachable table entry.
  The symbolic proof then compares its ties-away packing/unpacking with the
  SIMD add-4096/clear-low-13-bits transformation.
* The rounded result is a finite normal half, represented exactly as FP32.
  An IEEE floating-point query additionally proves FP32 -> FP16 -> FP32 is
  bit-identical for **every rounding mode**. Thus these conversions need no
  rounding, underflow, or overflow and add no conversion exception flags under
  the documented x86 conversion semantics. The preceding FP32 multiplication
  retains its original operation; its own exceptions are not suppressed.

The upper endpoint matters: extending it through `0x477ff000` allows conversion
to half infinity and can add a hardware overflow flag that the scalar integer
packer does not raise. The negative controls also reject treating infinity as
NaN and changing the rounding increment.

## Accumulation invariants

`accumulation.py` instantiates the same inductive argument for the following
ordered graph shapes. Width is the arithmetic lane width, not the input width.

| Graph | Lane bits x lanes | Chains | Updates/chain/block |
| --- | --- | --- | --- |
| Scalar FP32 / widened FP16 | 32 x 1 | 1 | 1 |
| SSE2 FP32 / widened FP16 | 32 x 4 | 1 | 1 |
| AVX2 FP32 ordered mul/add or FMA | 32 x 8 | 1 | 1 |
| AVX2 F16C FP16 FMA | 32 x 8 | 1 | 2 |
| AVX-512 FP32 ordered mul/add or FMA | 32 x 16 | 1 | 1 |
| AVX-512 F16C FP16 FMA | 32 x 16 | 1 | 2 |
| AVX-512 VNNI namespace raw FP16 L2 | 32 x 16 | 2 | 1 |
| Native AVX-512 FP16 | 16 x 32 | 2 | 1 |
| NEON FP32 / widened FP16 | 32 x 4 | 2 | 1 |
| NEON FP16 widened IP/L2 | 32 x 4 | 4 | 1 |
| NEON FP16 native cosine | 16 x 8 | 4 | 1 |

The invariant is: **after any number of complete blocks, every row/chain/lane
has the same raw bits, and the final accumulated sticky flags are equal**, when
the optimization only interleaves independent rows while keeping their ordered
instruction graphs unchanged.

An instruction sequence is modeled as deterministic functions returning raw
bits and raised flag bits, parameterized by its FP environment. No real-number
identities or assumptions about NaN payload selection are used. The solver
checks one arbitrary block from arbitrary incoming accumulator bits, allowing
row order and chain order to interleave, and checks address injectivity for
arbitrary row/chain/lane indices. Identical +0 initialization establishes the
base case; the block lemma extends it by induction. Adjacent independent-row
swaps extend the two-row lemma to any finite batch size. Ordered reduction and
tail graphs preserve equality by congruence. Repeated shared-query conversion
raises the same final sticky flags because bitwise OR is idempotent.

The instruction functions are opaque on purpose. The theorem holds for *any*
fixed deterministic NaN payload and exception semantics, including flush modes,
but requires the same ordered instruction sequence on both sides. It does not
prove FMA equal to separate multiply/add, choose a compiler's FMA operand form,
or equate distinct single-row/batch or CPU-family implementations. The source
mapping and actual compiler output remain trusted/reviewed boundaries. The
token manifest covers the source implementations and their local helper headers;
a new matching source is also rejected until reviewed.

The one-update shapes also apply to unpruned split kernels after reconstructing
the same logical input block. This does not prove the split loader's pointer
arithmetic or bounds; the existing guard-page tests check that integration.

Negative controls detect moving updates into the wrong chain, assuming
commutative NaN operand selection, and replacing FMA with separate mul/add.
The latter uses an actual IEEE counterexample, not an uninterpreted function.

## Proof boundary

The trusted source-to-model translation is deliberately small and reviewable.
`sources.json` pins the token stream of each modeled production source. Source
changes fail closed until the translation and its obligations are reviewed;
comments and formatting do not invalidate it. **Updating a fingerprint is not
a proof of the changed source.** There is no automatic fingerprint-update
option in the checker.

The AST checks reduce the trusted source-to-model boundary for the selected
expressions and recurrences. They do not translate arbitrary C++ control flow,
aliasing, split-pointer reconstruction, every initializer or helper into a
verified intermediate language. Source fingerprints remain required too.

The IR/assembly checks are practical compiler regression checks, **not formal
translation validation of every emitted instruction**. The register validator
covers physical-register rewriting, not the preceding allocator's splitting,
rematerialization, spilling or FMA commuting transformations. Those receive
compiler structural checks, not independent semantic translation validation.
Earlier instruction selection, later pseudo expansion, horizontal-reduction
lowering and CPU behavior remain outside this independent check. GCC does not
have the LLVM dump comparison. Native bit/exception comparisons in the parent
directory remain necessary. In particular, SMT-LIB values merge NaN payloads and
quiet/signaling NaNs; we use raw bitvectors for classification and only use the
IEEE conversion model inside a proved finite normal domain.

The checks do not claim that finite-cutoff pruning evaluates rejected rows or
reproduces exception flags from skipped arithmetic. They do not cover malformed
index files, memory safety, unmasked floating-point traps, or changing the FP
environment asynchronously during a call.

Semantics: [SMT-LIB bitvectors](https://smt-lib.org/theories-FixedSizeBitVectors.shtml)
and [floating point](https://smt-lib.org/theories-FloatingPoint.shtml).
Compiler boundaries: LLVM's [register rewriter](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/VirtRegMap.cpp),
[machine verifier](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/MachineVerifier.cpp)
and [FMA commuting rules](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrFMA.td).
