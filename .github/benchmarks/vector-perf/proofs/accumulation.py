"""Inductive row-scheduling lemmas over raw bits and sticky exception flags.

Instruction functions are intentionally uninterpreted: no commutativity,
associativity, NaN canonicalization, or equivalence of FMA and mul/add is assumed.
"""


# family, lane bits, lanes, independent chains, updates to each chain per block.
# These describe each family's own graph, never equality between families or
# between a family's distinct single-distance and batch implementations.
FAMILIES = (
    ("scalar/fp32", 32, 1, 1, 1),
    ("scalar/fp16", 32, 1, 1, 1),
    ("sse2/fp32", 32, 4, 1, 1),
    ("sse2/fp16", 32, 4, 1, 1),
    ("avx2/fp32", 32, 8, 1, 1),
    ("avx2/fp16-f16c-fma", 32, 8, 1, 2),
    ("avx512/fp32", 32, 16, 1, 1),
    ("avx512/fp16-f16c-fma", 32, 16, 1, 2),
    ("avx512_vnni/raw-fp16", 32, 16, 2, 1),
    ("avx512_fp16/native-fp16", 16, 32, 2, 1),
    ("neon/fp32", 32, 4, 2, 1),
    ("neon/fp16", 32, 4, 2, 1),
    ("neon_fp16/widened-ip-l2", 32, 4, 4, 1),
    ("neon_fp16/native-cosine", 16, 8, 4, 1),
)


def union(flags):
    value = "initial_flags"
    for flag in flags:
        value = f"(bvor {value} {flag})"
    return value


def schedule(chains, updates, interleave, wrong_chain=False):
    """Two arbitrary rows and one arbitrary lane; preserve each row's graph."""
    states = [[f"a{row}_{chain}" for chain in range(chains)] for row in range(2)]
    flags = []
    operations = [(row, chain, step) for row in range(2)
                  for step in range(updates) for chain in range(chains)]
    if interleave:
        operations.sort(key=lambda item: (item[2], item[1], item[0]))
    for row, chain, step in operations:
        target = (chain + 1) % chains if wrong_chain else chain
        old = states[row][target]
        # A whole ordered lane update includes any widening, subtraction,
        # FMA or separate mul/add instructions of this family and metric.
        args = f"mode q{chain}_{step} x{row}_{chain}_{step} {old}"
        flags.append(f"(raised {args})")
        states[row][target] = f"(value {args})"
    return states, union(flags)


def obligations(proofs):
    for family, width, lanes, chains, updates in FAMILIES:
        declarations = ["(declare-const mode (_ BitVec 32))",
                        "(declare-const initial_flags (_ BitVec 8))"]
        sort = f"(_ BitVec {width})"
        declarations += [f"(declare-fun value ((_ BitVec 32) {sort} {sort} {sort}) {sort})",
                         f"(declare-fun raised ((_ BitVec 32) {sort} {sort} {sort}) (_ BitVec 8))"]
        for chain in range(chains):
            for row in range(2):
                declarations.append(f"(declare-const a{row}_{chain} {sort})")
            for step in range(updates):
                declarations.append(f"(declare-const q{chain}_{step} {sort})")
                for row in range(2):
                    declarations.append(f"(declare-const x{row}_{chain}_{step} {sort})")
        original, flags = schedule(chains, updates, False)
        batched, batched_flags = schedule(chains, updates, True)
        differences = [f"(not (= {original[r][c]} {batched[r][c]}))"
                       for r in range(2) for c in range(chains)]
        differences.append(f"(not (= {flags} {batched_flags}))")
        proofs.add(f"{family}/inductive-block-bits-and-flags", "\n".join(declarations),
                   f"(or {' '.join(differences)})")
        # A scalar tail or partial block is the same lemma with one chain and
        # one update. A deterministic ordered reduction is a congruence on the
        # complete vector state, not an algebraic rewrite of its additions.
        bits = width * lanes * chains
        proofs.add(f"{family}/ordered-reduction", f"""
(declare-const a (_ BitVec {bits}))
(declare-const b (_ BitVec {bits}))
(declare-const mode (_ BitVec 32))
(declare-fun reduce ((_ BitVec 32) (_ BitVec {bits})) (_ BitVec 32))
(declare-fun reduce_flags ((_ BitVec 32) (_ BitVec {bits})) (_ BitVec 8))
""", "(and (= a b) (or (not (= (reduce mode a) (reduce mode b))) "
             "(not (= (reduce_flags mode a) (reduce_flags mode b)))))")
        proofs.add(f"{family}/lane-and-chain-address-injectivity", f"""
(declare-const row Int) (declare-const row2 Int)
(declare-const chain Int) (declare-const chain2 Int)
(declare-const lane Int) (declare-const lane2 Int)
""", f"""(and (<= 0 chain) (< chain {chains}) (<= 0 chain2) (< chain2 {chains})
  (<= 0 lane) (< lane {lanes}) (<= 0 lane2) (< lane2 {lanes})
  (= (+ (* row {chains * lanes}) (* chain {lanes}) lane)
     (+ (* row2 {chains * lanes}) (* chain2 {lanes}) lane2))
  (or (not (= row row2)) (not (= chain chain2)) (not (= lane lane2))))""")
        if chains > 1:
            broken, _ = schedule(chains, updates, True, wrong_chain=True)
            proofs.add(f"negative-control/{family}/wrong-chain", "\n".join(declarations),
                       f"(not (= {original[0][0]} {broken[0][0]}))", "sat")

    proofs.add("all-families/shared-query-conversion-sticky-flags", """
(declare-const before (_ BitVec 8))
(declare-const query_flags (_ BitVec 8))
(declare-const row_flags (_ BitVec 8))
""", "(not (= (bvor before (bvor query_flags (bvor row_flags query_flags))) "
         "(bvor before (bvor query_flags row_flags))))")
    proofs.add("all-families/induction-base", """
(declare-const flags (_ BitVec 8))
""", "(not (= (concat flags #x00000000) (concat flags #x00000000)))")
    proofs.add("negative-control/ordered-operands", """
(declare-const a (_ BitVec 32)) (declare-const b (_ BitVec 32))
(declare-const s (_ BitVec 32))
(declare-fun fma ((_ BitVec 32) (_ BitVec 32) (_ BitVec 32)) (_ BitVec 32))
""", "(not (= (fma a b s) (fma b a s)))", "sat")
    proofs.add("negative-control/fma-is-not-mul-add", """
(define-fun a () (_ FloatingPoint 8 24) ((_ to_fp 8 24) #x3f800001))
(define-fun b () (_ FloatingPoint 8 24) ((_ to_fp 8 24) #x3f7ffffe))
(define-fun c () (_ FloatingPoint 8 24) ((_ to_fp 8 24) #xbf800000))
""", "(not (= (fp.fma RNE a b c) (fp.add RNE (fp.mul RNE a b) c)))", "sat")
