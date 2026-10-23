"""Validate ordered machine operands across LLVM's physical-register rewrite.

This deliberately checks one compiler pass, not arbitrary assembly semantics.
No floating-point operation, including FMA, is treated as commutative.
"""

import re


VERIFY = [item for option in ("verify-regalloc", "verify-coalescing",
          "verify-machineinstrs", "verify-misched") for item in ("-mllvm", "-" + option)]
DUMP = [item for stage in ("before", "after")
        for item in ("-mllvm", f"-print-{stage}=virtregrewriter")]
REG = re.compile(r"%\d+(?:\.(?:sub_\w+|[bhsdq]sub))?(?::[\w]+)?|\$[\w]+")
FLAGS = re.compile(r"\b(?:renamable|killed|dead|undef|internal|early-clobber)\s+|\(tied-def \d+\)")


def physical(register):
    """Storage identity and view width; reject unknown aliasing conventions."""
    name = register.removeprefix("$")
    match = re.fullmatch(r"([xyz]mm)(\d+)", name)
    if match:
        return "vector" + match[2], {"xmm": 128, "ymm": 256, "zmm": 512}[match[1]]
    match = re.fullmatch(r"([bhsdqwx])(\d+)", name)
    if match:
        kind, index = match.groups()
        return ("gpr" if kind in "wx" else "vector") + index, dict(b=8, h=16, s=32, d=64, q=128, w=32, x=64)[kind]
    for root, aliases in {"rax": ("al", "ax", "eax", "rax"), "rbx": ("bl", "bx", "ebx", "rbx"),
                          "rcx": ("cl", "cx", "ecx", "rcx"), "rdx": ("dl", "dx", "edx", "rdx"),
                          "rsi": ("sil", "si", "esi", "rsi"), "rdi": ("dil", "di", "edi", "rdi"),
                          "rbp": ("bpl", "bp", "ebp", "rbp"), "rsp": ("spl", "sp", "esp", "rsp")}.items():
        if name in aliases:
            return root, (8, 16, 32, 64)[aliases.index(name)]
    match = re.fullmatch(r"r(\d+)([bwd]?)", name)
    if match:
        return "r" + match[1], {"b": 8, "w": 16, "d": 32, "": 64}[match[2]]
    # Fixed architectural registers are compared literally, never renamed.
    return name, None


def virtual(register):
    match = re.fullmatch(r"%(\d+)(?:\.(sub_\w+|[bhsdq]sub))?(?::(\w+))?", register)
    return match.groups() if match else None


def view_width(sub, cls):
    if sub:
        widths = {"sub_8bit": 8, "sub_16bit": 16, "sub_32bit": 32,
                  "sub_xmm": 128, "sub_ymm": 256, "sub_b": 8, "sub_h": 16,
                  "sub_s": 32, "sub_d": 64, "bsub": 8, "hsub": 16,
                  "ssub": 32, "dsub": 64, "qsub": 128, "sub_32": 32}
        if sub not in widths:
            raise ValueError(f"unmodeled register view: {sub}")
        return widths[sub]
    if re.match(r"fr(?:16|32|64)(?:\D|$)", cls or ""):
        return 128  # x86 scalar FP classes occupy an XMM register.
    match = re.match(r"(?:vr|fr|fpr|gr|gpr)(8|16|32|64|128|256|512)(?:\D|$)", cls or "")
    return int(match[1]) if match else None


def instruction(line):
    slot = re.match(r"\s*(\d+)B\s+", line)
    line = re.sub(r"^\d+B\s*", "", line.strip())
    if not line or line.startswith(("#", ";", "bb.", "successors:", "liveins:")):
        return None
    # Instruction lines always have an opcode. Ignore function metadata.
    match = re.match(r"(?:(?:[^=]+) = )?(?:(?:nofpexcept|nuw|nsw|exact|disjoint|reassoc|contract|nnan|ninf|nsz|arcp|afn) )*([A-Z][A-Za-z0-9_]*)\b", line)
    if not match:
        if slot:
            raise ValueError(f"unmodeled machine instruction: {line}")
        return None
    opcode = match[1]
    explicit, _, memory = line.partition(" :: ")
    # Rewriter adds super-register liveness annotations. MachineVerifier checks
    # those; retain FP state uses here because they affect arithmetic semantics.
    implicit = re.findall(r"implicit(?:-def)? (?:\w+ )*\$(?:mxcsr|fpcr|fpsr)\b", explicit)
    explicit = re.split(r", implicit(?:-def)?\b", explicit)[0]
    explicit = FLAGS.sub("", explicit)
    explicit = re.sub(r"\s+", " ", explicit).strip()
    if not slot:
        raise ValueError("machine instruction has no stable slot index")
    return opcode, explicit, memory, implicit, int(slot[1])


def dumps(text):
    result = {}
    ended = set()
    phase = name = block = None
    for line in text.splitlines():
        if line.startswith("# *** IR Dump "):
            match = re.match(r"# \*\*\* IR Dump (Before|After) Virtual Register Rewriter", line)
            phase = match[1] if match else None
            name = block = None
        elif phase and line.startswith("# Machine code for function "):
            name = line.split("function ", 1)[1].split(":", 1)[0]
            key = phase, name
            if key in result:
                raise ValueError("duplicate register-rewrite dump")
            result[key] = {}
        elif line.startswith("# End machine code"):
            ended.add((phase, name))
            name = block = None
        elif name:
            match = re.match(r"\s*(?:\d+B\s*)?bb\.(\d+)\b", line)
            if match:
                block = match[1]
                if block in result[phase, name]:
                    raise ValueError("duplicate machine block")
                result[phase, name][block] = []
            elif block is not None:
                parsed = instruction(line)
                if parsed:
                    result[phase, name][block].append(parsed)
    if ended != result.keys():
        raise ValueError("truncated register-rewrite dump")
    return result


def audit(text):
    records = dumps(text)
    names = {name for phase, name in records if phase == "Before"}
    if not names or set(records) != {(phase, name) for name in names for phase in ("Before", "After")}:
        raise ValueError("missing register-rewrite dump")
    counts = dict(functions=len(names), instructions=0, fma=0, copies=0, stack_operands=0)
    fma_opcodes = {}
    for name in sorted(names):
        before, after = records["Before", name], records["After", name]
        if not before or not after or list(before) != list(after):
            raise ValueError(f"changed machine CFG: {name}")
        if not any(before.values()) or not any(after.values()):
            raise ValueError(f"empty machine function: {name}")
        mapping, roots = {}, {}

        def bind(old, new):
            info = virtual(old)
            if not info:
                if old != new:
                    raise ValueError(f"changed fixed register: {old} -> {new}")
                return
            if not new.startswith("$"):
                raise ValueError("virtual register survived rewriting")
            number, sub, cls = info
            root, width = physical(new)
            expected_width = view_width(sub, cls)
            if expected_width and width != expected_width:
                raise ValueError(f"changed register width: {old} -> {new}")
            key = number, sub
            if ((key in mapping and mapping[key] != new)
                    or (number in roots and roots[number] != root)):
                raise ValueError(f"inconsistent register assignment: {old} -> {new}")
            mapping[key], roots[number] = new, root

        def compare(old, new):
            if old[0] != new[0] or old[2:] != new[2:]:
                raise ValueError(f"changed opcode, memory operand or FP state: {old} -> {new}")
            a, b = REG.findall(old[1]), REG.findall(new[1])
            if len(a) != len(b) or REG.sub("@", old[1]) != REG.sub("@", new[1]):
                raise ValueError(f"changed ordered machine operands: {old} -> {new}")
            for left, right in zip(a, b):
                bind(left, right)

        pairs = []
        for block in before:
            a = [i for i in before[block] if i[0] not in ("COPY", "KILL")]
            b = [i for i in after[block] if i[0] not in ("COPY", "KILL")]
            if len(a) != len(b):
                raise ValueError(f"changed instruction count: {name}/bb.{block}")
            for left, right in zip(a, b):
                compare(left, right)
                counts["instructions"] += 1
                if re.match(r"(?:VFMADD|FMLA|FMADD)", left[0]):
                    counts["fma"] += 1
                    fma_opcodes[left[0]] = fma_opcodes.get(left[0], 0) + 1
                counts["stack_operands"] += len(re.findall(r"%(?:stack|fixed-stack)\.\d+", left[1] + left[2]))
            pairs.append((before[block], after[block]))

        views = {physical(reg): reg for blocks in (before, after) for items in blocks.values()
                 for item in items for reg in REG.findall(item[1]) if reg.startswith("$")}

        # Match surviving copies in their original gaps. Only copies whose two
        # views are identical may disappear. This includes ABI input copies.
        for a, b in pairs:
            remaining = iter(i for i in b if i[0] != "KILL")
            current = next(remaining, None)
            for old in (i for i in a if i[0] != "KILL"):
                if old[0] == "COPY":
                    regs = REG.findall(old[1])
                    def resolved(reg):
                        info = virtual(reg)
                        if not info:
                            return reg
                        number, sub, cls = info
                        return mapping.get(info[:2]) or views.get((roots.get(number), view_width(sub, cls)))
                    left, right = map(resolved, regs)
                    survives = current is not None and current[0] == "COPY" and old[-1] == current[-1]
                    if survives:
                        compare(old, current)
                    else:
                        if left is None and right is not None:
                            bind(regs[0], right)
                            left = right
                        elif right is None and left is not None:
                            bind(regs[1], left)
                            right = left
                    if not survives and left is not None and left == right:
                        counts["copies"] += 1
                        continue
                    if not survives:
                        raise ValueError(f"unproved eliminated copy: {old}")
                    counts["copies"] += 1
                elif current is None or old[0] != current[0]:
                    raise ValueError("changed instruction order")
                current = next(remaining, None)
            if current is not None:
                raise ValueError("extra rewritten instruction")
    return dict(counts, fma_opcodes=fma_opcodes)
