"""Bounded translation checks for LLVM greedy allocation, before register rewrite.

Frame checks cover every parsed block. Local value checks produce explicit entry
contracts; they do not silently promote a block lemma to a whole-CFG proof.
"""

from collections import Counter
import re

import register_check as registers


DUMP = ['-mllvm', '-print-before=greedy', '-mllvm', '-print-after=greedy']
STACK = re.compile(r'%stack\.(\d+)\b')
FP = re.compile(r'^(?:V?(?:ADD|SUB|MUL|DIV|SQRT|HADD|HSUB|CMP|MIN|MAX|RCP|RSQRT|UCOMI|COMI)(?:PS|SS|PH|SH|PD|SD)|VF(?:N?MADD|N?MSUB)|VCVT|[SU]CVTF|F(?:ADD|SUB|MUL|DIV|SQRT|MLA|MLS|MADD|MSUB|NMADD|NMSUB|CVT|CMP|MAX|MIN|RINT))')


def parse(text):
    pairs = []
    pending = None
    current = block = phase = None
    for line in text.splitlines():
        if line.startswith('# *** IR Dump '):
            match = re.match(r'# \*\*\* IR Dump (Before|After) Greedy Register Allocator', line)
            phase = match[1] if match else None
            current = block = None
        elif phase and line.startswith('# Machine code for function '):
            name = line.split('function ', 1)[1].split(':', 1)[0]
            current = dict(name=name, blocks={}, successors={}, frames={}, opaque={})
        elif current and line.startswith('# End machine code'):
            if not current['blocks'] or not any(current['blocks'].values()):
                raise ValueError('empty allocator dump')
            if phase == 'Before':
                if pending is not None:
                    raise ValueError('missing allocator output')
                pending = current
            else:
                if pending is None or pending['name'] != current['name']:
                    raise ValueError('unpaired allocator output')
                pairs.append((pending, current))
                pending = None
            current = block = None
        elif current:
            frame = re.match(r'\s*fi#(\d+): size=(\d+), align=(\d+)', line)
            header = re.match(r'\s*(?:\d+B\s*)?bb\.(\d+)\b', line)
            if frame:
                current['frames'][int(frame[1])] = (int(frame[2]), int(frame[3]))
            elif header:
                block = header[1]
                if block in current['blocks']:
                    raise ValueError('duplicate allocator block')
                current['blocks'][block] = []
                current['successors'][block] = []
                current['opaque'][block] = False
            elif block is not None:
                if line.strip().startswith('successors:'):
                    current['successors'][block] = re.findall(r'%bb\.(\d+)', line.split(';')[0])
                else:
                    item = registers.instruction(line)
                    if item:
                        current['blocks'][block].append(item)
                        uses = re.findall(r'implicit (?:\w+ )*\$(\w+)', line)
                        defs = re.findall(r'implicit-def (?:\w+ )*\$(\w+)', line)
                        branch = item[0].startswith(('JCC', 'Bcc', 'RET'))
                        if ('undef ' in line
                                or (not branch and any(reg not in ('mxcsr', 'fpcr', 'fpsr') for reg in uses))
                                or any(reg not in ('eflags', 'nzcv', 'mxcsr', 'fpcr', 'fpsr') for reg in defs)):
                            current['opaque'][block] = True
    if current is not None or pending is not None or not pairs:
        raise ValueError('missing or truncated allocator dump')
    return pairs


def access(item, frames):
    """Decode stack MMO byte ranges, retaining folded reloads as reads."""
    opcode, explicit, memory, _, _ = item
    matches = list(STACK.finditer(explicit))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f'unmodeled multiple stack addresses: {item}')
    slot = int(matches[0][1])
    if opcode in ('LEA64r', 'ADDXri') and not memory and slot in frames:
        return 'address', slot, 0, 0
    mmo = re.search(r'\((?:[a-z-]+ )*(load|store) \(s(\d+)\)(.*?)\)', memory)
    if len(re.findall(r'\b(?:load|store) \(s\d+\)', memory)) != 1:
        raise ValueError('unmodeled multiple or atomic stack memory operands')
    if mmo is None or slot not in frames:
        raise ValueError(f'unmodeled stack access: {item}')
    bits = int(mmo[2])
    if STACK.search(memory) and {int(ref) for ref in STACK.findall(memory)} != {slot}:
        raise ValueError('stack slot differs from memory annotation')
    location = re.search(r'%stack\.' + str(slot) + r'(?: \+ (\d+))?', memory)
    if bits % 8:
        raise ValueError(f'unmodeled stack width/address: {item}')
    size = bits // 8
    # Check the executable address agrees with the MMO. Target frame-index
    # offsets are unscaled on x86 and scaled by access width on AArch64.
    tail = explicit[matches[0].end():].split(',')
    if len(tail) >= 5 and tail[1].strip() in ('1', '2', '4', '8'):
        if tail[2].strip() != '$noreg':
            return 'indexed', slot, 0, 0
        encoded = int(tail[3].strip())
    elif len(tail) >= 2 and re.match(r'^(?:LDR|STR)[A-Z]+ui$', opcode):
        encoded = int(tail[1].strip()) * size
    elif len(tail) >= 2 and re.match(r'^(?:LDUR|STUR)[A-Z]+i$', opcode):
        encoded = int(tail[1].strip())
    else:
        raise ValueError(f'unmodeled frame-index address: {item}')
    offset = int(location[1] or 0) if location else encoded
    if encoded != offset:
        raise ValueError('stack address differs from memory annotation')
    if offset < 0 or offset + size > frames[slot][0]:
        raise ValueError(f'out-of-bounds spill/reload: slot {slot}, {offset}+{size}')
    return mmo[1], slot, offset, size


def frame_check(before, after):
    if list(before['blocks']) != list(after['blocks']) or before['successors'] != after['successors']:
        raise ValueError('allocator changed CFG')
    frames = after['frames']
    for slot, description in before['frames'].items():
        if frames.get(slot) != description:
            raise ValueError('allocator changed an existing frame object')
    new = frames.keys() - before['frames'].keys()
    accesses = {b: [(item, access(item, frames)) for item in items]
                for b, items in after['blocks'].items()}
    # Must-defined byte sets form a finite lattice. Starting non-entry blocks
    # at top computes the greatest fixed point of predecessor intersection.
    universe = {(slot, byte) for slot in new for byte in range(frames[slot][0])}
    predecessors = {b: [] for b in accesses}
    for block, successors in after['successors'].items():
        for successor in successors:
            if successor not in predecessors:
                raise ValueError('unknown allocator CFG successor')
            predecessors[successor].append(block)
    entry = next(iter(accesses))
    incoming = {b: set() if b == entry or not predecessors[b] else universe.copy() for b in accesses}
    outgoing = {b: universe.copy() for b in accesses}
    for _ in range(len(universe) * len(accesses) + len(accesses) + 1):
        changed = False
        for block in accesses:
            state = (set.intersection(*(outgoing[p] for p in predecessors[block]))
                     if block != entry and predecessors[block] else set())
            end = state.copy()
            for _, transfer in accesses[block]:
                if transfer and transfer[0] == 'store' and transfer[1] in new:
                    _, slot, offset, size = transfer
                    end.update((slot, byte) for byte in range(offset, offset + size))
            if state != incoming[block] or end != outgoing[block]:
                incoming[block], outgoing[block] = state, end
                changed = True
        if not changed:
            break
    else:
        raise ValueError('spill initialization analysis did not converge')
    counts = Counter(new_slots=len(new), spill_stores=0, reloads=0)
    widths = set()
    for block, items in accesses.items():
        state = incoming[block].copy()
        for _, transfer in items:
            if not transfer or transfer[1] not in new:
                continue
            kind, slot, offset, size = transfer
            if kind in ('address', 'indexed'):
                raise ValueError('allocator spill-slot address escapes the transfer model')
            widths.add(size * 8)
            span = {(slot, byte) for byte in range(offset, offset + size)}
            if kind == 'load' and not span <= state:
                raise ValueError(f'reload without a dominating full-width store: bb.{block}, slot {slot}')
            if kind == 'store':
                state.update(span)
            counts['spill_stores' if kind == 'store' else 'reloads'] += 1
    return counts, widths


def projection(value, width):
    if value[0] == 'low':
        return ('low', min(width, value[1]), value[2])
    return ('low', width, value)


def trace(items, frames, original_registers):
    """Local raw-value graph, with opaque arithmetic and explicit entry leaves."""
    values, stack, events = {}, {}, []
    epoch = 0
    def read(reg):
        info = registers.virtual(reg)
        key = '%' + info[0] if info else reg
        value = values.get(key, ('entry' if key in original_registers or key.startswith('$') else 'split-entry', key))
        if info and info[1]:
            return projection(value, registers.view_width(info[1], info[2]))
        return value
    for item in items:
        opcode, explicit, memory, implicit, _ = item
        if opcode == 'KILL':
            continue
        lhs, sep, rhs = explicit.partition(' = ')
        defs = registers.REG.findall(lhs) if sep else []
        rhs = rhs if sep else explicit
        rhs = re.sub(r'^.*?\b' + re.escape(opcode) + r'\b\s*', '', rhs)
        rhs = re.split(r',?\s+implicit(?:-def)?\b', rhs)[0]
        args = [part.strip() for part in rhs.split(',') if part.strip()]
        parsed = access(item, frames)
        if parsed and parsed[0] not in ('address', 'indexed'):
            kind, slot, offset, size = parsed
            start = next(i for i, arg in enumerate(args) if STACK.fullmatch(arg))
            if kind == 'load':
                value = stack.get((slot, offset, size), ('split-entry', f'stack:{slot}:{offset}:{size}'))
                span = 5 if len(args) >= start + 5 and args[start + 1] == '1' else 2
                args[start:start + span] = [value]
                if re.match(r'^(?:V?MOV.*rm|LDR)', opcode):
                    result = value
                    for dest in defs:
                        values['%' + registers.virtual(dest)[0] if registers.virtual(dest) else dest] = result
                    continue
                opcode = re.sub(r'm$', 'r', opcode)
            else:
                source = args[-1] if start == 0 else args[0]
                for old in list(stack):
                    if old[0] == slot and old[1] < offset + size and offset < old[1] + old[2]:
                        del stack[old]
                if registers.REG.fullmatch(source):
                    stack[slot, offset, size] = read(source)
                elif re.fullmatch(r'-?\d+', source):
                    stack[slot, offset, size] = ('constant', int(source), size * 8)
                else:
                    raise ValueError(f'unmodeled spill store source: {item}')
                continue
        resolved = []
        for arg in args:
            if isinstance(arg, tuple):
                resolved.append(arg)
            elif registers.REG.fullmatch(arg):
                resolved.append(read(arg))
            else:
                # Address expressions retain every register and immediate.
                pieces = registers.REG.split(arg)
                refs = registers.REG.findall(arg)
                resolved.append(('text', tuple(pieces), tuple(read(reg) for reg in refs)))
        if opcode == 'COPY':
            if len(resolved) != 1 or len(defs) != 1:
                raise ValueError('unmodeled allocator copy')
            result = resolved[0]
        elif FP.match(opcode):
            if re.search(r'(?:SS|SH)(?:[rZmY]|$)', opcode) or re.match(r'F(?:MADD|MSUB|ADD|SUB|MUL|DIV)[SH]', opcode):
                width = 16 if 'SH' in opcode or opcode.endswith('Hrrr') else 32
                resolved = [projection(arg, width) for arg in resolved]
            events.append((opcode, tuple(resolved), tuple(implicit)))
            result = ('event', len(events) - 1)
        else:
            result = ('pure', opcode, tuple(resolved), memory,
                      0 if '%const.' in explicit or 'constant-pool' in memory else epoch)
        for index, dest in enumerate(defs):
            info = registers.virtual(dest)
            values['%' + info[0] if info else dest] = result if len(defs) == 1 else ('result', index, result)
        if opcode.startswith(('CALL', 'BL')):
            # Local calls are barriers, not rematerializable pure operations.
            for name in list(values):
                if name.startswith('$'):
                    del values[name]
        if '(store' in memory or opcode.startswith(('CALL', 'BL', 'INLINEASM')):
            epoch += 1
    return events


def local_check(before, after, strict=True):
    """Check local FP graphs and return unresolved entry/commuting contracts."""
    original = {'%' + info[0] for items in before['blocks'].values() for item in items
                for reg in registers.REG.findall(item[1]) if (info := registers.virtual(reg))}
    counts = Counter(fp_events=0, exact_fp_blocks=0, conditional_fp_blocks=0,
                     non_fp_blocks=0, entry_contracts=0, priority_unproved=0)
    reasons = Counter()
    for block in before['blocks']:
        a = trace(before['blocks'][block], before['frames'], original)
        b = trace(after['blocks'][block], after['frames'], original)
        if len(a) != len(b):
            raise ValueError(f'allocator duplicated/removed FP operations: bb.{block}')
        counts['fp_events'] += len(a)
        if not a:
            counts['non_fp_blocks'] += 1
            continue
        contracts = {}
        both = before['blocks'][block] + after['blocks'][block]
        conditional = before['opaque'][block] or after['opaque'][block] or any(item[0].startswith(('CALL', 'BL', 'INLINEASM', 'IMPLICIT_DEF',
                          'CMOV', 'CSEL', 'CSINC', 'CSNEG', 'ADC', 'SBC', 'SBB', 'SET'))
                          or (' = ' in item[1] and '.' in item[1].split(' = ', 1)[0]) for item in both)
        if conditional:
            reasons['call, conditional/undefined value or partial-register definition needs a richer model'] += 1
        def equal(left, right):
            nonlocal conditional
            if left == right:
                return True
            if isinstance(right, tuple) and right and right[0] == 'split-entry':
                conditional = True
                if right in contracts:
                    return contracts[right] == left
                contracts[right] = left
                return True
            return (isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right)
                    and all(equal(x, y) for x, y in zip(left, right)))
        for left, right in zip(a, b):
            if left[0] != right[0]:
                fma = lambda op: re.sub(r'(132|213|231)', 'XYZ', op)
                if left[0].startswith('VFMADD') and fma(left[0]) == fma(right[0]):
                    counts['priority_unproved'] += 1
                    conditional = True
                    reasons['FMA encoding changed; NaN priority not independently proved'] += 1
                    continue
                raise ValueError(f'allocator changed FP operation: {left[0]} -> {right[0]}')
            if not equal(left, right):
                match = re.match(r'VFMADD(132|213|231)', left[0])
                pair = {'132': (0, 2), '213': (0, 1), '231': (1, 2)}[match[1]] if match else None
                if left[0].startswith(('FMLA', 'FMLS')):
                    pair = (1, 2)
                elif left[0].startswith(('FMADD', 'FMSUB', 'FNMADD', 'FNMSUB')):
                    pair = (0, 1)
                if pair and len(right[1]) == 3:
                    swapped = list(right[1])
                    i, j = pair
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    if left[1] == tuple(swapped):
                        counts['priority_unproved'] += 1
                        reasons['FMA multiplicands commuted; NaN priority not independently proved'] += 1
                simple = re.compile(r'^(?:COPY|KILL|AVX(?:512)?_SET0|VMOV(?:UPS|APS)(?:Y|Z)?(?:rm|mr)|VFMADD\d+(?:PS|PH)(?:Y|Z)?[rm]|LDRQui|STRQui|FMLAv(?:4f32|8f16))$')
                if strict and not conditional and not contracts and all(simple.fullmatch(item[0]) for item in both):
                    raise ValueError('allocator changed a fully modeled local FP value graph')
                # A failed local model is visible in coverage; it is never an
                # exact-bit certificate. Full CFG value validation is separate.
                conditional = True
                reasons['local graph differs or needs a richer lane/address model'] += 1
        counts['entry_contracts'] += len(contracts)
        counts['conditional_fp_blocks' if conditional else 'exact_fp_blocks'] += 1
    return dict(counts), dict(reasons)


def constant_recipes(record):
    """Only operand-free integer constants and immutable constant-pool loads."""
    recipes = Counter()
    for items in record['blocks'].values():
        for opcode, explicit, memory, _, _ in items:
            if ' = ' not in explicit or FP.match(opcode):
                continue
            rhs = explicit.split(' = ', 1)[1]
            rhs = re.split(r',?\s+implicit(?:-def)?\b', rhs)[0]
            uses = registers.REG.findall(rhs)
            if any(reg not in ('$noreg', '$rip') for reg in uses):
                continue
            if memory and '%const.' not in explicit and 'constant-pool' not in memory:
                continue
            if STACK.search(explicit):
                continue
            recipes[opcode, rhs, memory] += 1
    return recipes


def audit(text, *, strict=True):
    pairs = parse(text)
    counts, reasons, widths = Counter(phases=len(pairs)), Counter(), set()
    for before, after in pairs:
        frame_counts, frame_widths = frame_check(before, after)
        counts.update(frame_counts)
        widths.update(frame_widths)
        local, pending = local_check(before, after, strict=strict)
        counts.update(local)
        reasons.update(pending)
        original_constants = constant_recipes(before)
        extra = constant_recipes(after) - original_constants
        counts['matching_constant_remats'] += sum(n for recipe, n in extra.items() if recipe in original_constants)
        counts['unmodeled_constant_recipes'] += sum(n for recipe, n in extra.items() if recipe not in original_constants)
        if before['frames'] == after['frames'] and all(
                [item[:4] for item in before['blocks'][b]] == [item[:4] for item in after['blocks'][b]]
                for b in before['blocks']):
            counts['unchanged_phases'] += 1
    if strict and (counts['priority_unproved'] or counts['unmodeled_constant_recipes']):
        raise ValueError('allocator requires a NaN-priority or constant-rematerialization proof: '
                         f"commutes={counts['priority_unproved']}, new recipes={counts['unmodeled_constant_recipes']}")
    return dict(counts, spill_widths=sorted(widths), unresolved=dict(reasons))


def obligations(proofs, reports):
    """Raw-byte transport lemmas, instantiated from the actual spill widths."""
    widths = set()
    for report in reports:
        widths.update(report.get('allocator', {}).get('spill_widths', []))
    for width in sorted(widths):
        declarations = [f'(declare-const value (_ BitVec {width}))',
                        '(declare-const base Int)', '(declare-const i Int)', '(declare-const j Int)']
        loaded = [f'((_ extract {byte * 8 + 7} {byte * 8}) value)' for byte in reversed(range(width // 8))]
        result = loaded[0] if len(loaded) == 1 else '(concat ' + ' '.join(loaded) + ')'
        different = f'(and (<= 0 i) (< i {width // 8}) (<= 0 j) (< j {width // 8}) (distinct i j))'
        counterexample = f'(or (not (= value {result})) (and {different} (= (+ base i) (+ base j))))'
        proofs.add(f'allocator/spill-{width}-raw-bits', '\n'.join(declarations), counterexample)
    if widths:
        # A single-byte store round trip plus noninterference is the induction
        # step for the finite, pairwise-distinct byte offsets proved above.
        declarations = '(declare-const m (Array Int (_ BitVec 8)))\n(declare-const x (_ BitVec 8))\n(declare-const i Int)\n(declare-const j Int)'
        proofs.add('allocator/byte-store-induction', declarations,
                   '(or (not (= (select (store m i x) i) x)) (and (distinct i j) (not (= (select (store m i x) j) (select m j)))))')
        proofs.add('allocator/negative-wrong-spill-offset',
                   '(declare-const m (Array Int (_ BitVec 8)))\n(declare-const x (_ BitVec 8))',
                   '(not (= (select (store m 0 x) 1) x))', expected='sat')
