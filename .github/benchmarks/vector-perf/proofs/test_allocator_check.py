"""Cheap negative controls for allocator frame and local value certificates."""

import unittest

import allocator_check as checker
import check


def dump(before, after, frames='  fi#0: size=16, align=16'):
    return '\n'.join(f'''# *** IR Dump {phase} Greedy Register Allocator (greedy) ***:
# Machine code for function kernel: TracksLiveness
{objects}
{body}
# End machine code for function kernel.
''' for phase, body, objects in [('Before', before, ''), ('After', after, frames)])


PREFIX = '''0B bb.0:
16B %0:vr128 = VMOVUPSrm $rdi, 1, $noreg, 0, $noreg :: (load (s128) from %ir.a)
32B %1:vr128 = VMOVUPSrm $rsi, 1, $noreg, 0, $noreg :: (load (s128) from %ir.b)
48B %2:vr128 = AVX_SET0'''
FMA = '96B %3:vr128 = nofpexcept VFMADD231PSr %2:vr128, %0:vr128, %1:vr128, implicit $mxcsr'
STORE = '64B VMOVAPSmr %stack.0, 1, $noreg, 0, $noreg, %0:vr128 :: (store (s128) into %stack.0)'
LOAD = '80B %4:vr128 = VMOVAPSrm %stack.0, 1, $noreg, 0, $noreg :: (load (s128) from %stack.0)'
BEFORE = PREFIX + '\n' + FMA
AFTER = PREFIX + '\n' + STORE + '\n' + LOAD + '\n' + FMA.replace('%0:vr128', '%4:vr128')


class AllocatorCheckTest(unittest.TestCase):
    def test_spill_reload_provenance_and_byte_widths(self):
        result = checker.audit(dump(BEFORE, AFTER))
        self.assertEqual((result['spill_stores'], result['reloads'], result['exact_fp_blocks']), (1, 1, 1))
        self.assertEqual(result['spill_widths'], [128])
        for bad in (AFTER.replace(STORE + '\n', ''),
                    AFTER.replace('store (s128)', 'store (s64)'),
                    AFTER.replace('0, $noreg :: (load', '16, $noreg :: (load'),
                    AFTER.replace('%stack.0)\n96B', '%stack.0 + 16)\n96B')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checker.audit(dump(BEFORE, bad))
        with self.assertRaisesRegex(ValueError, 'out-of-bounds'):
            checker.audit(dump(BEFORE, AFTER, '  fi#0: size=8, align=8'))

    def test_cfg_requires_initialization_on_every_path(self):
        before = '0B bb.0:\n successors: %bb.1, %bb.2\n16B JCC_1 %bb.1, 1\n32B bb.1:\n successors: %bb.3\n48B JMP_1 %bb.3\n64B bb.2:\n successors: %bb.3\n80B JMP_1 %bb.3\n96B bb.3:\n112B RET64'
        after = before.replace('48B JMP', STORE + '\n48B JMP').replace('112B RET64', LOAD + '\n112B RET64')
        with self.assertRaisesRegex(ValueError, 'dominating full-width'):
            checker.audit(dump(before, after))

    def test_same_bits_rematerialization_and_corrupt_constant(self):
        after = AFTER.replace('48B %2:vr128 = AVX_SET0\n', '').replace('96B', '88B %5:vr128 = AVX_SET0\n96B')
        after = after.replace('VFMADD231PSr %2:vr128', 'VFMADD231PSr %5:vr128')
        result = checker.audit(dump(BEFORE, after))
        self.assertEqual(result['exact_fp_blocks'], 1)
        bad = after.replace('%5:vr128 = AVX_SET0', '%5:vr128 = MOV32ri 7')
        result = checker.audit(dump(BEFORE, bad), strict=False)
        self.assertEqual(result['exact_fp_blocks'], 0)
        self.assertGreater(result['unmodeled_constant_recipes'], 0)
        with self.assertRaisesRegex(ValueError, 'rematerialization proof'):
            checker.audit(dump(BEFORE, bad))

    def test_wrong_reload_value_is_not_certified(self):
        bad = AFTER.replace('0, $noreg, %0:vr128 :: (store', '0, $noreg, %1:vr128 :: (store')
        result = checker.audit(dump(BEFORE, bad), strict=False)
        self.assertEqual(result['exact_fp_blocks'], 0)
        self.assertGreater(result['conditional_fp_blocks'], 0)
        with self.assertRaisesRegex(ValueError, 'fully modeled'):
            checker.audit(dump(BEFORE, bad))

    def test_fma_commuting_never_gets_an_implicit_nan_proof(self):
        result = checker.audit(dump(BEFORE, AFTER.replace('VFMADD231', 'VFMADD213')), strict=False)
        self.assertEqual((result['priority_unproved'], result['exact_fp_blocks']), (1, 0))
        with self.assertRaisesRegex(ValueError, 'NaN-priority'):
            checker.audit(dump(BEFORE, AFTER.replace('VFMADD231', 'VFMADD213')))
        bad = AFTER.replace('%4:vr128, %1:vr128, implicit', '%1:vr128, %4:vr128, implicit')
        with self.assertRaises(ValueError):
            checker.audit(dump(BEFORE, bad))
        with self.assertRaisesRegex(ValueError, 'duplicated/removed'):
            checker.audit(dump(BEFORE, AFTER + '\n' + FMA.replace('96B', '112B')))
        for opcode, pair in [('FMLAv4f32', (1, 2)), ('FMLAv8f16', (1, 2)), ('FMADDSrrr', (0, 1))]:
            prefix = '0B bb.0:\n' + '\n'.join(f'{16*(i+1)}B %{i}:fpr128 = LDRQui $x{i}, 0' for i in range(3))
            args = ['%0:fpr128', '%1:fpr128', '%2:fpr128']
            before = prefix + f'\n64B %3:fpr128 = {opcode} ' + ', '.join(args)
            i, j = pair
            args[i], args[j] = args[j], args[i]
            after = prefix + f'\n64B %3:fpr128 = {opcode} ' + ', '.join(args)
            with self.subTest(opcode=opcode), self.assertRaises(ValueError):
                checker.audit(dump(before, after))

    def test_entry_assumptions_and_calls_are_visible(self):
        result = checker.audit(dump('0B bb.0:\n' + FMA, '0B bb.0:\n' + FMA.replace('%0:', '%8:')))
        self.assertEqual((result['entry_contracts'], result['conditional_fp_blocks']), (1, 1))
        result = checker.audit(dump(BEFORE + '\n112B CALL64pcrel32 @f', AFTER + '\n112B CALL64pcrel32 @f'))
        self.assertEqual(result['exact_fp_blocks'], 0)

    def test_missing_and_truncated_diagnostics_fail_closed(self):
        for text in ('', dump('', ''), dump(BEFORE, AFTER).rsplit('# End machine code', 1)[0]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                checker.audit(text)

    def test_arbitrary_raw_bits_transport_and_negative_control(self):
        proofs = check.Proofs()
        checker.obligations(proofs, [{'allocator': {'spill_widths': [8, 16, 32, 64, 128, 256, 512]}}])
        self.assertEqual(len(proofs.cases), 9)
        proofs.run()


if __name__ == '__main__':
    unittest.main()
