"""Adversarial controls for the physical-register rewrite validator."""

from pathlib import Path
import platform
import re
import subprocess
import tempfile
import unittest

import register_check as checker


def dump(before, after):
    return '\n'.join(f'''# *** IR Dump {phase} Virtual Register Rewriter (virtregrewriter) ***:
# Machine code for function kernel: TracksLiveness
0B bb.0:
{body}
# End machine code for function kernel.
''' for phase, body in [('Before', before), ('After', after)])


BEFORE = '''16B %0:vr256 = VMOVUPSYrm $rdi, 1, $noreg, 0, $noreg :: (load from %ir.a)
32B %1:vr256 = VMOVUPSYrm $rsi, 1, $noreg, 0, $noreg :: (load from %ir.b)
48B %2:vr256 = COPY %0:vr256
64B %2:vr256 = nofpexcept VFMADD213PSYr %2:vr256(tied-def 0), %1:vr256, %0:vr256, implicit $mxcsr
80B VMOVUPSYmr %stack.0, 1, $noreg, 0, $noreg, %2:vr256 :: (store into %stack.0)'''
AFTER = '''16B $ymm0 = VMOVUPSYrm $rdi, 1, $noreg, 0, $noreg :: (load from %ir.a)
32B $ymm1 = VMOVUPSYrm $rsi, 1, $noreg, 0, $noreg :: (load from %ir.b)
64B $ymm0 = nofpexcept VFMADD213PSYr $ymm0, $ymm1, $ymm0, implicit $mxcsr
80B VMOVUPSYmr %stack.0, 1, $noreg, 0, $noreg, $ymm0 :: (store into %stack.0)'''


class RegisterCheckTest(unittest.TestCase):
    def test_ordered_fma_renaming_and_spills(self):
        result = checker.audit(dump(BEFORE, AFTER))
        self.assertEqual((result['fma'], result['copies'], result['stack_operands']), (1, 1, 2))
        for bad in (AFTER.replace('213PS', '231PS'),
                    AFTER.replace('$ymm0, $ymm1, $ymm0', '$ymm1, $ymm0, $ymm0'),
                    AFTER.replace('32B $ymm1', '32B $ymm0'),
                    AFTER.replace('%stack.0', '%stack.1'),
                    AFTER.replace('$ymm1', '$xmm1'),
                    AFTER.replace('implicit $mxcsr', 'implicit $fpcr')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checker.audit(dump(BEFORE, bad))

    def test_arm_multiplicands_are_not_commuted(self):
        before = '\n'.join(f'{16 * (i + 1)}B %{i}:fpr128 = LDRQui $x{i}, 0' for i in range(3))
        after = '\n'.join(f'{16 * (i + 1)}B $q{i} = LDRQui $x{i}, 0' for i in range(3))
        for opcode in ('FMLAv4f32', 'FMLAv8f16'):
            a = before + f'\n64B %0:fpr128 = {opcode} %0:fpr128, %1:fpr128, %2:fpr128, implicit $fpcr'
            b = after + f'\n64B $q0 = {opcode} $q0, $q1, $q2, implicit $fpcr'
            self.assertEqual(checker.audit(dump(a, b))['fma'], 1)
            with self.assertRaisesRegex(ValueError, 'inconsistent register assignment'):
                checker.audit(dump(a, b.replace('$q0, $q1, $q2', '$q0, $q2, $q1')))

    def test_copies_keep_stable_slots_and_register_views(self):
        before = '16B %0:gr64 = COPY $r9\n32B %1:gr64 = COPY $r8\n48B TEST64rr %1:gr64, %1:gr64\n64B $r9 = COPY %0:gr64'
        after = '32B $rbx = COPY $r8\n48B TEST64rr $rbx, $rbx'
        checker.audit(dump(before, after))
        checker.audit(dump('16B %0:fpr128 = MOVIv2d_ns 0\n32B %1:fpr128 = FCVTLv4i16 %0.dsub:fpr128',
                           '16B $q0 = MOVIv2d_ns 0\n32B $q1 = FCVTLv4i16 $d0'))
        with self.assertRaises(ValueError):
            checker.audit(dump(before, after.replace('32B', '16B')))
        with self.assertRaisesRegex(ValueError, 'register width'):
            checker.audit(dump('16B %0:fpr128 = MOVIv2d_ns 0\n32B %1:fpr128 = FCVTLv4i16 %0.dsub:fpr128',
                               '16B $q0 = MOVIv2d_ns 0\n32B $q1 = FCVTLv4i16 $q0'))

    def test_missing_unknown_and_truncated_diagnostics_fail_closed(self):
        for bad in ('', dump('', ''), dump(BEFORE, AFTER).rsplit('# End machine code', 1)[0],
                    dump(BEFORE, AFTER + '\n96B unsupported_opcode $ymm0')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checker.audit(bad)

    def test_actual_compiler_and_corrupted_fma(self):
        target = 'x86_64-apple-macos14' if platform.system() == 'Darwin' else 'x86_64-linux-gnu'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kernel.cc'
            path.write_text('''#include <immintrin.h>
extern "C" __m256 kernel(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}''')
            result = subprocess.run(['clang++', '--target=' + target, '-march=core-avx2', '-O3',
                *checker.VERIFY, *checker.DUMP, '-S', str(path), '-o', str(path.with_suffix('.s'))],
                text=True, capture_output=True, check=True)
            self.assertEqual(checker.audit(result.stderr)['fma'], 1)
            before, after = result.stderr.split('# *** IR Dump After', 1)
            after, count = re.subn(r'VFMADD(132|213|231)', lambda m: 'VFMADD' + ('231' if m[1] != '231' else '213'), after)
            self.assertEqual(count, 1)
            with self.assertRaisesRegex(ValueError, 'changed opcode'):
                checker.audit(before + '# *** IR Dump After' + after)


if __name__ == '__main__':
    unittest.main()
