"""Small adversarial controls for source and generated-code audits."""

import json
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest

import compiler_check as checker


def ref(name):
    return {"kind": "DeclRefExpr", "referencedDecl": {"name": name},
            "type": {"qualType": "__m256"}}


def assignment(feedback="accumulator"):
    return json.dumps({"kind": "BinaryOperator", "opcode": "=", "inner": [
        ref("accumulator"), {"kind": "CallExpr", "inner": [
            ref("_mm256_fmadd_ps"), ref("query"), ref("row"), ref(feedback)]}]})


class CompilerCheckTest(unittest.TestCase):
    def test_ast_feedback_is_checked(self):
        self.assertEqual(len(checker.source_trace(assignment())), 1)
        with self.assertRaisesRegex(ValueError, "wrong accumulation feedback"):
            checker.source_trace(assignment("sum1"))
        node = json.loads(assignment())
        member = lambda obj: {'kind': 'MemberExpr', 'name': 'sum0', 'inner': [ref(obj)],
                              'type': {'qualType': '__m256'}}
        node['inner'][0] = member('a')
        node['inner'][1]['inner'][3] = member('b')
        with self.assertRaisesRegex(ValueError, "wrong accumulation feedback"):
            checker.source_trace(json.dumps(node))

    def test_decoder_ast_bounds_and_rounding_are_checked(self):
        root = checker.check.ROOT
        target = "x86_64-apple-macos14" if platform.system() == "Darwin" else "x86_64-linux-gnu"
        command = ['clang++', '-std=c++17', '--target=' + target, '-march=core-avx2']
        command += ['-I' + str(root / p) for p in ('src/include', 'src', 'src/turbo', 'src/turbo/distance')]
        ast = subprocess.check_output([*command, '-fsyntax-only', '-Xclang', '-ast-dump=json',
            '-Xclang', '-ast-dump-filter=zvec::turbo', str(root / 'src/turbo/conversion/avx2/cosine_decode.cc')], text=True)
        checker.decoder_trace(ast)
        roots = list(checker.ast_roots(ast))
        for node in (node for root in roots for node in checker.walk(root)):
            if node.get('kind') == 'IntegerLiteral' and node.get('value') == '4096':
                node['value'] = '2048'
        with self.assertRaisesRegex(ValueError, 'source/model mismatch'):
            checker.decoder_trace('\n'.join(json.dumps(root) for root in roots))
        roots = list(checker.ast_roots(ast))
        for node in (node for root in roots for node in checker.walk(root)):
            if node.get('kind') == 'IfStmt':
                node['inner'][0] = {'kind': 'IntegerLiteral', 'value': '1'}
        with self.assertRaisesRegex(ValueError, 'both proved guards'):
            checker.decoder_trace('\n'.join(json.dumps(root) for root in roots))
        roots = list(checker.ast_roots(ast))
        for node in (node for root in roots for node in checker.walk(root)):
            if node.get('kind') == 'IfStmt' and any(
                    child.get('kind') == 'VarDecl' and child.get('name') == 'rounded'
                    for child in checker.walk(node['inner'][-1])):
                body = node['inner'][-1]
                node['hasElse'] = True
                node['inner'][-1] = {'kind': 'CompoundStmt', 'inner': []}
                node['inner'].append(body)
                break
        with self.assertRaisesRegex(ValueError, 'both proved guards'):
            checker.decoder_trace('\n'.join(json.dumps(root) for root in roots))

    def test_unsafe_ir_and_half_arithmetic_are_rejected(self):
        base = 'define void @kernel() {\n %s = fadd <4 x float> %a, %b\n ret void\n}'
        checker.audit_ir(base, False)
        for bad in (base.replace("fadd", "fadd nnan"),
                    base.replace("float", "half"),
                    base.replace("fadd", "fadd contract").replace("float", "half"),
                    base + '\n attributes #0 = { "no-signed-zeros-fp-math"="true" }'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                checker.audit_ir(bad, False)

    def test_baseline_isa_and_environment_writes_are_checked(self):
        checker.audit_assembly('\taddps %xmm0, %xmm1\n', 'sse2')
        for family, text in [('sse2', '\tvaddps %ymm0, %ymm1, %ymm2\n'),
                             ('avx2', '\tvfmadd213ps %zmm0, %zmm1, %zmm2\n'),
                             ('neon', '\tfadd s0, s1, s2\n\tmsr fpcr, x0\n'),
                             ('neon_fp16', '\tfmla v0.8h, v1.8h, v2.8h\n')]:
            with self.subTest(family=family), self.assertRaises(ValueError):
                checker.audit_assembly(text, family)

    def test_actual_compiler_rejects_wrong_feedback_and_fast_math(self):
        target = "x86_64-apple-macos14" if platform.system() == "Darwin" else "x86_64-linux-gnu"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.cc"
            path.write_text('''#include <immintrin.h>
namespace zvec::turbo::avx2 {
void bad(__m256 &sum0, __m256 sum1, __m256 q, __m256 row) {
  sum0 = _mm256_fmadd_ps(q, row, sum1);
}
}''')
            base = ['clang++', '-std=c++17', '--target=' + target, '-march=core-avx2', str(path)]
            ast = subprocess.check_output([*base, '-fsyntax-only', '-Xclang', '-ast-dump=json',
                                           '-Xclang', '-ast-dump-filter=zvec::turbo'], text=True)
            with self.assertRaisesRegex(ValueError, "wrong accumulation feedback"):
                checker.source_trace(ast)
            ir = subprocess.check_output([*base, '-O3', '-ffast-math', '-S', '-emit-llvm', '-o', '-'], text=True)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                checker.audit_ir(ir, False)

    def test_actual_build_flags_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = root / 'commands.json'
            entry = {'file': str(root / 'src/turbo/kernel.cc'), 'directory': str(root),
                     'arguments': ['clang++', '-O3', '-c', 'kernel.cc']}
            commands.write_text(json.dumps([entry]))
            self.assertEqual(checker.audit_build_commands(commands, root), 1)
            entry['arguments'].insert(1, '-ffast-math')
            commands.write_text(json.dumps([entry]))
            with self.assertRaisesRegex(ValueError, "unsafe production compile options"):
                checker.audit_build_commands(commands, root)

    def test_cache_tracks_headers_compiler_and_artifact_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header, artifact = root / 'header.h', root / 'kernel.s'
            header.write_text('original')
            artifact.write_text('assembly')
            digest = lambda path: checker.hashlib.sha256(path.read_bytes()).hexdigest()
            (root / 'stamp.json').write_text(json.dumps({'compiler': 'c1',
                'dependencies': {str(header): digest(header)}, 'artifacts': {artifact.name: digest(artifact)}}))
            self.assertTrue(checker.cache_hit(root, 'c1', (artifact.name,)))
            self.assertFalse(checker.cache_hit(root, 'c2', (artifact.name,)))
            header.write_text('modified')
            self.assertFalse(checker.cache_hit(root, 'c1', (artifact.name,)))
            header.write_text('original')
            artifact.write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'changed compiler-cache'):
                checker.cache_hit(root, 'c1', (artifact.name,))


if __name__ == '__main__':
    unittest.main()
