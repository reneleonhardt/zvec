"""Negative controls for proof/source drift and the solver protocol."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check


class ProofCheckerTest(unittest.TestCase):
    def test_source_guard_ignores_comments_but_rejects_math_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "kernel.cc"
            path.write_text("sum = fma(a, b, sum); // original\n")
            manifest = {path.name: check.fingerprint(path.read_text())}
            path.write_text("/* comment */ sum=fma(a,b,sum);\n")
            check.check_sources(root, manifest)
            path.write_text("sum=fma(b,a,sum);\n")
            with self.assertRaisesRegex(ValueError, "source/model review"):
                check.check_sources(root, manifest)
        self.assertNotEqual(check.fingerprint('f("//", a);'),
                            check.fingerprint('f("//", b);'))

    def test_unknown_errors_and_missing_answers_fail_closed(self):
        proofs = check.Proofs()
        proofs.add("p", "", "false")
        for output in ("p\nunknown\n", "p\nunsat\n(error bad)\n", "p\n"):
            with self.subTest(output=output), patch("check.subprocess.run") as run:
                run.return_value = check.subprocess.CompletedProcess([], 0, output, "")
                with self.assertRaisesRegex(RuntimeError, "proof failed"):
                    proofs.run()

    def test_false_lemma_and_wrong_expected_status_are_rejected(self):
        proofs = check.Proofs()
        proofs.add("false-lemma", "(declare-const x Bool)", "x")
        with self.assertRaisesRegex(RuntimeError, "proof failed"):
            proofs.run()

    def test_valid_lemma(self):
        proofs = check.Proofs()
        proofs.add("identity", "(declare-const x Bool)", "(not (= x x))")
        proofs.run()

    def test_source_inventory_includes_decoder_and_scalar_oracle(self):
        required = check.required_sources(check.ROOT)
        self.assertIn("src/ailego/utility/float_helper.cc", required)
        self.assertIn("src/turbo/conversion/avx512/cosine_decode.cc", required)
        self.assertTrue(required <= check.json.loads((check.HERE / "sources.json").read_text()).keys())

    def test_scalar_table_changes_are_rejected(self):
        path = "src/ailego/utility/float_helper.cc"
        helper = check.source(check.ROOT, path).replace("0x0400, 0x0800, 0x0C00", "0x0401, 0x0800, 0x0C00")
        self.assertNotEqual(helper, check.source(check.ROOT, path))
        with patch("check.source", return_value=helper):
            with self.assertRaisesRegex(ValueError, "normal-half table"):
                check.decoder(check.Proofs(), check.ROOT)


if __name__ == "__main__":
    unittest.main()
