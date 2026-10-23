import unittest
from reuse_candidate import ALLOWED, compatible


class ReuseCandidateTest(unittest.TestCase):
    def test_only_reviewed_tooling_changes_can_reuse_proofs(self):
        self.assertTrue(compatible([]))
        self.assertTrue(compatible(ALLOWED))
        for changed in ('src/turbo/turbo.cc', 'tests/turbo/turbo_avx2_distance_test.cc',
                        '.github/benchmarks/vector-perf/exact_bits.py',
                        '.github/benchmarks/vector-perf/fixture.h'):
            self.assertFalse(compatible([*ALLOWED, changed]))


if __name__ == '__main__':
    unittest.main()
