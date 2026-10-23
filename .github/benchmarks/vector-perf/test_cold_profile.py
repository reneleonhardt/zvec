import unittest
from cold_profile import summarize


class ColdProfileTest(unittest.TestCase):
    def test_phase_channels_and_resumed_calls(self):
        trace = '''10 0.800000 link("/tmp/ip32.index", "/tmp/cold-10-0.index") = 0 <0.000002>
10 0.900000 fsync(4) = 0 <0.030000>
10 1.000000 openat(AT_FDCWD, "/proc/self/io", O_RDONLY) = 3</proc/10/io> <0.000001>
10 1.100000 openat(AT_FDCWD, "/tmp/cold-10-0.index", O_RDONLY|O_DIRECT) = 4</tmp/cold-10-0.index> <0.000002>
10 1.200000 pread64(4</tmp/cold-10-0.index>,  <unfinished ...>
10 1.400000 <... pread64 resumed>"data", 4, 0) = 4 <0.200000>
10 1.500000 openat(AT_FDCWD, "/proc/self/io", O_RDONLY) = 3</proc/10/io> <0.000001>
10 1.600000 preadv(4</tmp/cold-10-0.index>, [], 0, 0) = 0 <0.010000>
10 1.700000 openat(AT_FDCWD, "/proc/self/io", O_RDONLY) = 3</proc/10/io> <0.000001>
10 1.800000 close(4</tmp/cold-10-0.index>) = 0 <0.000003>'''
        trial, = summarize(trace.splitlines())
        self.assertEqual(.03, trial['preparation']['fsync:other']['seconds'])
        self.assertEqual(.2, trial['open']['pread64:direct']['seconds'])
        self.assertEqual(.01, trial['search']['preadv:direct']['max_seconds'])
        self.assertEqual(1, trial['cleanup']['close:direct']['calls'])
        self.assertNotIn('openat:other', trial['search'])

    def test_missing_marker_is_an_error(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == '__main__':
    unittest.main()
