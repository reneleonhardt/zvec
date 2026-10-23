import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from split_exact_bits import ADAPTER


class SplitGuardTest(unittest.TestCase):
    def test_each_span_ends_at_an_inaccessible_page(self):
        source = '#include <cstring>\n' + ADAPTER.split('static size_t boundary_case', 1)[0] + r'''
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
int main() {
  const rlimit limit{0, 0};
  if (setrlimit(RLIMIT_CORE, &limit)) return 1;
  GuardedSpans arena;
  const unsigned char value = 42;
  if (arena.copy(0, nullptr, 0)) return 2;
  for (size_t slot : {0U, 1U, 64U}) {
    auto *p = static_cast<volatile unsigned char *>(arena.copy(slot, &value, 1));
    if (*p != value) return 3;
    const pid_t child = fork();
    if (child < 0) return 4;
    if (!child) { const auto invalid = p[1]; (void)invalid; _exit(5); }
    int status;
    if (waitpid(child, &status, 0) != child || !WIFSIGNALED(status) ||
        (WTERMSIG(status) != SIGSEGV && WTERMSIG(status) != SIGBUS)) return 6;
  }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / 'guard.cc').write_text(source)
            root = Path(__file__).resolve().parents[3]
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17',
                            '-I' + str(root / 'src/turbo'), str(path / 'guard.cc'),
                            '-o', str(path / 'guard')], check=True)
            subprocess.run([str(path / 'guard')], check=True, stderr=subprocess.PIPE)


if __name__ == '__main__':
    unittest.main()
