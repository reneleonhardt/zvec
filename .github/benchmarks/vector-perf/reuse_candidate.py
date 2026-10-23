"""Reuse cached production/proof binaries only for the listed tooling changes."""
from pathlib import Path
import subprocess
import sys

ALLOWED = {
    '.github/workflows/vector-perf-temp.yml',
    '.github/benchmarks/vector-perf/reuse_candidate.py',
    '.github/benchmarks/vector-perf/test_reuse_candidate.py',
    '.github/benchmarks/vector-perf/fenv_check.py',
}


def compatible(paths):
    return set(paths) <= ALLOWED


def main(root, directory, current):
    directory = Path(directory)
    provenance = directory / 'binary-source-sha.txt'
    original = (provenance if provenance.exists() else directory / 'source-sha.txt').read_text().strip()
    changed = subprocess.check_output(
        ['git', '-C', root, 'diff', '--name-only', original, current, '--'], text=True).splitlines()
    if not compatible(changed):
        print('Cached candidate has different production, tests or compiled proof inputs')
        return 1
    provenance.write_text(original + '\n')
    (directory / 'source-sha.txt').write_text(current + '\n')
    print('Reusing compatible candidate and proof binaries from', original)
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
