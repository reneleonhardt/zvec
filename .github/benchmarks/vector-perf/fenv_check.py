"""Run existing frozen-reference proofs with all x86 FTZ/DAZ combinations."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

PROBE = r'''
#include <immintrin.h>
#include <stdlib.h>
#include <unistd.h>
__attribute__((constructor)) static void configure(void) {
  const char *value = getenv("VECTOR_PERF_MXCSR");
  if (!value) _exit(78);
  unsigned mode = (unsigned)strtoul(value, 0, 10);
  if (mode & ~0x8040u) _exit(78);
  _mm_setcsr((_mm_getcsr() & ~(0x8040u | 0x3fu)) | mode);
  if ((_mm_getcsr() & 0x8040u) != mode) _exit(78);
}
'''


def main(directory, output, *binaries):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binaries = binaries or ('main-exact-bits', 'split-exact-bits', 'exact-bits')
    failed = False
    with tempfile.TemporaryDirectory(prefix='vector-fenv-', dir=output) as temporary:
        source = Path(temporary) / 'probe.c'
        library = Path(temporary) / 'probe.so'
        source.write_text(PROBE)
        subprocess.run(['cc', '-shared', '-fPIC', '-O2', str(source), '-o', str(library)], check=True)
        for mode in (0, 0x40, 0x8000, 0x8040):
            for binary in binaries:
                executable = directory / binary
                with (output / f'{Path(binary).name}-mxcsr-{mode:04x}.txt').open('w') as log:
                    result = subprocess.run([str(executable)], stdout=log, stderr=subprocess.STDOUT,
                                            env=os.environ | {'LD_PRELOAD': str(library),
                                                              'VECTOR_PERF_MXCSR': str(mode)})
                failed |= result.returncode != 0
                print(f'{binary} MXCSR control={mode:04x}: exit {result.returncode}', flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
