from pathlib import Path
import os
import subprocess
import sys
import tempfile

source = Path(__file__).resolve().parent
binary = Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix='vector-perf-smoke-') as temporary:
    root = Path(temporary)
    env = os.environ | {'PERF_FIXTURE_COUNT': '257'}
    subprocess.run([sys.executable, str(source / 'generate_fixture.py'), str(root)], env=env, check=True)
    def run(mode, kind, storage, filtered=0):
        subprocess.run([str(binary), mode, str(root), kind, storage, str(filtered)], check=True)
    run('prepare', 'flat', 'mmap')
    run('clonecheck', 'flat', 'mmap')
    for kind in ('flat', 'fp16', 'ivf', 'physical', 'raw16', 'cos32', 'ip32'):
        run('fetchoracle', kind, 'mmap')
        for storage in ('mmap', 'bufferpool'):
            run('fetchcheck', kind, storage)
            for filtered in (0, 1):
                run('oracle', kind, storage, filtered)
                run('check', kind, storage, filtered)
