"""Profile unchanged warm workloads separately from paired timing runs."""
import os
from pathlib import Path
import subprocess
import sys


def run(before, candidate, output):
    output = Path(output).resolve()
    perf = 'perf'
    for arch in ('auto', 'avx2'):
        env = os.environ | {'VECTOR_PERF_ARCH': arch}
        for version, directory in (('before', before), ('candidate', candidate)):
            for count in (1, 4):
                stem = output / f'warm-{arch}-{version}-n{count}'
                command = [str(Path(directory).resolve() / 'vector_system_search'),
                           'warm', str(output / arch), 'ip32', 'bufferpool', '0', str(count), '1', '3']
                for mode in ('stat', 'record'):
                    args = ([perf, 'stat', '-e', 'task-clock,context-switches,cpu-migrations,page-faults',
                             '-x', ';', '-o', str(stem) + '-stat.txt'] if mode == 'stat' else
                            [perf, 'record', '-e', 'cpu-clock:u', '-g', '--call-graph', 'dwarf',
                             '-o', str(stem) + '.perf'])
                    with Path(str(stem) + f'-{mode}.csv').open('w') as out, \
                            Path(str(stem) + f'-{mode}.log').open('w') as err:
                        subprocess.run([*args, '--', *command], env=env, stdout=out, stderr=err, check=True)
                with Path(str(stem) + '-profile.txt').open('w') as out:
                    subprocess.run([perf, 'report', '-i', str(stem) + '.perf', '--stdio',
                                    '--no-children', '--sort', 'symbol', '--percent-limit', '.2'],
                                   stdout=out, check=True)
            stem = output / f'warm-{arch}-{version}-futex'
            with stem.with_suffix('.csv').open('w') as out, stem.with_suffix('.log').open('w') as err:
                subprocess.run(['strace', '-f', '-ttt', '-T', '-o', str(stem) + '.trace',
                                '-e', 'trace=futex,openat,write',
                                str(Path(directory).resolve() / 'vector_system_search'),
                                'warm', str(output / arch), 'ip32', 'bufferpool', '0', '4', '1', '1'],
                               env=env, stdout=out, stderr=err, check=True)


if __name__ == '__main__':
    run(*sys.argv[1:])
