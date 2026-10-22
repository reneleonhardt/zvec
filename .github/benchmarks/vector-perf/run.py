from pathlib import Path
from statistics import median
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys

main, before, candidate, output = map(lambda s: Path(s).resolve(), sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parent
summary = {}
base_env = os.environ.copy()
base_env.pop('LD_PRELOAD', None)


def invoke(exe, args, prefix, env):
    with prefix.with_suffix('.csv').open('w') as out, prefix.with_suffix('.log').open('w') as err:
        subprocess.run([str(exe), *map(str, args)], env=env, stdout=out, stderr=err, check=True)
    return list(csv.DictReader(prefix.with_suffix('.csv').open()))


def compare(arch, reference_name, reference, mode, kind, storage, filtered=0, count=1):
    label = f'{arch}-{reference_name}-{mode}-{kind}-{storage}-f{filtered}-n{count}'
    rows = [[], []]
    data = output / arch
    env = base_env | {'VECTOR_PERF_ARCH': arch}
    work = mode in ('write', 'writealloc')
    if work:
        args = ['alloc' if mode == 'writealloc' else 'write', data, kind, storage, 8193, 7, 128]
        binary = 'vector_pool_workloads'
    else:
        args = [mode, data, kind, storage, filtered, count]
        args += [3] if 'cold' in mode else [.35, 3]
        binary = 'vector_system_search'
    if 'alloc' in mode:
        env['LD_PRELOAD'] = str(candidate / 'libvector_alloc_probe.so')
    for stage, directory in enumerate([reference, candidate, candidate, reference]):
        result = invoke(directory / binary, args, output / f'{label}-{stage}', env)
        if not result:
            raise RuntimeError(f'empty benchmark result: {label}-{stage}')
        if work:
            by_trial = {}
            for row in result:
                values = by_trial.setdefault(row['trial'], {})
                for field in ('wall_ns', 'cpu_ns', 'allocations', 'allocated_bytes'):
                    values[field] = values.get(field, 0) + float(row[field]) * int(row['operations'])
            result = list(by_trial.values())
        rows[stage in (1, 2)].extend(result)
    fields = ('allocations', 'allocated_bytes') if 'alloc' in mode else ('cpu_ns', 'wall_ns')
    if 'cold' in mode:
        fields += ('open_allocations', 'open_allocated_bytes', 'total_allocations', 'total_allocated_bytes') if 'alloc' in mode else ('open_ns', 'open_cpu_ns', 'total_wall_ns', 'total_cpu_ns')
    values = {}
    for field in fields:
        a, b = [median(float(row[field]) for row in side) for side in rows]
        values[field] = {'reference': a, 'candidate': b, 'delta_percent': 100 * (b / a - 1) if a else None}
    summary[label] = values
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(label, {field: round(v['delta_percent'], 2) if v['delta_percent'] is not None else None for field, v in values.items()}, flush=True)


subprocess.run([sys.executable, str(source / 'generate_fixture.py'), str(output / 'fixture')], check=True)
metadata = {'baseline': os.environ['BASELINE_SHA'], 'before': os.environ['BEFORE_SHA'],
            'candidate': os.environ['GITHUB_SHA'], 'order': ['reference', 'candidate', 'candidate', 'reference'],
            'allocation_scope': 'caller threads for warm read/search; entire process for cold I/O and writes',
            'cpu_scope': 'entire process',
            'avx2_scope': 'AVX512 distance dispatch flags masked before initialization; other hardware features remain native', 'cold_scope': 'new file with zero resident Linux page-cache pages before open'}
for label, directory in [('main', main), ('before', before), ('candidate', candidate)]:
    metadata[label + '_sha256'] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('vector_system_search', 'vector_pool_workloads', 'libvector_alloc_probe.so')
    }
(output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
kinds = ('flat', 'fp16', 'ivf', 'physical', 'cos32', 'ip32')
cpu_flags = Path('/proc/cpuinfo').read_text().lower()
arches = ('auto', 'avx2') if platform.machine() == 'x86_64' and 'avx512f' in cpu_flags else ('auto',)
for arch in arches:
    data = output / arch
    data.mkdir()
    (data / 'fixture.bin').symlink_to(output / 'fixture' / 'fixture.bin')
    env = base_env | {'VECTOR_PERF_ARCH': arch}
    invoke(main / 'vector_system_search', ['prepare', data, 'flat', 'mmap', 0], output / f'{arch}-prepare', env)
    for kind in kinds:
        invoke(main / 'vector_system_search', ['fetchoracle', data, kind, 'mmap', 0], output / f'{arch}-fetchoracle-{kind}', env)
        for storage in ('mmap', 'bufferpool'):
            for filtered in (0, 1):
                invoke(main / 'vector_system_search', ['oracle', data, kind, storage, filtered], output / f'{arch}-oracle-{kind}-{storage}-{filtered}', env)
                for version, directory in [('main', main), ('before', before), ('candidate', candidate)]:
                    invoke(directory / 'vector_system_search', ['check', data, kind, storage, filtered],
                           output / f'{arch}-check-{version}-{kind}-{storage}-{filtered}', env)
    for name, reference in [('main', main), ('before', before)]:
        for kind in kinds:
            for storage in ('mmap', 'bufferpool'):
                for threads in (1, 4):
                    for mode in ('warm', 'alloc', 'fetchwarm', 'fetchalloc'):
                        compare(arch, name, reference, mode, kind, storage, count=threads)
                for mode in ('warm', 'alloc'):
                    compare(arch, name, reference, mode, kind, storage, filtered=1)
                for batch in (1, 64):
                    for mode in ('cold', 'coldalloc', 'fetchcold', 'fetchcoldalloc'):
                        compare(arch, name, reference, mode, kind, storage, count=batch)
        for kind in ('flat', 'ivf'):
            for storage in ('mmap', 'bufferpool'):
                for mode in ('write', 'writealloc'):
                    compare(arch, name, reference, mode, kind, storage)
