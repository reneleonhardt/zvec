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
    command = [str(exe), *map(str, args)]
    if args[0] in ('prepare', 'fetchoracle', 'fetchcheck', 'oracle', 'check'):
        command = ['nice', '-n', '10', *command]
    with prefix.with_suffix('.csv').open('w') as out, prefix.with_suffix('.log').open('w') as err:
        subprocess.run(command, env=env, stdout=out, stderr=err, check=True)
    return list(csv.DictReader(prefix.with_suffix('.csv').open()))


def compare(arch, reference_name, reference, mode, kind, storage, filtered=0, count=1, reference_arch=None, data_name=None, replica=0):
    label = f'{data_name or arch}-{reference_name}-{mode}-{kind}-{storage}-f{filtered}-n{count}'
    if replica:
        label += f'-r{replica}'
    rows = [[], []]
    data = output / (data_name or arch)
    env = base_env | {'VECTOR_PERF_ARCH': arch}
    work = mode in ('write', 'writealloc')
    if work:
        args = ['alloc' if mode == 'writealloc' else 'write', data, kind, storage, 8193, 7, 128]
        binary = 'vector_pool_workloads'
    else:
        args = [mode, data, kind, storage, filtered, count]
        args += [int(os.environ.get('VECTOR_PERF_COLD_TRIALS', '3'))] if 'cold' in mode else [.35, 3]
        binary = 'vector_system_search'
    if 'alloc' in mode:
        env['LD_PRELOAD'] = str(candidate / 'libvector_alloc_probe.so')
    for stage, directory in enumerate([reference, candidate, candidate, reference]):
        stage_env = env | {'VECTOR_PERF_ARCH': reference_arch} if reference_arch and stage in (0, 3) else env
        result = invoke(directory / binary, args, output / f'{label}-{stage}', stage_env)
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


def prepare(arch, data_name, fixture, checked_kinds):
    data = output / data_name
    data.mkdir()
    (data / 'fixture.bin').symlink_to(output / fixture / 'fixture.bin')
    env = base_env | {'VECTOR_PERF_ARCH': arch}
    invoke(main / 'vector_system_search', ['prepare', data, 'flat', 'mmap', 0], output / f'{data_name}-prepare', env)
    for kind in checked_kinds:
        invoke(main / 'vector_system_search', ['fetchoracle', data, kind, 'mmap', 0], output / f'{data_name}-fetchoracle-{kind}', env)
        for storage in ('mmap', 'bufferpool'):
            for version, directory in [('main', main), ('before', before), ('candidate', candidate)]:
                invoke(directory / 'vector_system_search', ['fetchcheck', data, kind, storage, 0],
                       output / f'{data_name}-fetchcheck-{version}-{kind}-{storage}', env)
            for filtered in (0, 1):
                invoke(main / 'vector_system_search', ['oracle', data, kind, storage, filtered], output / f'{data_name}-oracle-{kind}-{storage}-{filtered}', env)
                for version, directory in [('main', main), ('before', before), ('candidate', candidate)]:
                    invoke(directory / 'vector_system_search', ['check', data, kind, storage, filtered],
                           output / f'{data_name}-check-{version}-{kind}-{storage}-{filtered}', env)


fixtures = {}
for fixture, dim in [('fixture', 768), ('fixture256', 256)]:
    subprocess.run(['nice', '-n', '10', sys.executable, str(source / 'generate_fixture.py'), str(output / fixture)],
                   env=base_env | {'PERF_FIXTURE_DIM': str(dim)}, check=True)
    fixtures[fixture] = json.loads((output / fixture / 'fixture.json').read_text())
metadata = {'baseline': os.environ['BASELINE_SHA'], 'before': os.environ['BEFORE_SHA'],
            'candidate': os.environ['GITHUB_SHA'], 'order': ['reference', 'candidate', 'candidate', 'reference'],
            'allocation_scope': 'caller threads for warm read/search; entire process for cold I/O and writes',
            'cpu_scope': 'entire process', 'suite': os.environ.get('VECTOR_PERF_SUITE', 'affected'), 'fixtures': fixtures,
            'cold_trials': int(os.environ.get('VECTOR_PERF_COLD_TRIALS', '3')),
            'cold_total_scope': 'open plus search; close/destruction excluded; traced separately',
            'avx2_scope': 'AVX512 distance dispatch flags masked before initialization; other hardware features remain native', 'cold_scope': 'new pathname and index/pool on an immutable inode; zero resident Linux page-cache pages before open, without payload-copy writeback'}
for label, directory in [('main', main), ('before', before), ('candidate', candidate)]:
    metadata[label + '_sha256'] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('vector_system_search', 'vector_pool_workloads', 'libvector_alloc_probe.so')
    }
    provenance = directory / 'binary-source-sha.txt'
    if provenance.is_file():
        metadata[label + '_binary_source'] = provenance.read_text().strip()
(output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
kinds = ('flat', 'fp16', 'ivf', 'physical', 'raw16', 'cos32', 'ip32')
cpu_flags = Path('/proc/cpuinfo').read_text().lower()
arches = ('auto', 'avx2') if platform.machine() == 'x86_64' and 'avx512f' in cpu_flags else ('auto',)
if metadata['suite'] == 'cold-ip':
    for arch in arches:
        prepare(arch, arch, 'fixture', ('ip32',))
        for name, reference in [('main', main), ('before', before)]:
            for count in (1, 4):
                compare(arch, name, reference, 'warm', 'ip32', 'bufferpool', count=count)
                compare(arch, name, reference, 'alloc', 'ip32', 'bufferpool', count=count)
            for count in (1, 64):
                for replica in range(2):
                    compare(arch, name, reference, 'cold', 'ip32', 'bufferpool', count=count, replica=replica)
                compare(arch, name, reference, 'coldalloc', 'ip32', 'bufferpool', count=count)
    sys.exit(0)
if metadata['suite'] == 'dot':
    if len(arches) != 2:
        raise RuntimeError('The AVX-512 dot experiment requires a compatible native runner')
    for arch in arches:
        prepare(arch, arch, 'fixture', ('flat', 'cos32', 'ip32'))
        for name, reference in [('main', main), ('before', before)]:
            for kind in ('cos32', 'ip32'):
                for count in (1, 4):
                    for replica in range(2):
                        compare(arch, name, reference, 'warm', kind, 'bufferpool',
                                count=count, replica=replica)
                    compare(arch, name, reference, 'alloc', kind, 'bufferpool', count=count)
                for mode in ('warm', 'alloc'):
                    compare(arch, name, reference, mode, kind, 'bufferpool', filtered=1)
                for count in (1, 64):
                    for replica in range(2):
                        compare(arch, name, reference, 'cold', kind, 'bufferpool',
                                count=count, replica=replica)
                    compare(arch, name, reference, 'coldalloc', kind, 'bufferpool', count=count)
    sys.exit(0)
full_suite = metadata['suite'] == 'all'
storages = ('bufferpool',) if metadata['suite'] == 'storage' else ('mmap', 'bufferpool')
if metadata['suite'] == 'cold':
    for arch in arches:
        prepare(arch, arch, 'fixture', ('flat', 'cos32', 'ip32'))
        for name, reference in [('main', main), ('before', before)]:
            for kind in ('flat', 'cos32', 'ip32'):
                for storage in ('mmap', 'bufferpool'):
                    for count in (1, 64):
                        for replica in range(2):
                            compare(arch, name, reference, 'cold', kind, storage,
                                    count=count, replica=replica)
                        compare(arch, name, reference, 'coldalloc', kind, storage, count=count)
    sys.exit(0)
for arch in arches:
    prepare(arch, arch, 'fixture', kinds)
    for name, reference in [('main', main), ('before', before)]:
        for kind in kinds:
            for storage in storages:
                for threads in (1, 4):
                    for mode in ('warm', 'alloc', 'fetchwarm', 'fetchalloc'):
                        if not full_suite and (mode.startswith('fetch') != (kind == 'physical')):
                            continue
                        compare(arch, name, reference, mode, kind, storage, count=threads)
                if full_suite or kind != 'physical':
                    for mode in ('warm', 'alloc'):
                        compare(arch, name, reference, mode, kind, storage, filtered=1)
                for batch in (1, 64):
                    for mode in ('cold', 'coldalloc', 'fetchcold', 'fetchcoldalloc'):
                        if not full_suite and (mode.startswith('fetch') != (kind == 'physical')):
                            continue
                        compare(arch, name, reference, mode, kind, storage, count=batch)
                        if metadata['suite'] == 'storage' and mode == 'cold' and kind in ('flat', 'cos32', 'ip32'):
                            compare(arch, name, reference, mode, kind, storage, count=batch, replica=1)
        if full_suite:
            for kind in ('flat', 'ivf'):
                for storage in ('mmap', 'bufferpool'):
                    for mode in ('write', 'writealloc'):
                        compare(arch, name, reference, mode, kind, storage)
    if arch == 'auto' and len(arches) > 1:
        for storage in storages:
            for count in (1, 4):
                for mode in ('fetchwarm', 'fetchalloc'):
                    compare(arch, 'candidate-avx2', candidate, mode, 'physical', storage,
                            count=count, reference_arch='avx2')
            for count in (1, 64):
                for mode in ('fetchcold', 'fetchcoldalloc'):
                    compare(arch, 'candidate-avx2', candidate, mode, 'physical', storage,
                            count=count, reference_arch='avx2')
    if metadata['suite'] == 'storage':
        # Restoring ordinary AVX-512 FP16 L2 also affects the mmap fallback.
        for name, reference in [('main', main), ('before', before)]:
            for kind in ('fp16', 'raw16'):
                for count in (1, 4):
                    for mode in ('warm', 'alloc'):
                        compare(arch, name, reference, mode, kind, 'mmap', count=count)
                for count in (1, 64):
                    for mode in ('cold', 'coldalloc'):
                        compare(arch, name, reference, mode, kind, 'mmap', count=count)
    data_name = f'{arch}-d256'
    prepare(arch, data_name, 'fixture256', ('raw16',))
    for name, reference in [('main', main), ('before', before)]:
        for storage in storages:
            for count in (1, 4):
                for mode in ('warm', 'alloc'):
                    compare(arch, name, reference, mode, 'raw16', storage, count=count, data_name=data_name)
            for mode in ('warm', 'alloc'):
                compare(arch, name, reference, mode, 'raw16', storage, filtered=1, data_name=data_name)
            for count in (1, 64):
                for mode in ('cold', 'coldalloc'):
                    compare(arch, name, reference, mode, 'raw16', storage, count=count, data_name=data_name)
