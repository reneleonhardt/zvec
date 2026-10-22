from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
import os

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
path = root / 'fixture.bin'
if path.exists():
    raise SystemExit('Fixture already exists; preserve the fixed baseline input')
rng = random.Random(421421)
count, dimension, query_count, clusters = int(os.environ.get("PERF_FIXTURE_COUNT", "8193")), 768, 64, 64
scales = [1 / math.sqrt(1 + d / 48) for d in range(dimension)]
bias = [rng.gauss(0, .35) * scale for scale in scales]
centers = [[rng.gauss(0, 1) * scale for scale in scales] for _ in range(clusters)]

def normalize(row):
    norm = math.sqrt(sum(x*x for x in row))
    return [x/norm for x in row]

vectors = array('f')
for i in range(count):
    if i and i % 50 == 0:
        source = (i - 1) * dimension
        row = [vectors[source+d] + rng.gauss(0, .0005) for d in range(dimension)]
    else:
        center = centers[rng.randrange(clusters)]
        noise = rng.uniform(.25, .55)
        row = [bias[d] + center[d] + rng.gauss(0, noise)*scales[d] for d in range(dimension)]
    vectors.extend(normalize(row))
queries = array('f')
for i in range(query_count):
    source = rng.randrange(count) * dimension
    queries.extend(normalize([vectors[source+d] + rng.gauss(0, .003)*scales[d] for d in range(dimension)]))
with path.open('wb') as output:
    output.write(struct.pack('<4I', 0x56454331, count, dimension, query_count))
    vectors.tofile(output)
    queries.tofile(output)
metadata = {
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'count': count, 'dimension': dimension, 'query_count': query_count,
    'clusters': clusters, 'seed': 421421,
    'description': 'Generated embedding-shaped fixture: anisotropic clustered unit vectors, shared bias, 2 percent near duplicates, and held-out noisy queries. This is not a real model embedding corpus.',
}
(root/'fixture.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2))
