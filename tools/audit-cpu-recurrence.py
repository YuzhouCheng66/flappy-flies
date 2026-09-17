"""Native CPU feasibility probe, NOT a browser/WASM benchmark or full policy.

Times the original 25.56M-edge CSR operator for all eight independent flies,
four recurrent updates per tick. No pruning, downsampling or reduced agents.
Omits sensory injection, policy readout, GBP and physics: a subsystem probe.
"""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import struct
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "graph/manifest.json").read_text())
n, edges = manifest["neurons"], manifest["edges"]
crow = np.zeros(n + 1, dtype=np.int32)
columns = np.empty(edges, dtype=np.int32)
values = np.empty(edges, dtype=np.float32)
cursor = 0
for entry in manifest["chunks"]:
    compressed = (ROOT / "graph" / entry["file"]).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == entry["sha256"]
    raw = gzip.decompress(compressed)
    start, rows, size, neurons = struct.unpack_from("<4I", raw, 8)
    degrees = np.frombuffer(raw, dtype="<u4", count=rows, offset=32)
    delta = np.frombuffer(raw, dtype="<u4", count=size, offset=32 + rows * 4)
    counts = np.frombuffer(raw, dtype="<u2", count=size, offset=32 + rows * 4 + size * 4)
    rowptr = np.concatenate(([0], np.cumsum(degrees, dtype=np.int64)))
    col = np.empty(size, dtype=np.int32)
    weight = np.empty(size, dtype=np.float32)
    for r in range(rows):
        lo, hi = rowptr[r:r + 2]
        col[lo:hi] = np.cumsum(delta[lo:hi])
        weight[lo:hi] = counts[lo:hi].astype(np.float32) / max(1, int(counts[lo:hi].sum())) * np.float32(.5)
    assert hashlib.sha256(col.tobytes()).hexdigest() == entry["columns_sha256"]
    assert hashlib.sha256(weight.tobytes()).hexdigest() == entry["weights_sha256"]
    crow[start:start + rows + 1] = rowptr + cursor
    columns[cursor:cursor + size] = col
    values[cursor:cursor + size] = weight
    cursor += size
assert cursor == edges
if '--reachability' in __import__('sys').argv:
    from scipy.sparse import csr_matrix
    adjacency = csr_matrix((np.ones(edges, dtype=np.float32), columns, crow), shape=(n,n))
    metadata = json.loads((ROOT/'model/model.json').read_text())
    reachable = np.zeros(n, dtype=bool)
    reachable[metadata['input_ids']] = True
    for iteration in range(n):
        updated = reachable | (adjacency @ reachable.astype(np.float32) > 0)
        if np.array_equal(reachable, updated): break
        reachable = updated
    print(json.dumps(dict(reachable_neurons=int(reachable.sum()),total_neurons=n,iterations=iteration,
                         reachable_edges=int(np.diff(crow)[reachable].sum()))))
    raise SystemExit
torch.set_num_interop_threads(1)
matrix = torch.sparse_csr_tensor(torch.from_numpy(crow), torch.from_numpy(columns), torch.from_numpy(values), size=(n, n), check_invariants=True)
torch.manual_seed(20260917)
initial = torch.rand(n, 8) * .1
results = []
with torch.inference_mode():
    for threads in [1, 4, 8, 16]:
        torch.set_num_threads(threads)
        durations = []
        for sample in range(8):
            state = initial.clone()
            start = time.perf_counter()
            for _ in range(4):
                state = .5 * state + .5 * torch.tanh(torch.sparse.mm(matrix, state))
            elapsed = (time.perf_counter() - start) * 1000
            if sample >= 2:
                durations.append(elapsed)
        result = dict(threads=threads, samples_ms=durations, median_ms=statistics.median(durations), max_ms=max(durations))
        results.append(result)
        print(json.dumps(result), flush=True)
report = dict(runtime="Native PyTorch CPU CSR, NOT browser WASM", torch=torch.__version__, neurons=n, edges=edges, agents=8, updates=4, game_budget_ms=80/3.3, results=results)
output = ROOT / "local-results/cpu-feasibility.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(report, indent=2))
