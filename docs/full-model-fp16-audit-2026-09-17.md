# Full-model GPU selection and FP16 audit

The original network remains the default. This is an optimization experiment,
not a public deployment or a claim that integrated-GPU real-time play is solved.

## GPU selection

Real standalone Chrome 152 was tested on this Windows laptop at 18:11 UTC.
All four standard requests (core high-performance, core default, core low-power,
compatibility high-performance) returned **Intel gen-12lp**, in both Window and
Dedicated Worker. Every adapter executed and passed a GPU compute test. WebGL2
also reported Intel UHD Graphics. This is GPU inference, not CPU fallback.

Separately, native Dawn selected the **RTX 4080 Laptop GPU** and executed the
original FP32 model at about 5.5 ms per full neural control tick. This is **not
browser evidence**. No browser flags, Windows GPU preferences or drivers were
changed. The measurements isolate an adapter-exposure/selection boundary; they
do not establish exactly which OS/driver preference caused that boundary.

For a controlled next test, the user can assign Chrome to the high-performance
GPU under Windows Settings → System → Display → Graphics, then save work and
fully relaunch Chrome. The same diagnostic should then be rerun; success means
it reports NVIDIA and actually computes, not merely that a setting was saved.
See [Microsoft's app GPU preference instructions](https://support.microsoft.com/en-us/windows/hardware/display-graphics/optimizations-for-windowed-games-in-windows-11)
and [Chrome's WebGPU troubleshooting](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips).

## What was preserved and changed

Preserved: 165,122 neurons, 25,563,197 directed graph edges, eight independent
agent states, four recurrent graph updates per control tick, frozen learned
weights, local GRU readout, actual Sheaf-GBP, bounded-force physics and exact
collision authority. No training, pruning, neuron replacement, inference-step
skipping, trajectory replay or game-speed reduction was used.

Experimental changes, all opt-in:

- `precision: 'f16'`: binary16 neural storage using `shader-f16` and `enable f16`.
- `precision: 'packed16'`: two half-precision values per u32, using WGSL
  `pack2x16float` / `unpack2x16float`. This does not require `shader-f16`.
- In both modes, multiplication, reduction, sensory drive, nonlinearities,
  readout and GRU remain FP32. The neural state pair falls from **10,567,808**
  to **5,283,904 bytes**. The 102,252,788-byte edge array does not shrink.
- Paired-lane kernels, subgroup scheduling and a lossless source-frequency
  memory permutation were tested. The permutation preserves each edge and the
  order of contributions within each row, remapping semantic IDs at all model
  boundaries. It was not consistently faster and was not enabled by default.

Native half storage and packed half storage did **not** produce identical
rounding on this stack. They are reported separately; neither is presented as
bit-exact FP32.

## Actual-browser timing

Each run used the same 80 recorded observation steps; the first eight steps
were excluded from timing statistics, but included in numerical-error checks.
Timing covers the entire neural `step`, including readback and JS decoding;
Sheaf-GBP, physics and rendering need additional time. Timestamp profiling was
enabled equally in these comparisons.

The finalist sequence was A, B, C, C, B, A. It was completed before the paired
closed-loop tests and native regression batch. Audit ID:
`46df8e55-d5be-4f6c-8f30-d46b2b8fc77e`.

| Full-network implementation | Median ms, two runs | P95 ms, two runs |
|---|---:|---:|
| A: tile16 FP32 | 133.97 / 135.33 | 146.55 / 153.22 |
| B: segmented FP32 | 116.28 / 117.22 | 130.53 / 132.12 |
| C: segmented packed FP16 | 101.69 / 101.98 | 116.54 / 118.10 |

C reduced elapsed time by about **24%** versus A in this repeated block
(approximately **1.32× throughput**). It is still over **4×** the **24.24 ms**
deadline for 3.30× game speed. The faster roughly 98 ms exploratory measurement
is not substituted for these repeat results. Ordinary `shader-f16` storage
was slower (tile16 approximately 150 ms; segmented approximately 126 ms).

Maximum absolute errors over the 80 Python fixture steps:

| Format | Readout neural features | Normalized wrench proposal | Precision |
|---|---:|---:|---:|
| FP32 | 2.09e-7 | 1.40e-3 | 3.71e-5 |
| shader-f16 | 3.60e-4 | 9.25e-4 | 3.22e-5 |
| packed16 | 7.94e-4 | 3.48e-3 | 2.05e-4 |

These are fixture discrepancies, not proof of policy equivalence. A smaller
error in one FP16 output does not mean the lower-precision model is better.

## New-feature compatibility finding

The GPU advertises `subgroup-size-control` and range [8,16]. Both a full model
shader and an independent minimal shader reproduced failures:

- `@subgroup_size(8)`: compilation reports an allowed range of [16,16].
- `@subgroup_size(16)`: DXC compilation fails with `E_FAIL`.

The minimal case uses only 64 integers and `subgroupAdd(1u)`, so this failure
does not require the fly model. It is a finding about this browser/driver stack,
not proof that the new WebGPU feature is generally broken. No unsafe flags or
validation bypasses were used. The feature stays off in the default model.
Its API was checked against the
[Chrome 151–152 release documentation](https://developer.chrome.com/blog/new-in-webgpu-151-152).

## Closed-loop regression

Both jobs finished. Seeds and complete records are in
`local-results/precision-audit-summary.json`.

| Runtime | tile16 FP32 | segmented packed16 | Seeds |
|---|---:|---:|---|
| Actual Chrome Worker, Intel | 2/2 | 2/2 | 92000–92001 |
| Native Dawn, RTX 4080 (not browser) | 20/20 | 19/20 | 92000–92019 |

The native packed16 candidate failed seed 92016 at stage 1, tick 556, with
zero contacts and positive minimum clearance (0.00453 m): it reached the
384-tick stage timeout, not a collision failure. Original FP32 completed this
seed in 764 ticks. Follow-up single-seed ablations both succeeded:
segmented FP32 (765 ticks) and tile16 packed16 (764 ticks). Thus the observed
regression is in the combined candidate; it is not established that either
half storage or the segmented reduction alone causes failure. Tiny numerical
differences can change a closed-loop trajectory. The candidate is not enabled
automatically.

The native and browser rollout batches overlapped on different adapters;
shared system/power effects are possible. These runs validate behavior, not
isolated timing. The separate A/B/C timing block above was completed first.
Two browser seeds are smoke coverage, not evidence of population success ≥90%;
even 20/20 native trials do not establish that lower confidence bound.

## Route toward a 25 ms tick while retaining the full brain

The first route is a real-browser NVIDIA test: unchanged FP32 already runs at
about 5.5 ms natively, but that result must not be represented as browser
performance. Browser adapter exposure is the immediate unresolved boundary.

For Intel, the latest packed16 fixture profile spends approximately 84.8 ms
in four sparse edge passes, out of 91.9 ms total. It would be insufficient to
optimize only the GRU, readback or rendering. A target of roughly 18–20 ms for
the neural graph, leaving time for the remaining controller, is an engineering
budget, not an achieved result.

Prioritized model-preserving research:

1. Improve graph locality and coalesced sparse-matrix/multiple-state access.
   All eight independent brains already share graph storage and are batched;
   this is not a new eightfold batching opportunity. Try graph/community-block
   layouts and cache-aware tiles beyond the source-frequency permutation that
   did not win here. Preserve all edges and remap sensory/readout IDs. Changing
   reduction order still requires numerical and closed-loop checks.
   [GE-SpMM](https://arxiv.org/abs/2007.03179) motivates coalesced row caching
   and avoiding redundant sparse-row loads, not a promised speedup here.
2. Profile mixed precision, retaining FP32 accumulation and sensitive state.
   Plain FP16 was slower on this stack; packed16 saved state bandwidth but
   did not shrink the 102 MB graph. Do not assume INT8 automatically accelerates
   irregular gathers or accept the current combined candidate's regression.
3. Only if exact full updates remain too slow, separately evaluate incremental
   graph propagation with an explicit error budget. This is a proposed
   approximate execution method, not an implemented optimization or replacement
   with a small GRU. Keep every neuron, edge, independent memory and Sheaf-GBP;
   cache the graph contribution and propagate changes. The identity
   `A h_new = A h_old + A (h_new - h_old)` is exact, but saving work by dropping
   small changes is approximate and helps only if changed-edge work is sparse.
   Measure that sparsity first; use bounded accumulated error and full refresh.

For the current positive row-normalized adjacency A and fixed sensory input d,
the graph update is `F(h) = 0.5 h + 0.5 tanh(0.5 A h + d)`. Its infinity-norm
Lipschitz bound is at most 0.75. If each approximate neural update introduces
at most epsilon additional infinity-norm error, then for the same input stream
`e[k+1] <= 0.75 e[k] + epsilon`, hence asymptotic error at most `4 epsilon`.
This is only a graph-state bound: readout normalization, GRU, feedback physics
and changing observations can amplify differences. It is not a task-success
guarantee and does not excuse skipping validation.

Neither deleting three of the four nonlinear updates nor copying one fly's
state to eight flies preserves this model. Four nonlinear recurrent steps also
cannot be collapsed into one multiplication by A^4. No evidence yet guarantees
an exact full-brain 25 ms tick on this Intel adapter.

## Reproduction and evidence

```powershell
node --test tests/engine.test.mjs tests/graph.test.mjs tests/backend.test.mjs tests/compact.test.mjs tests/graph-layout.test.mjs tests/precision.test.mjs
python tools/serve.py
# In real Chrome: http://127.0.0.1:8812/tests/precision-browser.html
# First Repeat finalists, then Paired rollout smoke test.
node tools/summarize-precision.mjs
```

- `/tests/adapter-diagnostics.html`: actual API requests and GPU compute checks.
- `/tests/subgroup-diagnostic.html`: minimal new-feature failure reproduction.
- `local-results/browser.jsonl`: append-only actual-browser measurements.
- `local-results/precision-native.jsonl`: separate native-Dawn measurements.
- `local-results/precision-audit-summary.json`: aggregated evidence plus final
  source hashes. Exploratory trials preceded the final source snapshot.

This is a local developer audit. Cached public startup ≤5 seconds, uninterrupted
live play at the original speed and ≥90% success on mainstream configurations
remain unaccepted. Whole-course precomputation is not a proposed remedy.
