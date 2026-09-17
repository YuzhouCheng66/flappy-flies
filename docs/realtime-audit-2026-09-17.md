# Real-time inference audit — 17 September 2026

## Verdict

**A labelled compact recurrent model is locally feasible. The full model is
not yet real-time on the browser's tested Intel adapter. This is not a claim
of mainstream-device compatibility or a public deployment.**

The original public model, playback rate, physics and success criteria remain
unchanged. The development preview is loopback-only and explicitly displays
“Compact · Research” / “Local GRU”, not the complete fly connectome.

## Measurements

Hardware: i9-13900HX, RTX 4080 Laptop, Intel gen-12lp browser adapter.
Native RTX timings are **not** browser GPU timings.

| Implementation | Actual measured scope | Result |
| --- | --- | --- |
| Full graph, native RTX 4080 | Original complete neural tick, 30 samples | median 5.43 ms, P95 5.59 ms |
| Full graph, segmented WebGPU | Actual Intel browser, 80-step parity run | 9.886 s total, approximately 124 ms/tick |
| Full graph, WASM SIMD + 8 pthread Workers | Actual isolated browser, 80-step parity run | approximately 338–390 ms/tick; rejects real-time gate |
| Compact local GRU + original GBP + physics | Actual browser, frozen 100-seed test | 97/100 successes; all episode control P95 values below 2.10 ms |
| Compact model, complete rendered game | Seed 91000 | 0.215 s page-to-Run; 3.297× actual vs 3.30× specified speed; zero buffer underruns |

The game deadline is 0.08 / 3.30 = **24.242 ms/control tick**. Acceptance
uses additional headroom (median <=15 ms and P95 <=20 ms).

The first rendered run starts with 24 ticks available, not a solved course.
At the time Run was clicked, 40 ticks were available; **732 additional ticks
were computed while playing**. The rolling buffer never exceeded 48 ticks
(1.164 s of displayed play). The course took 18.732 wall seconds / 61.760
simulation seconds. Rendering frame median/P95 were 16.67/33.33 ms: the
game clock was preserved, but occasional dropped display frames remain.
Cached localhost startup was measured at 0.215 and 0.184 seconds; this does
not establish public-network startup on other machines.

## What changed, and what did not

Full-model optimization preserves all **165,122 neurons**, **25,563,197
directed edges**, eight independent agent states, and four neural updates per
control tick. It partitions CSR rows into bounded edge segments and reduces
their partial sums. No graph pruning, skipped neural updates, replayed
trajectory or playback slowdown was used. Reachability analysis found only
nine unreachable edges, so trivial pruning would not solve the bottleneck.

The WASM port includes the complete graph, sensory projection, MLP and GRU
readout. It uses SIMD and persistent, edge-balanced pthread Workers.
Numerical tests against Python passed (80 steps: feature maximum error
approximately 1.5e-7, wrench error approximately 0.0014). Eight Workers were
not faster than one in this implementation. Merely enabling threads is not
sufficient; this backend is **not** accepted as a playable fallback.

The compact candidate is a **different model**, distilled from the original:

- Each agent independently receives the existing 27-dimensional local observation.
- Linear 27→128 + SiLU, GRU-128, Linear 128→128 + SiLU, Linear 128→6.
- Outputs are desired local twist (3) and log precision (3).
- 119,942 learned parameters; separate recurrent memory for each of eight agents.
- The unchanged runtime runs actual radius-constrained directed Sheaf-GBP,
  bounded force allocation and conservative exact-component SAT collision checks.
- No seed, map ID, stage ID or future trajectory is supplied to the neural model.
- The preview visualizes its actual 128 GRU activations in a grid; it does not
  fabricate activity in a 165,122-neuron anatomical brain.

## Training and frozen evaluation

Original teacher rollouts: seeds 94000–94047, 46/48 successes. Failed original
teacher rollouts were reported but excluded from cloning. Two genuine DAgger
rounds collected teacher labels on compact-policy-induced states, including
failed student trajectories, on 26 training seeds each. Total: 98 episodes.
All seeds congruent to 5 modulo 6 remained supervised validation data.

Initial cloning was 0/20 closed-loop. First-round DAgger reached 17/20.
Retraining from scratch after the second round regressed, despite lower
validation MSE. The retained model was instead warm-started with fixed
normalization, full recurrent sequences and AdamW learning rate 5e-5.
Checkpoint selection used closed-loop development results, not MSE alone.

Frozen checkpoint:

`3668b9bce38da54c1af7768465645e4a3d8618355b32da797b7561de1d883055`

Selected as the earliest snapshot with 10/10 on 96000–96009, then confirmed
20/20 on 97000–97019. Those are **development**, not final test seeds.
Before final testing, `local-results/compact/frozen-evaluation.json` fixed the
checkpoint and all 100 final seeds (98000–98099). No model selection or
retraining used this final set.

The actual-browser final result is **97/100**, zero accepted penetrations.
Failures: 98056 at gate 2, 98058 at gate 3, 98061 at gate 0. Minimum accepted
SAT clearance was 2.623e-6 m. Wilson's 95% interval is approximately
91.5–99.0%, conditional on the sampled distribution; it is not a guarantee
for every geometry or device. This is not a matched claim of superiority
over the full model, which was not evaluated on these same 100 seeds here.

## Backend and hosting boundary

GPU selection no longer infers speed from vendor. Adapter-selection helpers
accept measured workload timings; backend-selection tests reject slow,
high-jitter and unvalidated candidates. A webpage still cannot force access
to a PCI GPU that the browser/OS does not expose. Automatic production
integration of the full/compact choices remains separate work.

Local `tools/serve.py` emits `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. The actual WASM browser test
verified `crossOriginIsolated: true` and eight Workers. Public hosting was
not reconfigured. Pthread deployment requires these protections, and a
non-isolated fallback needs a separate non-threaded build, per the
[official Emscripten documentation](https://emscripten.org/docs/porting/pthreads.html).

Still needed before a general release: automatic measured backend selection,
public cached startup tests, deployed isolation if selecting pthreads,
Intel/AMD/Apple and lower-end CPU browser coverage, longer rendering-jitter
and input-latency tests, and explicit full-vs-compact product labelling.
Full-course precomputation is **not** the solution demonstrated here.

## Reproduction

Run from the repository root. Data and checkpoints intentionally stay in
ignored `local-results/`; no large experiment files are uploaded.

```powershell
node --test tests/engine.test.mjs tests/graph.test.mjs tests/backend.test.mjs tests/compact.test.mjs
node tests/compact-node.mjs 97000,97001 local-results/compact/model-3668b9bce38d.json
node tools/summarize-realtime-audit.mjs
python tools/serve.py
```

Browser tests:

- `/tests/browser.html?kernel=segmented`: full graph parity and timing.
- `/tests/wasm-browser.html`: isolated SIMD/pthread test.
- `/tests/compact-browser.html?model=3668b9bce38d`: compact closed-loop test.
- `/?candidate=3668b9bce38d`: labelled, live, short-buffer game preview.

The immutable model exports include dataset hashes, parent checkpoint hash,
normalization, parameter arrays and trainer hash. `acceptance.json` records
final failures, timings, rendered-game telemetry and current source hashes.
