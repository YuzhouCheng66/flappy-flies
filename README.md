# Flappy Flies

[**Play in your browser →**](https://yuzhoucheng.com/flappy-flies/)

Race eight connectome-based flies carrying a T through **30 selected courses**, with
one to five gates and progressively tighter docking.
**WASD / arrows** move; **Q / E** rotate. **Switch** swaps the two views.
**Run** / **Pause** control the race. Only **YOU WIN** unlocks the next level;
losing briefly shows the result, then resets the same course to its stationary
start. Press **Run** to retry. Progress is saved in this browser.
Early docking is forgiving: 0.18 m / 30° with a 0.10 s hold. These tighten
linearly to 0.03 m / 3.5° and 0.40 s at level 30. Flies start 20% slower than
the calibrated replay pace and linearly return to that pace by level 30;
player movement speed is unchanged.

## Campaign mode (default in this source build)

Player physics runs live locally; flies replay curated, collision-audited
**full-model rollouts**, including synchronized neural activity and real GBP
message strengths. Only the current level is fetched (0.45–2.74 MB compressed).
No WebGPU adapter, neural weights or graph download is needed for this mode.
The source 100-level library totals approximately 159 MB, **not** a startup
download. The game selects 30 of those recordings, preserving their identities.
Replay pacing is calibrated per course; this is not a claim of faster neural
inference or altered physical model performance. Curated success is not a
population success-rate estimate.

Click **01–08** below the CNS to pin any fly's 2,048 sampled neuron states,
including during play. Selection persists across retries and level changes.
Each fly has its own authentic recording; additional traces load on demand
and do not pause the race. The eight lower bars show whole-CNS RMS values.
All additional traces were re-recorded using each course's original full model,
with zero pose difference against every original recorded frame.
The two anatomical labels are **Brain** and **Ventral nerve cord**.

This local build does not itself update the public URL above. Deployment is a
separate step. `node tools/build.mjs` validates and packages the 30 selected files;
their generation must be completed under `local-results/campaign/` first.

There are two single-wall tutorials, then six two-wall, seven three-wall,
seven four-wall and eight five-wall courses. The last visited unlocked level
is remembered; every earlier level can be entered. Loss always retries the
current course. Loopback-only `?try=28`, `?try=29`, and `?try=30` allow independent
hard-course playtests without reading or changing campaign progression.

## Optional live full-model mode (`?mode=live`)

No account, installation, Python server, cloud GPU, or recorded fly trajectory.
The complete neural network runs in WebGPU; local observations, Sheaf-GBP,
force allocation, and collision-checked physics run in a browser worker.
Rendering and controls stay on the main thread. Both players share one clock.

The game appears first and stays still while approximately **65 MB** of
losslessly compressed graph data and model assets load automatically. Graph
chunks are checked and cached when browser storage permits. Starting the real
race resets both players. **Run** is disabled until ready; there is no practice
mode and no automatic start. GPU details records page-to-model-ready and
page-to-Run-ready seconds. This is automatic web loading, not zero download.

Use an up-to-date desktop Chrome or Edge with WebGPU and hardware acceleration.
A discrete GPU is strongly recommended. Some dual-GPU laptops select their
Intel GPU despite the high-performance request. **GPU details** shows the actual
adapter, inference time, and active mode; the website cannot force the OS to
expose an NVIDIA device. Choosing the browser's high-performance GPU in the
operating system may be necessary on such laptops.

The race always runs at the original **3.30× simulation rate**. There is no
throughput-dependent slow motion. Fast GPUs stream real inference with a
128-tick lookahead. Slow GPUs compute this seed's full fly run locally first,
clearly labelled **local precomputation**, then race at full speed. That fallback
is not real-time inference; both arenas stay at the starting pose during preparation.
An unexpected streaming underrun explicitly pauses and prepares the remainder,
instead of silently altering the clock. Unsupported/no-WebGPU devices show
an unavailable message and cannot start a race. No CPU neural
surrogate is claimed. Mobile/touch play is not validated.

## What is actually simulated

- Eight independent **165,122-neuron** states share the full **25,563,197-edge**
  frozen MaleCNS-derived graph. Four leaky-tanh sparse updates per control tick.
- Per fly: 27 local sensor values → 2,048 sensory neurons → 512 neural readouts
  → trained MLP + 128-dimensional GRU → waypoint delta and Gaussian precision.
- Local inverse dynamics → analytic **16→3** wrench restrictions → four
  directed diagonal-GBP sweeps on the radius graph → bounded handle forces.
- Real message magnitudes drive the glowing packets. The neural panel displays
  sampled actual states at measured soma locations, not an animated fake brain.

The selected V17 checkpoint and all graph edges are preserved. Degree-sorted
work scheduling, losslessly packed column/count pairs, vectorized eight-fly
state access, and cooperative 16-way sparse-row reduction accelerate the
bottleneck. Float32 reduction order differs; numerical and closed-loop tests
are required. Four neural updates still run on **every** physical control tick.
No teacher, route lookup, pruning, quantization, or remote inference is used.
This is a connectome-inspired rate model, not a biological-fidelity fly brain.

## Validation

Optimized-kernel checks: **20/20** preserved Python courses, **5/5** new portable
courses, and the default course succeeded in native Dawn on the RTX 4080 Laptop,
with zero accepted penetrations. Complete control/physics averaged roughly
**5.5 ms/tick** there, versus the **24.24 ms/tick** race budget. The Intel browser
is substantially slower: its optimized full-course test succeeded **1/1**,
zero accepted penetrations, at approximately **116 ms/control tick** without
the game renderer. It uses the explicitly labelled preparation fallback;
it is not claimed to meet real-time inference. In the actual game UI, after
local preparation, the four-gate race completed in **18.70 wall seconds** for
**61.68 simulated seconds**, confirming the original **3.30×** rate.
These are different hardware and runtimes; see the separate browser evidence.

See [machine-readable evidence](validation.json) for exact seeds, hardware,
checkpoint/source hashes, parity errors, contact counts, and terminal states.
Native Dawn tests and actual browser tests are reported separately. GPU
floating-point differences can change trajectories; this is not bitwise
cross-device equivalence or a guarantee for every random seed. Trials holding
neural decisions for 2 or 4 physics ticks failed and were **not shipped** as a
runtime mode. Optimizations do not skip decisions to obtain a misleading speedup.

The Python reference layouts are preserved for seed 91000 and 92000–92019.
Other seeds use documented Mulberry32 sampling over the same Twist ranges;
they do not reproduce NumPy's seed-to-map mapping. Collision rejection uses
conservative swept floating-point SAT on both component rectangles. Touching
or rejected attempts are not accepted penetration.

## Develop and maintain

```sh
python tools/serve.py                 # http://127.0.0.1:8812
node --test tests/engine.test.mjs tests/graph.test.mjs
node tools/build.mjs                 # clean static artifact: dist/
```

No npm packages are needed to play, build, or run CPU/graph tests. For optional
compute-only GPU tests, install `webgpu@0.6.1`, then run
`node tests/native-webgpu.mjs 91000,92000`. A supported Dawn GPU is required;
this harness is not a substitute for browser UI testing. The original Python
parity fixture is intentionally not distributed; its test skips when absent.

Pushes to `main` run regression tests and publish the whitelisted artifact to
GitHub Pages. No training traces, original 309 MB graph archive, private paths,
or extra checkpoints are shipped. `.bin.gz` files must be served as opaque
bytes, **without** automatic `Content-Encoding: gzip`: the loader verifies and
decompresses them itself. HTTPS is required outside localhost.

## Credits

MaleCNS v1.0: FlyEM / HHMI Janelia, Cambridge, MRC LMB, Google Research,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Frozen reservoir inspired by [Flyhard](https://github.com/MarkUnthank/flyhard).
See [full attribution and notices](THIRD_PARTY_NOTICES.md).
