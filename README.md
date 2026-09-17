# Flappy Flies

[**Play in your browser →**](https://yuzhoucheng66.github.io/flappy-flies/)

Race eight connectome-based flies carrying a T through four narrow gates.
**WASD / arrows** move; **Q / E** rotate. **Switch** swaps the two views.
**Random** makes a new course; **Run**, **Pause**, and **Reset** control the race.

## Runs on your computer

No account, installation, Python server, cloud GPU, or recorded fly trajectory.
The complete neural network runs in WebGPU; local observations, Sheaf-GBP,
force allocation, and collision-checked physics run in a browser worker.
Rendering and controls stay on the main thread. Both players share one clock.

The game appears first. You can practice while approximately **65 MB** of
losslessly compressed graph data and model assets load automatically. Graph
chunks are checked and cached when browser storage permits. Starting the real
race resets both players. This is automatic web loading, not zero download.

Use an up-to-date desktop Chrome or Edge with WebGPU and hardware acceleration.
A discrete GPU is strongly recommended. Some dual-GPU laptops select their
Intel GPU despite the high-performance request; choosing the browser's
high-performance GPU in the operating system can help. Slow GPUs slow the
shared game clock, not the model or physics. Unsupported devices retain human
practice but cannot race the neural flies. Mobile/touch play is not validated.

## What is actually simulated

- Eight independent **165,122-neuron** states share the full **25,563,197-edge**
  frozen MaleCNS-derived graph. Four leaky-tanh sparse updates per control tick.
- Per fly: 27 local sensor values → 2,048 sensory neurons → 512 neural readouts
  → trained MLP + 128-dimensional GRU → waypoint delta and Gaussian precision.
- Local inverse dynamics → analytic **16→3** wrench restrictions → four
  directed diagonal-GBP sweeps on the radius graph → bounded handle forces.
- Real message magnitudes drive the glowing packets. The neural panel displays
  sampled actual states at measured soma locations, not an animated fake brain.

The selected V17 checkpoint and all graph edges are preserved. No teacher,
route lookup, pruning, quantization, or remote inference is used during play.
This is a connectome-inspired rate model, not a biological-fidelity fly brain.

## Validation

Release checks: **20/20** preserved Python courses and **5/5** new portable
courses succeeded in native Dawn on the RTX 4080 Laptop; a full **1/1** Chrome
course succeeded on Intel gen-12lp. All had zero accepted penetrations.
Full neural inference measured approximately **14 ms/tick** on native 4080
Dawn versus **400 ms/tick** in the tested Intel browser. These are different
hardware/runtime measurements, not a browser-versus-native speed comparison.

See [machine-readable evidence](validation.json) for exact seeds, hardware,
checkpoint/source hashes, parity errors, contact counts, and terminal states.
Native Dawn tests and actual browser tests are reported separately. GPU
floating-point differences can change trajectories; this is not bitwise
cross-device equivalence or a guarantee for every random seed.

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
