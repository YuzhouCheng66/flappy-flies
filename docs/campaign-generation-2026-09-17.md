# Full-brain campaign generation — 100 accepted local levels

## Delivered locally

All 100 requested levels have accepted full-model recordings and independently
replay-verified digital keyboard solutions. Each group of 20 adds one gate,
from one to five. Late levels have two wide gates interleaved with narrow gates.
The compressed recordings occupy 159,044,295 bytes in total. They can be fetched
per level rather than as one download; frontend integration is not yet done.
No public deployment or Git push was performed.

All accepted recordings use local RTX 4080 Laptop native Dawn, full
FP32 tile16 inference, all 165,122 neurons, 25,563,197 edges, eight independent
states and four recurrences per control tick. Levels 1–60 retain the original
recordings/weights. Levels 61–100 use the adapted existing readout checkpoint
`e8fac4388178f82b50199719a28e643ddfccb43979b10eeff266996390fc3be2`.
The graph and GRU recurrence remain unchanged; identities are stored per level.
There is no teacher action replacement. These are curated accepted episodes,
not a claim about the model's unconditioned success rate.

Each recording contains every float32 physics pose, force, eight RMS activity
values, all 20 directed communication links across four sweeps (actual
message-derived visual strength), and all 2,048 displayed neuron activations
on EVERY control tick. Only display activations are quantized to signed int16;
maximum measured absolute error is approximately 1.526e-5. Physics poses are
checked for exact float32 round-trip preservation. The decoded file is checked
with independent conservative swept collision tests, not endpoint-only checks.

The full GBP natural/precision tensors and all 165,122 neuron histories are
not stored in the compact playback file. Their displayed quantities are real
recordings, not synthetic animations. The frozen model and scene can regenerate
the full tensors.

## Timing is bounded, not declared optimal

Player physics is unchanged: speed cap 0.40 m/s, angular cap 0.40 rad/s,
gravity 0.24, linear damping 8, angular damping 24, and 1/120 s integration.
The global 3.30x presentation rate makes the displayed horizontal cap 1.32 m/s.

A simple rigorous optimistic bound for these stationary starts is
`max(horizontal_distance/speed, vertical_distance/vertical_cap,
required_rotation/turn_speed) + docking_hold`, accounting for terminal
tolerances. This ignores obstacle-induced detours and acceleration/braking,
so it is NOT an achievable-time claim.

An independent controller follows collision-checked shortcuts through each
recording using only actual digital WASD/QE inputs. Its complete key sequence
is saved and replayed again through unmodified player physics. All 100 accepted
levels pass this second check. This supplies an executable upper bound, not a
proof of optimality, and rapid digital key switching is not evidence that an
ordinary human will match that score.

Proposed fly race time is the executable reference time multiplied by a
curriculum slack declining from 1.80 at level 1 to 1.03 at level 100. The final
3% slack is relative to the reference, NOT to a proved human optimum. Difficulty
still needs actual playtesting. Speed factors are metadata only: they have not
been applied to the public game. Retiming a prerecorded route is an arcade
presentation feature, not a new physically simulated high-speed model result.
The original simulation duration remains available in the manifest.

## Original blocker and its resolution

The requested four/five-gate courses include a second wide gate amid narrow
ones. The original frozen model repeatedly failed these mixed transitions: some runs
pass a wall but settle approximately 10–12 cm away from the required 3 cm dock;
others contact/stall. Changing spacing, putting the wide gates consecutively,
and advancing intermediate gates upon rear clearance did not solve the tested
cases. The latter is an opt-in experiment, not enabled in the accepted levels.

Three-, four-, and five-gate all-narrow diagnostic courses can succeed. This
does not satisfy the requested mixed-width distribution and they were not used
to fill the missing campaign levels. No failure was relabelled as success.

The user approved training with strict docking preserved. Full-model readout
adaptation and model-induced DAgger then passed all 40 missing layouts without
teacher intervention at inference. Every newly recorded trajectory passed the
independent swept replay audit and keyboard reference replay. The manifest now
records `complete: true`, with no pending IDs. See
[the training audit](full-campaign-dagger-2026-09-17.md) for failures, changes,
separate policy success rates and later collision-recovery experiments.

## Artifacts and reproduction

- `local-results/campaign/manifest.json`: accepted IDs, hashes, sizes, source
  identity, difficulty, timing bounds, pending IDs and explicit semantics.
- `level-NNN.bin.gz`: compact full-model recording, with per-file weight identity.
- `level-NNN.json`: source episode, geometry, success and collision audit.
- `level-NNN.player.json`: executable keyboard reference, complete key sequence.
- `trials.jsonl`: includes unsuccessful probes and generation attempts.

```powershell
# These commands use only the local NVIDIA GPU / CPU, no browser automation.
node tools/generate-campaign.mjs --levels=1,2,3 --attempts=4
node tools/benchmark-player.mjs
node tools/package-campaign.mjs
node --test tests/engine.test.mjs tests/campaign.test.mjs tests/replay-codec.test.mjs
```

The generator resumes existing accepted level files. Probe mode does not
populate production level IDs. All generation artifacts remain under ignored
`local-results/`; they have not inflated the Git repository.

## Local frontend integration

The playable campaign is now a 30-course selection from the unchanged 100-course
source library (`campaign-selection.mjs` contains the exact mapping). Walls by
display level: 1–2: one; 3–8: two; 9–15: three; 16–22: four; 23–30: five.
Difficulty endpoints now span 30 display levels, not source IDs. The build
manifest retains source IDs/hashes and adds `displayOrder`. Loopback `?try=N`
provides isolated manual testing, with no progression writes. Normal mode
resumes the last entered unlocked course and permits all earlier levels.

### All-eight CNS display upgrade

The missing CNS 02–08 display traces have now been re-recorded for all 30 selected
courses, using each source recording's exact original checkpoint and FP32 tile16
full-graph inference. Every pose and stage matched the source recording with
zero pose error. CNS 01 matched within its original int16 visualization encoding
error (maximum approximately 1.526e-5). `brains.json` links every sidecar to the
source trajectory and weights SHA-256. Seven distinct per-agent traces are
stored per course, not copies or RMS-scaled substitutes. The trace at tick zero
is the actual zero initialization. Existing pose/action recordings are unchanged.

Buttons 01–08 pin the selected CNS during playback and retain selection across
retries and level changes. CNS 02–08 files load only when selected; the game is
not paused for downloading. Until a trace is available, the anatomy is neutral
and the selected label indicates loading, never another agent's activation.
This supersedes the earlier one-CNS recording limitation described below.

The default client now loads the selected verified recording, not an inference
worker. The built preview is at `http://127.0.0.1:8812/dist/`. Each course hash
is checked before decoding. Recorded motion, agent-01 neural activity and actual
GBP strengths share the same replay clock; player physics remains live.
Only a human victory unlocks the next level. No Random control or unrestricted
level skipping remains. Progress is stored in browser-local storage.
The arcade difficulty layer now eases early human docking from 0.18 m / 30° /
0.10 s to 0.03 m / 3.5° / 0.40 s at level 100, with corresponding terminal-speed
tolerances. A separate linear fly playback multiplier rises from 0.80 to 1.00.
Source recordings/manifests remain unchanged; player controls and collision
checks are unchanged. Previously recorded reference timings describe the source
calibration, not this additional easing layer.

The original `?mode=live` route remains available. The public website has not
been deployed by this change. The current recording format stores agent 01's
2,048 sampled neuron activities and all eight agents' RMS, **not** eight separate
per-neuron traces; CNS switching needs additional genuine recording data.
