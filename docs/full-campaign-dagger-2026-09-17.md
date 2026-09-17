# Full-brain campaign adaptation

This is local RTX 4080 research. No website deployment, checkpoint replacement,
or Git push is implied by these experiments. The original 60 accepted recordings
remain untouched. Completing a curated campaign is not an unbiased success-rate
evaluation.

## What is preserved

All 165,122 neurons, 25,563,197 directed edges, eight independent neural states,
four neural recurrence steps per control tick, local sensory inputs and genuine
Sheaf-GBP remain active. Collision authority, forces, stage timeouts and final
3 cm docking tolerances are unchanged. There is no privileged runtime controller.

Initial experiments train the existing 198,662-parameter feed-forward decoder.
The subsequent readout experiment also trains the existing 774-parameter GRU
output projection: 199,436 parameters total. The sensory projection, embedding,
GRU recurrence and full neural graph remain frozen. This is not a compact-brain
substitution. Frozen exported weights are checked bit-for-bit.

## Training data and oracle

The collector always runs the full neural network. In `student` mode, its own
actions are executed without teacher replacement. In `guided` mode, a teacher
may replace an action while still updating the real neural state on the resulting
observations. These modes are recorded separately and guided success is never
counted as student success.

A direct goal-feedback teacher is queried only when the entire straight SE(2)
motion to that goal passes conservative swept collision certification. Elsewhere
the original decoder supplies preservation targets. This teacher has privileged
geometry during training; the student does not receive new inputs. It is not a
complete collision-recovery oracle: an already stuck state without a direct
safe path can remain uncorrected. That limitation remains important.

Each agent record stores 512 actual neural features, six teacher proposal values,
six actual student proposal values and a label flag. Complete episode boundaries,
model identities, geometry, mode and binary hashes are retained. Validation level
IDs 63, 71, 83, 91 and 99 never enter optimization. Repeated screening on these IDs
makes them validation, not an untouched final test set.

The first collection contains 31 episodes. The next DAgger round adds student-
induced histories plus guided efficient transit examples. On easy levels, newer
safe-path labels supersede obsolete imitation labels rather than requesting
contradictory behavior. Necessary narrow-gate motion remains anchored.

## Loss and numerical verification

Raw proposal regression was harmful: proposals are saturated before execution,
so equivalent directions can have very different pre-limit magnitudes. Training
therefore compares executed normalized translations/rotation. Braking also needs
an unsaturated magnitude term, because saturation has zero radial gradient.

The readout trainer reconstructs the frozen GRU from each complete feature
history. Before training, it must reproduce the recorded native GPU proposals
within 2e-4 maximum absolute error. Initial measured maximum was 4.59e-6.

Temporal supervision compares successive action changes for the same agent
within one episode. It does not simply penalize rotation or force every action
to be constant. Closed-loop evaluation records path length, backward distance,
total rotation, stalled ticks and per-component force-change RMS, alongside
success and clearance. Better loss alone is not acceptance.

## Screening results before the second DAgger optimization

Common screening IDs: 1, 21, 41, 60, 61, 63, 71, 81, 83, 91, 99, 100.

| Candidate | Independent closed-loop result | Interpretation |
|---|---:|---|
| Raw loss, 2k steps | 0/8 on the initial subset | Rejected |
| Executed-action loss, 20k | 6/12 | Four-wall mixed courses now possible |
| MLP-only braking extension, 20k | 0/12 | Rejected; stopping destabilized |
| MLP + original memory readout, 30k | 7/12 | Five-wall success exists; not stable enough |

The last candidate independently passed levels 61, 63, 99 and 100. Level 100
completed all five walls in 1,322 control ticks, zero contacts, minimum recorded
clearance 0.009325 m. Level 99 completed in 1,198 ticks with two rejected contact
attempts. These are true model executions, not guided demonstrations. The same
candidate failed level 41, so old-scene non-regression has NOT been established.

Checkpoint identity for this milestone:
`da601f4ea90512907f12cd6f66f0710598560887cf939904d42ba3d9f3881561`.

## Second-round DAgger milestone

The collection grew to 60 episodes: 21 autonomous student episodes and 39 guided
episodes, 355,848 agent-time records in total. After superseding contradictory
easy-scene labels, optimization uses 262,168 training and 71,256 validation rows.
The existing readouts are trained for another 30,000 steps with temporal-change
supervision. Graph, GRU recurrence and physics remain unchanged.

`readouts1-step-30000` passed **11/12** of the same screening courses. All eight
four/five-wall courses in that screen passed; level 41 still stalls. These are
screening results, not a claim of >=90% held-out population performance. A full
100-layout audit was started next, including all original 60 layouts.

On matching layouts, compared with `readouts0-step-30000`:

| Level | Old ticks → new ticks | Old path → new path (m) | Old backward → new backward (m) |
|---|---:|---:|---:|
| 1 | 339 → 164 | 2.725 → 2.085 | 0.149 → 0 |
| 61 | 910 → 764 | 10.245 → 9.517 | 0.139 → 0.000495 |
| 100 | 1,322 → 998 | 13.465 → 12.199 | 0.225 → 0 |

Level 100 remains zero-contact. The unadapted original level-1 recording took
177 ticks and 2.256 m, so its improvement is smaller than the comparison with
the first adapted candidate. This distinction prevents overstating progress.
Minimum positive clearance can be small; successful new recordings still need
the independent swept replay audit before campaign acceptance.

### Full 100-layout audit of the second-round candidate

The complete audit yielded **93/100**: 53/60 old layouts and **40/40 four/five-wall
layouts**. Failed IDs: 26, 29, 32, 35, 41, 44 and 46. This is a fixed campaign
audit, partly overlapping development data, not an unbiased generalization
estimate. The non-regression requirement for replacing the original model is
still unmet. Original recordings/default model remain untouched.

Weight identity:
`e8fac4388178f82b50199719a28e643ddfccb43979b10eeff266996390fc3be2`.

### Collision-recovery DAgger

For these actual failed poses, a training-only waypoint oracle uses successful
single-gate segments from training levels 21/30/41/50/60, transformed into the
current gate frame. Every proposed motion is swept-certified against current
walls. It may select a sideways/backward target instead of repeating the blocked
goal-directed action. Source recordings are SHA-256 verified.

The CPU physics oracle probe recovered all seven failed states. A full-neural
collector then reproduced each student's prefix until the first collision and
applied the oracle only after that collision; all seven such recovery rollouts
completed. Those seven results are teacher-assisted, NOT new policy successes.
Separate autonomous student trajectories query labels without intervention.

The label flag is 0 for preservation, 1 for direct safe-goal guidance, and 2 for
route-recovery guidance. Recovery states are deliberately oversampled. No oracle
or route library is included in the inference/game modules.

`readouts2-step-10000` starts from the second-round checkpoint and trains the
same existing readouts for 10,000 steps at learning rate 1e-5. The expanded
collection comprises 74 episodes (28 autonomous student, 39 guided, 7 triggered
recovery), with 318,352 optimization rows and 71,256 validation rows after
supersession. Each 512-row batch includes 64 recovery-label samples when present.

Independent screening passed **18/18**, including all seven prior failures.
Six of those seven had zero contact attempts; level 41 had two. Level 1 improved
to 158 ticks and 0.107 rad cumulative rotation, versus 177 ticks / 0.440 rad in
the original accepted recording. Level 100 completed in 983 ticks with two
contact attempts, so it is faster but NOT more contact-free than the preceding
candidate's zero-contact level-100 rollout.

### Final fixed-campaign audit — complete

`readouts2-step-10000` completed **100/100** fixed layouts: **60/60** original
layouts and **40/40** four/five-wall layouts. No further training is running.
This is a development/regression campaign, not an untouched generalization test.
The autonomous policy uses the original full connectome and real Sheaf-GBP;
no teacher or waypoint library is used at evaluation time.

Weight SHA-256:
`8a5027114ab01dd905ccabaaf0345306b2b7378bfed27b9e151a8f1bd5fc24da`.

On the same original 60 successful layouts, original → final mean metrics:

| Metric | Original | Final | Change |
|---|---:|---:|---:|
| Physics ticks | 374.42 | 367.80 | -1.77% |
| Path length (m) | 4.7492 | 4.5904 | -3.34% |
| Accumulated rotation (rad) | 2.6571 | 2.3330 | -12.19% |
| Contact attempts | 0.6667 | 0.4833 | -27.50% |
| Backward distance (m) | 0.00993 | 0.00133 | -86.62% |
| Per-component force-change RMS | 0.16207 | 0.13851 | -14.54% |

Artifacts: `readouts2-step-10000.eval.json` and
`readouts2-step-10000.behavior.json` in the checkpoint folder. The completed
100-level replay library remains separately versioned: original weights for
levels 1–60, `readouts1` for 61–100. Do not attribute that existing library to
`readouts2`. Neither default public weights nor the live website were replaced.

## Reproduce locally

```powershell
node tools/collect-full-campaign.mjs --mode=student --round=1 --label-all-safe --levels=61,81,100 --parent=C:\absolute\checkpoint.json
python tools/train-full-readouts.py --tag=readouts1 --steps=30000 --parent=C:\absolute\checkpoint.json
node tools/eval-full-campaign.mjs C:\absolute\checkpoint.json "1,21,41,60,61,63,71,81,83,91,99,100"
```

Candidate files and all dataset binaries stay under ignored
`local-results/full-campaign-training`. Candidate replay generation accepts
`--parent=...`; every recording stores its own weight/checkpoint hash. Original
recordings must not be misattributed to an adapted checkpoint or vice versa.
