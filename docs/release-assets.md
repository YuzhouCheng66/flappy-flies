# Release assets

The game uses 30 selected source recordings plus seven additional CNS recordings
per course. Keep only these 242 campaign files in Git:

- `manifest.json` and `brains.json`;
- the 30 `level-NNN.bin.gz` files referenced by `COURSE_IDS`;
- the 210 per-agent files referenced by the selected entries in `brains.json`.

`node tools/audit-release-assets.mjs` checks this set. Add `--untrack-unused`
to remove unused campaign files from the Git index while preserving every
local research file. Do not force-add the whole `local-results` directory.

The September 18 cleanup removed 287 unused recordings, probe results and
intermediate research records from the current tree (119.43 MB). Their local
copies remain available. Previous commits still retain Git history; this change
does not rewrite history or reduce a complete historical clone by the same amount.

The approximately 63 MB graph and original model remain available for optional
live inference and reproducibility. Browser campaign play downloads only the
chosen course and requested CNS traces, rather than the whole repository.
