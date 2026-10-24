# Full TabArena run implementation record

Final scope authorized 2026-10-24: AlloyGBM only, 200 samples plus public default,
51 official Lite outer splits / 51 datasets, published competitors only.
27-hour total allocation cap ($13.50 at $0.50/hour), early cleanup on completion.
The 816-split full run was explicitly declined after the measured cost estimate.
Include Morph, DRO and PL in a versioned search space; preserve first 25 v1
candidates for exact-key reuse. Six legal combinations: manual/morph crossed
with constant-standard / constant-DRO / linear-standard. Never linear+DRO.
No model/library changes; no use of outer scores for candidate selection.

- [x] Baseline Rust workspace passed; Python 1259 tests +64 subtests passed.
- [x] Inspect pinned bundle: custom config generators are an official input.
- [x] Versioned search space; real fit compatibility tests; official full grid.
- [x] Durable incremental checkpoints, local verified sync, restore test: all
  1,326 actual caches/10,608 children restored after simulated server loss.
- [x] Explicit fit import with per-origin provenance; reject changed jobs/build sources.
- [x] Lite-200/full acceptance/report scopes and run scripts; docs + validated skill.
- [ ] Full validation, commit/PR, provision only once ready, bounded Linux pilot.
- [ ] Enforce absolute creation-based 27-hour allocation deadline; stop fits
  with two hours reserved for reporting/export/transfer. Heartbeat tracks backups,
  cost and early cleanup. Retain partial work at the deadline, no budget extension.

Old Lite VM 612107127 was deleted; do not connect to its old IP. Verified final
archive: /Users/lashby/AlloyGBM-evidence/2026-10-23-tabarena/final/alloy-final-evidence.tar.gz.
SHA256 b7280e987618a27e2da4551d78e35b21009595f273fb05e4bb9e74cf3c322a4b.
Worktree /Users/lashby/.codex/worktrees/tabarena-benchmark/AlloyGBM; original
library/source commit f1c6cfcdf4308db7061888e0b0f4d3adafa93ad8. Current task has
not created a replacement VM. Preserve baseline mode sampler for historical reports.
