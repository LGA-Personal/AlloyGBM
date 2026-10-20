# AlloyGBM benchmarks

The active benchmark is [standard TabArena](tabarena/README.md). Use its isolated pinned environment and official 8×1 protocol for local regression, binary and multiclass comparisons, with current public defaults and validation-selected tuned/ensemble variants.

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --execution local --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-hpo --execution upstream
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory
```

[Installation, profiles, resume, reporting and acceptance](tabarena/README.md) document the full workflow. Smoke is a pipeline check; full Lite has 5,355 outer jobs / 42,840 child fits. Local results are not official leaderboard placements or submission approval.

The previous bespoke harness, datasets, gates, tests and CI definition are preserved in [the historical archive](../docs/archive/benchmarks/pre-tabarena-2026-10-20/README.md), with an exact path/hash manifest. Archived code is historical evidence and is not an active or supported execution path. The retained [benchmark notes](../docs/benchmarks/README.md) distinguish historical reports from standard TabArena evidence.

TabArena does not cover AlloyGBM's ranking, temporal or special modes in this integration. Their library correctness tests remain active; retired benchmark quality/speed gates must not be described as ongoing validation.

Use the [disposable droplet workflow](tabarena/DROPLET.md) to create a dedicated host only when needed, run with an explicit cap, save and verify results, then destroy it. Acceptance requires upstream Ray on Linux x86_64, at least 16 available CPUs, 60 GiB usable RAM (nominal 64 GB), and no smaller thread caps. Published comparisons add only AlloyGBM to upstream entrants and disclose candidate budgets.
