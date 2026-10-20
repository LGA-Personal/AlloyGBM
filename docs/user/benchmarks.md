# Benchmarks

AlloyGBM uses [standard TabArena](../../benchmarks/tabarena/README.md) for local CPU regression, binary and multiclass benchmarking. The integration pins upstream source and dependencies, uses official task splits with enforced **8 folds × 1 set**, and delegates preprocessing, selection, ensembling, scoring and comparison to TabArena.

The declared default retains AlloyGBM's current public configuration, including automatic rounds, with supplied fold validation. A frozen author search space provides sampled candidates; TabArena chooses the tuned single model and tuned + ensemble using validation predictions. Outer test scores are never used for selection. LightGBM, XGBoost and CatBoost use their upstream wrappers/search spaces under the same selected local resources.

Follow the [installation/build instructions](../../benchmarks/tabarena/README.md) before running:

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --execution local --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-defaults --execution upstream
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-hpo --execution upstream
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory
```

Smoke covers the official example's three tasks with one sampled candidate. Lite defaults covers all 51 datasets with the default only; Lite HPO adds 25 sampled candidates per booster (5,355 jobs / 42,840 child fits, including default RF). Full scope uses the full official outer collection and upstream candidate budget. All profiles retain 8×1 bagging. Smoke and filtered subsets do not establish competitiveness or full acceptance.

Each run stores immutable build/dependency/resource identity, exact tasks/configs, raw upstream caches and a durable failure ledger. Resume requires identical identity; cached reports can be regenerated without fitting. Failures and pending outcomes remain in coverage. Inspect common-task scope and upstream imputation fields alongside ranks. The matched local pool adds a default-only RandomForest for upstream RF calibration/fallback; optional published-context reports use upstream defaults and label mixed hardware and candidate budgets.

Upstream Ray is the default; native/debug execution is a diagnostic option that runs each outer job in a fresh process to avoid a reproduced macOS cross-library warmup crash. Explicit upstream/Ray mode is also available. Timings apply to the recorded hardware/resources/execution; local ranks are pool-relative and are not official leaderboard placements. These commands do not upload results or certify submission acceptance. Full Lite cutover acceptance also requires `report --published-context`, preserving upstream calibration/fallback and binding that report to the terminal run ledger.

The [historical harness archive](../archive/benchmarks/pre-tabarena-2026-10-20/README.md) preserves old runners, dataset fixtures, custom benchmark gates and their tests. [Historical reports](../benchmarks/README.md) retain their original context and are not standard TabArena evidence. Library correctness tests remain active. Ranking, temporal and special-mode comparative benchmarking are outside the new integration.

Use the [disposable droplet workflow](../../benchmarks/tabarena/DROPLET.md) to create a dedicated host only when needed, run with an explicit cap, save and verify results, then destroy it. Acceptance requires upstream Ray on Linux x86_64, at least 16 available CPUs, 60 GiB usable RAM (nominal 64 GB), and no smaller thread caps. Published comparisons add only AlloyGBM to upstream entrants and disclose candidate budgets.
