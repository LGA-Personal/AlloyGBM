# Benchmarks

AlloyGBM uses [standard TabArena](../../benchmarks/tabarena/README.md) for local CPU regression, binary and multiclass benchmarking. The integration pins upstream source and dependencies, uses official task splits with enforced **8 folds × 1 set**, and delegates preprocessing, selection, ensembling, scoring and comparison to TabArena.

The declared default retains AlloyGBM's current public configuration, including automatic rounds, with supplied fold validation. A frozen author search space provides sampled candidates; TabArena chooses the tuned single model and tuned + ensemble using validation predictions. Outer test scores are never used for selection. LightGBM, XGBoost and CatBoost use their upstream wrappers/search spaces under the same selected local resources.

Follow the [installation/build instructions](../../benchmarks/tabarena/README.md) before running:

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-defaults --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-hpo --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory
```

Smoke covers the official example's three tasks with one sampled candidate. Lite defaults covers all 51 datasets with the default only; Lite HPO adds 25 sampled candidates per method (5,304 jobs / 42,432 child fits). Full scope uses the full official outer collection and upstream candidate budget. All profiles retain 8×1 bagging. Smoke and filtered subsets do not establish competitiveness or full acceptance.

Each run stores immutable build/dependency/resource identity, exact tasks/configs, raw upstream caches and a durable failure ledger. Resume requires identical identity; cached reports can be regenerated without fitting. Failures and pending outcomes remain in coverage. Inspect common-task scope and upstream imputation fields alongside ranks. The local four-method pool explicitly disables absent RF calibration/fallback; optional published-context reports use upstream defaults and label mixed hardware and candidate budgets.

Local native/debug execution runs each outer job in a fresh process to avoid a reproduced macOS cross-library warmup crash. Explicit upstream/Ray mode is also available. Timings apply to the recorded hardware/resources/execution; local ranks are pool-relative and are not official leaderboard placements. These commands do not upload results or certify submission acceptance. Full Lite cutover acceptance also requires `report --published-context`, preserving upstream calibration/fallback and binding that report to the terminal run ledger.

The [historical harness archive](../archive/benchmarks/pre-tabarena-2026-10-20/README.md) preserves old runners, dataset fixtures, custom benchmark gates and their tests. [Historical reports](../benchmarks/README.md) retain their original context and are not standard TabArena evidence. Library correctness tests remain active. Ranking, temporal and special-mode comparative benchmarking are outside the new integration.
