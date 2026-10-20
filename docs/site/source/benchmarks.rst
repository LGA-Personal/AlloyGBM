Benchmarks
==========

AlloyGBM uses standard TabArena for local CPU regression, binary and multiclass
benchmarking. The integration pins upstream source and dependencies, uses official
task splits with enforced **8 folds × 1 set**, and delegates preprocessing,
selection, ensembling, scoring and comparison to TabArena.

The declared default retains current public configuration, including automatic
rounds, with supplied fold validation. A frozen author search space supplies
sampled candidates; TabArena selects the tuned single model and tuned + ensemble
from validation predictions. Outer test scores never drive selection. LightGBM,
XGBoost and CatBoost retain their upstream wrappers and search spaces.

See the `complete installation and workflow guide
<https://github.com/LGA-Personal/AlloyGBM/blob/main/benchmarks/tabarena/README.md>`_
for the isolated Python 3.12 environment, release build and provenance receipt.
Run from the repository root after setup:

.. code-block:: bash

   .venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --threads 1
   .venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-defaults --threads 1
   .venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-hpo --threads 1
   .venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory

Smoke uses the official example's three tasks and one sampled candidate. Lite
defaults uses all 51 datasets with the default only; Lite HPO adds 25 sampled
candidates per method (5,304 jobs / 42,432 child fits). Full scope uses the full
official outer collection and upstream candidate budget. Every profile retains
8×1 bagging. Smoke and filtered subsets do not establish competitiveness.

Runs retain immutable build/dependency/resource identity, exact tasks/configs,
raw upstream caches and a durable failure ledger. Resume requires identical
identity; cached reports regenerate without fitting. Read failure/pending
coverage, common-task scope and upstream imputation fields alongside ranks.
The local four-method pool explicitly disables absent RF calibration/fallback;
optional published context uses upstream defaults and labels mixed hardware and
candidate budgets.

Local native/debug execution isolates each outer job to avoid a reproduced
macOS cross-library warmup crash. Explicit upstream/Ray execution is available.
Timings apply to recorded hardware/resources/execution. Local ranks are
pool-relative, not official leaderboard placements; these commands do not upload
results or certify submission acceptance. Full Lite cutover acceptance requires
``report --published-context`` with upstream calibration/fallback and evidence
bound to the terminal run ledger.

The `historical archive
<https://github.com/LGA-Personal/AlloyGBM/tree/main/docs/archive/benchmarks/pre-tabarena-2026-10-20>`_
preserves retired runners, fixtures, custom gates and their tests. Reports under
``docs/benchmarks/`` retain historical context and are not standard TabArena
evidence. Library correctness tests remain active. Ranking, temporal and
special-mode comparative benchmarking remain outside this integration.
