# Standard TabArena benchmark

This integration uses pinned TabArena model contributions, official tasks/splits, enforced **TabArena-v0.1 8 folds × 1 set**, its experiment bundle and its result processor. It benchmarks CPU regression, binary and multiclass classification. Ranking, temporal evaluation, GPU and special modes are outside this suite.

AlloyGBM's declared default is the current public constructor configuration, including automatic rounds and policy, with TabArena's supplied validation partition. The sampled configurations use the frozen author search space in `submission/alloygbm/hpo.py`. TabArena selects the tuned model from validation predictions and produces its usual tuned + ensemble result. Neither selection uses outer test scores. Internal training stopping follows AutoGluon’s resolved stopping metric. At this pin, ROC AUC evaluation defaults to log-loss stopping, just as for the peer wrappers; explicit AUC stopping is supported. Regression uses RMSE, and multiclass uses native softmax log loss. Candidate selection uses the official task metric. Do not optimize this search space against observed outer test scores.

“Public defaults” refers to constructor parameters. TabArena supplies validation and the stopping metric: binary and regression callbacks use absolute improvement with the configured patience, and do **not** apply the public `1e-3` relative-improvement stopping tolerance. Multiclass uses native softmax log loss and retains that relative stopping rule. Consequently, the default arm can choose different training lengths from a direct public estimator fit with `eval_set`. Reports disclose this stopping-policy distinction; the library's public behavior is unchanged.

For an on-demand dedicated Linux machine, use the [disposable droplet workflow](DROPLET.md): create, bootstrap, run with an explicit cap, copy and verify evidence, then destroy.

## Install and build

Use Python 3.12 and Rust 1.92.0. Run commands from the repository root. Keep this environment separate from normal library development.

```bash
uv venv --python 3.12 .venv-tabarena
uv pip install --python .venv-tabarena/bin/python --prerelease=allow -r benchmarks/tabarena/requirements.lock
.venv-tabarena/bin/python -m benchmarks.tabarena.prepare_upstream
uv pip install --python .venv-tabarena/bin/python --no-deps -e .cache/tabarena/upstream/packages/tabarena -e .cache/tabarena/upstream/packages/bencheval
VIRTUAL_ENV="$PWD/.venv-tabarena" .venv-tabarena/bin/maturin develop --release
.venv-tabarena/bin/python -m benchmarks.tabarena.provenance record-build
.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena benchmarks/tests/test_tabarena_cloud.py -q
```

The universal lock pins dependencies and platform markers. TabArena itself is pinned at `041f9b3774d1191ffcdcf137cbe4e1e08159b972`; AlloyGBM's installable submission requirement is the deadline-support commit. The final editable release build deliberately replaces that dependency with the selected development source. Record a new build receipt immediately after each rebuild. A receipt is a declaration of that build, not an independent proof of compilation flags; never issue one to bless an old binary.

`prepare_upstream` installs the canonical payload in `tabarena.models.alloygbm`; there is no second local wrapper. It permits only the three registration/extra/family edits and model payload. It refuses unknown edits. `--check` verifies those bytes and the pinned revision. For an intentional payload update, recreate the generated clone after preserving any unrelated work. Do not change upstream validation/scoring code. `registration.patch` records the required tracked registration edits for a future contribution.

The `[benchmark]` dependency union includes upstream's full fitting/preprocessing support and can consume several GB. OpenML data and upstream artifacts are cached under `.cache/tabarena/`; local runs live under `benchmarks/results/tabarena/`. Both are ignored by Git. Cached pickle files must come from trusted local runs.

## Run

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --execution local --threads 1 --plan-only
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --execution local --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-defaults --execution upstream
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-hpo --execution upstream
# Full official outer scope, with the upstream full candidate budget:
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile full --execution upstream
```

| Profile | Outer scope | Random candidates per method |
|---|---|---:|
| smoke | Official example's three tasks, first official split | 1 |
| lite-defaults | All 51 Lite datasets, first official split | 0 |
| lite-hpo | All 51 Lite datasets, first official split | 25 |
| full | Full official task collection | Upstream `all` (currently 200) |

Every profile retains 8×1 inner bagging. The default candidate is additional to sampled candidates. The default entrant is **AlloyGBM only**. Its defaults require **51 jobs / 408 child fits**; Lite HPO requires **1,326 jobs / 10,608 child fits**, including those defaults. Existing competitors come from published TabArena artifacts. Peer reruns are optional via `--models AlloyGBM LightGBM XGBoost CatBoost RandomForest` (5,355 jobs); they are useful for matched hardware timing, not required for Lite acceptance. `--datasets NAME ...` narrows the immutable scope and is not full Lite acceptance. `--threads` applies consistently to all entrants and numerical thread environment variables. Omitting it inherits upstream CPU allocation and records the resolved fit resources. Keep the machine otherwise idle for meaningful timing comparisons.

The default is `--execution upstream`. Acceptance requires Linux x86_64, at least 16 available CPUs, at least 28 GiB usable RAM (nominal 32 GB), no numerical thread caps below 16, and upstream Ray execution. This is the project’s compute policy for the selected DigitalOcean CPU-optimized host, not a claim that TabArena forbids other hardware.

`--execution local` is an explicit development option and uses upstream's native backend and documented debug/quickstart execution with sequential local fold fitting. For local/macOS diagnostics, each *outer job* executes in a fresh process: macOS crashes when these particular AlloyGBM/LightGBM OpenMP warmups share process state. Dataset loading, preprocessing, warmup, eight folds, fit limits, predictions and scoring remain upstream code. On Linux, `--execution upstream` reuses one upstream context and Ray runtime across outer jobs, while preserving per-job logs; it uses upstream parallel fold scheduling and defaults. The framework divides the whole-machine CPU budget among children and can reduce concurrency for memory. It does not allocate 16 CPUs to each of eight concurrent children. Outer jobs run one at a time to keep each configuration’s resource budget intact. Sequential one-thread runs can complete fewer rounds or time out differently under the fixed one-hour budget; their accuracy and timings are evidence for those recorded resources only; official maintainers can require a full Ray/cluster rerun on their hardware.

The runner stores immutable configs, tasks, serialized jobs, code/build/dependency identity and an fsynced job ledger. Worker stdout/stderr, including crash tracebacks and separate retry-attempt markers, persists in `job_logs/<job-key>.log`. Completion requires a cache with eight fitted children, enforced standard protocol, finite aligned predictions and disjoint training/OOF and test rows. Failures are recorded, never converted to scores. Interrupts leave an interrupted `started` entry; resume reruns it through upstream's cache path. Torn final ledger writes are recoverable. Concurrent drivers for one run directory are rejected.

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --execution local --threads 1 --resume /absolute/run/directory
# Retry recorded failures explicitly; retain the original history:
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --execution local --threads 1 --resume /absolute/run/directory --retry-failed
```

Supply the same profile/scope/resources/execution on resume. Changed source, extension, wrapper, dependencies, lock, protocol or configs fail identity validation. Completed jobs with missing caches fail loudly. An upstream cache hit is separately recorded; it is not a fresh timing measurement. The library checks fit deadlines at complete-round boundaries; the adapter reserves 5% and retains a usable cooperative partial model even if its final round slightly overruns. No completed native round raises an upstream timeout. AutoGluon owns the bag-level budget.

## Report and interpret

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory
# Recommended comparison; downloads published upstream artifacts without fitting peers:
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory --published-only
# Optional matched local pool plus published context (requires suitable local coverage):
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory --published-context
```

Reporting never fits models. The local artifact root is carried explicitly in method metadata so Ray workers and the driver write to the same cache. Reports record their processor code hash as well as the original run identity. Only ledger-completed, conformant caches enter upstream `EndToEnd` processing. Normal outputs include per-split results, leaderboard, method info, win-rate/Pareto plots, and default/tuned/tuned + ensemble variants; configuration and variant CSVs are exported beside `coverage.csv`. Tuned and ensemble weight CSVs replay upstream selection from cached predictions using each variant's recorded settings; their scores must match the standard output. This diagnostic never replaces the upstream scores. The one-candidate smoke is a workflow check, not competitive evidence. A zero-candidate profile has no actual tuning, even if upstream emits the variant names.

An optional matched local pool can include the four boosters and a **default-only RandomForest** fitted on the same host/scope. Its upstream context uses `[Local] RF (default)` for calibration and failure fallback. RF must complete every scoped task before local ranking is allowed. A deliberately narrowed run without RF can generate a diagnostic ranking only with complete coverage; failures cannot silently disappear. Entirely unsuccessful entrants block local ranking. Coverage retains the actual failures even when upstream ranking imputes RF scores. Local Elo remains pool-relative.

The recommended `--published-only` context adds **only AlloyGBM** to the published entrants and does not require local RF or other peer fits. `--published-context` additionally produces the optional local pool. Local LightGBM/XGBoost/CatBoost/RF reruns remain in the matched local pool. Published comparisons retain upstream default RF calibration/fallback and record source metadata and artifact hashes. Lite AlloyGBM has **25 sampled candidates plus default**, while published tuned baselines generally use **200**; this budget difference belongs beside tuned-variant claims. Published timings mix hardware and versions and cannot establish matched local speedups. No command uploads results or marks them verified. The current wrapper deliberately fails closed when a training partition lacks a global class, instead of silently changing prediction columns; inspect such failures. A static memory estimator remains a future profiling task permitted by upstream’s submission guidance.

## Migration gate

The old harness remains active until standard Lite acceptance: all 1,326 expected AlloyGBM jobs have recorded terminal outcomes on the required execution/resources, all three AlloyGBM variants process against the published pool with standard published RF calibration/fallback, failures have explanations, resume does not refit completed jobs, and cached report regeneration works. The retirement PR is separate and must not merge before that evidence exists. Preserve library correctness tests when retiring benchmark-only assertions. Full-suite completion must never be inferred from smoke success.

After the full Lite run, regenerate the standard published-context comparison before acceptance (the local diagnostic pool alone cannot satisfy the gate):

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/lite-hpo/run --published-only
.venv-tabarena/bin/python -m benchmarks.tabarena.acceptance /absolute/lite-hpo/run
# If there were failures, supply a JSON object mapping each failed ledger key to
# an examined explanation. Adapter/protocol defects must be fixed and rerun.
.venv-tabarena/bin/python -m benchmarks.tabarena.acceptance /absolute/lite-hpo/run --failure-explanations explanations.json
```

Archive the resulting `acceptance.json` and provenance with development evidence before merging the retirement PR. This mechanical audit does not approve failure explanations or establish official submission eligibility.

Review follow-up and validation: [2026-10-20 resolutions](../../docs/reviews/2026-10-20-tabarena-resolutions.md).

## Staged execution and metrics

`--stage-datasets NAME ...` and `--stage-candidates N` select work from a full,
immutable plan without changing its identity, splits, candidate parameters or
seeds. A limit of 0 executes only the default; 1 executes default plus the first
sample. Keep `--profile lite-hpo`, the output root, source and resources unchanged
between stages. Completed, checksum-verified fits are reused; only pending work
runs. Do not confuse these execution selectors with `--datasets`, which changes
the benchmark scope. Coverage and summaries mark the unfinished full plan partial.
The droplet session script uses these selectors for smoke, large-data, default
and HPO stages, sharing one cache. Reporting/ensembling uses saved predictions and
never refits configurations.

The official Lite protocol is the first official outer split for each of 51
datasets, with eight inner folds and one bagging set. It is not an arbitrary
multi-seed mean. Upstream varies model seeds across folds and configurations;
we preserve that schedule (default base 0, first sampled base 8, then disjoint
8-seed blocks). Search-space sampling uses 1234. No blanket seed of 42 is imposed.
The full official collection has additional outer splits; Lite is the standard
initial author evaluation, not the final verified hosted entry.

The upstream published-context report produces Elo, ranks, normalized error,
improvability, win-rate comparisons, training/inference timing and Pareto plots,
plus per-dataset/split scores and the default/tuned/tuned+ensemble variants.
Retain `coverage.csv`, per-config/variant CSVs, selection weights, raw validation
and test predictions, manifests, dependency/build identity and published source
hashes. Local tuning has 25 sampled candidates plus default; published methods
usually have 200. Elo is relative to the recorded comparison pool; timing against
published methods mixes hardware and versions. Failed/pending work is visible,
never silently omitted or treated as a measured score. The published RF fallback
is applied by upstream; acceptance still requires every job to reach a terminal
outcome and every failure to have an examined explanation.
