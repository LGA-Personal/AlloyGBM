# Standard TabArena benchmark

This integration uses pinned TabArena model contributions, official tasks/splits, enforced **TabArena-v0.1 8 folds × 1 set**, its experiment bundle and its result processor. It benchmarks CPU regression, binary and multiclass classification. Ranking, temporal evaluation, GPU and special modes are outside this suite.

AlloyGBM's declared default is the current public constructor configuration, including automatic rounds and policy, with TabArena's supplied validation partition. The sampled configurations use the frozen author search space in `submission/alloygbm/hpo.py`. TabArena selects the tuned model from validation predictions and produces its usual tuned + ensemble result. Neither selection uses outer test scores. Internal training stopping follows AlloyGBM's public objective/metric behavior; candidate selection uses the task's official metric. Do not optimize this search space against observed outer test scores.

## Install and build

Use Python 3.12 and Rust 1.92.0. Run commands from the repository root. Keep this environment separate from normal library development.

```bash
uv venv --python 3.12 .venv-tabarena
uv pip install --python .venv-tabarena/bin/python --prerelease=allow -r benchmarks/tabarena/requirements.lock
.venv-tabarena/bin/python -m benchmarks.tabarena.prepare_upstream
uv pip install --python .venv-tabarena/bin/python --no-deps -e .cache/tabarena/upstream/packages/tabarena -e .cache/tabarena/upstream/packages/bencheval
VIRTUAL_ENV="$PWD/.venv-tabarena" .venv-tabarena/bin/maturin develop --release
.venv-tabarena/bin/python -m benchmarks.tabarena.provenance record-build
.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena -q
```

The universal lock pins dependencies and platform markers. TabArena itself is pinned at `041f9b3774d1191ffcdcf137cbe4e1e08159b972`; AlloyGBM's installable submission requirement is the deadline-support commit. The final editable release build deliberately replaces that dependency with the selected development source. Record a new build receipt immediately after each rebuild. A receipt is a declaration of that build, not an independent proof of compilation flags; never issue one to bless an old binary.

`prepare_upstream` installs the canonical payload in `tabarena.models.alloygbm`; there is no second local wrapper. It permits only the three registration/extra/family edits and model payload. It refuses unknown edits. `--check` verifies those bytes and the pinned revision. For an intentional payload update, recreate the generated clone after preserving any unrelated work. Do not change upstream validation/scoring code. `registration.patch` records the required tracked registration edits for a future contribution.

The `[benchmark]` dependency union includes upstream's full fitting/preprocessing support and can consume several GB. OpenML data and upstream artifacts are cached under `.cache/tabarena/`; local runs live under `benchmarks/results/tabarena/`. Both are ignored by Git. Cached pickle files must come from trusted local runs.

## Run

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --threads 1 --plan-only
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-defaults --threads 1
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile lite-hpo --threads 1
# Full official outer scope, with the upstream full candidate budget:
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile full --threads 1 --execution upstream
```

| Profile | Outer scope | Random candidates per method |
|---|---|---:|
| smoke | Official example's three tasks, first official split | 1 |
| lite-defaults | All 51 Lite datasets, first official split | 0 |
| lite-hpo | All 51 Lite datasets, first official split | 25 |
| full | Full official task collection | Upstream `all` (currently 200) |

Every profile retains 8×1 inner bagging. The default candidate is additional to sampled candidates. Lite HPO with four methods entails **5,304 outer jobs / 42,432 child fits**; large runs can take days or longer. `--datasets NAME ...` and `--models AlloyGBM LightGBM XGBoost CatBoost` narrow development scope; such runs are explicitly subsets, not full Lite acceptance. `--threads` applies consistently to all entrants and numerical thread environment variables. Omitting it inherits upstream CPU allocation and records the resolved fit resources. Keep the machine otherwise idle for meaningful timing comparisons.

`--execution local` uses upstream's native backend and documented debug/quickstart execution with sequential local fold fitting. Each *outer job* executes in a fresh process: macOS crashes when these particular AlloyGBM/LightGBM OpenMP warmups share process state. Dataset loading, preprocessing, warmup, eight folds, fit limits, predictions and scoring remain upstream code. `--execution upstream` uses upstream Ray execution and defaults. Local timings are evidence for the recorded execution/resources only; official maintainers can require a full Ray/cluster rerun on their hardware.

The runner stores immutable configs, tasks, serialized jobs, code/build/dependency identity and an fsynced job ledger. Completion requires a cache with eight fitted children, enforced standard protocol, finite aligned predictions and disjoint training/OOF and test rows. Failures are recorded, never converted to scores. Interrupts leave an interrupted `started` entry; resume reruns it through upstream's cache path. Torn final ledger writes are recoverable. Concurrent drivers for one run directory are rejected.

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --threads 1 --resume /absolute/run/directory
# Retry recorded failures explicitly; retain the original history:
.venv-tabarena/bin/python -m benchmarks.tabarena.run --profile smoke --threads 1 --resume /absolute/run/directory --retry-failed
```

Supply the same profile/scope/resources/execution on resume. Changed source, extension, wrapper, dependencies, lock, protocol or configs fail identity validation. Completed jobs with missing caches fail loudly. An upstream cache hit is separately recorded; it is not a fresh timing measurement. The library checks fit deadlines at complete-round boundaries; the adapter reserves 5% and rejects a fit that overruns its allocated wrapper budget.

## Report and interpret

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory
# Optional broader context; downloads published upstream artifacts:
.venv-tabarena/bin/python -m benchmarks.tabarena.report /absolute/run/directory --published-context
```

Reporting never fits models. Only ledger-completed, conformant caches enter upstream `EndToEnd` processing. Normal outputs include per-split results, leaderboard, method info, win-rate/Pareto plots, and default/tuned/tuned + ensemble variants; configuration and variant CSVs are exported beside `coverage.csv`. The one-candidate smoke is a workflow check, not competitive evidence. A zero-candidate profile has no actual tuning, even if upstream emits the variant names.

The primary four-method local pool has no RandomForest calibration/fallback entrant. Its context explicitly sets `calibration_method=None` and `fillna_method=None`; the upstream comparison engine otherwise owns metrics, ranks, normalization, aggregation and ties. Coverage and `summary.json` disclose partial scope and failures; inspect upstream `imputed` fields and common-task scope before interpreting ranks. An entrant with no successful fit still appears in coverage. Local-pool Elo is pool-relative, not an official leaderboard position.

The optional published context preserves upstream's default calibration/fallback behavior. Its timings mix source hardware, package versions and candidate budgets; use them as context, never as matched local speedups. No result is uploaded or marked verified by these commands. A future official contribution requires the self-contained model payload, registration, dependency pin, audit evidence, complete requested scope and maintainer review/reruns.

## Migration gate

The old harness remains active until standard Lite acceptance: all expected jobs have recorded terminal outcomes, all variants process, failures have explanations, resume does not refit completed jobs, and cached report regeneration works. The retirement PR is separate and must not merge before that evidence exists. Preserve library correctness tests when retiring benchmark-only assertions. Full-suite completion must never be inferred from smoke success.

After the full Lite run and report regeneration:

```bash
.venv-tabarena/bin/python -m benchmarks.tabarena.acceptance /absolute/lite-hpo/run
# If there were failures, supply a JSON object mapping each failed ledger key to
# an examined explanation. Adapter/protocol defects must be fixed and rerun.
.venv-tabarena/bin/python -m benchmarks.tabarena.acceptance /absolute/lite-hpo/run --failure-explanations explanations.json
```

Archive the resulting `acceptance.json` and provenance with development evidence before merging the retirement PR. This mechanical audit does not approve failure explanations or establish official submission eligibility.
