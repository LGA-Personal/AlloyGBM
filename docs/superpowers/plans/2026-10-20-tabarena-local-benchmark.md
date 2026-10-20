# TabArena Benchmark Replacement Implementation Plan

> **2026-10-20 review addendum:** the implementation decisions in the follow-up
> below supersede the original post-fit-overrun rejection and uncalibrated local
> pool. See `docs/reviews/2026-10-20-tabarena-resolutions.md` for evidence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing benchmark system with standard TabArena evaluation of AlloyGBM defaults, tuned models and tuned ensembles, then archive the legacy system.

**Architecture:** Keep an importable AutoGluon model adapter and small configuration, protocol, provenance, execution and reporting modules under `benchmarks/tabarena/`. Delegate datasets, splits, bagging, prediction caching, tuning and benchmark metrics to pinned TabArena APIs. Use the enforced upstream protocol and normal default/tuned/tuned + ensemble processing; archive the legacy benchmark system after replacement acceptance.

**Tech Stack:** Python 3.12, uv, maturin release builds, TabArena and bencheval pinned to `041f9b3774d1191ffcdcf137cbe4e1e08159b972`, AutoGluon Tabular, ConfigSpace, pandas, pytest, CPU LightGBM/XGBoost/CatBoost.

**Spec:** `docs/benchmarks/2026-10-20-tabarena-local-design.md`. Read it, `docs/benchmarks/2026-10-20-tabarena-submission-audit.md` and `docs/alloygbm_handoff.md` before implementation. The spec/audit record the source mismatch, required submission-template gates and authoritative upstream sources.

## Global Constraints

- User decisions: the fixed arm uses **current public defaults**; use **standard TabArena benchmarking**; **replace and retire the old benchmark system after validation**.
- Start with CPU regression, binary classification and multiclass classification.
- Standard TabArena-v0.1 outer splits and inner **8 folds × 1 set**, enforced by the upstream context; no custom holdout lane.
- Use upstream runner/backend/fold-fitting/evaluation defaults, **zero GPUs** for the four CPU entrants, and the same resolved local CPU budget for each entrant.
- Dedicated **Python 3.12** environment, `.venv-tabarena/`; leave the existing `.venv` and public runtime dependencies alone.
- ConfigSpace seed **1234**; N means **additional sampled configurations**; total search size is **N+1**.
- Include upstream default, tuned and tuned + ensemble outputs; no public default changes, upstream submission, publication or cloud execution.
- Test scores are reporting only; configuration and ensemble selection use inner validation predictions.
- Honor the actual remaining per-child `time_limit` received by the wrapper, subtract preprocessing and leave 5% headroom; deadline support is an acceptance blocker, not logging-only.
- `unsafe_code = "forbid"`; Rust edition 2024; MSRV 1.92.0; preserve 1.x artifact compatibility. The only planned library API/engine change is cooperative fit-time deadline support required by the upstream wrapper contract; unrestricted fits remain unchanged.
- Before and after implementation run `cargo test --workspace` and `.venv/bin/python -m pytest bindings/python/tests/ -q`; retain outcomes in the integration evidence document.

## Review Focus

- A stale native extension or experimental environment must fail provenance checks rather than receive the current commit's label (Task 1).
- Integer-only numeric data, missing/unseen categories, and rare classes must retain valid training/prediction semantics or produce an explicit recorded failure (Task 3).
- The default arm must not acquire sampled parameters, stopping rules or an internal holdout through wrapper defaults (Tasks 3–4).
- Interrupted or changed-build runs must not reuse incompatible cached results, lose failed jobs or silently refit completed jobs (Task 5).
- Validation/test disagreement, missing results and published hardware/budget differences must preserve upstream evaluation semantics and visible imputation/failure metadata (Task 6).

---

## File map and interfaces

Create these focused modules, implement required fit-time control, validate the replacement, then archive all legacy benchmark payload in Task 8. Estimator code and library correctness tests remain active:

| File | Responsibility |
|---|---|
| `benchmarks/tabarena/__init__.py` | Lightweight package; no heavyweight imports |
| `benchmarks/tabarena/requirements.in`, `requirements.lock` | Isolated pinned environment input and resolved lock |
| `benchmarks/tabarena/provenance.py` | Release-build receipt, runtime identity and content fingerprint |
| `benchmarks/tabarena/submission/alloygbm/{__init__,model,hpo,info}.py` | Self-contained, submission-shaped registered model payload |
| `benchmarks/tabarena/configs.py` | Thin import of canonical submitted HPO generator; frozen config serialization |
| `benchmarks/tabarena/protocol.py` | `RunSpec`, `ResolvedRun`, task/exposure manifest and upstream experiment construction |
| `benchmarks/tabarena/run.py` | Thin upstream run/resume CLI and provenance/job ledger |
| `benchmarks/tabarena/report.py` | Upstream standard result processing/comparison plus coverage metadata |
| `benchmarks/tabarena/prepare_upstream.py`, `registration.patch` | Install canonical model payload/registration edits into an otherwise unchanged pinned clone |
| `benchmarks/tabarena/README.md` | Reproducible setup, commands and interpretation |
| `benchmarks/tests/tabarena/test_{provenance,model,protocol,run,report,integration,migration,upstream}.py` | Offline contracts plus explicitly enabled live integration |
| `docs/benchmarks/tabarena_local_v1.md` | Measured smoke/Lite evidence and migration acceptance, written after actual runs |

Modify `.gitignore` for `.venv-tabarena/`, `.cache/tabarena/`, and `benchmarks/results/tabarena/`. Replace the active benchmark README at cutover. Task 8 also updates CI, user/Sphinx docs, contribution guidance and both agent guides, and creates the historical archive manifest. This plan lives under a currently ignored `docs/superpowers/` directory: force-add only this plan if committing it, not unrelated ignored material.

Data interfaces below use JSON-compatible dictionaries for upstream metadata/records. Preserve raw upstream objects only in memory. Persist stable IDs, parameters and hashes instead of trying to JSON-serialize model classes.

### Task 1: Establish an isolated, identifiable benchmark build

**Files:** Create `__init__.py`, `requirements.in`, `requirements.lock`, `provenance.py`, `README.md`, and `test_provenance.py` at the paths above; modify `.gitignore`.

**Interfaces:**
- `record_build_receipt(source_root: Path, output: Path) -> dict[str, object]`: run immediately after a clean release build; bind source/tree, extension hash, toolchain and build flags.
- `collect_provenance(source_root: Path, lock_path: Path, receipt_path: Path) -> dict[str, object]`: collect and validate source/build identity, imports, environment, versions and hardware.
- `fingerprint(payload: dict[str, object]) -> str`: SHA256 of canonical sorted JSON; caller supplies only identity fields.
- CLI: `python -m benchmarks.tabarena.provenance record-build --output PATH`.

- [ ] **Step 1: Run and record the pre-change baseline suites.**

Run the two Global Constraints commands and `.venv/bin/python -m pytest benchmarks/tests/ -q`. Record actual failures; do not infer success from historical logs. Do not change unrelated code to make this benchmark plan proceed.

- [ ] **Step 2: Write provenance contract tests.**

```python
def test_fingerprint_is_order_independent_and_build_sensitive():
    assert fingerprint({"extension": "a", "lock": "b"}) == fingerprint({"lock": "b", "extension": "a"})
    assert fingerprint({"extension": "a"}) != fingerprint({"extension": "c"})
```

Also write `test_stale_extension_receipt_rejected` asserting `ValueError` contains `extension`, `test_source_change_requires_new_receipt` asserting a changed source tree fails preflight, and `test_experiment_or_profile_env_rejected` parameterized over an `ALLOYGBM_EXPERIMENT_` key and `ALLOYGBM_PROFILE`. Supply fake receipt/files via `tmp_path` and monkeypatch metadata collection; no real rebuild is needed for these tests.

- [ ] **Step 3: Run the new tests to establish failure.**

Run `.venv/bin/python -m pytest benchmarks/tests/tabarena/test_provenance.py -q`. Expected: collection/import failure for the missing module.

- [ ] **Step 4: Implement provenance and the environment inputs.**

Pin git requirements for `tabarena[benchmark,plot]` and `bencheval[plot]` to the same spec revision, plus the actual supported AutoGluon peer extras, maturin and pytest. Keep all TabArena imports out of `__init__.py` and provenance helpers. Hash imported native extension files, not only `alloygbm.__file__`; compare receipt against current source/build. Include dirty state rather than silently attributing it to HEAD. Check in source paths/toolchain/build flags with the receipt schema, but keep generated machine-specific receipts ignored.

- [ ] **Step 5: Resolve, install and verify the real isolated build.**

Create `.venv-tabarena` with `uv venv --python 3.12 .venv-tabarena`; compile the lock with pre-releases allowed; install the lock using `uv pip sync --python .venv-tabarena/bin/python ...`. Build AlloyGBM in release mode with `VIRTUAL_ENV` pointing to this environment, then record the receipt. Document the exact successful commands and locked versions. Import all four libraries and TabArena in this environment; installation failures remain explicit blockers to live comparisons. Run the provenance tests there; expected PASS.

- [ ] **Step 6: Commit the environment/provenance deliverable.**

Run `git add .gitignore benchmarks/tabarena benchmarks/tests/tabarena/test_provenance.py` and `git commit -m "bench: isolate and identify TabArena benchmark builds"`.

### Task 2: Honor TabArena fit deadlines in the library

**Files:** Modify `crates/engine/src/trainer/mod.rs`, `crates/engine/src/types.rs`, `crates/engine/src/lib.rs`, `bindings/python/src/train.rs`, `bindings/python/src/params.rs` as needed for keyword plumbing, `bindings/python/alloygbm/_regressor/_core.py`, `bindings/python/alloygbm/classifier.py`, `CHANGELOG.md` and fit API documentation. Create `crates/engine/src/fit_deadline.rs` and `bindings/python/tests/test_fit_time_limit.py`; add focused Rust deadline tests in the engine test modules. Update affected struct constructors/exhaustive enum matches without changing artifact sections.

**Consumes:** Existing complete-round training/early-stopping machinery and monotonic clocks; no new validation split or tuning algorithm.

**Produces:** Keyword-only `time_limit: float | None = None` appended to regression/classification `fit`; internal optional `max_fit_seconds` bridge argument; a validated `Trainer::with_fit_time_limit_seconds(self, limit: Option<f64>) -> EngineResult<Self>` builder; `IterationStopReason::TimeBudgetReached`; trained-state publication only for a complete retained model. This is a fit-time control, not a tuned constructor parameter.

- [ ] **Step 1: Write deadline and unrestricted-equivalence tests.**

Parameterize regression, binary and multiclass. Assert zero, negative, NaN and infinite limits fail validation; omitted and explicit `None` produce identical artifact/prediction bytes; a sufficiently large limit preserves those bytes. Use a deterministic clock seam in Rust to expire before a round and during a multiclass round: assert no half-round/class tree is committed and the stop reason records a budget stop. A timeout with completed rounds retains a loadable complete model and valid diagnostics; no completed model maps to a clear timeout outcome. Include preprocessing consuming the budget and deadline interaction with validation-best-model truncation. Time-sensitive tests assert safe-point behavior rather than exact wall milliseconds.

- [ ] **Step 2: Run focused tests to establish failure.**

Run `.venv/bin/python -m pytest bindings/python/tests/test_fit_time_limit.py -q` and focused engine deadline tests. Expected FAIL because fit has no time-limit keyword and the engine has no deadline machinery.

- [ ] **Step 3: Implement monotonic cooperative budget checks.**

Start a monotonic fit budget before Python input preparation, subtract elapsed preparation at every bridge entry, and pass only the remaining duration into the native trainer. Append the optional budget field at the end of trainer state with default `None`; validate finite positive values. Check before starting work and at iteration/tree safe points in single-output and multiclass training. Commit only complete rounds; do not interrupt between class updates into an inconsistent model. Reuse the existing retained-best validation model logic when budget-stopping, and expose actual retained rounds/stop reason. No clock work or output change in the unrestricted branch. Document possible one-safe-point overshoot; the wrapper will reserve headroom and reject a fit exceeding its original budget. Do not misuse a custom evaluation metric to trigger stopping, since that changes early-stopping semantics and does not cover all task paths.

- [ ] **Step 4: Verify library behavior and rebuild both benchmark/native environments.**

Run the focused Python/Rust tests, `cargo test --workspace`, `cargo clippy --workspace --all-targets` and the full Python suite after a release rebuild. Verify old 1.x artifacts still load unchanged. Rebuild the benchmark environment and issue a new source/build receipt; an old receipt or dependency pin must fail provenance. Expected PASS. No accepted standard benchmark can precede this gate.

- [ ] **Step 5: Commit deadline support separately.**

Stage only the deadline API/engine/bridge tests, CHANGELOG and documentation files; commit with `feat: honor fit time limits for benchmark training`. Use that exact installable commit in the submission dependency metadata; no release publication is implied.

### Task 3: Wrap the current AlloyGBM estimator without changing its defaults

**Files:** Create `benchmarks/tabarena/submission/alloygbm/{__init__,model,hpo,info}.py`, `benchmarks/tabarena/prepare_upstream.py`, `benchmarks/tabarena/registration.patch`, `benchmarks/tests/tabarena/test_model.py` and `benchmarks/tests/tabarena/test_upstream.py`. The local source clone is ignored under `.cache/tabarena/upstream/`.

**Consumes:** Task 1 environment and Task 2 fit-time deadline support; current `GBMRegressor`, `GBMClassifier` and pinned `AbstractModel`.

**Produces:** Importable `AlloyGBMModel` with `ag_key="TA-ALLOYGBM"`, `ag_name="TA-AlloyGBM"`, `ag_priority=65`, `seed_name="seed"`, class attribute `_supported_problem_types` for `regression`, `binary`, `multiclass`; methods:
- `_preprocess(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame`.
- `_fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=1, num_gpus=0, **kwargs) -> None`.
- `_predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray` following the prediction contract in the spec.
- `fit_metadata(self) -> dict[str, object]`: effective constructor parameters, rounds, best round, stopping reason, policy and cap-hit status.

- [ ] **Step 1: Write adapter tests with spy estimators and real tiny fits.**

Define spy estimator classes inside `test_model.py`; monkeypatch the imported AlloyGBM constructor to capture calls. `test_default_fit_preserves_public_params` asserts that constructor overrides contain only `seed` and `n_jobs`, and there is no wrapper-added `early_stopping_rounds`, `n_estimators` or `training_policy`. `test_resources_and_seed_forwarded` asserts `n_jobs=2`, the supplied seed, and rejection of `num_gpus=1`.

Parameterize real tiny numeric fits over all three problem types; assert prediction shapes `(n,)`, `(n,)`, `(n,3)`, finite values, probabilities in `[0,1]`, and multiclass row sums within `1e-6` of one. Add `test_integer_numeric_stays_continuous` asserting post-preprocess numeric dtype float32 and learned continuous binning. Add a categorical/NaN/unseen-category save/load test asserting identical predictions before/after loading. Add `test_missing_expected_class_fails_explicitly` asserting an informative `ValueError` before malformed probabilities can enter scoring.

Add stopping tests: missing validation plus configured patience raises `ValueError`; validation is forwarded as a tuple and never concatenated; an early-stopped save/load preserves predictions and the exact best/completed-round metadata, including a zero-round result via a spy. Add `test_required_upstream_warmup_succeeds` asserting that upstream warmup completes for this importable model before its timed fit; any unsupported capability is declared rather than silently bypassed.

- [ ] **Step 2: Run adapter tests to establish failure.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_model.py -q`. Expected: missing `AlloyGBMModel` import.

- [ ] **Step 3: Implement the adapter methods.**

Use the current upstream declarative class attributes, including `_default_auxiliary_params_extra` for numeric/category raw types, and the current estimator API; no obsolete method overrides or mutation of `self.params`/`self.params_aux`. Keep optional library imports lazy. Preserve category columns/NaNs and cast only numeric columns after superclass preprocessing. `_set_default_params` must not install boosting/stopping defaults. Set permitted raw types explicitly to numeric/category; remove framework-only control keys using upstream parameter extraction. Use already encoded labels and assert the global class dimension. Attach fit diagnostics to serializable adapter state. Subtract wrapper preprocessing from the received child budget, reserve 5% headroom and forward the remaining time to the new `fit(time_limit=...)` argument. Map exhausted budgets to upstream `TimeLimitExceeded`; reject post-fit budget overruns instead of silently reporting success. Use the upstream trained-parameter/refit-capability conventions and declare unsupported capabilities honestly; do not globally disable framework behavior or add further AlloyGBM native/public changes beyond Task 2. Declare model warmup imports/cheap parameters through the upstream warmup API and test that required warmup succeeds.

- [ ] **Step 4: Run adapter and provenance tests.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_model.py benchmarks/tests/tabarena/test_provenance.py benchmarks/tests/tabarena/test_upstream.py -q`. Prepare a fresh clone at the pinned revision, copy canonical submission files and apply only the three required integration edits (lazy exports, dependency extra/extended, website family); install it editable after locked dependency resolution. Metadata uses `MethodMetadata.config`, `model_key="ALLOYGBM"`, local storage, MIT license, `verified=False`, matching `pip_extra` and an exact AlloyGBM source pin including Task 2. The payload may not import `benchmarks.*`.

Run from that clone: `pytest -m models -k AlloyGBM`, `pytest -m models tests/tabarena/models/test_warmup_coverage.py -k AlloyGBM`, `python -P -m tabarena.tools.audit_warmup --model AlloyGBM`, `python -m tabarena.tools.sync_pyproject_extras --check`, Ruff check and format check of the new files. Assert selected model tests actually passed rather than skipped or selected zero tests. Add model-specific tests inside the submitted model folder; no shared test-infrastructure edits. Assert registry discovery, exact config-default resolution from processed output, tree-based family and Ray bagging pickle/save/load equivalence. `test_upstream_protocol_files_unchanged` hashes validation, runner, preprocessing and scoring files against the pinned commit and permits only model/registration edits. Expected PASS; failed warmup, an incompatible dependency pin or missing registry model blocks substantive evaluation.

- [ ] **Step 5: Commit the adapter deliverable.**

Stage the canonical submission payload, preparation helper/registration patch and adapter/upstream tests; commit with `bench: add submission-shaped AlloyGBM TabArena integration`.

### Task 4: Freeze configuration generation and enforce the standard TabArena protocol

**Files:** Create `configs.py`, `protocol.py`, `test_protocol.py`.

**Consumes:** `AlloyGBMModel`; pinned TabArena bundle/context and upstream peer generators.

**Produces:**
- `sample_alloy_configs(n: int, seed: int = 1234) -> list[dict[str, object]]`: sampled configs only, no `_c1` default.
- `alloy_generator() -> AGConfigGenerator`: manual public default `[{}]` plus the search generator.
- Frozen `RunSpec(profile: str = "smoke", num_cpus: int | None = None, dataset_names: tuple[str, ...] | None = None)`.
- `ResolvedRun` dataclass fields `spec: RunSpec`, `tasks: list[dict]`, `configs: dict[str, list[dict]]`, `experiments: list[Experiment]`, `expected_jobs: list[dict]`, `exposure_audit: dict`.
- `audit_exposure(tasks: list[dict], protected_membership: dict, historical_exposure: list[dict]) -> dict`: annotated identities, overlap flags and unknowns.
- `resolve_run(spec: RunSpec) -> ResolvedRun`: resolve metadata/configs and construct upstream jobs without fitting. Download only needed metadata at this stage.

- [ ] **Step 1: Write deterministic configuration and protocol tests.**

```python
def test_sampled_configs_are_stable_and_leave_default_separate():
    a = sample_alloy_configs(5)
    assert a == sample_alloy_configs(25)[:5]
    assert len(a) == 5
    assert all(c["training_policy"] == "manual" for c in a)
    assert all(c["n_estimators"] == 10000 and c["early_stopping_rounds"] == 50 for c in a)
    assert alloy_generator().manual_configs == [{}]
```

Assert all eight search dimensions and fixed mode values match the spec, generated configs validate against current public estimator parameters, and sampled dictionaries contain no framework/foreign-library keys. Use fake metadata and monkeypatched constructors for protocol tests. Assert smoke resolves 3 task splits, 2 configs per library, 24 outer jobs and 192 child fits; Lite uses the first split of every official dataset and 25 sampled configs; full inherits all splits and the bundle default of 200 sampled configs. Assert official validation enforcement, protocol key `8x1`, and no holdout/outer experiments. Assert unspecified CPU allocation and runner options are delegated upstream; reject invalid profiles and nonpositive explicit CPUs. Exposure annotations must never remove tasks from the official suite.

Write `test_exposure_matches_ids_aliases_and_flags_shape_review` with protected-ID match, normalized-name alias, shape-only ambiguity and unknown exposure; assert none receives a false clean-development label. Assert outer test targets are not read by config/task resolution. Mock blocked metadata retrieval and assert the audit is unknown rather than clean.

- [ ] **Step 2: Run protocol tests to establish failure.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_protocol.py -q`. Expected: missing configuration/protocol imports.

- [ ] **Step 3: Implement generators and metadata resolution.**

Use ConfigSpace single-configuration sampling in a seeded sequence to retain prefix stability, and convert NumPy scalars to JSON scalars. Implement `CustomAGConfigGenerator` once in the canonical submission `hpo.py`, with the public default as the first manual config; local `configs.py` imports it without redefining the space. Materialize and save all peer configurations too; never resample during resume. Use the registered `AlloyGBM` model for final runs; compare generated experiments against the custom-generator quickstart as a parity test, not a separate benchmark implementation. Use the pinned v0.1 bundle/context defaults for preprocessing, validation, seeds, fitting and warmup. Pass a CPU override only when supplied, and zero GPUs for CPU entrants; retain upstream resource detection otherwise. Never enable holdout/outer modes or disable official validation enforcement. Resolve exact task/split IDs and profile counts from metadata; expose child-fit estimates separately from job counts. Snapshot protected suite membership and historical-exposure evidence with retrieval status.

- [ ] **Step 4: Run protocol tests and inspect a real no-fit job plan.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_protocol.py -q`. Expected PASS. In a Python invocation call `resolve_run(RunSpec())` and print task names, problem types, metric names, default/sample configs and job/child counts. Verify no estimator fit and no baseline prediction-repository download occurs.

- [ ] **Step 5: Commit configuration/protocol delivery.**

Run `git add benchmarks/tabarena/configs.py benchmarks/tabarena/protocol.py benchmarks/tests/tabarena/test_protocol.py` and `git commit -m "bench: freeze TabArena configurations and local protocols"`.

### Task 5: Execute and resume local jobs with visible failures

**Files:** Create `run.py`, `test_run.py`; update the setup README with the run CLI.

**Consumes:** `RunSpec`, `ResolvedRun`, provenance/fingerprint functions and upstream job caches.

**Produces:**
- `apply_thread_limits(num_cpus: int) -> None`: set OMP/OpenBLAS/MKL/BLIS/vecLib/NumExpr limits before scientific imports.
- `run_experiments(resolved: ResolvedRun, *, output_root: Path, receipt_path: Path, resume: Path | None = None, retry_failed: bool = False) -> Path`: directory with immutable manifests, upstream raw cache and append-only job ledger.
- CLI `python -m benchmarks.tabarena.run --profile PROFILE [--threads N] --output-root PATH --build-receipt PATH [--datasets NAME ...] [--resume RUN_DIR] [--retry-failed] [--plan-only]`.

- [ ] **Step 1: Write runner contracts using a counting fake executor.**

Define the fake executor in `test_run.py`, returning the same raw result shape as the pinned runner. `test_resume_does_not_refit_successes` runs two fake jobs, interrupts after one and resumes; assert each successful key was fit once. `test_failed_job_stays_visible_until_explicit_retry` asserts failed status/error persist and only `retry_failed=True` calls that job again. Parameterize changed extension/config order/adapter source/task scope/CPU/protocol hashes; assert resume raises before execution. `test_thread_limits_precede_scientific_imports` runs the CLI in a subprocess with an import hook and asserts all six thread limits match an explicit `--threads 2` at first NumPy import; unspecified budgets retain upstream resource handling. `test_plan_only_never_fits` asserts zero fake calls. `test_budget_overrun_is_failure` asserts expired/overrunning child fits produce upstream time-limit failures and visible ledger outcomes, never successful out-of-budget scores. `test_missing_peer_is_failure_not_success` asserts nonzero exit and explicit dependency status.

- [ ] **Step 2: Run runner tests to establish failure.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_run.py -q`. Expected: missing runner functions.

- [ ] **Step 3: Implement orchestration and persistence.**

Use stdlib-only imports until thread limits are applied. Write `run_manifest.json`, `tasks.json`, `configs.json` atomically before fitting. Include expected job grid and fingerprint; reject mismatched resumes. Invoke upstream `build_and_run_jobs`/`run_jobs` using default backend/fold fitting and its per-job cache; keep the ledger observational, without introducing a second scheduler or alternate retry/tuning engine; after recovery, verify cached results before adopting them into the ledger. Persist started/completed/failed/skipped records with config/task identity, elapsed time, overruns, exception and raw-cache location. On interruption leave pending jobs discoverable. Check protocol/build cache identity before cache hits. Do not catch exceptions as successful zero-score/zero-time rows.

- [ ] **Step 4: Run contracts and the real plan-only CLI.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_run.py -q` and the CLI with `--profile smoke --plan-only` plus output/receipt paths. Expected PASS; manifests show 24 jobs/192 child fits and no trained result cache. Assert the plan uses enforced `8x1` validation and inherits upstream backend/resource settings. Repeat with an explicit CPU budget and verify that override is recorded.

- [ ] **Step 5: Commit resumable runner.**

Run `git add benchmarks/tabarena/run.py benchmarks/tabarena/README.md benchmarks/tests/tabarena/test_run.py` and `git commit -m "bench: run and resume local TabArena jobs"`.

### Task 6: Use standard upstream processing and leaderboard outputs

**Files:** Create `benchmarks/tabarena/report.py` and `benchmarks/tests/tabarena/test_report.py`.

**Consumes:** Saved manifests/config order, upstream raw caches and pinned `EndToEnd`, `MethodSimulator` and context APIs.

**Produces:**
- `process_standard_results(run_dir: Path) -> list[MethodMetadata]`: upstream raw-to-method processing with normal default/tuned/tuned + ensemble variants.
- `write_report(run_dir: Path, *, published_context: bool = False) -> Path`: upstream leaderboard/CSV/plots plus provenance, diagnostics and coverage sidecars.
- CLI `python -m benchmarks.tabarena.report RUN_DIR [--published-context]`.

- [ ] **Step 1: Write processing-parity and adversarial-selection tests.**

Build a tiny in-memory repository using pinned upstream repository fixtures. `test_standard_variants_match_upstream` compares wrapper results directly with upstream processing: identical method subtypes, metrics, weights, default selection, tuned selection and ensemble behavior. Assert upstream currently uses one selection iteration for tuned and 40 for tuned + ensemble without a local override. `test_selection_uses_validation_not_test` makes configuration A best on validation and B best on test; assert the tuned row chooses A. Change only test outcomes and assert both tuned selection and ensemble weights are unchanged.

`test_compare_inherits_upstream_defaults` spies on context comparison and asserts no replacement scoring/tie/bootstrap/calibration/imputation defaults. `test_missing_results_preserve_imputation_flags` checks the upstream fallback and flags remain intact beside a ledger of measured versus failed jobs; no imputed row is labeled measured. `test_perfect_scores_pass_through` checks zero/perfect scores are handed unchanged to upstream evaluation. `test_search_cost_distinct_from_selected_fit` checks attempted fit cost 2+3+4=9 seconds is separate from the selected model fit. `test_published_hardware_and_budget_labels` asserts source metadata is displayed and published times are not claimed as local speedups.

- [ ] **Step 2: Run report tests to establish failure.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_report.py -q`. Expected: missing reporting functions.

- [ ] **Step 3: Implement the thin upstream reporting wrapper.**

Use upstream cache-only loading and `EndToEnd`/context registration, then normal processing of all three result variants and `TabArenaContext.compare`. Inherit upstream evaluation defaults, metrics, failure/imputation handling, plots and output filenames. Do not implement a custom HPO simulator or a success-intersection headline leaderboard. Scope comparison using upstream task filters, without replacing the standard dataset suite. Write extra `run_manifest.json`, coverage, effective-parameter and search-cost sidecars without changing upstream table semantics. N=0 must be labeled as no HPO. Keep local peer reports and published-context reports distinguishable by source hardware/budget; include wider published entrants through normal upstream comparison. No fitting during report regeneration.

- [ ] **Step 4: Verify parity and fit-free regeneration.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_report.py -q`. Patch estimator `fit` to raise while regenerating a report from synthetic caches; assert upstream CSVs/plots and all three normal variants are produced. Expected PASS and identical upstream headline values.

- [ ] **Step 5: Commit standard reporting.**

Run `git add benchmarks/tabarena/report.py benchmarks/tests/tabarena/test_report.py` and `git commit -m "bench: use standard TabArena processing and reports"`.

### Task 7: Validate standard smoke and Lite evaluation

**Files:** Create `benchmarks/tests/tabarena/test_integration.py` and `docs/benchmarks/tabarena_local_v1.md`; finish `benchmarks/tabarena/README.md`.

**Consumes:** All previous upstream integration contracts.

**Produces:** Measured standard-protocol smoke/Lite results, upstream-template verification artifacts and explicit replacement acceptance evidence. Legacy system remains available until acceptance, then Task 8 retires it.

- [ ] **Step 1: Write offline and opt-in live integration tests.**

`test_offline_run_resume_standard_report` uses cached synthetic results to assert enforced `8x1`, default/tuned/tuned + ensemble rows, stable resumed fit counts, standard leaderboard parity and artifact names. `test_live_smoke` runs only when `ALLOYGBM_TABARENA_LIVE=1`; resolve the three upstream example datasets and verify all expected jobs, all three problem types, valid prediction shapes and complete terminal outcomes. No network in ordinary test runs. New tests may skip for missing optional dependencies in the old environment but must execute in the dedicated locked environment.

- [ ] **Step 2: Run offline integration and resolve remaining wiring defects.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_integration.py -q`. Expected PASS for offline integration, SKIP for live smoke without its explicit environment switch. Run affected component tests after any fixes.

- [ ] **Step 3: Execute upstream smoke, then standard TabArena-Lite.**

Run `python -m benchmarks.tabarena.run --profile smoke` with output/build receipt arguments in the locked environment. Verify all registered-template checks from Task 3 passed, child fit deadlines are enforced, required warmup succeeds, recorded requested/resolved/actual protocol and child counts agree, and all three result variants process correctly. Run reporting against local peers and published context, then resume/report-only to prove completed jobs are not refitted. Run `--profile lite-hpo`: the first split of every official dataset, default plus 25 sampled configurations per method. Preserve model/resource failures as outcomes; repair unexplained adapter/protocol failures before cutover. Full all-split/200-config evaluation is supported but not a prerequisite to local Lite-based replacement acceptance.

- [ ] **Step 4: Document actual results and cutover readiness.**

Write exact setup/build/run/resume/report commands, the fixed public-default configuration, standard preprocessing/validation/tuning/ensemble behavior, completed Lite task/config counts, failures/imputation, effective resource budgets and versions/source hashes. Include the 192 smoke and 42,432 Lite planned child-fit counts for four methods; ensemble processing reuses cached predictions. Record actual elapsed times and disk use. Include a portable result/coverage excerpt and artifact hashes, with local raw runs ignored. Save an unsubmitted PR-template evidence bundle with exact dependency pin, canonical model files/registration patch, hardware/RAM/resources, Lite results, raw expname paths, exceptions/time-limit hits and an unfilled future author sign-off field. Explain that maintainer review, full benchmark-hardware rerun and author sign-off are still required for leaderboard acceptance. State supported benchmark task types and explicit ranking/temporal/special-mode coverage gaps. Do not infer competitiveness from smoke.

- [ ] **Step 5: Run pre-cutover regression checks and commit acceptance evidence.**

Run `cargo test --workspace`, `.venv/bin/python -m pytest bindings/python/tests/ -q`, `.venv/bin/python -m pytest benchmarks/tests/ -q`, and `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/ -q`. Expected library behavior unchanged, new integration passing and Lite terminal outcomes/reports recorded. Commit `benchmarks/tabarena/README.md`, `benchmarks/tests/tabarena/test_integration.py` and `docs/benchmarks/tabarena_local_v1.md` with message `bench: validate standard TabArena replacement`.

### Task 8: Retire and archive the previous benchmark system

**Files:** Create `docs/archive/benchmarks/pre-tabarena-2026-10-20/{README.md,manifest.json,benchmarks/,scripts/}`, `benchmarks/tests/tabarena/test_migration.py`, and root `pytest.ini` if no active pytest configuration exists. Modify `benchmarks/README.md`, `.github/workflows/ci.yml`, other discovered workflow consumers, `README.md`, `CONTRIBUTING.md`, `docs/user/benchmarks.md`, `docs/site/source/benchmarks.rst`, `docs/benchmarks/README.md`, `requirements-dev.txt`, `CLAUDE.md` and `AGENTS.md`. Move every tracked legacy benchmark file, its benchmark-only tests/fixtures/baselines and `scripts/benchmark_avx2_compare.sh` into the archive. Inventory additional consumers before fixing the complete move list.

**Consumes:** Task 7 acceptance: standard Lite jobs have terminal outcomes, all three variants report successfully, registered upstream smoke/warmup/lint/drift/parallel checks pass, deadlines are honored, no unexplained adapter/protocol failures remain and library correctness suites pass.

**Produces:** One active benchmark system under `benchmarks/tabarena/`, its tests and README; historical system preserved with exact path/hash provenance. No active CI/tests/imports execute the archive.

- [ ] **Step 1: Freeze the archive inventory and write migration assertions.**

Build `manifest.json` from `git ls-files benchmarks scripts .github` plus tracked references: each moved file records `old_path`, `archive_path`, `sha256` and original source commit; include original benchmark README and legacy benchmark test packages. Define the retained active benchmark allowlist as `benchmarks/tabarena/**`, `benchmarks/tests/tabarena/**`, `benchmarks/README.md` and generated ignored results. Inventory tracked imports outside the old harness and any correctness assertions that need a new active home.

Write `test_legacy_files_archived_with_identical_hashes` asserting every moved manifest entry exists at its destination, content hash is unchanged and the old active path is absent. `test_only_tabarena_is_active_benchmark` asserts every tracked benchmark implementation/test file is on the new allowlist. `test_active_consumers_do_not_execute_archive` checks active workflow commands, imports and contribution instructions for retired entry points/archive imports. `test_archive_excluded_from_pytest_collection` invokes collection and asserts no archived node IDs. For each unique correctness invariant found, add a focused test under `bindings/python/tests/` or the relevant Rust crate before archiving its old harness; do not port old timing/quality gates.

- [ ] **Step 2: Run migration tests to establish failure.**

Run `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/test_migration.py -q`. Expected FAIL because the old active system still exists and its archive has not been created.

- [ ] **Step 3: Archive legacy payload after verifying Task 7 acceptance.**

Use the inventory as the exact move list; preserve file contents/relative paths and manifest hashes. Keep historical benchmark evidence docs with an index declaring the old system retired, and repair links through archived destinations. Add archive metadata and reproduction limitations; no compatibility shims or fallback runners remain active. Configure pytest to exclude `docs/archive` from collection. Existing library tests remain active. This archival cutover is explicitly authorized by the user's replacement request; no separate approval is needed when acceptance is satisfied.

- [ ] **Step 4: Switch active CI and documentation to TabArena.**

Remove CI commands/jobs that run old competitiveness/gating/allocation benchmark harnesses and scheduled comparator runs; add a locked-environment job for the new offline TabArena adapter/protocol/report contracts. Keep expensive OpenML evaluation manual/local. Replace current benchmark instructions in README, CONTRIBUTING, user docs and Sphinx mirror; retain old numeric results only as labeled historical evidence. Update both agent guides consistently. Remove now-unused benchmark-only developer dependency declarations after checking active library tests; public package dependencies remain unchanged. No current instructions should tell contributors to add an arm to `run_model_comparison.py`.

- [ ] **Step 5: Verify complete cutover and regression preservation.**

Run migration tests, `cargo test --workspace`, `.venv/bin/python -m pytest bindings/python/tests/ -q`, `.venv-tabarena/bin/python -m pytest benchmarks/tests/tabarena/ -q`, repository pytest collection and `git diff --check`. Validate modified workflow YAML and build the changed Sphinx docs. Search active imports/workflows/guides with `rg` for retired entry points and archive imports; historical text may link the archive but may not execute it. Confirm every archived file hash matches the manifest and all new benchmark outputs/environments are ignored.

- [ ] **Step 6: Commit the archive/cutover as one logical change.**

Stage the manifest-listed moves, new migration tests and updated consumers explicitly; commit with `chore: retire legacy benchmarks in favor of TabArena`. Final evidence states that TabArena is the only active benchmark system and links the preserved archive.

## Self-review and execution handoff

Spec coverage: environment/build (Task 1), required cooperative native fit deadlines (Task 2), default-preserving model integration and upstream capability conventions (Task 3), standard protocol/search/data metadata (Task 4), upstream run/cache handling (Task 5), upstream three-variant reporting (Task 6), real smoke/Lite acceptance (Task 7), and mandatory CI/docs/archive cutover (Task 8). All Review Focus conditions have owning tests. Deadline capability and full registered-template tests are mandatory gates; no custom holdout protocol, replacement scoring engine or parallel legacy benchmark system survives completion.

Recommended execution: **Native**, because the eight tasks share the same upstream integration and the final archive depends on measured replacement acceptance. Use a fresh whole-branch reviewer after implementation. The user has clarified replacement scope but has not yet chosen an execution method; this revision remains a plan, and legacy files have not moved.

## Review follow-up and on-demand Linux workflow

The user authorized fixing the remaining findings and preparing disposable
infrastructure documentation; actual provisioning is outside this execution.
Implement inline, preserving the existing stacked PRs and author-owned review.

- [x] Native deadline correctness: reproduce completed-prefix truncation and
  classifier refit-state corruption; fix in trainer/classifier; verify all four
  estimator paths and full Rust/Python suites; commit to PR #172.
- [x] Adapter fidelity: honor AutoGluon's resolved stopping scorer on raw margins,
  retain cooperative partial fits, leave public defaults unchanged; test against
  upstream metrics, serialization, timeout and warmup contracts.
- [x] Execution/acceptance: default to upstream Ray; gate retirement on Linux
  x86_64, >=16 available/allocated CPUs, nominal 64-GB RAM and no smaller thread
  caps. Reject diagnostic resources without preventing local smoke work.
- [x] Matched reporting: add default-only RF, use standard calibration/fallback,
  refuse uncalibrated incomplete rankings, add only AlloyGBM to published
  comparisons, scope to planned tasks, and record actual candidate budgets.
- [x] Disposable workflow: add `cloud.py` preflight/export/verify and Ubuntu
  bootstrap/bounded-session scripts. Test archive recovery/corruption and shell
  exit/export behavior; document console provisioning, exact-commit setup,
  same-host resume, verified download and manual destruction. Never destroy a
  droplet before off-server evidence verification. Add portable Ubuntu CI tests.
- [x] Ruling: missing-class folds stay explicit failures; no fabricated training
  samples. Keep public-default rounds. Defer a static memory estimate until Linux
  measurements can support it; inspect this during the pilot.
- [ ] Next operational gate, when a droplet is requested: verify bootstrap on a
  fresh selected Linux host, run the full three-task Ray pilot, inspect CPU/RAM,
  choose a measured budget, then run substantive Lite. Full Lite remains required
  before merging retirement; local smoke does not satisfy it.

File ownership: `resources.py` defines the project's acceptance resource policy;
`protocol.py` adds only the RF default; `report.py` delegates imputation and
scoring to upstream; `cloud.py` handles provider-independent evidence lifecycle;
`remote/*.sh` bootstrap/run on an existing host; `DROPLET.md` describes provider
creation/destruction. No cloud tokens, API clients or public leaderboard uploads.
