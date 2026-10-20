# TabArena benchmark replacement design

> Review follow-up supersedes the initial execution/report choices below: upstream Ray is the default, acceptance uses the chosen Linux 16-vCPU / nominal 64-GB policy, default RF supplies matched local calibration/fallback, and published context adds only AlloyGBM over the planned task scope. Cooperative partial models are retained. See [resolutions](../reviews/2026-10-20-tabarena-resolutions.md) and the [droplet runbook](../../benchmarks/tabarena/DROPLET.md). Full Lite HPO now includes RF: 5,355 outer jobs / 42,840 children.

Date: 2026-10-20. Status: implementation in progress; see the operational addendum below for executed evidence.

## Purpose and scope

Answer two development questions: how competitive is AlloyGBM with its current public defaults, and how competitive is it when TabArena selects its hyperparameters? Start with CPU regression, binary classification and multiclass classification. Rerun LightGBM, XGBoost and CatBoost locally for the primary comparison; use TabArena's downloaded results for broader predictive-quality context.

TabArena becomes the sole active benchmark system. Use its standard model protocol, result processing and leaderboard outputs, with local execution and a thin AlloyGBM integration. Once the replacement passes the migration acceptance checks, deprecate and archive the entire previous benchmark system inside the repository, including bespoke runners, fixtures, benchmark-only gates/tests and their CI entry points. Preserve library unit/correctness tests; they are not the benchmark system. BeyondArena remains follow-up work; standard tuned + ensemble results are included now. Public submission, publication and changes to library defaults remain outside this work.

User decisions: the fixed arm uses **current public defaults**; use **standard TabArena benchmarking**; **replace and retire the old benchmark system after validation**. A separate frozen-preset facility is unnecessary for this replacement.

## What the investigation established

Read `docs/alloygbm_handoff.md` (the supplied path omitted `.md`), both agent guides, the competitiveness harness, estimator implementations and review/resolution documents. Treat `CLAUDE.md` and `AGENTS.md` as the same project guidance; differing historical text is not evidence that the code has a feature.

Initial inspection found stale local `main` at `d1d706dd`. Execution fetched current `origin/main` at `4e5af80787610714f52415a09d5182fd1373ca98`, which contains the handoff's public automatic-round behavior. Implementation uses that current baseline in an isolated worktree. The Python 3.12 benchmark environment is built from the selected source with a hashed native-extension receipt. Current public defaults are automatic rounds, not the stale checkout's fixed 100 rounds.

Current benchmark entry points include `benchmarks/run_model_comparison.py` and `benchmarks/competitiveness/run.py`. They are migration inputs and historical evidence, and will leave the active benchmark tree at cutover. The replacement must not copy their parameter-matching rules, quality gates or custom ranking formulae into TabArena. The September competitiveness review's source/build provenance findings remain relevant.

Upstream TabArena was inspected at **`041f9b3774d1191ffcdcf137cbe4e1e08159b972`**. Its runnable custom-model example, bundle, context, configuration generator, simulator and validation code establish the integration below. These are pre-release APIs: pin the revision rather than depending on moving `main` or copying older examples from an index.

## Method and result identities

| Result | Meaning | Training and selection |
|---|---|---|
| `AlloyGBM (public defaults)` | Current package behavior under the stated wrapper preprocessing | Empty model hyperparameters, except injected seed and CPU budget; no wrapper-added stopping or parameter retuning |
| `AlloyGBM (TabArena tuned, N candidates)` | Best single configuration according to inner validation | Default configuration plus N sampled configurations; TabArena performs selection using validation predictions |
| `AlloyGBM (tuned + ensemble)` | Standard TabArena ensemble over tuned configurations | Upstream validation-based ensemble selection and upstream default iteration count |

Use an importable `AlloyGBMModel(AbstractModel)` adapter with `ag_key="TA-ALLOYGBM"`, `ag_name="TA-AlloyGBM"`, `ag_priority=65` and `seed_name="seed"`. For submission parity, prepare and test the full in-tree integration in a local pinned TabArena clone: `models/alloygbm/{__init__,model,hpo,info}.py`, lazy class export, exact-pinned dependency extra/extended registration and tree-model website family. Use class attributes `_supported_problem_types` and `_default_auxiliary_params_extra`, lazy library imports and no mutation of resolved framework parameter dictionaries. Do not copy obsolete helper-method overrides from older examples.

Store the self-contained model payload in `benchmarks/tabarena/submission/alloygbm/` and the small registration patch alongside it. A setup helper installs these exact bytes in the pinned clone; local runs import the registered class from `tabarena.models.alloygbm`, never a second independent wrapper. The submission payload cannot import AlloyGBM's benchmark package. Only the new model files and required registration/extra/family edits may differ from pinned upstream; validation, scoring, runner and shared test infrastructure remain untouched. A direct custom-generator quickstart remains useful as a parity check, but is not the final acceptance path.

Use **one protocol: standard TabArena-v0.1**, with official outer splits and inner **8 folds × 1 set**, inherited from `TabArenaContext` with enforcement enabled. There is no custom holdout lane, custom splitter or custom tuning/evaluation engine. Use upstream processing to produce its three normal variants: default, tuned and tuned + ensemble. Upstream currently uses one selection iteration for tuned and 40 for tuned + ensemble; inherit those values rather than replacing the processing pipeline.

The requested public-default AlloyGBM configuration is the declared method-author default `_c1`; the framework permits model-specific defaults. The implementation guide recommends high iteration caps with early stopping for boosters. Our fixed public-default arm deliberately retains the user's requested behavior, and this choice must be disclosed in the submission evidence rather than claimed to be an upstream-mandated booster default. Sampled candidates follow the high-cap/early-stopping convention. Do not change the fixed configuration without changing its declared identity and rerunning it. Peer defaults remain their upstream TabArena wrapper defaults. Bagging applies even to the public-default result.

## Adapter contract

- Select `GBMRegressor` for regression and `GBMClassifier` for binary/multiclass; let AutoGluon encode target labels and supply problem type/number of classes.
- Use the bundle's `default` preprocessing pipeline. After `super()._preprocess`, preserve pandas category columns so AlloyGBM's existing categorical inference/encoding operates. Convert numeric columns to float32 to avoid accidentally invoking the integer/pre-binned input path. Preserve numeric NaNs. Do not zero-fill missing values or target-encode outside the estimator.
- Fit all model-specific transformations on the child training partition only. Never concatenate validation/test rows or labels into training. Preserve feature order/category mappings on prediction; test unseen categories, missing categories and serialization explicitly.
- Forward `num_cpus` to `n_jobs`; accept no GPU allocation. Inject the framework seed through `seed_name`, rather than hardcoding it in defaults. Remove framework-only keys before constructing AlloyGBM. Do not silently ignore unknown estimator parameters.
- Forward `X_val,y_val` as `eval_set=(X_val,y_val)` only when present. For sampled candidates with stopping enabled, absence of validation is an error. Automatic public defaults also require supplied validation; a hidden internal holdout is forbidden.
- Adapt predictions to the AutoGluon convention: regression `(n,)`, binary positive-class probability `(n,)`, multiclass `(n,K)` in global encoded class order. Check finiteness/range/row sums. A child missing an expected class must either be supported correctly by the library or fail explicitly; never pretend a smaller probability matrix matches K classes.
- Preserve fitted estimator state through AutoGluon save/load. Record actual completed rounds, `best_iteration_`, `stop_reason_` and resolved training policy when available. The current best-round field counts rounds, including the possible zero-round initial predictor; do not unconditionally add one. Follow upstream wrapper conventions for trained parameters, refit capability, warmup and resource handling; do not globally disable framework behavior or claim support the adapter lacks.
- Honor the per-child remaining `time_limit` received by `_fit`, which is not simply the bundle's 3600-second configuration budget. Subtract preprocessing time and leave 5% headroom. The inspected AlloyGBM cannot currently stop on a deadline: this is a blocking integration gap, not an accepted logging-only limitation. Add a fit-time `time_limit` argument defaulting to `None` to regression/classification fit and a native cooperative deadline before standard benchmark acceptance. Check it at iteration/tree safe points, preserve only complete rounds (complete K-class rounds for multiclass), and map exhausted/no-valid-model outcomes to AutoGluon's `TimeLimitExceeded`. Reject a result that finishes beyond the original child budget instead of reporting it as a successful fit. Document safe-point granularity rather than promising preemption within a histogram kernel. No deadline/clock checks may change unrestricted fits or artifact compatibility.

## Search space v1

The default configuration `_c1` is `{}` and stays unchanged. Generate N sampled configurations deterministically with ConfigSpace seed **1234**, and save the fully materialized ordered list before fitting. Each sampled configuration fixes `training_policy="manual"`, `n_estimators=10000`, `early_stopping_rounds=50`, `tree_growth="level"`, `leaf_model="constant"`, `boosting_mode="standard"` and `training_mode="manual"`. Manual policy ensures explicit sampled values are actually used. Preserve library defaults for parameters outside this space.

| Parameter | Distribution |
|---|---|
| `learning_rate` | log-uniform `[0.01, 0.2]` |
| `max_depth` | integer uniform `[3, 10]` |
| `min_data_in_leaf` | integer log-uniform `[1, 128]` |
| `row_subsample` | uniform `[0.6, 1.0]` |
| `col_subsample` | uniform `[0.6, 1.0]` |
| `lambda_l1` | categorical `[0.0, 0.01, 0.1, 1.0, 10.0]` |
| `lambda_l2` | categorical `[0.0, 0.01, 0.1, 1.0, 10.0, 100.0]` |
| `continuous_binning_max_bins` | categorical `[64, 128, 256]` |

Do not also search iteration caps, DART, GOSS, PL leaves, DRO, native-categorical thresholds or undocumented experiment flags in v1. The 10,000-round cap is headroom for early stopping, not a matched-capacity claim. Record cap hits. The initial space is a preregistered engineering choice, not a claim of an optimal AlloyGBM search space. Improvements get a new search-space hash/version and new runs.

Use upstream peer generators, each with the same N sampled configurations plus its default. Materialize and hash peer configurations too. Count N as **additional sampled configurations**; total search size is **N+1**. Search configuration seed 1234 is distinct from child seeds controlled by TabArena's `fold-config-wise` policy.

## Data discipline

Use TabArena's task IDs, dataset IDs/versions, metrics and split indices exactly. Store the complete resolved task manifest before training, including problem type, row/feature counts, categorical metadata and hashes of split row identifiers. Model selection consumes only validation predictions. Outer test scores are for reporting.

The handoff reserves OpenML-CC18 (99) and OpenML-CTR23 (353) for confirmation. Obtain and snapshot their membership metadata; audit TabArena overlap by dataset ID, normalized name/known aliases and shape (a shape match flags review, not proof of identity). Annotate each selected task as protected, previously used for development, or unclassified, without removing tasks from the standard TabArena suite. The absent dev manifest cannot be assumed to exist: inventory actual checked-in historical evidence and mark unknown exposure honestly. A temporary inability to fetch overlap metadata must not produce a false “clean development set” label.

Per-task HPO on the inner train/validation portion is legitimate. Changing defaults or the search space after inspecting outer test results makes those datasets development evidence. Log that exposure; do not later call the same tasks pristine held-out confirmation. Initial work measures frozen AlloyGBM behavior and does not promote parameter changes.

## Local environment and provenance

Use a dedicated **Python 3.12** environment, `.venv-tabarena/`; leave the existing `.venv` and public runtime dependencies alone. TabArena supports Python 3.11–3.13 at the inspected revision. Use uv, pre-release resolution and the upstream recommended benchmark installation (`tabarena[benchmark]`, plus plotting and the selected CPU peer dependencies). Lock TabArena and bencheval to the same inspected revision. Dependency installation does not imply running every model; the first locally fitted entrants are AlloyGBM, LightGBM, XGBoost and CatBoost, while downloaded results provide the wider field.

Commit a resolved lock generated and smoke-tested on this macOS arm64 host. The setup guide must show `uv venv --python 3.12 .venv-tabarena`, locked installation, then a release `maturin develop --manifest-path bindings/python/Cargo.toml --release` into that environment. Confirm upstream extra names during resolution rather than silently dropping a peer when resolution fails.

Record source commit and dirty diff hash, Rust/toolchain/build flags, imported package paths, compiled extension SHA256, the clean release-build receipt, all Python distribution versions, lock hash, platform/CPU/RAM, CPU allocation, thread variables, task manifest, protocol, preprocessing, search-space/config list hashes and command line. Package version `1.0.0` alone cannot identify unreleased code. A stale extension must fail preflight. Reject active `ALLOYGBM_EXPERIMENT_*` or profiling variables so an experiment cannot masquerade as defaults.

Use upstream runner, backend, fold-fitting, warmup, preprocessing, time-limit and evaluation defaults. Allocate **zero GPUs** to the four CPU entrants. Expose a matched local CPU budget through upstream resource arguments; when unspecified use upstream resource detection rather than hardcoding a one-core benchmark. Record resolved CPU allocations/thread limits and any explicit operational overrides. Local execution and reduced task/config counts for smoke/Lite do not redefine the validation protocol. No cloud jobs or services are required.

Use `benchmarks/tabarena/` as the sole active benchmark implementation, with ignored paths `benchmarks/results/tabarena/` for runs and `.cache/tabarena/` for datasets/artifacts. Set caches through upstream configuration. Start downloaded comparisons from the small results tier; fetching baseline prediction repositories is unnecessary for the initial published-context comparison.

## Run stages and resumability

| Profile | Outer tasks | Extra random configs/library | Approximate arena child fits, four libraries |
|---|---|---:|---:|
| `smoke` | First split of `blood-transfusion-service-center`, `QSAR_fish_toxicity`, `anneal` | 1 | 192 |
| `lite-defaults` | First split of all 51 datasets at the pinned revision | 0 | 1,632 |
| `lite-hpo` | Same Lite tasks | 25 | 42,432 |
| `full` | All official outer splits | Upstream bundle default: 200 | Derived from resolved metadata; explicit profile only |

These are planned child-fit counts, not time estimates, and exclude failures and framework overhead. Ensemble selection reuses cached predictions rather than multiplying trained configuration counts. Smoke uses upstream's example and is an adapter check, not competitiveness evidence. `lite-hpo` follows upstream's recommended initial evaluation: first split of every dataset, default plus about 25 random configurations. `full` inherits the v0.1 bundle's 200 sampled-config default and all outer splits; Lite is not labeled full. Reduced smoke/Lite scopes remain explicitly labeled subsets of the standard protocol. Default CLI profile is `smoke` for setup verification; the first substantive benchmark is `lite-hpo`.

Complete smoke, then the standard `lite-hpo` evaluation. An optional `lite-defaults` run can produce fixed-configuration results sooner. Estimate cost using actual smoke/task metadata and record uncertainty; no bespoke pilot protocol is needed. Validate replacement operation on all Lite tasks before archiving the legacy system. Full benchmark runs can follow on the user's local hardware without implying official-hardware timing equivalence.

## Submission parity gates

Follow the actual upstream PR template, not just the runnable quickstart. Set `MethodMetadata.config` and `ModelInfo` for method `AlloyGBM`, `model_key="ALLOYGBM"`, CPU/tree-based classification, MIT license, supported problem types and an exact installable AlloyGBM source commit/version. The initial inspected source is `d1d706dd06d333cda2798258938fb4830e143dd3`; any necessary deadline-support commit supersedes that pin and all receipts/metadata must record the new measured revision. A plain `alloygbm==1.0.0` must not stand in for unreleased source. Keep `verified=False` and storage local until maintainer verification/hosting. The dependency extra and `pip_extra` must match.

Derive the declared processed `config_default` from actual processed output, not the raw class name: the inspected v0.1 bundle adds the default-preprocessing suffix, normally yielding `AlloyGBM_c1_default_BAG_L1`. Validate that it resolves to the exact intended frozen configuration. Record all requested/resolved/actual validation-protocol fields and child counts; a declared `8x1` flag alone is insufficient. Test that no library-side internal HPO/validation split/ensemble selection occurs for any default or sampled configuration; future automatic-round implementations must use the supplied validation data or be disabled for a model entrant.

Before substantive runs, execute the registered upstream checks in the prepared clone: `pytest -m models -k AlloyGBM`, warmup coverage and `python -P -m tabarena.tools.audit_warmup --model AlloyGBM`, dependency-extra drift checks, registry/config-default/family assertions, Ruff lint/format and model-specific serialization tests. Selected upstream tests must run and pass, not disappear through a missing optional dependency or zero-item selection. Verify real parallel bagging can import, pickle, save/load and predict with this exact class. Do not make shared test-infrastructure changes to obtain a pass.

Keep a local submission evidence bundle matching the PR template: model files/registration patch, exact dependency pin, frozen search space, official-protocol Lite results (default plus approximately 25 random configurations), hardware/RAM/resources, exact entry-point command, raw `expname` caches, failures/time-limit hits and a future author sign-off field. No PR, upload or author sign-off is performed now. Official leaderboard placement still requires maintainer review, a full run on benchmark hardware, comparison with Lite results and author sign-off; a local result does not establish that acceptance.

Compute an experiment fingerprint from protocol, tasks, configs, adapter source, dependency lock, build identity and resources. Within a matching fingerprint, reuse upstream per-job result caching. A changed build, profile scope, config order, space or preprocessing cannot resume the previous run. On interruption, completed jobs survive; retry failed jobs only on explicit `--retry-failed`. Record expected, successful, failed, skipped and pending jobs with exception/reason. Do not count failure, timeout/overrun, unsupported input or a missing dependency as a successful low-cost result. Preserve raw predictions/results and parameters locally for later reanalysis without refitting. Keep a slim successful-job ledger so reporting can be rerun independently of training.

## Reports and acceptance criteria

Every run writes `run_manifest.json`, `configs.json`, `tasks.json`, `jobs.jsonl` and upstream raw caches. Reporting preserves upstream output names such as `results_per_split.csv`, `tabarena_leaderboard.csv`, `method_info.csv`, `winrate_matrix.csv`, Pareto outputs and figures, supplemented by `coverage.csv` and `summary.md`. Include coverage status even for a method with zero successful fits.

**Primary reports:** use upstream result processing and `TabArenaContext.compare` for default, tuned and tuned + ensemble across all four libraries. Preserve upstream metrics, normalization, ranking, calibration, tie handling, aggregation and imputation defaults; do not replace them with the old benchmark's ratios, 0.5% tie rule, bootstrap scheme or custom success-only leaderboard. Export the normal upstream CSVs/plots, selected configurations/ensemble weights and adapter diagnostics. N=0 has no actual tuning and must be labeled accordingly. Keep requested budgets, actual search cost, selected configuration fit time and local wall time distinguishable, without overwriting upstream metric meanings.

**Coverage audit:** retain the expected task/config grid, errors, successful fits and upstream `imputed` flags beside the standard report. Explain any missing data and the upstream fallback used; never relabel an imputed score as measured. An explicitly labeled non-imputed diagnostic may supplement the standard report but cannot replace it or change its leaderboard semantics. Cutover requires every Lite job to have a recorded terminal outcome, all three result variants to process successfully and no unexplained adapter/protocol failures. Genuine model/resource failures are visible outcomes, not a reason to quietly remove datasets.

**Published context:** use the upstream comparison flow against pinned downloaded results on matching task keys, including standard default/tuned/ensemble variants where available. Store source suite, tuning budgets, versions and artifact hashes. Preserve upstream outputs while labeling published hardware and candidate-budget differences. Present matched-hardware local comparisons separately; do not claim a cross-hardware speedup. Subset Elo/ranks are results for that upstream evaluation pool, not official leaderboard placement.

Acceptance: build the selected local source, complete smoke and standard Lite evaluation, resume without refitting completed jobs, regenerate upstream reports without fitting, and trace every score to its task/config/build. Tests prove public-default fits preserve modeling parameters and both tuning/ensembling use validation predictions. Baseline Rust/Python correctness tests pass before and after; offline integration tests and live evidence cover the new benchmark workflow.

## Deprecation and archive cutover

Inventory all tracked files under the existing `benchmarks/`, benchmark-specific scripts, CI jobs, documentation links and imports before moving anything. Record exact source/destination paths and hashes in `docs/archive/benchmarks/pre-tabarena-2026-10-20/manifest.json`. After replacement acceptance, move all legacy benchmark payload into that archive, preserving relative paths and a historical README with the source commit, retirement date and replacement commands. Move legacy benchmark-only tests out of active collection. Preserve historical evidence docs under `docs/benchmarks/` with an archive index; repair historical links or point them through the manifest. Do not promise archived scripts remain executable.

Retire old benchmark jobs in `.github/workflows/ci.yml` and any other consumers; add the new offline adapter/protocol/report tests to CI. Keep expensive OpenML runs manual/local. Replace `benchmarks/README.md`, `docs/user/benchmarks.md`, its Sphinx mirror, and current README/CONTRIBUTING guidance with the TabArena workflow. Update agent-guide benchmark paths as part of the same cutover, keeping CLAUDE.md/AGENTS.md semantically aligned. No active command, CI job or test may import the archive or execute the retired benchmark framework. Keep Rust and Python library correctness tests active; if a legacy benchmark test contains a unique library correctness invariant, port that focused assertion into the library tests before archiving its harness. TabArena does not newly validate ranking, temporal or special modes merely because their old harnesses have been retired; document those benchmark coverage gaps explicitly.

Final active layout: `benchmarks/tabarena/`, its integration tests and the benchmark README; generated outputs remain ignored. Archive creation and active-tree retirement are required deliverables, not optional follow-up work. During planning, only these design/plan documents change; no legacy files move yet.

## Upstream references (pinned when APIs matter)

- [README: installation, model versus system, artifact tiers](https://github.com/autogluon/tabarena/tree/041f9b3774d1191ffcdcf137cbe4e1e08159b972)
- [Custom-model runnable quickstart](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/examples/benchmarking/run_quickstart_tabarena_model.py)
- [Standard experiment bundle and configuration-count defaults](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/src/tabarena/benchmark/experiment/bundle.py)
- [Validation protocol](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/src/tabarena/benchmark/validation_protocol.py)
- [Context: task filters, running, cache registration, comparison and HPO](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/src/tabarena/contexts/abstract_arena_context.py)
- [Configuration generators](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/src/tabarena/utils/config_utils.py)
- [HPO simulator: tuned uses one selection iteration](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/src/tabarena/simulation/repo_simulator.py)
- [Minimal dependencies and optional extras](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/pyproject.toml)
- [Exporting per-split results](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/examples/plots/run_export_results_per_split.py)

## Execution rulings (2026-10-20)

The canonical implementation and commands are in `benchmarks/tabarena/README.md`. The earlier proposal's assumptions are superseded by these measured findings:

- Native deadline support is implemented in PR #172; Rust workspace, the full bindings suite and its GitHub CI passed. Automatic rounds use the supplied fold validation, never an extra holdout.
- Fresh-process isolation is required for the measured macOS AlloyGBM/LightGBM warmup interaction. Local execution uses the documented upstream native/debug quickstart (sequential local folds). The explicit upstream/Ray mode retains upstream defaults. Execution mode is part of identity and report metadata.
- A four-entrant local pool lacks RF calibration/fallback results. Its comparison explicitly disables those two RF-dependent options, retaining upstream processing/ranking otherwise. Published-context comparisons use upstream defaults and are labeled mixed hardware. This is a disclosed local evaluation pool, not official Elo.
- Raw registered model names include `TA-`: the measured default is `TA-AlloyGBM_c1_default_BAG_L1`. Per-method registered metadata is supplied to the processor so the family key remains `ALLOYGBM` and each peer's actual `_c1_default` is its declared default.
- Upstream's report worker-count default calls Linux-only `os.sched_getaffinity`; passing an explicit report worker count avoids that macOS error without patching shared code.
- Upstream's global dependency-extra checker fails on unrelated entries in an untouched clone as well as the registered clone. AlloyGBM's extra/pin/extended/family registration is checked directly. This does not constitute a passing global checker.
- Retirement is a separate PR, prepared for review but gated from merge until the actual full Lite acceptance audit passes. This preserves the original acceptance requirement while making all migration changes reviewable.

Final acceptance ruling: require `report --published-context` and bind its source-artifact manifest to the terminal ledger and processor revision. This preserves unchanged upstream calibration/fallback in acceptance evidence; the no-RF local entrant pool is an explicitly labeled diagnostic, never a substitute for that standard comparison. Cached tuned/ensemble weights are exported by replaying upstream selection and checking score agreement.
