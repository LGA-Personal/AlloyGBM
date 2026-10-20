# TabArena submission-protocol audit

Date: 2026-10-20. Scope: source/documentation audit of the replacement design, not a completed benchmark or certification of an implemented wrapper.

## Conclusion

The source audit confirmed the planned validation, Lite scope, tuning, ensembling and evaluation approach, but found two material omissions: recording budget overruns is insufficient, and a quickstart-compatible custom wrapper is not the complete submission integration. Both are now explicit implementation gates. The plan also requires current declarative AutoGluon configuration, canonical metadata, exact dependency/config-default identity and the actual registered upstream model tests.

The initial audit found missing wrapper integration and native deadline control. Those features and the actual registered tests are now implemented; see the execution addendum. Full Lite acceptance and official submission review remain outstanding, so **the implementation is not yet certified submission-ready**.

## Sources inspected

Actual local git clone: `/tmp/alloygbm-tabarena-inspect`, commit `041f9b3774d1191ffcdcf137cbe4e1e08159b972`. Recheck against the selected upstream revision before a future submission; this audit does not promise acceptance under future rules.

- [Submission PR template](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/.github/pull_request_template.md): required files, wrapper contract, pinned dependency, tests, Lite evidence and maintainer-run process.
- [Add-model guide](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/.claude/skills/add-model/SKILL.md) and [annotated patterns](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/.claude/skills/add-model/references/model_patterns.md): model versus system, validation, remaining time budget/headroom, declarative class attributes, resources, seeds, optional imports, metadata/config naming and warmup. Some generic snippets predate the explicit contract; the explicit contract and current tests take precedence.
- [Validation protocol](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/src/tabarena/benchmark/validation_protocol.py), v0.1 context and experiment bundle: enforced `8x1`, outer task splits, preprocessing and configuration counts.
- [Standard simulator](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/packages/tabarena/src/tabarena/simulation/repo_simulator.py), method processor and context comparison: default/tuned/tuned + ensemble output and unchanged leaderboard semantics.
- [Registered smoke test](https://github.com/autogluon/tabarena/blob/041f9b3774d1191ffcdcf137cbe4e1e08159b972/tests/tabarena/models/test_all_models.py), registry discovery, warmup coverage/audit and dependency-extra checker: actual model acceptance path.
- Existing CPU wrappers, particularly ChimeraBoost's categorical/resource/deadline callback handling. Existing wrappers illustrate implementation; an old API or acknowledged oversight is not a protocol exemption.

Local AlloyGBM revision inspected: `d1d706dd06d333cda2798258938fb4830e143dd3`. Read regression/classification fit signatures, Python/native training calls, metric callbacks, iteration controls and single/multiclass trainer loops. These establish the missing deadline API; they do not prove behavior of unimplemented changes.

## Requirements and disposition

| Requirement | Audited requirement and implementation gate |
|---|---|
| Model category | One registered model; arena owns splits, HPO and ensemble selection. Library-internal selection must use supplied validation data or be disabled. |
| Splits/bagging | Official dataset collection unchanged, inner `8 folds × 1 set`, enforcement enabled. Record requested, resolved and actual protocol/child counts. Exposure annotations cannot filter the suite. |
| Lite/full | Initial substantive evaluation: first split of every dataset, default plus approximately 25 sampled configs. Full: all official splits/full search space, pinned bundle default 200 sampled configs. Example-dataset smoke is not competitiveness evidence. |
| Processing | Upstream default/tuned/tuned + ensemble; current selection iteration counts 1/40. Preserve metrics, normalization, aggregation, calibration and imputation; retain measured/imputed flags. |
| Submission integration | Model folder `{__init__,model,hpo,info}.py`, lazy class export, dependency extra/extended entry and tree-family classification. Local runs use these exact canonical bytes in a pinned clone. No duplicate local wrapper or modifications to protocol/shared tests. |
| Class conventions | `TA-ALLOYGBM`, `TA-AlloyGBM`, method `AlloyGBM`, key `ALLOYGBM`; `_supported_problem_types` and `_default_auxiliary_params_extra`. Lazy optional imports; no mutation of resolved params. |
| Default identity | Validate processed `_c1`, `TA-AlloyGBM_c1_default_BAG_L1`, against actual output and the intended frozen configuration. Do not confuse raw class/config names with processed IDs. |
| Time budget | Implemented in PR #172. Verify cooperative fit-time deadlines for regression/binary/multiclass; subtract preprocessing, leave 5% headroom, preserve complete rounds and reject expired/over-budget outcomes through upstream timeout handling. Logging alone does not comply. |
| Resources/seed | Forward actual child CPU allocation and framework seed. No all-core override, hidden GPU budget, extra internal holdout or global monkeypatch. |
| Warmup/parallelism | Upstream data-independent warmup must pass. Verify actual Ray worker import/pickling, fit and save/load prediction equivalence; native/debug-only success is insufficient. |
| Dependencies/provenance | Exact installable source commit/version, matching `pip_extra`, measured extension/build and dependency identity. A locally committed but remotely unavailable revision is not yet a reproducible submission dependency. |
| Upstream tests | Actual registered smoke, warmup audit/coverage, extra drift, lint/format and model-specific tests must run/pass. No skipped AlloyGBM case, zero-item selection or shared-test changes to obtain a pass. |
| Evidence | Entry-point command, exact source/search space, hardware/RAM/resources, Lite results, raw expname caches, failures/time-limit hits and future author sign-off. Keep local storage and `verified=False`. |

## Public defaults and search space

The framework permits method-author default configurations, while its booster guidance recommends a high iteration cap with validation early stopping. The user chose current AlloyGBM public defaults for the fixed arm. Declaring those as `_c1` is a deliberate author choice, not proof that the public automatic-round configuration is the recommended or best-performing booster setup. Disclose this in later maintainer review. Sampled configurations use the high-cap/stopping approach. Never silently tune the fixed arm or add another internal holdout.

Search-space distributions are model-author choices; TabArena supplies no universal AlloyGBM space. Freeze them before test evaluation, support the full sampling budget, and share the exact space between local and prospective submission runs. Standard protocol execution does not independently establish that the space is optimal.

## Completion and leaderboard limits

Implementation acceptance requires deadline tests, unchanged unrestricted outputs, the canonical registered integration, passing upstream checks without skipped AlloyGBM cases, official-protocol Lite results/all three processed variants, exact reproducible dependency/build identity, and complete failure/imputation records. Only then may legacy benchmarks be archived.

Local acceptance does not create an official entry. The PR template specifies maintainer review, a full rerun on benchmark hardware, comparison against Lite, author confirmation, hosted artifacts/merge and leaderboard regeneration. Prepare the evidence locally; no PR, upload or official-verification claim is authorized now.

## Execution addendum (supersedes the initial implementation-status paragraph)

Deadline support and the self-contained registered adapter/search/metadata/extra/family integration have now been implemented. The actual registered upstream model smoke and isolated warmup coverage cases pass without modifying shared tests. The adapter passes regression, binary and multiclass save/load checks, sampled warmup, protocol, provenance and selection tests. Real official-split caches have been processed through upstream `EndToEnd` and `TabArenaContext.compare` into all three variants.

Those results are partial development evidence, not a completed Lite benchmark or official submission approval. Local native/debug execution and fresh outer-job processes address a reproduced macOS OpenMP warmup crash; official Ray/cluster evidence and the complete required scope must be evaluated separately. The local four-method report explicitly disables absent RF calibration/fallback; published-context output uses upstream defaults. Full Lite evidence and subsequent author/maintainer review remain required. See the runner README and execution-evidence document for final tested scope.

The global upstream extra checker currently reports the same unrelated drift on the untouched pinned revision. AlloyGBM's row is `OK`, and its exact pin/extra/family have direct tests; do not report the whole checker as passing.
