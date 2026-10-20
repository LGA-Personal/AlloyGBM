# TabArena replacement execution evidence

Date: 2026-10-20. This is implementation evidence, not an official leaderboard result.

- Baseline: `origin/main` at `4e5af80787610714f52415a09d5182fd1373ca98`.
- Native deadline source: `ecbafa1fb06130d5fe8e481e87d6aaa9fbb79a09` (PR #172), exact installable dependency in the submission metadata.
- TabArena and bencheval: `041f9b3774d1191ffcdcf137cbe4e1e08159b972`, with only the registered AlloyGBM payload and three allowed registration changes. No validation/scoring/shared-test edits.
- Machine: macOS arm64, Apple M4, 10 physical/logical cores, 24 GiB RAM; Python 3.12.10. Exact dependency versions, native hash, source hash, CPU/thread allocation and execution are in each run manifest. Smoke timings are workflow checks and are not comparative claims.

## Completed checks

- Native deadline change: full Rust workspace **909 passed / 3 ignored**, full bindings suite **1,254 passed / 64 subtests**, clippy all targets with warnings denied. All PR #172 GitHub CI checks passed.
- Registered adapter (31 local integration tests): regression/binary/multiclass fit, NaNs/categories (including unseen and missing values), prediction shape/finite checks, save/load equivalence, CPU forwarding, rejection of hidden validation, native timeout translation, and fresh-interpreter registered-class serialization. Complete worker stdout/stderr is retained per job, including an actual malformed-job traceback test.
- Protocol: official enforced 8×1, three example tasks / 24 smoke jobs, 51 Lite tasks / 204 default jobs / 5,304 HPO jobs, immutable default identity and deterministic search-prefix stability.
- Execution/report contracts: changed-identity rejection, torn-write recovery, exclusive driver ownership, failure/pending coverage, checksum/conformance verification, full-scope acceptance guard, and adversarial upstream selection test (changing test targets does not change validation-selected weights).
- Actual upstream registered tests: `test_all_models.py` and `test_warmup_coverage.py`, `-m models -k AlloyGBM`: **2 passed / 102 deselected**. No AlloyGBM case skipped.
- Ruff check/format pass on the integration and its tests. Universal Python 3.12 dependency lock resolves; install dry-run differs only in the three deliberately editable source packages.
- Real caches from an exploratory official-split run were processed with upstream `EndToEnd` and `TabArenaContext.compare`. All three variants and standard CSV/plot outputs were generated. This partial evidence does not satisfy Lite acceptance.

- Real upstream/Ray execution: both AlloyGBM default and sampled configurations completed on official `blood-transfusion-service-center` task 363621, first outer split, with eight fitted children each. Run identity `c0055a5d5437fc4bd42103b0c6d5bede799fb2de8fe31cf9542d9d7178a1f69c`. Upstream Ray processing regenerated all three variants and standard local-pool reports. This is one-task worker evidence, not a full Ray benchmark.
- The isolated TabArena GitHub CI job passed a clean pinned environment install, release build/attestation, integration contracts and registered upstream tests. The complete first integration PR CI matrix passed.

- Current runner regression execution: public default and sampled AlloyGBM completed the official `QSAR_fish_toxicity` first split with eight children each (`18a702bd83340d48`). Current-code resume skipped both completed jobs with an unchanged ledger; per-job logs persisted. Standard cached report regeneration passed.
- Published-context comparison passed on the matching binary task with upstream default RF calibration/fallback. Its 60 source-method metadata records and 110 downloaded result-artifact hashes are recorded in `report-published/sources.json`; timings are explicitly mixed hardware and budgets. This one-task comparison is a workflow check, not a competitive ranking claim.

## Measured issues and scope limits

The global upstream dependency-extra checker reports 21 drifting entries even on the untouched pinned revision. Registered AlloyGBM's row is `OK`; the only union-row difference is its required addition to `extended`. Direct tests check the extra, exact pin, registry and tree-family entry. The global checker is not passing.

macOS reproduces a native crash when AlloyGBM warmup precedes LightGBM warmup in the same interpreter. Isolating each standard outer job into a fresh process permits the peer fits to execute. Local mode retains the upstream quickstart native/debug runner and eight sequential child folds. Explicit Ray mode has the one-task AlloyGBM evidence above; a complete Ray/cluster run remains separate. No OpenMP duplicate-library bypass or global runtime monkeypatch is used.

On macOS, upstream warns that a one-CPU allocation has a ten-CPU affinity mask. This platform lacks Linux cpuset enforcement; the explicit one-CPU child budgets and numerical thread variables are recorded. These timings are local development evidence, not an official hardware-normalized result. Upstream reporting also needs an explicit worker count to avoid its Linux-only default affinity query.

The four-method local pool explicitly disables absent RF calibration/fallback. Its ranking engine remains upstream; published context preserves upstream defaults and is labeled mixed hardware/budgets. The fixed AlloyGBM default preserves public automatic rounds and stopping metric behavior; HPO configuration selection uses official task validation predictions. Neither arm adds an internal holdout.

Ray result-processing workers do not inherit the driver's process-local TabArena cache setting. Passing the local cache root explicitly in serialized method metadata fixes a reproduced missing `model_predictions` failure without modifying upstream. Reports record their own code hash, allowing cached predictions from a recorded fit revision to be reprocessed transparently after a report-only fix. Resume still requires the recorded implementation identity.

Cached selection diagnostics also export the tuned single-config and ensemble weights. They replay upstream selection per task using recorded variant settings and assert that validation/test scores match the standard processor output. Adversarial tests confirm test-target changes do not alter selected weights.

The retirement gate requires the published-context report with unchanged upstream calibration/fallback, bound to the same terminal ledger and report-code revision. The four-method local diagnostic pool alone cannot satisfy acceptance.

## Acceptance status

The earlier settings above describe the initial development runs. They are
superseded by the review follow-up below. Full Lite HPO acceptance, complete
official submission scope and maintainer approval have **not** been established.
The retirement PR remains a draft. Generated caches/manifests remain ignored;
no leaderboard upload is performed.

## Review follow-up: standard resources, calibration and disposable Linux runs

See [resolutions](../reviews/2026-10-20-tabarena-resolutions.md) and the
[droplet runbook](../../benchmarks/tabarena/DROPLET.md). Upstream Ray is now the
default; retirement requires the chosen Linux 16-vCPU / nominal 64-GB resource
policy. Local debug execution remains diagnostic. The matched pool adds a
default-only RF calibrator; published comparisons add only AlloyGBM and preserve
the planned task scope even when an entire task has failed AlloyGBM fits.
The complete Lite HPO plan now has 5,355 jobs / 42,840 children, including RF.

The adapter uses AutoGluon's resolved stopping metric. At this pin, AUC evaluation
defaults to log-loss stopping; explicit AUC stopping is supported. Completed
cooperative fits are retained, and native deadlines without early stopping keep
the completed prefix. These changes supersede the earlier post-fit rejection.

Verified follow-up evidence:

- Rust before/after native fixes: 909 passed, 3 ignored; formatting and Clippy
  with warnings denied passed.
- Python baseline on the retirement tree: 1,251 + 64 subtests. After native fixes,
  the integration tree: 1,259 + 64 subtests; deadline-specific suite: 27 passed.
  Final rebased retirement tree: 1,256 + 64 subtests. Sphinx with warnings as
  errors and Clippy across all targets with warnings denied passed.
- Benchmark/cloud contracts: 50 passed. Registered upstream tests: 2 passed,
  102 unrelated cases deselected. Ruff lint/format and shell syntax passed.
- Run `4d176b4128ab175d`, official binary task, upstream Ray, development CPU
  budget 2: 9/9 jobs completed, 72 children validated. Resume made no fits or
  ledger additions: 18 original start/completion events remain.
- Cached report generation and regeneration passed. Local report: 13 variants,
  RF Elo calibration exactly 1000. Published: 106 rows, 60 method metadata records,
  only AlloyGBM added locally, exactly the one planned dataset/split. Both reports
  bind the same ledger and current processor hash; sampled-candidate count is 1.
- Portable export tests recover source and predictions after removing a
  simulated server checkout and reject a corrupted transfer. Session tests cover
  successful fits, failed fits and timeouts while preserving exit codes and
  exporting diagnostics. They require no cloud account or heavy model packages.

No droplet was provisioned. A fresh Ubuntu bootstrap and full Linux Ray pilot
remain unexecuted; full Lite acceptance remains pending. This macOS binary smoke
does not establish competitive rank or satisfy the migration resource gate.
