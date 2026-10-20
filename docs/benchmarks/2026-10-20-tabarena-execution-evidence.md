# TabArena replacement execution evidence

Date: 2026-10-20. This is implementation evidence, not an official leaderboard result.

- Baseline: `origin/main` at `4e5af80787610714f52415a09d5182fd1373ca98`.
- Native deadline source: `ecbafa1fb06130d5fe8e481e87d6aaa9fbb79a09` (PR #172), exact installable dependency in the submission metadata.
- TabArena and bencheval: `041f9b3774d1191ffcdcf137cbe4e1e08159b972`, with only the registered AlloyGBM payload and three allowed registration changes. No validation/scoring/shared-test edits.
- Machine: macOS arm64, Apple M4, 10 physical/logical cores, 24 GiB RAM; Python 3.12.10. Exact dependency versions, native hash, source hash, CPU/thread allocation and execution are in each run manifest. Smoke timings are workflow checks and are not comparative claims.

## Completed checks

- Native deadline change: full Rust workspace **909 passed / 3 ignored**, full bindings suite **1,254 passed / 64 subtests**, clippy all targets with warnings denied. All PR #172 GitHub CI checks passed.
- Registered adapter: regression/binary/multiclass fit, NaNs/categories, prediction shape/finite checks, save/load equivalence, CPU forwarding, rejection of hidden validation, native timeout translation, and fresh-interpreter registered-class serialization.
- Protocol: official enforced 8×1, three example tasks / 24 smoke jobs, 51 Lite tasks / 204 default jobs / 5,304 HPO jobs, immutable default identity and deterministic search-prefix stability.
- Execution/report contracts: changed-identity rejection, torn-write recovery, exclusive driver ownership, failure/pending coverage, checksum/conformance verification, full-scope acceptance guard, and adversarial upstream selection test (changing test targets does not change validation-selected weights).
- Actual upstream registered tests: `test_all_models.py` and `test_warmup_coverage.py`, `-m models -k AlloyGBM`: **2 passed / 102 deselected**. No AlloyGBM case skipped.
- Ruff check/format pass on the integration and its tests. Universal Python 3.12 dependency lock resolves; install dry-run differs only in the three deliberately editable source packages.
- Real caches from an exploratory official-split run were processed with upstream `EndToEnd` and `TabArenaContext.compare`. All three variants and standard CSV/plot outputs were generated. This partial evidence does not satisfy Lite acceptance.

## Measured issues and scope limits

The global upstream dependency-extra checker reports 21 drifting entries even on the untouched pinned revision. Registered AlloyGBM's row is `OK`; the only union-row difference is its required addition to `extended`. Direct tests check the extra, exact pin, registry and tree-family entry. The global checker is not passing.

macOS reproduces a native crash when AlloyGBM warmup precedes LightGBM warmup in the same interpreter. Isolating each standard outer job into a fresh process permits the peer fits to execute. Local mode retains the upstream quickstart native/debug runner and eight sequential child folds. Explicit Ray mode remains available and requires its own execution evidence. No OpenMP duplicate-library bypass or global runtime monkeypatch is used.

On macOS, upstream warns that a one-CPU allocation has a ten-CPU affinity mask. This platform lacks Linux cpuset enforcement; the explicit one-CPU child budgets and numerical thread variables are recorded. These timings are local development evidence, not an official hardware-normalized result. Upstream reporting also needs an explicit worker count to avoid its Linux-only default affinity query.

The four-method local pool explicitly disables absent RF calibration/fallback. Its ranking engine remains upstream; published context preserves upstream defaults and is labeled mixed hardware/budgets. The fixed AlloyGBM default preserves public automatic rounds and stopping metric behavior; HPO configuration selection uses official task validation predictions. Neither arm adds an internal holdout.

## Acceptance status

The frozen smoke run is in progress. Full Lite HPO acceptance, complete official submission scope and maintainer approval have **not** been established. The separate retirement PR must remain a draft until the documented acceptance audit passes. Generated raw caches and manifests remain ignored local artifacts; no leaderboard upload is performed.
