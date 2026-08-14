# PR #137 DART Expected-Drop Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

| Date | Author | Production base | Status |
|---|---|---|---|
| 2026-08-14 | OpenAI Codex | `8a76ccb` | Implemented locally; Task 5 verification complete; PR publication intentionally omitted |

**Goal:** Select and ship a lower public `dart_max_drop` default only if a fixed five-seed,
multi-task A/B matrix proves materially lower long-fit dropout work without material held-out
quality regressions.

**Architecture:** A subprocess benchmark evaluates explicit caps `2`, `5`, `10`, `20`, and
incumbent `50`, then applies the approved predeclared selection rule. Because calibration arms are
explicit, matrix results do not depend on the installed default. Separate pre/post compatibility
captures prove explicit `50` retains production bytes and the new default equals its selected
explicit value.

**Tech Stack:** Python 3.11-3.13, NumPy, scikit-learn, pytest, Rust 1.92, PyO3 0.29, maturin,
Sphinx, subprocess RSS measurement, deterministic SHA-256 evidence.

## Global Constraints

- Follow `docs/reviews/2026-08-14-pr-137-dart-expected-drop-calibration-design.md` exactly.
- Candidate caps are fixed at `2`, `5`, `10`, `20`; incumbent is `50`; full seeds are `0..4`.
- Select the largest passing cap; retain `50` if no candidate passes.
- Do not change dropout RNG, forced-one behavior, probabilities, truncation ordering,
  normalization, warm-start replay, native `BoostingMode`, artifacts, or prediction.
- Explicit `dart_max_drop=50` must reproduce production artifacts and predictions exactly.
- Keep `dart_max_drop` as the only public cap. Add no alias, hidden policy, dependency, or unsafe.
- Use test-first red/green cycles and retain every rejected candidate reason in JSON evidence.

---

### Task 1: Build the calibration harness and capture production evidence

**Files:**
- Create: `benchmarks/dart_policy_calibration.py`
- Create: `benchmarks/tests/test_dart_policy_calibration.py`
- Create: `benchmarks/results/pr137_dart_policy_matrix.json`
- Create: `benchmarks/results/pr137_dart_policy_production_compat.json`

**Interfaces:**
- Produce immutable `FixtureSpec`, `DartPolicyRecord`, `CandidateAssessment`, and
  `DartPolicyDecision` dataclasses.
- Produce `full_specs()`, `quick_specs()`, `configured_dropout_pressure()`,
  `evaluate_candidate_caps()`, deterministic JSON I/O, and `run`, `run-compat`, `compare`, and
  internal `record` CLI commands.
- Later tasks consume `DartPolicyDecision.selected_cap` without reinterpretation.

- [x] **Step 1: Write failing benchmark-contract tests**

Pin constants `(2, 5, 10, 20)`, incumbent `50`, and seeds `(0, 1, 2, 3, 4)`. Require these cases:

| Name | Task | Rows | Features | Rounds | Rate | Policy | Growth | Stress |
|---|---|---:|---:|---:|---:|---|---|---|
| `reg-small-narrow` | regression | 640 | 8 | 100 | .10 | uniform/tree | level | no |
| `reg-small-wide` | regression | 640 | 64 | 100 | .10 | uniform/tree | level | no |
| `reg-tall-narrow` | regression | 4096 | 12 | 200 | .10 | uniform/tree | level | yes |
| `reg-tall-wide-leaf` | regression | 3072 | 64 | 200 | .10 | uniform/tree | leaf | yes |
| `reg-long-stress` | regression | 2048 | 24 | 300 | .20 | uniform/tree | level | yes |
| `binary-medium` | binary | 2048 | 24 | 150 | .10 | uniform/tree | level | no |
| `multiclass-four` | multiclass | 1600 | 20 | 100 | .10 | uniform/tree | level | yes |
| `ranking-groups` | ranking | 2400 | 16 | 120 | .10 | uniform/tree | level | no |
| `reg-weighted` | regression | 1536 | 16 | 200 | .10 | weighted/tree | level | yes |
| `reg-forest` | regression | 1536 | 16 | 200 | .10 | uniform/forest | level | yes |

Pin multiclass pressure with `trees_per_round=4`, proving it exceeds the matching scalar pool.
Create synthetic records and independently reject median quality `1.0201`, one-seed quality
`1.1001`, accuracy loss `0.0201`, NDCG loss `0.0101`, pressure `0.5001`, stress time `0.8501`,
excess RSS, non-finite/incomplete fits, and missing/duplicate keys. Pin largest-pass selection and
the no-pass fallback to 50 with reasons.

- [x] **Step 2: Run tests and verify RED**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  benchmarks/tests/test_dart_policy_calibration.py -q
```

Expected: missing harness/import failure.

- [x] **Step 3: Implement deterministic fixtures, metrics, and subprocess records**

Use local seeded 80/20 data splits. Regression records RMSE/MAE; binary and multiclass record log
loss/accuracy; ranking records NDCG@10. Use `training_policy="manual"`, `learning_rate=0.06`,
`max_depth=4`, `lambda_l2=1.0`, deterministic quantile binning, explicit DART policy values, and
`max_leaves=16` for leaf-wise. Ranking uses 80 contiguous groups and passes training group sizes.

Normalize subprocess `ru_maxrss` to bytes. Record elapsed fit seconds, completed rounds, metric
values, configured pressure, prediction SHA-256, and artifact SHA-256. Multiclass pressure uses
the four-class tree pool. Reject schema mismatch and non-finite JSON on read; write sorted stable
records with indent 2.

- [x] **Step 4: Implement the exact selection algorithm**

Orient lower metrics as `candidate/incumbent` and higher metrics as `incumbent/candidate`. Apply all
eight design gates without rounding. Aggregate pressure/time over `stress=True` fixtures. Select
`max(passing_caps)`, otherwise 50. Preserve every ratio and rejection reason.

- [x] **Step 5: Implement and test CLI commands**

```bash
python benchmarks/dart_policy_calibration.py run \
  --caps 2 5 10 20 50 --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr137_dart_policy_matrix.json
python benchmarks/dart_policy_calibration.py run-compat \
  --arms default cap50 --seeds 0 1 2 \
  --output benchmarks/results/pr137_dart_policy_production_compat.json
python benchmarks/dart_policy_calibration.py compare \
  benchmarks/results/pr137_dart_policy_matrix.json \
  --production-compat benchmarks/results/pr137_dart_policy_production_compat.json \
  --output /tmp/pr137-decision.json
```

Compatibility uses `reg-small-wide`, `binary-medium`, `multiclass-four`, and `ranking-groups`.
Production default and explicit cap50 hashes must match within the capture.

- [x] **Step 6: Build release extension, run GREEN, capture and commit evidence**

Run the complete explicit-cap matrix on base behavior. Do not override the generated decision.

```bash
git add -f benchmarks/dart_policy_calibration.py \
  benchmarks/tests/test_dart_policy_calibration.py \
  benchmarks/results/pr137_dart_policy_matrix.json \
  benchmarks/results/pr137_dart_policy_production_compat.json
git commit -m "bench: calibrate DART expected-drop default"
```

### Task 2: Apply the evidence-selected public default

**Files:**
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/tests/test_dart.py`
- Modify: `bindings/python/tests/test_regressor_contract.py`
- Modify: `bindings/python/tests/test_sklearn_conformance.py` only if needed
- Create: `benchmarks/results/pr137_dart_policy_candidate_compat.json`
- Create: `benchmarks/results/pr137_dart_policy_comparison.json`

**Interfaces:**
- Produce the selected default on regressor, classifier, and ranker through the shared constructor.
- Multi-label independent mode inherits the ranker default; joint mode still forwards explicit
  kwargs. If selection returns 50, leave constructor literals unchanged and continue evidence/docs.

- [x] **Step 1: Write failing public-default tests from the generated selected cap**

For all three estimators, pin constructor attribute, `get_params`, clone, repr, pickle, and explicit
50 override. Keep zero/negative validation and `set_params` coverage.

- [x] **Step 2: Run focused test and verify RED**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_dart.py -k default -q
```

If the generated decision retains `50`, the existing default assertion is expected to remain
GREEN. In that branch, add the generated-decision and explicit-override assertions without forcing
an artificial production change; the retained-default evidence is the behavior under test.

- [x] **Step 3: Change only both Python constructor default literals**

Replace both `dart_max_drop: int = 50` declarations in `_core.py` with the selected integer. Do not
change assignment, validation, parameter order, forwarding, or native code.

- [x] **Step 4: Build and run focused DART/conformance tests GREEN**

```bash
maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_dart.py bindings/python/tests/test_dart_warm_start.py \
  bindings/python/tests/test_multiclass_dart.py bindings/python/tests/test_ranker.py \
  bindings/python/tests/test_regressor_contract.py \
  bindings/python/tests/test_sklearn_conformance.py -q
```

- [x] **Step 5: Capture candidate compatibility and final comparison**

Run compatibility arms `default cap-selected cap50` over seeds `0 1 2`. Require candidate cap50
hashes equal production cap50, candidate default equals explicit selected cap, and all fits finish.
Embed the Task 1 decision and rejected-cap reasons in final comparison JSON.

- [x] **Step 6: Commit the default and evidence**

```bash
git add bindings/python/alloygbm/_regressor/_core.py bindings/python/tests/test_dart.py \
  bindings/python/tests/test_regressor_contract.py \
  bindings/python/tests/test_sklearn_conformance.py
git add -f benchmarks/results/pr137_dart_policy_candidate_compat.json \
  benchmarks/results/pr137_dart_policy_comparison.json
git commit -m "feat: recalibrate the DART drop cap default"
```

### Task 3: Protect explicit behavior and supported combinations

**Files:**
- Modify: `bindings/python/tests/test_dart.py`
- Modify: `bindings/python/tests/test_multiclass_dart.py`
- Modify: `bindings/python/tests/test_dart_warm_start.py`
- Modify: `bindings/python/tests/test_joint_multilabel.py`
- Modify: `benchmarks/tests/test_dart_policy_calibration.py`

- [x] **Step 1: Add deterministic artifact regression cases**

Pin repeated explicit-50 artifacts for regression, multiclass, and ranking. Pin default versus
explicit-selected artifacts under level-wise and leaf-wise regression. For deterministic fixtures,
pin prediction and artifact equality across `n_jobs=1` and `n_jobs=2` for the selected explicit cap.

- [x] **Step 2: Add warm-start and multi-label forwarding coverage**

Use existing DART tolerance for uninterrupted versus warm-start predictions under the selected
default. Assert multi-label omission inherits the ranker default and explicit 50 reaches joint
native kwargs unchanged.

- [x] **Step 3: Run focused suites and commit**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_dart.py bindings/python/tests/test_multiclass_dart.py \
  bindings/python/tests/test_dart_warm_start.py bindings/python/tests/test_joint_multilabel.py \
  benchmarks/tests/test_dart_policy_calibration.py -q
git add bindings/python/tests/test_dart.py \
  bindings/python/tests/test_multiclass_dart.py \
  bindings/python/tests/test_dart_warm_start.py \
  bindings/python/tests/test_joint_multilabel.py \
  benchmarks/tests/test_dart_policy_calibration.py
git commit -m "test: protect calibrated DART drop policy"
```

### Task 4: Publish evidence and close the July finding

**Files:**
- Create: `docs/benchmarks/dart_policy_calibration_pr137.md`
- Modify: `README.md`, `CHANGELOG.md`, `benchmarks/README.md`
- Modify: `docs/user/gbmregressor.md`, `docs/user/benchmarks.md`
- Modify: `docs/site/source/estimator.rst`, `docs/site/source/benchmarks.rst`
- Modify: `docs/roadmap/current.md`
- Modify: `docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md`
- Modify: both PR #137 design/plan documents

- [x] **Step 1: Write the evidence report**

Include commits, commands, host/packages, fixtures, fixed thresholds, every cap verdict, per-fixture
median quality, stress pressure/time/RSS, hashes, rejected reasons, and selected default. State that
same-host timing is not universal.

- [x] **Step 2: Update public and review documentation**

Document selected default, expected-work formula, forced-one rule, multiclass class-tree pool, and
`dart_max_drop=50` migration override. Do not claim universal optimality. Replace the open July
finding, add Unreleased/roadmap entries, benchmark links, and implemented document statuses.

- [x] **Step 3: Verify docs and commit**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/sphinx-build -W -b html \
  docs/site/source /tmp/alloygbm-pr137-sphinx
git diff --check
git add README.md CHANGELOG.md benchmarks/README.md docs
git commit -m "docs: close DART expected-drop calibration"
```

### Task 5: Final review, verification, and draft PR

- [x] **Step 1: Run complete verification**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
maturin develop --release
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/ benchmarks/tests/ -q
/Users/lashby/Projects/AlloyGBM/.venv/bin/sphinx-build -W -b html \
  docs/site/source /tmp/alloygbm-pr137-sphinx
python benchmarks/dart_policy_calibration.py compare \
  benchmarks/results/pr137_dart_policy_matrix.json \
  --production-compat benchmarks/results/pr137_dart_policy_production_compat.json \
  --candidate-compat benchmarks/results/pr137_dart_policy_candidate_compat.json \
  --output /tmp/pr137-dart-policy-comparison.json
git diff --check 8a76ccb...HEAD
git status --short
```

- [x] **Step 2: Inspect scope**

Confirm no Rust production, artifact, or predictor diff; only two Python production literals may
change. Confirm complete finite evidence, exact cap50 production hashes, and generated cap choice.

- [ ] **Step 3: Push and create draft PR #137** (intentionally omitted per user instruction)

Use the PR template. Include selected cap, algorithm, rejected caps, quality/performance evidence,
explicit-50 migration, compatibility, verification counts, and document links. Preserve the
worktree and stop before merge.
