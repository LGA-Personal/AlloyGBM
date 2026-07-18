# Approximate Quantile Sketches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound quantile-cut preprocessing cost on very large matrices with deterministic sampled sketches while preserving exact behavior below a configured threshold.

**Architecture:** Select exact sorting or deterministic bounded-row sampling inside the native quantile path. Return the native cut metadata to every estimator mode, including joint multi-output, and expose fitted method diagnostics. Prediction continues using persisted cut values and existing quantization semantics.

**Tech Stack:** Rust quantization bridge, PyO3 estimator parameters/metadata, NumPy contract tests.

## Global Constraints

- Public option: `quantile_sketch_max_rows: int | None = None`; `None` keeps exact sorting.
- Candidate benchmark uses `65_536` sampled rows per feature.
- Sampling and cuts must be deterministic across supported Rayon thread counts.
- Preserve strictly increasing finite cuts and the missing-bin sentinel contract.

---

### Task 1: Pin The Existing Upper-Tail Bin Contract

**Files:**
- Modify: `bindings/python/src/quantization.rs`
- Test: `bindings/python/src/tests.rs`
- Test: `bindings/python/tests/test_regressor_contract.py`

- [ ] **Step 1: Add a failing parity test** comparing native and Python quantization for values above the final cut at `max_bins=256`, including NaN.
- [ ] **Step 2: Make both paths reserve the missing sentinel and clamp ordinary values to the same maximum data bin.**
- [ ] **Step 3: Run bridge and Python contract tests; commit this correctness fix separately.**

Commit: `git commit -am "Align native quantile upper-tail bins"`

### Task 2: Add Sketch Configuration And Activation Diagnostics

**Files:**
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/multi_label_ranker.py`
- Modify: `bindings/python/src/pyclasses.rs`
- Modify: `bindings/python/src/train.rs`
- Test: `bindings/python/tests/test_regressor_contract.py`

**Interfaces:**
- Adds constructor/get-set/repr parameter `quantile_sketch_max_rows`.
- Adds fitted `feature_quantile_cut_methods_`, one value of `"exact"` or `"sketch"` per feature.

- [ ] **Step 1: Write failing API round-trip, validation, persistence, and diagnostic tests.**
- [ ] **Step 2: Thread the optional positive row limit to every quantile-training bridge.**
- [ ] **Step 3: Keep exact behavior and report `"exact"` until the sketch implementation lands.**
- [ ] **Step 4: Run tests and commit.**

Commit: `git commit -am "Add quantile sketch configuration"`

### Task 3: Implement Deterministic Bounded Sampling

**Files:**
- Modify: `bindings/python/src/quantization.rs`
- Test: `bindings/python/src/tests.rs`

**Interfaces:**
- Adds `derive_dense_feature_quantile_cuts_sampled(values, rows, features, max_bins, max_rows)`.

- [ ] **Step 1: Add failing tests** for exact fallback, deterministic sampled indices, skewed rank error, duplicate plateaus, constants, NaNs, and 1-thread/4-thread equality.
- [ ] **Step 2: For each feature, select a deterministic evenly distributed row sample capped at `max_rows`, discard NaNs, sort with `f32::total_cmp`, and reuse the exact cut selector.**
- [ ] **Step 3: Keep feature-level Rayon collection ordered; do not use unordered reductions.**
- [ ] **Step 4: Return method diagnostics with cut metadata; run bridge tests and commit.**

Commit: `git commit -am "Derive quantile cuts from deterministic samples"`

### Task 4: Keep Joint Training And Prediction On One Cut Set

**Files:**
- Modify: `bindings/python/src/joint.rs`
- Modify: `bindings/python/alloygbm/multi_label_ranker.py`
- Test: `bindings/python/tests/test_joint_multilabel.py`

- [ ] **Step 1: Add a failing joint sketch test** proving fitted native cuts equal prediction/persistence cuts after save/load.
- [ ] **Step 2: Return native continuous-binning metadata from joint training and remove Python re-derivation for that fit.**
- [ ] **Step 3: Preserve exact behavior when the threshold is `None` or row count is below it.**
- [ ] **Step 4: Run joint tests and commit.**

Commit: `git commit -am "Reuse native quantile cuts in joint mode"`

### Task 5: Verify Accuracy And Performance

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate --scenario quantile_sketches \
  --baseline benchmarks/results/architectural_backlog_baseline.json --gate
```

- [ ] **Step 1: Require activation, mean/p99/max rank error at or below 0.0025/0.0075/0.01, and RMSE no worse than 1%.**
- [ ] **Step 2: Require native bridge preparation at most 60% of baseline, total fit no worse than 5%, and at least 32 MiB end-to-end RSS reduction.**
- [ ] **Step 3: Run full fmt, clippy, Rust, and Python suites; commit evidence separately.**
