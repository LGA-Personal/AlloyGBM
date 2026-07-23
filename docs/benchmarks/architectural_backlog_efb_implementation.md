# Exclusive Feature Bundling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle mutually exclusive sparse/one-hot numeric features during training without changing emitted tree feature identities or prediction/explanation contracts.

**Architecture:** Build deterministic training-only bundles before histogram construction. Bundle bins occupy disjoint ranges, while split candidates decode back to original feature indices and thresholds before stumps are emitted. Start with dense NumPy one-hot inputs; do not add a sparse matrix API in the same project.

**Tech Stack:** Rust binning/core/backend crates, PyO3 estimator configuration, existing artifact/predictor/SHAP contracts.

## Global Constraints

- Public option: `feature_bundling="off" | "exact"`, default `"off"` in the first release.
- Skip categorical, monotone-constrained, interaction-constrained, or conflict-unsafe features rather than weakening their semantics.
- Limit candidates to contiguous original-column groups with at most 25%
  occupancy per feature. This first contract targets encoded one-hot blocks
  rather than arbitrary conflict-graph coloring.
- Emitted stumps retain original feature indices, so no artifact section is added.
- Deterministic input and seed must produce deterministic bundle maps and artifacts.

---

### Task 1: Add Configuration And Diagnostics

**Files:**
- Modify: `bindings/python/alloygbm/_regressor/_core.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/multi_label_ranker.py`
- Modify: `bindings/python/src/train.rs`
- Test: `bindings/python/tests/test_regressor_contract.py`

**Interfaces:**
- Adds `feature_bundling: str = "off"`.
- Adds fitted `feature_bundling_diagnostics_` with original/effective feature counts, bundle count, skipped-feature count, and observed conflict count.

- [x] **Step 1: Write failing constructor/get-set/repr/persistence tests.**
- [x] **Step 2: Add validation and thread the enum to native training without changing behavior.**
- [x] **Step 3: Return inactive diagnostics for `"off"`; run contract tests and commit.**

Commit: `git commit -am "Add EFB configuration contract"`

### Task 2: Build Deterministic Zero-Conflict Bundles

**Files:**
- Create: `crates/core/src/feature_bundling.rs`
- Modify: `crates/core/src/lib.rs`
- Test: `crates/core/src/tests/main.rs`

**Interfaces:**
- Produces `FeatureBundleMap` with original feature id, bundle id, bin offset, and bin span.
- Consumes nonzero row sets plus excluded-feature masks and accepts only
  canonical contiguous groups.

- [x] **Step 1: Add failing tests** for perfect exclusivity, deterministic first-fit ordering, excluded features, all-zero columns, NaNs, and conflicts.
- [x] **Step 2: Implement stable greedy bundling ordered by descending nonzero count then original feature id.**
- [x] **Step 3: Permit only canonical contiguous zero-conflict bundles in this first implementation.**
- [x] **Step 4: Run core tests and commit.**

Commit: `git commit -am "Build deterministic exclusive feature bundles"`

### Task 3: Integrate Bundles Into Binning And Split Search

**Files:**
- Modify: `bindings/python/src/quantization.rs`
- Modify: `crates/core/src/binned.rs`
- Modify: `crates/backend_cpu/src/lib.rs`
- Modify: `crates/engine/src/traits.rs`
- Test: `crates/backend_cpu/src/tests/main.rs`
- Test: `bindings/python/tests/test_regressor_contract.py`

- [x] **Step 1: Add a failing equivalence test** where bundled and unbundled zero-conflict one-hot data select the same original feature and threshold.
- [x] **Step 2: Encode each original feature into its bundle's disjoint bin range.**
- [x] **Step 3: Scan bundle segments as original-feature candidate ranges; decode accepted candidates before engine/artifact handoff.**
- [x] **Step 4: Ensure missing bins and zero/default bins cannot alias another feature's segment.**
- [x] **Step 5: Run workspace tests and commit.**

Commit: `git commit -am "Train with exclusive feature bundles"`

### Task 4: Verify Public Contracts And Benchmark

- [x] **Step 1: Add persistence, feature importance, SHAP, categorical-skip, monotone-skip, and interaction-skip tests.**
- [x] **Step 2: Run full Rust/Python suites.**
- [x] **Step 3: Run the benchmark.**

```bash
.venv/bin/python -m benchmarks.architectural_backlog.run \
  --profile full --mode candidate --scenario efb \
  --baseline benchmarks/results/architectural_backlog_baseline.json --gate
```

- [x] **Step 4: Require candidate activation, byte-equivalent zero-conflict artifacts, deterministic conflict refusal, and either 15% total-fit or 20% RSS improvement.**
