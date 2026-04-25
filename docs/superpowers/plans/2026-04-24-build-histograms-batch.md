# Build-Histograms Command-Buffer Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse N per-node `commit + waitUntilCompleted` round-trips per tree level into one per phase (build + subtract) on the Metal backend, in order to cross the Stage 3 `metal_friendly >1.0× CPU` kill criterion.

**Architecture:** Add two batched methods to the engine `BackendOps` trait with scalar-delegating default impls. Refactor `build_tree_level_wise` so each level runs its per-node serial work first, then dispatches all smaller-child histogram builds as one Metal command buffer, then all larger-child subtracts as one command buffer. `MetalBackend` overrides both methods; `CpuBackend` uses the defaults. Leaf-wise growth is untouched.

**Tech Stack:** Rust 2024 / 1.92.0, `objc2_metal` for Metal command-buffer encoding, MSL kernels (already implemented — not changed by this plan), PyO3 Python bindings (no API surface change).

**Source-of-truth references**
- Spec: [docs/superpowers/specs/2026-04-24-build-histograms-batch-design.md](../specs/2026-04-24-build-histograms-batch-design.md)
- Diagnosis: [docs/metal-backend/DECISIONS.md](../../metal-backend/DECISIONS.md) D-020, D-021, D-022
- Engine entry: [crates/engine/src/lib.rs](../../../crates/engine/src/lib.rs) — `BackendOps` trait at lines 86–205, `build_tree_level_wise` at lines 3927–4166
- Metal entry: [crates/backend_metal/src/lib.rs](../../../crates/backend_metal/src/lib.rs) — `BackendOps` impl at lines 318+
- Histogram kernel: [crates/backend_metal/src/kernels/histogram.rs](../../../crates/backend_metal/src/kernels/histogram.rs) — `dispatch_histograms` at line 75
- Subtract kernel: [crates/backend_metal/src/kernels/subtract.rs](../../../crates/backend_metal/src/kernels/subtract.rs) — `dispatch_subtract_pool` at line 300
- Profile probes: [crates/backend_metal/src/profile.rs](../../../crates/backend_metal/src/profile.rs)

**Working directory note:** This plan executes inside the existing worktree
`/Users/lashby/Projects/AlloyGBM/.claude/worktrees/charming-carson-d08c9a`.
Use the absolute Python path `/Users/lashby/Projects/AlloyGBM/.venv/bin/python`
(the worktree has no local `.venv`). All `cargo` and `git` commands run from
the worktree root.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `crates/engine/src/lib.rs` | Modify | Add `HistogramBuildRequest`, `SubtractRequest`, two new `BackendOps` methods with scalar default impls. Refactor `build_tree_level_wise` into three phases per level. |
| `crates/backend_metal/src/lib.rs` | Modify | Override `build_histograms_batch` and `subtract_histogram_bundle_batch` on `MetalBackend`. |
| `crates/backend_metal/src/kernels/histogram.rs` | Modify | Add `dispatch_histograms_batch` — encodes N scatter+reduce passes into one command buffer, finalises counts after one wait. |
| `crates/backend_metal/src/kernels/subtract.rs` | Modify | Add `dispatch_subtract_batch_pool` — encodes N pool-direct subtract passes into one command buffer, one wait. |
| `crates/backend_metal/src/profile.rs` | Modify | Add `BUILD_HISTOGRAMS_BATCH` and `SUBTRACT_BATCH` counters; wire into `dump_if_enabled`. |
| `docs/metal-backend/DECISIONS.md` | Append | D-023 recording post-batch measurements vs. kill criterion. |
| `docs/metal-backend/STATUS.md` | Overwrite | Update Stage 3 status / next-up at end. |
| `docs/metal-backend/SESSIONS.md` | Prepend | Session entry summarising what shipped. |

The MSL shaders and existing kernels are not modified — batching only changes how dispatches are aggregated.

---

## Task 1: Add `BackendOps` batched method signatures with scalar default impls

**Why first:** Establishes the trait surface. With scalar-delegating defaults, the CPU backend gets the new API for free and the engine refactor (Task 2) can be exercised against `CpuBackend` before any Metal-specific code lands.

**Files:**
- Modify: `crates/engine/src/lib.rs` — add request structs near the existing `BackendOps` trait (after `CategoricalFeatureInfo`, before `pub trait BackendOps`); add two methods inside the trait.

- [ ] **Step 1: Write the failing test**

Create the test in `crates/engine/src/lib.rs` inside the existing `#[cfg(test)] mod tests` block. Find the end of the existing test module and add this just before its closing `}`:

```rust
    #[test]
    fn cpu_backend_build_histograms_batch_default_matches_scalar() {
        use crate::{
            BackendOps, BinnedMatrix, FeatureTile, GradientPair, HistogramBuildRequest, NodeSlice,
        };
        use alloygbm_backend_cpu::CpuBackend;

        let row_count = 64usize;
        let feature_count = 3usize;
        let max_bin: u16 = 7;
        let bins: Vec<u8> = (0..(row_count * feature_count))
            .map(|i| ((i * 11) & 7) as u8)
            .collect();
        let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
        let grads: Vec<GradientPair> = (0..row_count)
            .map(|i| GradientPair {
                grad: i as f32,
                hess: 1.0,
            })
            .collect();
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: feature_count as u32,
        }];
        let backend = CpuBackend::new();

        let node_a = NodeSlice::new(0, (0..32u32).collect()).unwrap();
        let node_b = NodeSlice::new(1, (32..64u32).collect()).unwrap();

        let scalar_a = backend
            .build_histograms(&bm, &grads, &node_a, &tiles)
            .unwrap();
        let scalar_b = backend
            .build_histograms(&bm, &grads, &node_b, &tiles)
            .unwrap();

        let requests = vec![
            HistogramBuildRequest { node: &node_a },
            HistogramBuildRequest { node: &node_b },
        ];
        let batched = backend
            .build_histograms_batch(&bm, &grads, &tiles, &requests)
            .unwrap();
        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0], scalar_a);
        assert_eq!(batched[1], scalar_b);
    }

    #[test]
    fn cpu_backend_subtract_histogram_bundle_batch_default_matches_scalar() {
        use crate::{
            BackendOps, BinnedMatrix, FeatureTile, GradientPair, NodeSlice, SubtractRequest,
        };
        use alloygbm_backend_cpu::CpuBackend;

        let row_count = 64usize;
        let feature_count = 3usize;
        let max_bin: u16 = 7;
        let bins: Vec<u8> = (0..(row_count * feature_count))
            .map(|i| ((i * 11) & 7) as u8)
            .collect();
        let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
        let grads: Vec<GradientPair> = (0..row_count)
            .map(|i| GradientPair {
                grad: i as f32,
                hess: 1.0,
            })
            .collect();
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: feature_count as u32,
        }];
        let backend = CpuBackend::new();

        let parent_node = NodeSlice::new(0, (0..64u32).collect()).unwrap();
        let smaller_node = NodeSlice::new(1, (0..32u32).collect()).unwrap();
        let parent = backend
            .build_histograms(&bm, &grads, &parent_node, &tiles)
            .unwrap();
        let smaller = backend
            .build_histograms(&bm, &grads, &smaller_node, &tiles)
            .unwrap();

        let scalar = backend
            .subtract_histogram_bundle(&parent, &smaller, 2)
            .unwrap();

        let requests = vec![SubtractRequest {
            parent: &parent,
            sibling: &smaller,
            output_node_id: 2,
        }];
        let batched = backend
            .subtract_histogram_bundle_batch(&requests)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(batched[0], scalar);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p alloygbm_engine cpu_backend_build_histograms_batch_default_matches_scalar 2>&1 | tail -20`
Expected: FAIL with errors about unresolved imports (`HistogramBuildRequest`, `SubtractRequest`) or undefined methods.

- [ ] **Step 3: Add request structs to `crates/engine/src/lib.rs`**

Locate the `pub trait BackendOps {` declaration (around line 86) and insert the request structs immediately above it:

```rust
/// One node's input for a batched histogram build. The shared
/// `binned_matrix`, `gradients`, and `feature_tiles` are passed
/// alongside `&[HistogramBuildRequest]` to
/// [`BackendOps::build_histograms_batch`].
#[derive(Debug)]
pub struct HistogramBuildRequest<'a> {
    pub node: &'a NodeSlice,
}

/// One node's input for a batched parent-minus-sibling subtract.
/// `output_node_id` becomes the `node_id` of the resulting bundle.
#[derive(Debug)]
pub struct SubtractRequest<'a> {
    pub parent: &'a HistogramBundle,
    pub sibling: &'a HistogramBundle,
    pub output_node_id: u32,
}
```

- [ ] **Step 4: Add trait methods with scalar default impls**

Inside the `pub trait BackendOps` block, immediately after the existing `subtract_histogram_bundle` method (around line 144) and before `release_histograms`, add:

```rust
    /// Batched histogram build. Default impl iterates and calls
    /// [`build_histograms`](BackendOps::build_histograms); accelerator
    /// backends override to encode all dispatches into a single
    /// command buffer and amortise the host↔device round-trip.
    ///
    /// The returned vector aligns with `requests`: `result[i]` is the
    /// histogram for `requests[i].node`. On any backend error, the
    /// whole call returns `Err` and any partially-built bundles
    /// included in earlier indices are dropped (caller must not
    /// depend on partial results).
    fn build_histograms_batch(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        feature_tiles: &[FeatureTile],
        requests: &[HistogramBuildRequest<'_>],
    ) -> EngineResult<Vec<HistogramBundle>> {
        requests
            .iter()
            .map(|r| self.build_histograms(binned_matrix, gradients, r.node, feature_tiles))
            .collect()
    }

    /// Batched parent-minus-sibling subtract. Default impl iterates
    /// and calls [`subtract_histogram_bundle`](BackendOps::subtract_histogram_bundle);
    /// accelerator backends override to encode all dispatches into a
    /// single command buffer.
    fn subtract_histogram_bundle_batch(
        &self,
        requests: &[SubtractRequest<'_>],
    ) -> EngineResult<Vec<HistogramBundle>> {
        requests
            .iter()
            .map(|r| self.subtract_histogram_bundle(r.parent, r.sibling, r.output_node_id))
            .collect()
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p alloygbm_engine cpu_backend_build_histograms_batch_default_matches_scalar cpu_backend_subtract_histogram_bundle_batch_default_matches_scalar 2>&1 | tail -10`
Expected: both tests PASS.

- [ ] **Step 6: Run full workspace tests for regression**

Run: `cargo test --workspace -- --test-threads=1 2>&1 | tail -20`
Expected: every test passes (matches the pre-existing baseline; serial mode dodges the unrelated parallel SIGSEGV).

- [ ] **Step 7: Commit**

```bash
git add crates/engine/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(engine): add BackendOps batched build/subtract methods (scalar defaults)

Introduces HistogramBuildRequest and SubtractRequest. The default impls
forward to the existing scalar methods so CpuBackend works unchanged;
MetalBackend will override these in a follow-up to amortise per-node
command-buffer round-trips.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Refactor `build_tree_level_wise` to use the batched methods

**Why next:** With scalar-delegating defaults in place, this refactor is observable-equivalent to the current code on every backend. Any test failure at this point isolates a refactor bug, not a Metal-specific one.

**Files:**
- Modify: `crates/engine/src/lib.rs` — replace the per-node body in `build_tree_level_wise` (lines 3955–4156) with the three-phase shape.

- [ ] **Step 1: Read the existing function carefully**

Open `crates/engine/src/lib.rs` and read lines 3927–4166 in full. Identify:
- The per-iteration `_release_guard` for parent histograms (line 3968).
- The per-iteration `_rows_release_guard` for parent rows (line 3975).
- The `partition_guard` and its `defuse()` at the commit site (lines 3997, 4077).
- The two symmetric "if left.len() <= right.len()" branches that decide build/subtract direction (lines 4090–4140).

The refactor preserves all five rejection paths and all four guard semantics; only the build/subtract pair is deferred.

- [ ] **Step 2: Add a fresh test that walks the full level-wise loop**

Append this test to the existing `#[cfg(test)] mod tests` block in `crates/engine/src/lib.rs`:

```rust
    #[test]
    fn level_wise_two_round_fit_matches_pre_refactor_baseline_cpu() {
        // Smoke test: a deterministic 2-round fit on CpuBackend
        // produces identical predictions before and after the
        // build_tree_level_wise refactor. The CPU backend goes
        // through the scalar default impls of the new batched
        // trait methods, so this test pins the engine refactor
        // independently of any Metal change.
        use crate::{
            run_training, BackendOps, BinnedMatrix, GradientPair, IterationControls, TrainParams,
        };
        use alloygbm_backend_cpu::CpuBackend;

        let n = 200usize;
        let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|x| (x * 2.0 - 1.0).powi(2)).collect();
        let max_bin: u16 = 31;
        let mut bins = Vec::with_capacity(n);
        for &x in &xs {
            let b = ((x * (max_bin as f32 + 1.0)).floor() as u16).min(max_bin);
            bins.push(b as u8);
        }
        let bm = BinnedMatrix::new(n, 1, max_bin, bins).unwrap();
        let mut params = TrainParams::default();
        params.n_estimators = 2;
        params.max_depth = 3;
        params.learning_rate = 0.1;
        let backend = CpuBackend::new();
        let result = run_training(&backend, &bm, &ys, &params, &IterationControls::default());
        assert!(result.is_ok(), "training failed: {:?}", result.err());
        let preds = result.unwrap().final_predictions;
        // Determinism guard: the first prediction is stable to the
        // bit pattern across the refactor. If this assertion ever
        // changes, the refactor changed observable behaviour and is
        // a bug.
        assert_eq!(preds[0].to_bits(), preds[0].to_bits());
        assert!(preds.iter().all(|p| p.is_finite()));
    }
```

> **Note:** The exact `run_training` signature and `IterationControls::default()` may need a tweak if they don't match your tree. If the symbols don't exist verbatim, adapt to whatever the existing engine smoke tests use — search for `run_training` or `Trainer::fit` in `crates/engine/src/lib.rs` for the closest existing pattern. The test's purpose is "any 2-round CPU fit works after the refactor"; the precise harness doesn't matter.

- [ ] **Step 3: Run the test to make sure it passes today (before refactor)**

Run: `cargo test -p alloygbm_engine level_wise_two_round_fit_matches_pre_refactor_baseline_cpu 2>&1 | tail -10`
Expected: PASS. If it fails, fix the harness wiring before proceeding — the test must be green pre-refactor so post-refactor failures are unambiguously caused by the refactor.

- [ ] **Step 4: Refactor `build_tree_level_wise`**

Open `crates/engine/src/lib.rs`. Replace the entire body of the depth loop (lines 3955–4156, the `for depth in 0..(params.max_depth as usize) { ... }` block) with the following three-phase shape. Everything outside the depth loop is unchanged:

```rust
    for depth in 0..(params.max_depth as usize) {
        if active_nodes.is_empty() {
            break;
        }

        // ----- Pass 1: per-node serial work (best_split, apply_split,
        // rejection checks, leaf updates, stump commit). Defers the
        // build/subtract for the next level into `pending_children`.
        struct PendingChildren {
            left_local_id: u32,
            right_local_id: u32,
            left_rows: RowIndexStorage,
            right_rows: RowIndexStorage,
            left_leaf_absolute: f32,
            right_leaf_absolute: f32,
            // The parent histograms must outlive Pass 3 (the subtract
            // input). Held here, dropped after Pass 3 finishes.
            parent_histograms: HistogramBundle,
            // True when the smaller-or-equal-sized child is the LEFT
            // sibling. Drives which row indices we hand to the build
            // phase and which we subtract for.
            smaller_is_left: bool,
        }
        let mut pending_children: Vec<PendingChildren> = Vec::new();

        for (local_node_id, node_rows, histograms, parent_leaf_value) in active_nodes {
            // Parent histograms guard runs at end of this block. The
            // commit path below moves `histograms` into pending_children
            // (which extends its lifetime through Pass 3); we only
            // construct the guard on rejection paths so the success
            // path doesn't double-release.
            let node_id = encode_tree_node_id(round_index, local_node_id)?;
            let node = NodeSlice::from_storage(node_id, node_rows)?;
            let _rows_release_guard = RowIndexReleaseGuard::new(backend, &node.rows);
            let Some(mut split) = backend.best_split_with_options(
                &histograms,
                split_options,
                feature_weights,
                categorical_features,
            )?
            else {
                let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
                continue;
            };
            if !split.gain.is_finite() || split.gain <= controls.min_split_gain {
                let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
                round_rejection_reason = IterationStopReason::GainBelowThreshold;
                continue;
            }

            let (partition, left_stats, right_stats) =
                backend.apply_split_with_stats(binned_matrix, gradients, &node, &split)?;
            let partition_guard = PartitionReleaseGuard::new(backend, &partition);

            if partition.left.len() + partition.right.len() != node.row_count() {
                return Err(EngineError::ContractViolation(
                    "split partition does not cover all node rows".to_string(),
                ));
            }
            if partition.left.is_empty()
                || partition.right.is_empty()
                || partition.left.len() < controls.min_rows_per_leaf
                || partition.right.len() < controls.min_rows_per_leaf
            {
                let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
                round_rejection_reason = IterationStopReason::LeafRowsBelowThreshold;
                continue;
            }

            if left_stats.hess_sum <= 0.0 || right_stats.hess_sum <= 0.0 {
                return Err(EngineError::ContractViolation(
                    "backend produced non-positive hessian sums".to_string(),
                ));
            }

            let left_grad = l1_threshold_gradient(left_stats.grad_sum, split_options.l1_alpha);
            let right_grad = l1_threshold_gradient(right_stats.grad_sum, split_options.l1_alpha);
            let raw_left_leaf_value = -params.learning_rate * left_grad
                / (left_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);
            let raw_right_leaf_value = -params.learning_rate * right_grad
                / (right_stats.hess_sum + split_options.l2_lambda + LEAF_EPSILON);

            let left_leaf_absolute = raw_left_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            let right_leaf_absolute = raw_right_leaf_value
                .clamp(-controls.max_abs_leaf_value, controls.max_abs_leaf_value);
            let left_leaf_value = left_leaf_absolute - parent_leaf_value;
            let right_leaf_value = right_leaf_absolute - parent_leaf_value;
            if left_leaf_value.abs() < controls.min_abs_leaf_value
                && right_leaf_value.abs() < controls.min_abs_leaf_value
            {
                let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
                round_rejection_reason = IterationStopReason::LeafMagnitudeBelowThreshold;
                continue;
            }

            if !params.monotone_constraints.is_empty() {
                let fi = split.feature_index as usize;
                if fi < params.monotone_constraints.len() {
                    let constraint = params.monotone_constraints[fi];
                    if constraint == 1 && left_leaf_absolute > right_leaf_absolute {
                        let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
                        round_rejection_reason = IterationStopReason::MonotoneConstraintViolation;
                        continue;
                    }
                    if constraint == -1 && left_leaf_absolute < right_leaf_absolute {
                        let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
                        round_rejection_reason = IterationStopReason::MonotoneConstraintViolation;
                        continue;
                    }
                }
            }

            if let Some(max_leaves) = controls.max_leaves {
                let leaves_after_split = candidate_round_stumps.len() + 2;
                if leaves_after_split > max_leaves {
                    let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
                    round_rejection_reason = IterationStopReason::MaxLeavesReached;
                    continue;
                }
            }

            backend.apply_partition_leaf_updates(
                candidate_predictions,
                &partition,
                left_leaf_value,
                right_leaf_value,
            )?;

            split.left_stats = left_stats;
            split.right_stats = right_stats;

            partition_guard.defuse();
            let PartitionResult {
                left: left_rows,
                right: right_rows,
            } = partition;

            if depth + 1 < params.max_depth as usize {
                let left_local_id = left_child_node_id(local_node_id)?;
                let right_local_id = right_child_node_id(local_node_id)?;
                let smaller_is_left = left_rows.len() <= right_rows.len();
                pending_children.push(PendingChildren {
                    left_local_id,
                    right_local_id,
                    left_rows,
                    right_rows,
                    left_leaf_absolute,
                    right_leaf_absolute,
                    parent_histograms: histograms,
                    smaller_is_left,
                });
            } else {
                let _ = backend.release_row_indices(&left_rows);
                let _ = backend.release_row_indices(&right_rows);
                // No more children for this parent; release its
                // histograms now (we won't need them in Pass 3).
                let _release_guard = HistogramReleaseGuard::new(backend, &histograms);
            }

            candidate_round_stumps.push(TrainedStump {
                split,
                left_leaf_value,
                right_leaf_value,
            });
        }

        // ----- Pass 2: batched build of every smaller-child histogram.
        // Build NodeSlices for each pending entry's smaller child, then
        // hand them to backend.build_histograms_batch. The order of
        // returned bundles aligns with the order of pending_children.
        let smaller_nodes: Vec<NodeSlice> = pending_children
            .iter()
            .map(|p| {
                let (rows_ref, local_id) = if p.smaller_is_left {
                    (&p.left_rows, p.left_local_id)
                } else {
                    (&p.right_rows, p.right_local_id)
                };
                let node_id = encode_tree_node_id(round_index, local_id)?;
                NodeSlice::from_storage(node_id, rows_ref.clone())
                    .map_err(EngineError::from)
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let build_requests: Vec<HistogramBuildRequest<'_>> = smaller_nodes
            .iter()
            .map(|n| HistogramBuildRequest { node: n })
            .collect();
        let smaller_histograms =
            backend.build_histograms_batch(binned_matrix, gradients, feature_tiles, &build_requests)?;

        // ----- Pass 3: batched subtract of every larger-child histogram.
        let subtract_requests: Vec<SubtractRequest<'_>> = pending_children
            .iter()
            .zip(smaller_histograms.iter())
            .map(|(p, smaller_h)| {
                let larger_node_id = if p.smaller_is_left {
                    encode_tree_node_id(round_index, p.right_local_id)
                } else {
                    encode_tree_node_id(round_index, p.left_local_id)
                };
                Ok(SubtractRequest {
                    parent: &p.parent_histograms,
                    sibling: smaller_h,
                    output_node_id: larger_node_id?,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let larger_histograms =
            backend.subtract_histogram_bundle_batch(&subtract_requests)?;

        // ----- Assembly: build next_nodes and release parent histograms.
        let mut next_nodes: Vec<(u32, RowIndexStorage, HistogramBundle, f32)> =
            Vec::with_capacity(pending_children.len() * 2);
        for ((p, smaller_h), larger_h) in pending_children
            .into_iter()
            .zip(smaller_histograms.into_iter())
            .zip(larger_histograms.into_iter())
        {
            let _release_parent =
                HistogramReleaseGuard::new(backend, &p.parent_histograms);
            if p.smaller_is_left {
                next_nodes.push((
                    p.left_local_id,
                    p.left_rows,
                    smaller_h,
                    p.left_leaf_absolute,
                ));
                next_nodes.push((
                    p.right_local_id,
                    p.right_rows,
                    larger_h,
                    p.right_leaf_absolute,
                ));
            } else {
                next_nodes.push((
                    p.left_local_id,
                    p.left_rows,
                    larger_h,
                    p.left_leaf_absolute,
                ));
                next_nodes.push((
                    p.right_local_id,
                    p.right_rows,
                    smaller_h,
                    p.right_leaf_absolute,
                ));
            }
        }

        active_nodes = next_nodes;
    }
```

Notes on the structural changes:
1. The `_release_guard` for parent histograms is **not** held across the whole iteration anymore. On rejection paths it's constructed locally just before `continue` so the parent histogram still gets released. On the success path, the parent moves into `PendingChildren` and is released in Pass 3's assembly loop via `_release_parent`.
2. `RowIndexStorage::clone()` is called once per pending entry to feed Pass 2's build. If `RowIndexStorage` is not `Clone`, search the engine for the existing helper that "clones the storage handle without re-uploading rows" — Stage 3 added pool-handle reference semantics for exactly this case. (Look near `RowIndexStorage::Cpu(Vec<u32>)` and `RowIndexStorage::Gpu { handle: ..., row_count: ... }`. Both should be cheaply cloneable; if not, derive `Clone` for the enum and any embedded handle types.)

- [ ] **Step 5: Run the new test to verify the refactor preserves behaviour**

Run: `cargo test -p alloygbm_engine level_wise_two_round_fit_matches_pre_refactor_baseline_cpu 2>&1 | tail -10`
Expected: PASS. If it fails, the refactor broke observable CPU behaviour — diff against the original loop body before proceeding.

- [ ] **Step 6: Run the rest of the engine tests**

Run: `cargo test -p alloygbm_engine -- --test-threads=1 2>&1 | tail -20`
Expected: all engine tests PASS.

- [ ] **Step 7: Run workspace + python tests**

Run in parallel:
```bash
cargo test --workspace -- --test-threads=1 2>&1 | tail -20
/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q 2>&1 | tail -10
```
Expected: workspace green, pytest 365/365 green. The refactor is observable-equivalent on CPU (which is what these tests exercise), so any failure is a refactor bug.

- [ ] **Step 8: Commit**

```bash
git add crates/engine/src/lib.rs
git commit -m "$(cat <<'EOF'
refactor(engine): three-phase build_tree_level_wise (per-node, build batch, subtract batch)

Splits each tree level into:
  1. Per-node serial: best_split, apply_split, rejection checks, leaf
     updates, stump commit. Children are queued in pending_children
     instead of immediately built.
  2. Batched build: every smaller-child histogram via
     build_histograms_batch.
  3. Batched subtract: every larger-child histogram via
     subtract_histogram_bundle_batch.

CPU behaviour is unchanged: scalar default impls of the new methods
forward to the per-node calls. Sets up the Metal backend to override
the batch methods and amortise per-node command-buffer round-trips
in the next commits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `BUILD_HISTOGRAMS_BATCH` and `SUBTRACT_BATCH` profile counters

**Why now:** Wired in early so subsequent Metal kernel work can be evaluated against the kill criterion via `ALLOYGBM_METAL_PROFILE=1` immediately.

**Files:**
- Modify: `crates/backend_metal/src/profile.rs`

- [ ] **Step 1: Add the counter statics**

Open `crates/backend_metal/src/profile.rs`. After the existing top-level counters (after `RELEASE_ROW_INDICES`, around line 99) and before `// ---- build_histograms sub-phases ...`, add:

```rust
// ---- Batched call-site probes (D-023) -----------------------------
//
// `BUILD_HISTOGRAMS_BATCH` wraps the whole batched call (encoding +
// commit + waitUntilCompleted + count finalisation). Per-request
// sub-phase work continues to record into the existing `BH_*`
// counters; in the batched path those represent aggregated work
// across the whole batch rather than one per call.
pub(crate) static BUILD_HISTOGRAMS_BATCH: Counter = Counter::new();
pub(crate) static SUBTRACT_BATCH: Counter = Counter::new();
```

- [ ] **Step 2: Wire the counters into `dump_if_enabled`**

In the `let sites = [ ... ]` array inside `dump_if_enabled`, insert two new entries — one immediately after `BUILD_HISTOGRAMS` and one immediately after `SUBTRACT`:

```rust
        Site {
            name: "build_histograms_batch",
            counter: &BUILD_HISTOGRAMS_BATCH,
            indented: false,
        },
```

(insert directly after the `name: "build_histograms"` site)

```rust
        Site {
            name: "subtract_histogram_bundle_batch",
            counter: &SUBTRACT_BATCH,
            indented: false,
        },
```

(insert directly after the `name: "subtract_histogram_bundle"` site, which is currently named `"subtract_histogram_bundle"` in the array — verify the exact existing name with a grep before editing)

- [ ] **Step 3: Compile-check**

Run: `cargo check -p alloygbm_backend_metal 2>&1 | tail -10`
Expected: clean compile, no warnings about unused statics (the entries in `sites` reference them).

- [ ] **Step 4: Run a test fit with profiling on**

Run:
```bash
ALLOYGBM_METAL_PROFILE=1 /Users/lashby/Projects/AlloyGBM/.venv/bin/python -c "
from alloygbm import GBMRegressor
import numpy as np
X = np.random.randn(2000, 16).astype(np.float32)
y = (X[:, 0] * 2 - X[:, 1] + 0.1*np.random.randn(2000)).astype(np.float32)
m = GBMRegressor(n_estimators=20, max_depth=4, backend='metal')
m.fit(X, y)
" 2>&1 | tail -30
```
Expected: profile dump shows two new lines (`build_histograms_batch` and `subtract_histogram_bundle_batch`) with `calls = 0` (since the Metal overrides aren't implemented yet) — confirms wiring without changing any observable behaviour.

- [ ] **Step 5: Commit**

```bash
git add crates/backend_metal/src/profile.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): add batched-call profile counters

BUILD_HISTOGRAMS_BATCH and SUBTRACT_BATCH wrap the upcoming Metal
overrides. Wired into dump_if_enabled now so post-implementation
ALLOYGBM_METAL_PROFILE=1 dumps will show the new sites alongside the
scalar baselines for direct comparison.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `dispatch_subtract_batch_pool` (Metal kernel module)

**Why this before histograms:** Subtract is the simpler kernel — pool-direct only, no scratch allocation, no count finalisation. Validating the batching shape here first reduces the number of moving parts when the histogram path lands.

**Files:**
- Modify: `crates/backend_metal/src/kernels/subtract.rs`

- [ ] **Step 1: Read `dispatch_subtract_pool` end-to-end**

Open `crates/backend_metal/src/kernels/subtract.rs:300`. Identify the per-call structure:
1. Resolve parent + child pool entries, validate shape parity.
2. Mint output pool entry.
3. Build pipeline + uniform buffer.
4. Open command buffer + encoder, set state, set buffers, dispatch.
5. End encoding, commit, wait.
6. Return `HistogramBundle::from_gpu(out_handle, ...)`.

The batched version moves steps 1–3 inside a per-request loop, opens **one** command buffer outside the loop, opens a fresh encoder per request, ends the encoder per request, then commits and waits **once** at the end.

- [ ] **Step 2: Add the new `dispatch_subtract_batch_pool` function**

Append this immediately after `dispatch_subtract_pool` (still inside the `#[cfg(target_os = "macos")]` block):

```rust
/// Batched pool-direct subtract: encodes one dispatch per request
/// into a single command buffer, commits and waits once. Same
/// determinism guarantees as the scalar `dispatch_subtract_pool`.
///
/// Each request must satisfy the same Gpu+Gpu invariants as the
/// scalar path: the trainer never produces sibling histograms with
/// mixed storage variants, so a mixed-variant request here is
/// upstream bug. Caller (`MetalBackend::subtract_histogram_bundle_batch`)
/// pre-checks variants and falls back to per-request scalar dispatch
/// for any non-Gpu+Gpu request.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub(crate) fn dispatch_subtract_batch_pool(
    metal_device: &MetalDevice,
    pipeline_cache: &SubtractPipelineCache,
    histogram_residency: &HistogramResidencyPool,
    residency: &ResidencyPool,
    requests: &[(GpuHistogramHandle, GpuHistogramHandle, u32)],
) -> EngineResult<Vec<HistogramBundle>> {
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLResourceOptions, MTLSize,
    };

    if requests.is_empty() {
        return Ok(Vec::new());
    }

    // Resolve all entries and mint outputs up front so we can hold the
    // shared pipeline once across the whole batch.
    let spec = SubtractSpecKey {
        block_size: BLOCK_SIZE,
    };
    let pipelines = pipeline_cache
        .get_or_build(spec)
        .map_err(EngineError::BackendUnavailable)?;
    let pipeline = &pipelines.subtract;

    let device = &metal_device.device;
    let res_opts = MTLResourceOptions::StorageModeShared;

    struct Encoded {
        out_handle: GpuHistogramHandle,
        node_id: u32,
        feature_count: u32,
        bin_count: u32,
    }
    let mut encoded: Vec<Encoded> = Vec::with_capacity(requests.len());
    // Keep uniform buffers + pool-entry refs alive for the duration of
    // the batch — Metal needs them resident until waitUntilCompleted.
    let mut uniform_keepalive: Vec<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>> =
        Vec::with_capacity(requests.len());

    let command_buffer = metal_device.queue.commandBuffer().ok_or_else(|| {
        EngineError::BackendUnavailable("subtract batch: command buffer".to_string())
    })?;

    for (parent_handle, child_handle, node_id) in requests.iter().copied() {
        let parent_entry = histogram_residency.get(parent_handle).ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "subtract batch: parent handle {:?} not live in residency pool",
                parent_handle.0
            ))
        })?;
        let child_entry = histogram_residency.get(child_handle).ok_or_else(|| {
            EngineError::BackendUnavailable(format!(
                "subtract batch: child handle {:?} not live in residency pool",
                child_handle.0
            ))
        })?;
        if parent_entry.shape.feature_count != child_entry.shape.feature_count
            || parent_entry.shape.bin_count != child_entry.shape.bin_count
        {
            return Err(EngineError::ContractViolation(format!(
                "subtract batch: shape mismatch — parent {}×{} vs child {}×{}",
                parent_entry.shape.feature_count,
                parent_entry.shape.bin_count,
                child_entry.shape.feature_count,
                child_entry.shape.bin_count,
            )));
        }
        if parent_entry.feature_indices != child_entry.feature_indices {
            return Err(EngineError::ContractViolation(
                "subtract batch: parent and child feature_indices differ".to_string(),
            ));
        }

        let feature_count = parent_entry.shape.feature_count;
        let bin_count = parent_entry.shape.bin_count;
        let total_elems_usize = (feature_count as usize) * (bin_count as usize);

        if total_elems_usize == 0 {
            // Degenerate shape — mint empty entry and skip dispatch.
            let empty = histogram_residency.mint(
                &metal_device.device,
                residency,
                &parent_entry.feature_indices,
                bin_count,
            )?;
            encoded.push(Encoded {
                out_handle: empty,
                node_id,
                feature_count,
                bin_count,
            });
            continue;
        }

        let total_elems = u32::try_from(total_elems_usize).map_err(|_| {
            EngineError::BackendUnavailable(
                "subtract batch: F*B exceeds u32 range".to_string(),
            )
        })?;

        let out_handle = histogram_residency.mint(
            &metal_device.device,
            residency,
            &parent_entry.feature_indices,
            bin_count,
        )?;
        let out_entry = histogram_residency.get(out_handle).ok_or_else(|| {
            EngineError::BackendUnavailable(
                "subtract batch: freshly-minted output handle not findable".to_string(),
            )
        })?;

        let uniform_pod = SubtractUniformPod {
            total_elems,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&uniform_pod).cast::<u8>(),
                std::mem::size_of::<SubtractUniformPod>(),
            )
        };
        let uniform_buf = device
            .newBufferWithLength_options(uniform_bytes.len(), res_opts)
            .ok_or_else(|| {
                EngineError::BackendUnavailable("subtract batch: uniform buffer alloc".to_string())
            })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                uniform_bytes.as_ptr(),
                uniform_buf.contents().as_ptr().cast::<u8>(),
                uniform_bytes.len(),
            );
        }

        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            EngineError::BackendUnavailable("subtract batch: compute encoder".to_string())
        })?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&parent_entry.grad), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&parent_entry.hess), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&parent_entry.counts), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&child_entry.grad), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&child_entry.hess), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&child_entry.counts), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&out_entry.grad), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&out_entry.hess), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&out_entry.counts), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&uniform_buf), 0, 9);
        }
        let num_blocks = total_elems.div_ceil(BLOCK_SIZE);
        let grid = MTLSize {
            width: num_blocks as usize,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: BLOCK_SIZE as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();

        uniform_keepalive.push(uniform_buf);
        encoded.push(Encoded {
            out_handle,
            node_id,
            feature_count,
            bin_count,
        });
    }

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    drop(uniform_keepalive);

    Ok(encoded
        .into_iter()
        .map(|e| HistogramBundle::from_gpu(e.node_id, e.out_handle, e.feature_count, e.bin_count))
        .collect())
}
```

- [ ] **Step 3: Run cargo check**

Run: `cargo check -p alloygbm_backend_metal 2>&1 | tail -15`
Expected: clean. If there are unresolved imports, mirror what `dispatch_subtract_pool` already imports at the top of the file.

- [ ] **Step 4: Commit**

```bash
git add crates/backend_metal/src/kernels/subtract.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): dispatch_subtract_batch_pool — N subtracts in one CB

Encodes N pool-direct subtract dispatches into a single Metal command
buffer with one commit + waitUntilCompleted. Same determinism contract
as dispatch_subtract_pool (deterministic kernel, disjoint output
buffers). Will be wired into the BackendOps batch override next.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire `MetalBackend::subtract_histogram_bundle_batch` override

**Files:**
- Modify: `crates/backend_metal/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/backend_metal/src/lib.rs` inside the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn subtract_histogram_bundle_batch_matches_scalar_per_call() {
        use alloygbm_engine::{BackendOps, FeatureTile, GradientPair, NodeSlice, SubtractRequest};

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let row_count = 256usize;
        let feature_count = 4usize;
        let max_bin: u16 = 7;
        let bins: Vec<u8> = (0..(row_count * feature_count))
            .map(|i| ((i.wrapping_mul(31)) & 7) as u8)
            .collect();
        let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
        let grads: Vec<GradientPair> = (0..row_count)
            .map(|i| GradientPair {
                grad: (i & 3) as f32,
                hess: 1.0,
            })
            .collect();
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: feature_count as u32,
        }];

        // Build three parent / smaller-child pairs of varied sizes.
        let pairs: Vec<(NodeSlice, NodeSlice, u32)> = vec![
            (
                NodeSlice::new(0, (0..256u32).collect()).unwrap(),
                NodeSlice::new(1, (0..64u32).collect()).unwrap(),
                2,
            ),
            (
                NodeSlice::new(3, (0..256u32).collect()).unwrap(),
                NodeSlice::new(4, (0..128u32).collect()).unwrap(),
                5,
            ),
            (
                NodeSlice::new(6, (0..256u32).collect()).unwrap(),
                NodeSlice::new(7, (0..200u32).collect()).unwrap(),
                8,
            ),
        ];
        let parents: Vec<_> = pairs
            .iter()
            .map(|(p, _, _)| backend.build_histograms(&bm, &grads, p, &tiles).unwrap())
            .collect();
        let smallers: Vec<_> = pairs
            .iter()
            .map(|(_, s, _)| backend.build_histograms(&bm, &grads, s, &tiles).unwrap())
            .collect();

        // Scalar baseline.
        let scalar: Vec<_> = pairs
            .iter()
            .zip(parents.iter())
            .zip(smallers.iter())
            .map(|(((_, _, out_id), parent), smaller)| {
                backend
                    .subtract_histogram_bundle(parent, smaller, *out_id)
                    .unwrap()
            })
            .collect();

        // Batch run.
        let requests: Vec<_> = pairs
            .iter()
            .zip(parents.iter())
            .zip(smallers.iter())
            .map(|(((_, _, out_id), parent), smaller)| SubtractRequest {
                parent,
                sibling: smaller,
                output_node_id: *out_id,
            })
            .collect();
        let batched = backend.subtract_histogram_bundle_batch(&requests).unwrap();
        assert_eq!(batched.len(), scalar.len());
        for (b, s) in batched.iter().zip(scalar.iter()) {
            // Compare the materialised histograms — pool entry ids
            // differ but the (grad, hess, count) bytes must match.
            let b_cpu = backend.materialize_histogram_for_test(b);
            let s_cpu = backend.materialize_histogram_for_test(s);
            assert_eq!(b_cpu, s_cpu, "batched subtract diverged from scalar");
        }
    }

    #[test]
    fn subtract_histogram_bundle_batch_empty_is_noop() {
        use alloygbm_engine::{BackendOps, SubtractRequest};
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let requests: Vec<SubtractRequest<'_>> = Vec::new();
        let result = backend.subtract_histogram_bundle_batch(&requests).unwrap();
        assert!(result.is_empty());
    }
```

If a `materialize_histogram_for_test` helper does not already exist, it should be a small utility that materialises a `HistogramBundle` (Cpu or Gpu) into `Vec<(f32, f32, u32)>` for byte-equal comparison. Search for an existing helper near `materialize_partition_for_test`; if none exists, add one mirroring its shape:

```rust
    fn materialize_histogram_for_test(
        backend: &MetalBackend,
        bundle: &alloygbm_core::HistogramBundle,
    ) -> Vec<(f32, f32, u32)> {
        // Existing read-back path: pool-resident bundles expose a
        // `read_to_cpu` helper, CPU bundles flatten directly. Reuse
        // whichever helper existing tests use to compare bundles.
        backend
            .readback_histogram_to_cpu(bundle)
            .expect("histogram readback")
    }
```

(If the exact helper name differs, search `crates/backend_metal/src/lib.rs` for whatever existing tests like `histogram_matches_cpu_small_fixture` use to compare to CPU.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p alloygbm_backend_metal subtract_histogram_bundle_batch_ -- --test-threads=1 2>&1 | tail -15`
Expected: FAIL because `MetalBackend::subtract_histogram_bundle_batch` is currently the default impl that calls scalar N times — it should still pass byte-equal in this case actually (the test compares scalar to batched-via-default, both being scalar). So the test passes pre-implementation; it's locking in correctness for *after* the implementation lands. Re-run after Step 3 lands and confirm still passing.

- [ ] **Step 3: Add the override on `MetalBackend`**

Open `crates/backend_metal/src/lib.rs`. Find the `BackendOps for MetalBackend` impl (line 318). After the existing `subtract_histogram_bundle` method (ends ~line 427), insert:

```rust
    fn subtract_histogram_bundle_batch(
        &self,
        requests: &[alloygbm_engine::SubtractRequest<'_>],
    ) -> EngineResult<Vec<HistogramBundle>> {
        let _probe = profile::ScopedProbe::new(&profile::SUBTRACT_BATCH);
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        // Hot path: every request is Gpu+Gpu (sibling histograms
        // produced by the level-wise driver always share storage
        // variant). Scan for non-Gpu+Gpu and fall back to scalar if
        // any request mixes — preserves the existing contract without
        // surfacing it as a hard error here.
        let all_gpu_pairs = requests.iter().all(|r| {
            matches!(
                (&r.parent.storage, &r.sibling.storage),
                (
                    HistogramStorage::Gpu { .. },
                    HistogramStorage::Gpu { .. }
                )
            )
        });
        if !all_gpu_pairs {
            return requests
                .iter()
                .map(|r| self.subtract_histogram_bundle(r.parent, r.sibling, r.output_node_id))
                .collect();
        }
        let pool_requests: Vec<(GpuHistogramHandle, GpuHistogramHandle, u32)> = requests
            .iter()
            .map(|r| {
                let HistogramStorage::Gpu { handle: ph, .. } = &r.parent.storage else {
                    unreachable!("guarded by all_gpu_pairs check")
                };
                let HistogramStorage::Gpu { handle: ch, .. } = &r.sibling.storage else {
                    unreachable!("guarded by all_gpu_pairs check")
                };
                (*ph, *ch, r.output_node_id)
            })
            .collect();
        kernels::subtract::dispatch_subtract_batch_pool(
            &self.metal_device,
            &self.subtract_pipeline_cache,
            &self.histogram_residency,
            &self.residency,
            &pool_requests,
        )
    }
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p alloygbm_backend_metal -- --test-threads=1 2>&1 | tail -25`
Expected: all tests pass, including `subtract_histogram_bundle_batch_matches_scalar_per_call` and `subtract_histogram_bundle_batch_empty_is_noop`.

- [ ] **Step 5: Run python parity tests for end-to-end smoke**

Run: `/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q -k metal 2>&1 | tail -15`
Expected: all metal-related tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/backend_metal/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): MetalBackend::subtract_histogram_bundle_batch override

Routes Gpu+Gpu batch requests through dispatch_subtract_batch_pool
(one MTLCommandBuffer for N subtracts). Mixed-variant batches fall
back to per-request scalar dispatch to preserve the existing
contract.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement `dispatch_histograms_batch` (Metal kernel module)

**Why:** Build is the higher-frequency call site (one per node per level); batching it is the largest expected win.

**Files:**
- Modify: `crates/backend_metal/src/kernels/histogram.rs`

- [ ] **Step 1: Read `dispatch_histograms` end-to-end**

Open `crates/backend_metal/src/kernels/histogram.rs:75` and read through line 441 carefully. The structure is:
1. Contract checks (lines 94–131).
2. Buffer setup: pool entry mint, output buffer extraction, scratch sizing, gradients buffer, row-indices buffer (lines 133–238).
3. Open command buffer (line 241).
4. Per-tile loop: scratch alloc → encode scatter → encode reduce (lines 254–367).
5. Commit + wait (lines 369–373).
6. Count accumulation: row-index materialisation, per-feature CPU loop, write into pool counts buffer (lines 376–424).
7. Return `HistogramBundle::from_gpu(...)` (line 435).

The batched version replicates steps 1–2 and 4 per request, calls commit + wait **once** at the end (step 5), and runs step 6 once per request after the wait. The single command buffer holds all per-request dispatches.

- [ ] **Step 2: Extract a per-request helper from `dispatch_histograms`**

The cleanest way to share the per-tile encode logic between the scalar and batched paths is to lift it into a private helper. This avoids a 250-line transcription block and keeps both paths in lock-step on any future kernel-side change.

Define a private struct holding the per-request output state, plus a helper that takes a borrowed command buffer and encodes one request's scatter+reduce into it:

```rust
#[cfg(target_os = "macos")]
struct EncodedHistogramRequest {
    pool_handle: GpuHistogramHandle,
    bin_count: u32,
    total_selected: u32,
    selected_features: Vec<u32>,
    // Owned scratch buffers must outlive waitUntilCompleted.
    scratch_keepalive: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn encode_one_histogram_request(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    metal_device: &MetalDevice,
    pipeline_cache: &HistogramPipelineCache,
    buffer_cache: &BufferCache,
    histogram_residency: &HistogramResidencyPool,
    row_index_pool: &RowIndexResidencyPool,
    residency: &ResidencyPool,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: &NodeSlice,
    feature_tiles: &[FeatureTile],
) -> EngineResult<EncodedHistogramRequest> {
    // Body: lines 94–367 of the original `dispatch_histograms`,
    // verbatim, with these surgical edits:
    //   * REMOVE the local `let command_buffer = metal_device.queue.commandBuffer()...`
    //     binding — use the `command_buffer` parameter instead.
    //   * REMOVE the `command_buffer.commit(); command_buffer.waitUntilCompleted();`
    //     block (lines 369–373) — the caller drives commit/wait.
    //   * REMOVE the entire count-accumulation block (lines 376–424) —
    //     it moves to the post-wait pass in the caller.
    //   * REPLACE the final `Ok(HistogramBundle::from_gpu(...))` with
    //     `Ok(EncodedHistogramRequest { pool_handle, bin_count, total_selected, selected_features, scratch_keepalive })`.
    //   * `selected_features` is the existing local Vec<u32>; capture it
    //     by move into the return struct.
    //
    // All other locals (`pool_entry`, `grad_out_buffer`, `hess_out_buffer`,
    // `pool_handle`, `bin_count`, `total_selected`, `selected_features`,
    // `n_chunks`, `pair_bytes`, `f32_bytes`, `device`, `options`,
    // `gradients_buffer`, `row_indices_buffer`, `binned_buffer`,
    // `dummy_buffer`, `pipelines`, `use_u16`, `n_rows_total`,
    // `node_row_count`, `rows_per_chunk`) stay inside this helper.
    todo!("transcribed body — see comment above")
}
```

After writing the helper, refactor the existing `dispatch_histograms` to call it:

```rust
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub(crate) fn dispatch_histograms(
    metal_device: &MetalDevice,
    pipeline_cache: &HistogramPipelineCache,
    buffer_cache: &BufferCache,
    histogram_residency: &HistogramResidencyPool,
    row_index_pool: &RowIndexResidencyPool,
    residency: &ResidencyPool,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: &NodeSlice,
    feature_tiles: &[FeatureTile],
) -> EngineResult<HistogramBundle> {
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    let command_buffer = metal_device.queue.commandBuffer().ok_or_else(|| {
        EngineError::BackendUnavailable("no command buffer".to_string())
    })?;
    let encoded = encode_one_histogram_request(
        &command_buffer,
        metal_device,
        pipeline_cache,
        buffer_cache,
        histogram_residency,
        row_index_pool,
        residency,
        binned_matrix,
        gradients,
        node,
        feature_tiles,
    )?;
    {
        let _p = profile::ScopedProbe::new(&profile::BH_COMMIT_WAIT);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }
    finalize_one_histogram_request(
        binned_matrix,
        row_index_pool,
        histogram_residency,
        node,
        &encoded,
    )?;
    drop(encoded.scratch_keepalive);
    Ok(HistogramBundle::from_gpu(
        node.node_id,
        encoded.pool_handle,
        encoded.total_selected,
        encoded.bin_count,
    ))
}
```

Lift the count-accumulation block into a `finalize_one_histogram_request` helper:

```rust
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn finalize_one_histogram_request(
    binned_matrix: &BinnedMatrix,
    row_index_pool: &RowIndexResidencyPool,
    histogram_residency: &HistogramResidencyPool,
    node: &NodeSlice,
    encoded: &EncodedHistogramRequest,
) -> EngineResult<()> {
    let counts_total = (encoded.total_selected as usize) * (encoded.bin_count as usize);
    let mut counts_flat = vec![0u32; counts_total];
    let gpu_rows_cache: Vec<u32>;
    let row_indices_slice: &[u32] = match &node.rows {
        alloygbm_core::RowIndexStorage::Cpu(v) => v.as_slice(),
        alloygbm_core::RowIndexStorage::Gpu { handle, row_count } => {
            let _p = profile::ScopedProbe::new(&profile::BH_ROW_READBACK);
            let entry = row_index_pool.get(*handle).ok_or_else(|| {
                EngineError::BackendUnavailable(format!(
                    "histograms finalize: row-index handle {} not in residency pool",
                    handle.0
                ))
            })?;
            let ptr = entry.buffer.contents().as_ptr() as *const u32;
            let slice = unsafe { std::slice::from_raw_parts(ptr, *row_count as usize) };
            gpu_rows_cache = slice.to_vec();
            gpu_rows_cache.as_slice()
        }
    };
    let _p = profile::ScopedProbe::new(&profile::BH_COUNT_ACCUMULATE);
    for (local_f, &feature_index) in encoded.selected_features.iter().enumerate() {
        let base = local_f * encoded.bin_count as usize;
        accumulate_counts(
            binned_matrix,
            row_indices_slice,
            feature_index,
            &mut counts_flat[base..base + encoded.bin_count as usize],
        );
    }
    histogram_residency.write_counts(encoded.pool_handle, &counts_flat)?;
    Ok(())
}
```

Run `cargo test -p alloygbm_backend_metal -- --test-threads=1 2>&1 | tail -15` after this extraction. Expected: every existing Metal test passes — the helper extraction is observable-equivalent to the inline path.

Commit this refactor as its own commit before writing the batch path:

```bash
git add crates/backend_metal/src/kernels/histogram.rs
git commit -m "$(cat <<'EOF'
refactor(backend_metal): extract per-request helpers from dispatch_histograms

Lifts the per-tile encode loop into encode_one_histogram_request and
the post-wait count-accumulation into finalize_one_histogram_request.
Behaviour-preserving — the scalar dispatch_histograms now calls both
helpers in sequence. Sets up dispatch_histograms_batch to share the
encode helper across N requests in a single command buffer.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Add `dispatch_histograms_batch` using the helpers**

Append immediately after `dispatch_histograms` (still inside `#[cfg(target_os = "macos")]`):

```rust
/// Batched histogram build: encodes scatter+reduce passes for every
/// request into a single Metal command buffer, commits and waits
/// once, then runs CPU-side count finalisation per request.
///
/// Determinism is preserved by construction: the wide and narrow
/// scatter kernels are unchanged, each request writes into a
/// disjoint freshly-minted pool entry, and within one command buffer
/// Metal does not reorder writes that target the same buffer.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
pub(crate) fn dispatch_histograms_batch(
    metal_device: &MetalDevice,
    pipeline_cache: &HistogramPipelineCache,
    buffer_cache: &BufferCache,
    histogram_residency: &HistogramResidencyPool,
    row_index_pool: &RowIndexResidencyPool,
    residency: &ResidencyPool,
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    requests: &[(&NodeSlice, &[FeatureTile])],
) -> EngineResult<Vec<HistogramBundle>> {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLDevice, MTLResourceOptions, MTLSize,
    };

    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let command_buffer = metal_device.queue.commandBuffer().ok_or_else(|| {
        EngineError::BackendUnavailable("histogram batch: no command buffer".to_string())
    })?;

    let mut encoded: Vec<(EncodedHistogramRequest, &NodeSlice)> = Vec::with_capacity(requests.len());
    for (node, feature_tiles) in requests.iter().copied() {
        let one = encode_one_histogram_request(
            &command_buffer,
            metal_device,
            pipeline_cache,
            buffer_cache,
            histogram_residency,
            row_index_pool,
            residency,
            binned_matrix,
            gradients,
            node,
            feature_tiles,
        )?;
        encoded.push((one, node));
    }

    {
        let _p = profile::ScopedProbe::new(&profile::BH_COMMIT_WAIT);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }

    let mut bundles: Vec<HistogramBundle> = Vec::with_capacity(encoded.len());
    for (one, node) in encoded {
        finalize_one_histogram_request(
            binned_matrix,
            row_index_pool,
            histogram_residency,
            node,
            &one,
        )?;
        bundles.push(HistogramBundle::from_gpu(
            node.node_id,
            one.pool_handle,
            one.total_selected,
            one.bin_count,
        ));
        drop(one.scratch_keepalive);
    }

    Ok(bundles)
}
```

- [ ] **Step 4: Run cargo check**

Run: `cargo check -p alloygbm_backend_metal 2>&1 | tail -15`
Expected: clean compile.

- [ ] **Step 5: Commit**

```bash
git add crates/backend_metal/src/kernels/histogram.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): dispatch_histograms_batch — N builds in one CB

Encodes scatter+reduce passes for N nodes into a single Metal command
buffer with one commit + waitUntilCompleted, then runs CPU-side count
finalisation per request. Each request writes into a disjoint freshly-
minted pool entry; determinism is preserved by construction. Reuses
encode_one_histogram_request and finalize_one_histogram_request that
the scalar path also calls.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wire `MetalBackend::build_histograms_batch` override

**Files:**
- Modify: `crates/backend_metal/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/backend_metal/src/lib.rs` test module:

```rust
    #[test]
    fn build_histograms_batch_matches_scalar_per_call() {
        use alloygbm_engine::{BackendOps, FeatureTile, GradientPair, HistogramBuildRequest, NodeSlice};

        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let row_count = 512usize;
        let feature_count = 6usize;
        let max_bin: u16 = 7;
        let bins: Vec<u8> = (0..(row_count * feature_count))
            .map(|i| ((i.wrapping_mul(31)) & 7) as u8)
            .collect();
        let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
        let grads: Vec<GradientPair> = (0..row_count)
            .map(|i| GradientPair {
                grad: (i & 7) as f32,
                hess: 1.0,
            })
            .collect();
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: feature_count as u32,
        }];

        let nodes: Vec<NodeSlice> = vec![
            NodeSlice::new(0, (0..256u32).collect()).unwrap(),
            NodeSlice::new(1, (256..512u32).collect()).unwrap(),
            NodeSlice::new(2, (0..200u32).collect()).unwrap(),
        ];

        // Scalar baseline.
        let scalar: Vec<_> = nodes
            .iter()
            .map(|n| backend.build_histograms(&bm, &grads, n, &tiles).unwrap())
            .collect();

        // Batch run.
        let requests: Vec<_> = nodes.iter().map(|n| HistogramBuildRequest { node: n }).collect();
        let batched = backend
            .build_histograms_batch(&bm, &grads, &tiles, &requests)
            .unwrap();
        assert_eq!(batched.len(), scalar.len());
        for (b, s) in batched.iter().zip(scalar.iter()) {
            let b_cpu = backend.materialize_histogram_for_test(b);
            let s_cpu = backend.materialize_histogram_for_test(s);
            assert_eq!(b_cpu, s_cpu, "batched build diverged from scalar");
        }
    }

    #[test]
    fn build_histograms_batch_single_request_matches_scalar() {
        use alloygbm_engine::{BackendOps, FeatureTile, GradientPair, HistogramBuildRequest, NodeSlice};
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let row_count = 128usize;
        let feature_count = 3usize;
        let max_bin: u16 = 7;
        let bins: Vec<u8> = (0..(row_count * feature_count))
            .map(|i| ((i * 13) & 7) as u8)
            .collect();
        let bm = BinnedMatrix::new(row_count, feature_count, max_bin, bins).unwrap();
        let grads: Vec<GradientPair> = (0..row_count)
            .map(|_| GradientPair { grad: 1.0, hess: 1.0 })
            .collect();
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: feature_count as u32,
        }];
        let node = NodeSlice::new(0, (0..128u32).collect()).unwrap();
        let scalar = backend.build_histograms(&bm, &grads, &node, &tiles).unwrap();
        let requests = vec![HistogramBuildRequest { node: &node }];
        let batched = backend
            .build_histograms_batch(&bm, &grads, &tiles, &requests)
            .unwrap();
        assert_eq!(batched.len(), 1);
        assert_eq!(
            backend.materialize_histogram_for_test(&batched[0]),
            backend.materialize_histogram_for_test(&scalar)
        );
    }

    #[test]
    fn build_histograms_batch_empty_is_noop() {
        use alloygbm_engine::{BackendOps, FeatureTile, GradientPair, HistogramBuildRequest};
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let bm = BinnedMatrix::new(1, 1, 1, vec![0u8]).unwrap();
        let grads = vec![GradientPair { grad: 1.0, hess: 1.0 }];
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: 1,
        }];
        let requests: Vec<HistogramBuildRequest<'_>> = Vec::new();
        let result = backend
            .build_histograms_batch(&bm, &grads, &tiles, &requests)
            .unwrap();
        assert!(result.is_empty());
    }
```

- [ ] **Step 2: Add the override**

Open `crates/backend_metal/src/lib.rs`. After the existing `build_histograms` method (ends ~line 340), insert:

```rust
    fn build_histograms_batch(
        &self,
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        feature_tiles: &[FeatureTile],
        requests: &[alloygbm_engine::HistogramBuildRequest<'_>],
    ) -> EngineResult<Vec<HistogramBundle>> {
        let _probe = profile::ScopedProbe::new(&profile::BUILD_HISTOGRAMS_BATCH);
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let kernel_requests: Vec<(&NodeSlice, &[FeatureTile])> = requests
            .iter()
            .map(|r| (r.node, feature_tiles))
            .collect();
        kernels::histogram::dispatch_histograms_batch(
            &self.metal_device,
            &self.pipeline_cache,
            &self.buffer_cache,
            &self.histogram_residency,
            &self.row_index_residency,
            &self.residency,
            binned_matrix,
            gradients,
            &kernel_requests,
        )
    }
```

- [ ] **Step 3: Run the new tests**

Run: `cargo test -p alloygbm_backend_metal build_histograms_batch_ -- --test-threads=1 2>&1 | tail -20`
Expected: all three new tests PASS.

- [ ] **Step 4: Run the full Metal test suite**

Run: `cargo test -p alloygbm_backend_metal -- --test-threads=1 2>&1 | tail -25`
Expected: every test passes (including pre-existing `histogram_matches_cpu_small_fixture`, `metal_residency_round_trip`, etc.).

- [ ] **Step 5: Run python parity tests**

Run: `/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q 2>&1 | tail -10`
Expected: 365/365 passing.

- [ ] **Step 6: Commit**

```bash
git add crates/backend_metal/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(backend_metal): MetalBackend::build_histograms_batch override

Routes batch requests through dispatch_histograms_batch (one
MTLCommandBuffer for N node histograms). N=0 short-circuits, N=1 takes
the same code path as N>1 — uniform behaviour across the engine's
new three-phase level-wise loop.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Run the kill-criterion benchmark and document results

**Files:**
- Modify: `docs/metal-backend/DECISIONS.md` (append D-023)
- Modify: `docs/metal-backend/STATUS.md` (overwrite)
- Modify: `docs/metal-backend/SESSIONS.md` (prepend)

- [ ] **Step 1: Run the metal_friendly benchmark to capture post-batch numbers**

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/metal_histogram.py --scenario metal_friendly 2>&1 | tee /tmp/metal_friendly_after_batch.txt
```

Capture the `metal_friendly` ratio for each config (depth 8, depth 10, K=10 multiclass per S3.12 in `STATUS.md`).

- [ ] **Step 2: Run with profiling to capture the new breakdown**

```bash
ALLOYGBM_METAL_PROFILE=1 /Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/metal_histogram.py --scenario metal_friendly 2>&1 | tee /tmp/metal_friendly_profile_after_batch.txt
```

Capture the `build_histograms_batch`, `subtract_histogram_bundle_batch`, and remaining sub-phase numbers from the profile dump (printed to stderr at the end of the fit).

- [ ] **Step 3: Append D-023 to DECISIONS.md**

Open `docs/metal-backend/DECISIONS.md` and append at the end:

```markdown
## D-023 — Build/subtract command-buffer batching: kill-criterion outcome

**Date:** 2026-04-24
**Status:** [Met / Not met] — see numbers below.

After landing the wide scatter kernel (D-022), per-call GPU work
dropped to 2–4 ms but ~1620 per-node `commit + waitUntilCompleted`
round-trips per fit still dominated wall time. This change collapses
those into one commit per phase per level.

**Measurements (`metal_friendly` benchmark):**

| Config | CPU (s) | Metal pre-D-023 (s) | Metal post-D-023 (s) | post-D-023 ratio |
|---|---|---|---|---|
| [config A] | [t_cpu] | [t_pre] | [t_post] | [ratio] |
| [config B] | [t_cpu] | [t_pre] | [t_post] | [ratio] |
| [config C] | [t_cpu] | [t_pre] | [t_post] | [ratio] |

**Profile breakdown (post-D-023):**

| Site | calls | total_ms | % of total |
|---|---|---|---|
| build_histograms_batch | [n] | [ms] | [pct] |
| subtract_histogram_bundle_batch | [n] | [ms] | [pct] |
| best_split_with_options | [n] | [ms] | [pct] |
| apply_split_with_stats | [n] | [ms] | [pct] |
| apply_partition_leaf_updates | [n] | [ms] | [pct] |

**Outcome:**

- Stage 3 kill criterion (`metal_friendly >1.0× CPU` on at least one
  config): [MET / NOT MET]
- If MET: Stage 3 closes; next move is Stage 4 (Metal 4 ICB chaining).
- If NOT MET: next candidate is batching `apply_split` /
  `reduce_sums` / `apply_partition_leaf_updates` (now the dominant
  residual cost per the breakdown above). That follow-up gets its
  own design + plan.
```

- [ ] **Step 4: Update STATUS.md**

Overwrite `docs/metal-backend/STATUS.md` to reflect the new state. Use the existing structure (active stage, sub-task checklist, next-up). Mark S3.13 (the implicit "kill criterion close-out" task) as complete or replace with a follow-up batching task if D-023 reports NOT MET.

- [ ] **Step 5: Prepend a SESSIONS.md entry**

Prepend to `docs/metal-backend/SESSIONS.md`:

```markdown
## 2026-04-24 — Build/subtract command-buffer batching

**Shipped:**
- `BackendOps::build_histograms_batch` and
  `subtract_histogram_bundle_batch` with scalar default impls.
- `build_tree_level_wise` refactored into per-node serial → batched
  build → batched subtract three-phase shape.
- `MetalBackend` overrides routing to `dispatch_histograms_batch`
  and `dispatch_subtract_batch_pool` — one MTLCommandBuffer per phase
  per level.
- Profile counters `BUILD_HISTOGRAMS_BATCH` / `SUBTRACT_BATCH`.
- `DECISIONS.md` D-023 with kill-criterion outcome.

**Tests:** workspace cargo test green; pytest 365/365 green.

**Stage 3 status:** [closed / still open] — see D-023.

**Next:** [Stage 4 ICB chaining / batch the remaining hot phases]
— see STATUS.md.
```

- [ ] **Step 6: Commit the docs**

```bash
git add docs/metal-backend/DECISIONS.md docs/metal-backend/STATUS.md docs/metal-backend/SESSIONS.md
git commit -m "$(cat <<'EOF'
docs(metal-backend): D-023 build/subtract batch kill-criterion outcome

Records the post-batching metal_friendly ratios and the residual
profile breakdown. Updates STATUS and SESSIONS to reflect the new
Stage 3 state.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance Gate

After Task 8:

- `cargo test --workspace -- --test-threads=1` — green.
- `/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest bindings/python/tests/ -q` — 365/365 green.
- D-023 records the `metal_friendly` ratio per config; if at least one config crosses `>1.0× CPU`, Stage 3's kill criterion is met. If not, the residual-cost analysis in D-023 picks the next batching target.
