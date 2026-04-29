# Metal Backend — Bug / Issue Tracker

Log issues as you discover them. Don't let anything slip into "I'll fix it later in my head."

## Severity legend
- **blocker** — must fix before the current sub-task is "done"
- **high** — must fix before the current stage is done
- **medium** — must fix before the next release
- **low** — nice-to-have / polish

## Status legend
- **open** — not started
- **in-progress** — actively being fixed
- **fixed** — fix is committed, awaiting verification
- **verified** — fixed and confirmed (include commit SHA)
- **wontfix** — explicit decision not to fix (note rationale in `DECISIONS.md`)

---

## Open

| ID | Severity | Opened | Stage | Symptom | Repro | Owner | Notes |
|----|----------|--------|-------|---------|-------|-------|-------|
| B-001 | blocker | 2026-04-21 | S3.7c | `partition.metal` fails library compile via Python-ext (release build) with "program scope variable must reside in constant address space" on the 4 file-scope `threadgroup` arrays (lines 166-169). `cargo test` (debug + release) compiles the same source cleanly, so the bug is context-dependent — likely differs on MSL language-version default between test-harness vs. dylib-loaded Metal context. MetalBackend init falls back to CPU; every Stage-3 Python test that asserts `trained_device == "metal"` fails. | `maturin develop --release --manifest-path bindings/python/Cargo.toml && .venv/bin/python -m pytest bindings/python/tests/test_metal_backend.py::MetalRegressionTests::test_metal_regression_records_trained_device -q` | Claude | Pre-existing on HEAD `4cf7d06`; introduced in `281ae1a` when partition.metal shipped. Fix: move the 4 file-scope `threadgroup` arrays into each kernel body (MSL spec is explicit — `threadgroup` is only valid as a local or parameter). Declaring per-kernel is fine because threadgroup memory isn't shared across kernels regardless. |
| B-002 | medium | 2026-04-25 | S3-killcrit | Mid-batch error in `dispatch_histograms_batch` and `dispatch_subtract_batch_pool` orphans already-minted `HistogramResidencyPool` entries until backend teardown. If `encode_one_histogram_request` (or the subtract encode loop) returns `Err` partway through a batch, earlier requests' freshly-minted pool handles are dropped on the floor — the entries stay registered in the pool but are never finalized or released. Engine treats the whole batch as failed and won't use returned bundles, so this is not a correctness bug. But pool capacity is bounded; on a constrained device or under error-injection, repeated orphans could compound into "pool full" failures on subsequent calls. | n/a (defensive — no organic repro yet). | Claude | Flagged by Task 6 code review (M-1). Consistent with the existing scalar `dispatch_subtract_batch_pool` orphan behaviour, so this is not a regression from Approach A. Fix when needed: collect minted handles in a guard struct that releases on drop, defuse on success. Defer until kill-criterion benchmark either lands Stage 3 or fails (Task 8). |
| B-004 | low | 2026-04-28 | S4b | `icb_partition` casts `uint` child-node ID to `uint16_t`. At depth ≥ 15 rightmost child (2×(2^14−2)+2 = 32766 < 65535, still safe; but depth ≥ 15 gives 2×(2^15−2)+2 = 65534 — close to limit, depth 16 overflows). Default depth=8 safe. Fix: add `params.max_depth <= 14` to `try_build_tree_level_wise` eligibility gate in Task 8. | Set max_depth=16 and inspect row_node_id readback. | Claude | **Fix in Task 8 eligibility gate.** |
| B-003 | low | 2026-04-25 | S3-killcrit | All three new `build_histograms_batch_*` tests in `crates/backend_metal/src/lib.rs` use `max_bin = 7`, which lands every dispatch on the same kernel path. The other path (different per-simdgroup threadgroup layout) is not exercised by the batched determinism assertion. The same gap exists for the pre-existing scalar `build_histograms` tests, so this is not a Task 7 regression — but the kill-criterion benchmark (Task 8) is the only place both paths are organically exercised. | n/a — pre-existing coverage gap. | Claude | Flagged by Task 7 code review (I-1). Cheap to fix: add a fourth `build_histograms_batch` test with `max_bin = 1023` (or whatever lands on the alternate path) using `u16` binned storage. Low priority because Task 8 benchmark will hit both paths on real datasets anyway. |

---

## Resolved

| ID | Severity | Opened | Resolved | Stage | Summary | Fix / Commit |
|----|----------|--------|----------|-------|---------|--------------|
| B-003 (ICB PSO) | blocker | 2026-04-29 | 2026-04-29 | S4b | `IcbPipelineCache` created PSOs via `newComputePipelineStateWithFunction_error`, which does NOT set `supportIndirectCommandBuffers=true`. Metal requires this for any PSO used inside an ICB command; without it, `setComputePipelineState` causes SIGSEGV. Fixed by using `MTLComputePipelineDescriptor` with `setSupportIndirectCommandBuffers(true)` for all three ICB kernels. | `de2b9d3` |
| B-004 (histogram layout) | blocker | 2026-04-29 | 2026-04-29 | S4b | Histogram buffer was sized for one level's worth of nodes, all levels sharing offset 0. Level N accumulated on top of level N-1 data, producing completely wrong histograms. Fixed: buffer sized for all levels combined (`(2^depth - 1) × F × B × 2` floats). Each level L binds the histogram at byte offset `(2^L - 1) × F × B × 2 × 4`, giving non-overlapping per-level regions. Single CPU zero before GPU commit covers all levels. | `de2b9d3` |
| B-005 (bin layout) | blocker | 2026-04-29 | 2026-04-29 | S4b | `icb_histogram` and `icb_partition` kernels used row-major bin access `bin_data[gid * feature_count + f]`. `BinnedMatrix::bins_col_adaptive` is column-major (`bins_col[feature * row_count + row]`). Fixed to `bin_data[f * c.row_count + gid]` in both kernels. | `de2b9d3` |
| B-006 (last-level leaf values) | blocker | 2026-04-29 | 2026-04-29 | S4b | Partition kernel was a no-op at max depth (zero-width dispatch), so rows stayed at level-(depth-1) split nodes. `update_candidate_predictions` then used `grad_total / hess_total` (whole-node average) instead of the correct left or right child value, causing prediction error proportional to the split quality. Fixed by passing column-major bin data to the readback function and determining left/right child assignment on the CPU using the row's bin value, the split threshold, and the NaN direction flag. | `de2b9d3` |
