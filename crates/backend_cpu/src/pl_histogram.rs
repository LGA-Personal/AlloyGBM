//! Piecewise-linear histogram construction for the CPU backend.
//!
//! Builds [`LinearHistogramBundle`] statistics — `(Xᵀg, XᵀHX)` per bin per
//! split feature — needed for the closed-form ridge-regression leaf-weight
//! solve `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.
//!
//! The SIMD path for standard (scalar-leaf) histograms is left completely
//! untouched.  This module is a parallel addition, not a modification.

use alloygbm_core::simd::f32x8;
use alloygbm_core::{
    BinnedMatrix, FeatureTile, GradientPair, LinearFeatureHistogram, LinearHistogramBin,
    LinearHistogramBundle, MAX_PL_REGRESSORS, NodeSlice,
};
use alloygbm_engine::{EngineError, EngineResult};

/// Build a [`LinearHistogramBundle`] for a single tree node.
///
/// # Arguments
/// * `binned_matrix` — pre-binned feature matrix
/// * `gradients` — gradient pairs `(g_i, h_i)` for all samples
/// * `node` — the node slice (which rows belong to this node)
/// * `feature_tiles` — which features to scan for splits
/// * `regressor_features` — indices of features used as linear regressors
/// * `raw_feature_values` — **row-major** float matrix: `[row * feature_count + feat]`
/// * `row_count` / `feature_count` — dimensions of `raw_feature_values`
#[allow(clippy::too_many_arguments)]
pub fn build_linear_histograms_cpu(
    binned_matrix: &BinnedMatrix,
    gradients: &[GradientPair],
    node: &NodeSlice,
    feature_tiles: &[FeatureTile],
    regressor_features: &[u32],
    raw_feature_values: &[f32],
    row_count: usize,
    feature_count: usize,
) -> EngineResult<LinearHistogramBundle> {
    let d = regressor_features.len();
    if d > MAX_PL_REGRESSORS {
        return Err(EngineError::ContractViolation(format!(
            "regressor_features.len() = {d} exceeds MAX_PL_REGRESSORS = {MAX_PL_REGRESSORS}"
        )));
    }
    if d == 0 {
        return Err(EngineError::ContractViolation(
            "regressor_features must be non-empty for linear histograms".to_string(),
        ));
    }

    let bin_count = binned_matrix.max_bin as usize + 2; // include missing bin

    // Collect all split feature indices from the tiles.
    let split_features: Vec<u32> = feature_tiles
        .iter()
        .flat_map(|tile| tile.start_feature..tile.end_feature)
        .collect();

    // Build one feature histogram per split feature.
    //
    // Inner loop is SIMD-vectorised via `wide::f32x8`: per row we compute
    // `b.xtg += g * x` as one `f32x8` op, then `b.xt_hx += h * x ⊗ x` as 8
    // `f32x8` ops (one per row of the 8×8 matrix).  `x_arr` is zero-padded
    // when `d < MAX_PL_REGRESSORS`, so the unused matrix slots multiply by
    // zero and stay zero — which is correct for the closed-form ridge solve.
    //
    // The outer product writes BOTH triangles of the symmetric matrix.  The
    // Cholesky in `compute_pl_gain_one_side` reads only the upper triangle
    // (and mirrors to fill the lower), so the lower-triangle SIMD writes are
    // mathematically harmless.  `subtract_linear_histogram_bundle` likewise
    // operates on all 64 entries to stay consistent.
    let feature_histograms: Vec<LinearFeatureHistogram> = split_features
        .iter()
        .map(|&split_feat_idx| {
            let mut bins = vec![LinearHistogramBin::default(); bin_count];

            for &row_idx in &node.row_indices {
                let row_idx = row_idx as usize;
                let bin = binned_matrix
                    .row_bin(row_idx * binned_matrix.feature_count + split_feat_idx as usize)
                    as usize;
                // Safety: bin_count includes the missing-bin sentinel.
                let b = &mut bins[bin];

                let gp = &gradients[row_idx];
                let g = gp.grad;
                let h = gp.hess;

                b.grad_sum += g;
                b.hess_sum += h;
                b.count += 1;

                // Pre-load the d regressor values into a zero-padded f32x8.
                // Slots beyond `d` stay zero; multiplications against them
                // contribute nothing to the active matrix block.
                let mut x_arr = [0.0_f32; MAX_PL_REGRESSORS];
                for (slot, &feat_raw) in x_arr.iter_mut().zip(regressor_features.iter()).take(d) {
                    let feat = feat_raw as usize;
                    *slot = if feat < feature_count && row_idx < row_count {
                        raw_feature_values[row_idx * feature_count + feat]
                    } else {
                        0.0
                    };
                }
                let x_v = f32x8::from(x_arr);

                // Xᵀg: b.xtg += g * x   (1 SIMD op).
                let xtg_v = f32x8::from(b.xtg) + f32x8::splat(g) * x_v;
                b.xtg = xtg_v.to_array();

                // XᵀHX: b.xt_hx += h * x ⊗ x   (8 SIMD ops, full outer product).
                // Layout is stride-8 row-major: row j occupies xt_hx[j*8..(j+1)*8].
                for (j, &x_j) in x_arr.iter().enumerate() {
                    let row_v = f32x8::splat(h * x_j) * x_v;
                    let base = j * MAX_PL_REGRESSORS;
                    let cur_arr: [f32; 8] =
                        b.xt_hx[base..base + 8].try_into().expect("8-entry slice");
                    let cur_v = f32x8::from(cur_arr);
                    let new_arr = (cur_v + row_v).to_array();
                    b.xt_hx[base..base + 8].copy_from_slice(&new_arr);
                }
            }

            // Replace NaN/Inf in accumulated values with 0 (defensive).
            for bin in &mut bins {
                sanitize_linear_bin(bin);
            }

            LinearFeatureHistogram {
                feature_index: split_feat_idx,
                bins,
            }
        })
        .collect();

    Ok(LinearHistogramBundle {
        node_id: node.node_id,
        num_regressors: d,
        regressor_features: regressor_features.to_vec(),
        feature_histograms,
    })
}

/// Zero out any non-finite entries in a bin's linear statistics.
///
/// Operates on all `MAX_PL_REGRESSORS` entries of `xtg` and all
/// `MAX_PL_MATRIX_ENTRIES` entries of `xt_hx`: with the SIMD-vectorised
/// histogram build, both triangles of `xt_hx` may be populated, and unused
/// slots are zero (already finite).  Iterating the full storage is
/// sufficient and matches the rest of the SIMD-friendly code paths.
#[inline]
fn sanitize_linear_bin(bin: &mut LinearHistogramBin) {
    if !bin.grad_sum.is_finite() {
        bin.grad_sum = 0.0;
    }
    if !bin.hess_sum.is_finite() {
        bin.hess_sum = 0.0;
    }
    for slot in &mut bin.xtg {
        if !slot.is_finite() {
            *slot = 0.0;
        }
    }
    for slot in &mut bin.xt_hx {
        if !slot.is_finite() {
            *slot = 0.0;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloygbm_core::subtract_linear_histogram_bundle;
    use alloygbm_core::{BinnedMatrix, FeatureTile, GradientPair, NodeSlice};

    /// Build a minimal BinnedMatrix for tests: 2 features, 4 rows, 3 bins each.
    fn fixture_binned() -> BinnedMatrix {
        // Bins: feature 0: [0,1,2,2], feature 1: [0,0,1,2]
        BinnedMatrix::new(4, 2, 2, vec![0u8, 0, 1, 0, 2, 1, 2, 2]).expect("valid")
    }

    fn fixture_gradients() -> Vec<GradientPair> {
        vec![
            GradientPair {
                grad: 1.0,
                hess: 1.0,
            },
            GradientPair {
                grad: 2.0,
                hess: 1.0,
            },
            GradientPair {
                grad: -1.0,
                hess: 2.0,
            },
            GradientPair {
                grad: 0.5,
                hess: 0.5,
            },
        ]
    }

    fn fixture_raw_values() -> Vec<f32> {
        // 4 rows × 2 features, row-major
        // row 0: x0=1.0, x1=2.0
        // row 1: x0=3.0, x1=4.0
        // row 2: x0=0.5, x1=1.5
        // row 3: x0=2.0, x1=0.5
        vec![1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.0, 0.5]
    }

    fn all_rows_node() -> NodeSlice {
        NodeSlice::new(1, vec![0, 1, 2, 3]).expect("valid")
    }

    fn all_feature_tile() -> Vec<FeatureTile> {
        vec![FeatureTile {
            start_feature: 0,
            end_feature: 2,
        }]
    }

    #[test]
    fn histogram_matches_brute_force() {
        let binned = fixture_binned();
        let grads = fixture_gradients();
        let raw = fixture_raw_values();
        let node = all_rows_node();
        let tiles = all_feature_tile();

        let bundle = build_linear_histograms_cpu(
            &binned,
            &grads,
            &node,
            &tiles,
            &[0], // single regressor: feature 0
            &raw,
            4,
            2,
        )
        .expect("build should succeed");

        assert_eq!(bundle.num_regressors, 1);
        assert_eq!(bundle.regressor_features, vec![0u32]);

        // Split feature 0 histogram.
        let fh = &bundle.feature_histograms[0];
        assert_eq!(fh.feature_index, 0);

        // Bin 0: row 0 (g=1.0, h=1.0, x0=1.0)
        let b0 = &fh.bins[0];
        assert!((b0.grad_sum - 1.0).abs() < 1e-6, "bin0 grad_sum");
        assert!((b0.hess_sum - 1.0).abs() < 1e-6, "bin0 hess_sum");
        assert_eq!(b0.count, 1);
        assert!((b0.xtg[0] - 1.0).abs() < 1e-6, "bin0 xtg"); // 1.0 * 1.0
        assert!((b0.xt_hx[0] - 1.0).abs() < 1e-6, "bin0 xt_hx"); // 1.0 * 1.0 * 1.0

        // Bin 1: row 1 (g=2.0, h=1.0, x0=3.0)
        let b1 = &fh.bins[1];
        assert!((b1.grad_sum - 2.0).abs() < 1e-6, "bin1 grad_sum");
        assert!((b1.xtg[0] - 6.0).abs() < 1e-6, "bin1 xtg"); // 2.0 * 3.0

        // Bin 2: rows 2 and 3 (g=−1.0+0.5=−0.5, x0=0.5 and 2.0)
        let b2 = &fh.bins[2];
        assert!((b2.grad_sum - (-1.0 + 0.5)).abs() < 1e-6, "bin2 grad_sum");
        // xtg = g2*x0_2 + g3*x0_3 = -1.0*0.5 + 0.5*2.0 = -0.5 + 1.0 = 0.5
        assert!((b2.xtg[0] - 0.5).abs() < 1e-6, "bin2 xtg");
    }

    #[test]
    fn direct_partition_leaf_pair_matches_histogram_leaf_pair() {
        let binned = fixture_binned();
        let grads = fixture_gradients();
        let raw = fixture_raw_values();
        let node = all_rows_node();
        let tiles = all_feature_tile();
        let missing_bin_index = binned.missing_bin() as usize;

        let bundle = build_linear_histograms_cpu(&binned, &grads, &node, &tiles, &[0], &raw, 4, 2)
            .expect("linear histogram build succeeds");
        let histogram_pair = crate::pl::leaf_linear_stats_for_split(
            &bundle.feature_histograms[0],
            0,
            missing_bin_index,
            true,
        );
        let histogram_left = crate::pl::solve_pl_leaf(
            &histogram_pair.0,
            &histogram_pair.1,
            histogram_pair.2,
            histogram_pair.3,
            0.05,
            0.01,
            &[0],
        );
        let histogram_right = crate::pl::solve_pl_leaf(
            &histogram_pair.4,
            &histogram_pair.5,
            histogram_pair.6,
            histogram_pair.7,
            0.05,
            0.01,
            &[0],
        );

        let direct = crate::pl::solve_pl_leaf_pair_from_partitions(
            &binned,
            &grads,
            &raw,
            2,
            0,
            0,
            true,
            &[0],
            &[0],
            &[1, 2, 3],
            0.05,
            0.01,
        )
        .expect("direct partition solve succeeds");

        assert!((histogram_left.intercept - direct.0.intercept).abs() < 1e-6);
        assert!((histogram_right.intercept - direct.1.intercept).abs() < 1e-6);
        assert!((histogram_left.weights[0] - direct.0.weights[0]).abs() < 1e-6);
        assert!((histogram_right.weights[0] - direct.1.weights[0]).abs() < 1e-6);
    }

    #[test]
    fn direct_partition_leaf_pair_matches_histogram_with_nan_regressor_bin() {
        let binned = fixture_binned();
        let grads = fixture_gradients();
        let mut raw = fixture_raw_values();
        // Row 2 is in split-feature-0 bin 2 with row 3. A NaN in regressor
        // feature 1 makes the old histogram path sanitize bin 2's affected
        // linear-stat slots after aggregation, so row 3's same-bin slot
        // contribution is discarded too. The direct partition solve preserves
        // that bin-level behavior for compatibility.
        raw[2 * 2 + 1] = f32::NAN;
        let node = all_rows_node();
        let tiles = all_feature_tile();
        let missing_bin_index = binned.missing_bin() as usize;

        let bundle = build_linear_histograms_cpu(&binned, &grads, &node, &tiles, &[1], &raw, 4, 2)
            .expect("linear histogram build succeeds");
        let histogram_pair = crate::pl::leaf_linear_stats_for_split(
            &bundle.feature_histograms[0],
            0,
            missing_bin_index,
            true,
        );
        let histogram_left = crate::pl::solve_pl_leaf(
            &histogram_pair.0,
            &histogram_pair.1,
            histogram_pair.2,
            histogram_pair.3,
            0.05,
            0.01,
            &[1],
        );
        let histogram_right = crate::pl::solve_pl_leaf(
            &histogram_pair.4,
            &histogram_pair.5,
            histogram_pair.6,
            histogram_pair.7,
            0.05,
            0.01,
            &[1],
        );

        let direct = crate::pl::solve_pl_leaf_pair_from_partitions(
            &binned,
            &grads,
            &raw,
            2,
            0,
            0,
            true,
            &[1],
            &[0],
            &[1, 2, 3],
            0.05,
            0.01,
        )
        .expect("direct partition solve succeeds");

        assert!((histogram_left.intercept - direct.0.intercept).abs() < 1e-6);
        assert!((histogram_right.intercept - direct.1.intercept).abs() < 1e-6);
        assert!((histogram_left.weights[0] - direct.0.weights[0]).abs() < 1e-6);
        assert!((histogram_right.weights[0] - direct.1.weights[0]).abs() < 1e-6);
    }

    #[test]
    fn subtraction_trick_recovers_parent() {
        let binned = fixture_binned();
        let grads = fixture_gradients();
        let raw = fixture_raw_values();
        let tiles = all_feature_tile();

        // Parent node = rows 0..3
        let parent_node = all_rows_node();
        let parent_bundle =
            build_linear_histograms_cpu(&binned, &grads, &parent_node, &tiles, &[0], &raw, 4, 2)
                .expect("parent build");

        // Smaller child = rows 0..1
        let smaller_node = NodeSlice::new(2, vec![0, 1]).expect("valid");
        let smaller_bundle =
            build_linear_histograms_cpu(&binned, &grads, &smaller_node, &tiles, &[0], &raw, 4, 2)
                .expect("smaller build");

        // Larger child via subtraction = rows 2..3
        let larger_bundle = subtract_linear_histogram_bundle(&parent_bundle, &smaller_bundle);
        let larger_node = NodeSlice::new(3, vec![2, 3]).expect("valid");
        let larger_direct =
            build_linear_histograms_cpu(&binned, &grads, &larger_node, &tiles, &[0], &raw, 4, 2)
                .expect("larger direct build");

        // Compare every bin of every feature histogram.
        for (lfh, dfh) in larger_bundle
            .feature_histograms
            .iter()
            .zip(larger_direct.feature_histograms.iter())
        {
            for (lb, db) in lfh.bins.iter().zip(dfh.bins.iter()) {
                assert!(
                    (lb.grad_sum - db.grad_sum).abs() < 1e-5,
                    "grad_sum mismatch: {} vs {}",
                    lb.grad_sum,
                    db.grad_sum
                );
                assert!(
                    (lb.hess_sum - db.hess_sum).abs() < 1e-5,
                    "hess_sum mismatch"
                );
                assert_eq!(lb.count, db.count, "count mismatch");
                for j in 0..1 {
                    assert!(
                        (lb.xtg[j] - db.xtg[j]).abs() < 1e-5,
                        "xtg[{j}] mismatch: {} vs {}",
                        lb.xtg[j],
                        db.xtg[j]
                    );
                    assert!(
                        (lb.xt_hx[j] - db.xt_hx[j]).abs() < 1e-5,
                        "xt_hx[{j}] mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn memory_footprint_reasonable() {
        // d=6, bins=256, features=20 should still fit comfortably in L2.
        let d = 6;
        let bins = 256;
        let features = 20;
        let bin_size = std::mem::size_of::<LinearHistogramBin>();
        let total_bytes = bin_size * bins * features;
        // LinearHistogramBin (stride-8 layout): 4+4+4 + 8*4 + 64*4 = 12 + 32 + 256
        // = 300 bytes (rounded up by alignment).  20 features × 256 bins × 300 B
        // ≈ 1.5 MB, well under the 10 MB target.
        assert!(
            total_bytes < 10 * 1024 * 1024,
            "footprint {total_bytes} bytes exceeds 10 MB for d={d}, bins={bins}, features={features}"
        );
    }

    /// Scalar reference implementation of the per-row inner loop, used to
    /// verify the SIMD path's correctness end-to-end.  Mirrors the pre-SIMD
    /// histogram build with the new stride-8 layout: writes the full upper
    /// triangle via `pl_matrix_index`.  The SIMD path additionally writes the
    /// lower triangle (full outer product), but the upper-triangle values are
    /// the only ones the Cholesky solver reads.
    #[allow(clippy::too_many_arguments)]
    fn build_linear_histograms_scalar_reference(
        binned_matrix: &BinnedMatrix,
        gradients: &[GradientPair],
        node: &NodeSlice,
        feature_tiles: &[FeatureTile],
        regressor_features: &[u32],
        raw_feature_values: &[f32],
        row_count: usize,
        feature_count: usize,
    ) -> LinearHistogramBundle {
        use alloygbm_core::pl_matrix_index;

        let d = regressor_features.len();
        let bin_count = binned_matrix.max_bin as usize + 2;
        let split_features: Vec<u32> = feature_tiles
            .iter()
            .flat_map(|tile| tile.start_feature..tile.end_feature)
            .collect();

        let feature_histograms: Vec<LinearFeatureHistogram> = split_features
            .iter()
            .map(|&split_feat_idx| {
                let mut bins = vec![LinearHistogramBin::default(); bin_count];
                for &row_idx in &node.row_indices {
                    let row_idx = row_idx as usize;
                    let bin = binned_matrix
                        .row_bin(row_idx * binned_matrix.feature_count + split_feat_idx as usize)
                        as usize;
                    let b = &mut bins[bin];
                    let gp = &gradients[row_idx];
                    let g = gp.grad;
                    let h = gp.hess;
                    b.grad_sum += g;
                    b.hess_sum += h;
                    b.count += 1;
                    for (j, &feat_j_raw) in regressor_features.iter().enumerate().take(d) {
                        let feat_j = feat_j_raw as usize;
                        let x_j = if feat_j < feature_count && row_idx < row_count {
                            raw_feature_values[row_idx * feature_count + feat_j]
                        } else {
                            0.0
                        };
                        b.xtg[j] += g * x_j;
                        for (k, &feat_k_raw) in
                            regressor_features.iter().enumerate().skip(j).take(d - j)
                        {
                            let feat_k = feat_k_raw as usize;
                            let x_k = if feat_k < feature_count {
                                raw_feature_values[row_idx * feature_count + feat_k]
                            } else {
                                0.0
                            };
                            b.xt_hx[pl_matrix_index(j, k)] += h * x_j * x_k;
                        }
                    }
                }
                LinearFeatureHistogram {
                    feature_index: split_feat_idx,
                    bins,
                }
            })
            .collect();

        LinearHistogramBundle {
            node_id: node.node_id,
            num_regressors: d,
            regressor_features: regressor_features.to_vec(),
            feature_histograms,
        }
    }

    /// End-to-end equivalence check: the SIMD `build_linear_histograms_cpu`
    /// must produce results that agree with the scalar reference on the
    /// upper-triangle entries (the only ones consumed by the Cholesky solve).
    /// Tested across 1000 randomised rows with d=6 regressors, 16 bins, and
    /// 5 split features — wide enough to exercise the bin-scatter pattern.
    #[test]
    fn build_linear_histograms_simd_matches_scalar_reference() {
        // Deterministic LCG for reproducibility.
        fn step(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        }
        fn next_f32(state: &mut u64) -> f32 {
            let s = step(state);
            ((s >> 32) as i32 as f32) / (i32::MAX as f32) * 2.0
        }

        let mut state: u64 = 0xDEADBEEF;
        let row_count = 1000usize;
        let feature_count = 8usize;
        let d = 6usize;
        let regressor_features: Vec<u32> = (0..d as u32).collect();
        let max_bin = 15u16;

        let mut bins = Vec::with_capacity(row_count * feature_count);
        for _ in 0..row_count * feature_count {
            let s = step(&mut state);
            bins.push(((s >> 56) as u8) % (max_bin as u8 + 1));
        }
        let binned = BinnedMatrix::new(row_count, feature_count, max_bin, bins).expect("ok");

        let gradients: Vec<GradientPair> = (0..row_count)
            .map(|_| GradientPair {
                grad: next_f32(&mut state),
                hess: next_f32(&mut state).abs() + 0.1,
            })
            .collect();

        let raw: Vec<f32> = (0..row_count * feature_count)
            .map(|_| next_f32(&mut state))
            .collect();
        let node = NodeSlice::new(1, (0..row_count as u32).collect()).expect("ok");
        let tiles = vec![FeatureTile {
            start_feature: 0,
            end_feature: 5, // 5 split features
        }];

        let simd_bundle = build_linear_histograms_cpu(
            &binned,
            &gradients,
            &node,
            &tiles,
            &regressor_features,
            &raw,
            row_count,
            feature_count,
        )
        .expect("simd ok");
        let scalar_bundle = build_linear_histograms_scalar_reference(
            &binned,
            &gradients,
            &node,
            &tiles,
            &regressor_features,
            &raw,
            row_count,
            feature_count,
        );

        assert_eq!(
            simd_bundle.feature_histograms.len(),
            scalar_bundle.feature_histograms.len()
        );
        for (sfh, rfh) in simd_bundle
            .feature_histograms
            .iter()
            .zip(scalar_bundle.feature_histograms.iter())
        {
            assert_eq!(sfh.bins.len(), rfh.bins.len());
            for (sb, rb) in sfh.bins.iter().zip(rfh.bins.iter()) {
                // Scalar accumulators (always bit-exact: identical add order).
                assert!(
                    (sb.grad_sum - rb.grad_sum).abs() < 1e-3,
                    "grad_sum: simd={} scalar={}",
                    sb.grad_sum,
                    rb.grad_sum
                );
                assert!(
                    (sb.hess_sum - rb.hess_sum).abs() < 1e-3,
                    "hess_sum: simd={} scalar={}",
                    sb.hess_sum,
                    rb.hess_sum
                );
                assert_eq!(sb.count, rb.count);
                for j in 0..d {
                    assert!(
                        (sb.xtg[j] - rb.xtg[j]).abs() < 1e-3,
                        "xtg[{j}]: simd={} scalar={}",
                        sb.xtg[j],
                        rb.xtg[j]
                    );
                    for k in j..d {
                        let idx = j * MAX_PL_REGRESSORS + k;
                        assert!(
                            (sb.xt_hx[idx] - rb.xt_hx[idx]).abs() < 1e-3,
                            "xt_hx[{j},{k}]: simd={} scalar={}",
                            sb.xt_hx[idx],
                            rb.xt_hx[idx]
                        );
                    }
                }
            }
        }
    }
}
