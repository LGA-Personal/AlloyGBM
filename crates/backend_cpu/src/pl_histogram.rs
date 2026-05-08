//! Piecewise-linear histogram construction for the CPU backend.
//!
//! Builds [`LinearHistogramBundle`] statistics — `(Xᵀg, XᵀHX)` per bin per
//! split feature — needed for the closed-form ridge-regression leaf-weight
//! solve `α* = -(XᵀHX + λI)⁻¹ Xᵀg`.
//!
//! The SIMD path for standard (scalar-leaf) histograms is left completely
//! untouched.  This module is a parallel addition, not a modification.

use alloygbm_core::{
    BinnedMatrix, FeatureTile, GradientPair, LinearFeatureHistogram, LinearHistogramBin,
    LinearHistogramBundle, MAX_PL_REGRESSORS, NodeSlice, pl_matrix_index,
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

                // Accumulate Xᵀg and XᵀHX using only the first `d` regressors.
                for j in 0..d {
                    let feat_j = regressor_features[j] as usize;
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
                        let idx = pl_matrix_index(j, k);
                        b.xt_hx[idx] += h * x_j * x_k;
                    }
                }
            }

            // Replace NaN/Inf in accumulated values with 0 (defensive).
            for bin in &mut bins {
                sanitize_linear_bin(bin, d);
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
/// `d` is the number of active regressors so we iterate only the active
/// `(j, k)` index pairs in `xt_hx` (the indices are non-contiguous for
/// `d < MAX_PL_REGRESSORS` due to how `pl_matrix_index` works).
#[inline]
fn sanitize_linear_bin(bin: &mut LinearHistogramBin, d: usize) {
    if !bin.grad_sum.is_finite() {
        bin.grad_sum = 0.0;
    }
    if !bin.hess_sum.is_finite() {
        bin.hess_sum = 0.0;
    }
    for j in 0..d {
        if !bin.xtg[j].is_finite() {
            bin.xtg[j] = 0.0;
        }
    }
    for j in 0..d {
        for k in j..d {
            let idx = pl_matrix_index(j, k);
            if !bin.xt_hx[idx].is_finite() {
                bin.xt_hx[idx] = 0.0;
            }
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
        // d=6, bins=256, features=20 should be < 10 MB.
        let d = 6;
        let bins = 256;
        let features = 20;
        let bin_size = std::mem::size_of::<LinearHistogramBin>();
        let total_bytes = bin_size * bins * features;
        // LinearHistogramBin: 4+4+4 + 8*4 + 36*4 = 12 + 32 + 144 = 188 bytes
        assert!(
            total_bytes < 10 * 1024 * 1024,
            "footprint {total_bytes} bytes exceeds 10 MB for d={d}, bins={bins}, features={features}"
        );
    }
}
