//! Joint multi-output training entry points.
//!
//! The three `fit_joint_multi_output*` functions are the public API
//! consumed by the PyO3 bridge; they all funnel through the private
//! `fit_joint_inner` workhorse. `build_joint_metadata` assembles the
//! `ModelMetadata` for a joint-trained artifact.

use alloygbm_core::{
    BinnedMatrix, Device, DroMetadataPayload, FactorExposureMatrix, GradientPair, LeafValue,
    ModelMetadata, TrainParams, TreeGrowth,
};

use crate::{TrainedModel, TrainedStump, encode_tree_node_id};

use super::TREE_NODE_STRIDE;
use super::build_round::{build_joint_round_inner, build_joint_round_leafwise};
use super::helpers::{
    effective_dro_config, effective_neutralization_config, select_joint_row_indices_for_round,
    walk_tree_into_predictions,
};
use super::types::{JointMorphContext, JointObjective, JointTrainingSummary, JointWarmStartState};

pub fn fit_joint_multi_output(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
) -> Result<JointTrainingSummary, String> {
    fit_joint_inner(
        params,
        feature_count,
        binned_matrix,
        targets_per_output,
        group_id,
        per_output_objective,
        n_estimators,
        /*categorical_features=*/ &[],
        /*warm_start=*/ None,
        /*factor_exposures=*/ None,
    )
}

/// Same as [`fit_joint_multi_output_with_categorical`] but accepts an
/// optional [`JointWarmStartState`] to continue training from a prior
/// fit. When `warm_start = None`, behavior is byte-identical to
/// `fit_joint_multi_output_with_categorical`.
#[allow(clippy::too_many_arguments)]
pub fn fit_joint_multi_output_with_warm_start(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    warm_start: Option<JointWarmStartState>,
    factor_exposures: Option<&FactorExposureMatrix>,
) -> Result<JointTrainingSummary, String> {
    fit_joint_inner(
        params,
        feature_count,
        binned_matrix,
        targets_per_output,
        group_id,
        per_output_objective,
        n_estimators,
        categorical_features,
        warm_start,
        factor_exposures,
    )
}

/// Same as [`fit_joint_multi_output`] but accepts a slice of
/// [`CategoricalFeatureInfo`] specifying which features should be treated
/// as native-categorical via multi-output Fisher-sort partitioning. An
/// empty slice means all features are numeric (byte-identical to
/// `fit_joint_multi_output`).
#[allow(clippy::too_many_arguments)]
pub fn fit_joint_multi_output_with_categorical(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
) -> Result<JointTrainingSummary, String> {
    fit_joint_inner(
        params,
        feature_count,
        binned_matrix,
        targets_per_output,
        group_id,
        per_output_objective,
        n_estimators,
        categorical_features,
        /*warm_start=*/ None,
        /*factor_exposures=*/ None,
    )
}

/// Inner implementation of the joint multi-output trainer. Public
/// callers route through `fit_joint_multi_output_with_categorical`
/// (cold start) or `fit_joint_multi_output_with_warm_start`
/// (continuation). Mirrors the pattern used by the single-output
/// engine where `fit_iterations*` variants all funnel through one
/// underlying impl.
#[allow(clippy::too_many_arguments)]
fn fit_joint_inner(
    params: &TrainParams,
    feature_count: usize,
    binned_matrix: &BinnedMatrix,
    targets_per_output: &[Vec<f32>],
    group_id: Option<&[u32]>,
    per_output_objective: &[JointObjective],
    n_estimators: usize,
    categorical_features: &[crate::CategoricalFeatureInfo],
    warm_start: Option<JointWarmStartState>,
    factor_exposures: Option<&FactorExposureMatrix>,
) -> Result<JointTrainingSummary, String> {
    let n_outputs = targets_per_output.len();
    if per_output_objective.len() != n_outputs {
        return Err(format!(
            "per_output_objective length {} != n_outputs {n_outputs}",
            per_output_objective.len()
        ));
    }
    let n_rows = binned_matrix.row_count;
    for (k, tg) in targets_per_output.iter().enumerate() {
        if tg.len() != n_rows {
            return Err(format!(
                "targets[{k}] length {} != n_rows {n_rows}",
                tg.len()
            ));
        }
    }
    if per_output_objective.iter().any(|o| o.requires_group()) && group_id.is_none() {
        return Err("at least one objective requires group_id".to_string());
    }

    // v0.10.6: pre_target neutralization — residualize each per-output target
    // through the factor exposures BEFORE baselines are computed.
    // Squared-error only (the only objective where residualize-target equals
    // residualize-gradient); validated at the Python boundary AND below so a
    // Rust caller calling `fit_joint_*` directly still gets the guard.
    let projected_targets_owned: Option<Vec<Vec<f32>>> =
        match (effective_neutralization_config(params), factor_exposures) {
            (Some(cfg), Some(exposures))
                if cfg.kind == alloygbm_core::NeutralizationKind::PreTarget =>
            {
                for (k, obj) in per_output_objective.iter().enumerate() {
                    if !matches!(obj, JointObjective::SquaredError) {
                        return Err(format!(
                            "neutralization='pre_target' requires every per-output \
                         objective to be 'squared_error' (output {k} is {obj:?}). \
                         Use neutralization='per_round_gradient' for ranking objectives."
                        ));
                    }
                }
                let projector = crate::FactorProjector::new(exposures, None, cfg.ridge_lambda)
                    .map_err(|err| format!("pre_target projector: {err:?}"))?;
                let mut projected: Vec<Vec<f32>> = Vec::with_capacity(n_outputs);
                let mut projection_scratch: Vec<f32> = Vec::with_capacity(n_rows);
                for tg in targets_per_output {
                    let mut owned = tg.clone();
                    projector
                        .residualize_values_in_place_with_scratch(
                            &mut owned,
                            &mut projection_scratch,
                        )
                        .map_err(|err| format!("pre_target residualize: {err:?}"))?;
                    projected.push(owned);
                }
                Some(projected)
            }
            (Some(cfg), None) if cfg.kind == alloygbm_core::NeutralizationKind::PreTarget => {
                return Err(
                    "factor_exposures are required when neutralization='pre_target'".to_string(),
                );
            }
            _ => None,
        };

    // `effective_targets` is the residualized view when pre_target is active;
    // otherwise it borrows the original targets. Every gradient/baseline site
    // below reads through this view so non-pre_target modes are unaffected.
    let effective_targets: Vec<&[f32]> = match &projected_targets_owned {
        Some(owned) => owned.iter().map(|v| v.as_slice()).collect(),
        None => targets_per_output.iter().map(|v| v.as_slice()).collect(),
    };

    // v0.10.3: warm-start branch — when `warm_start` is `Some`, the
    // prior fit's baselines win (re-seed `predictions` from them) and
    // its stumps are prepended to `all_stumps`. The cold-start branch
    // computes per-output baselines as before. v0.10.4: also carries
    // `initial_ema_stats` for MorphBoost warm-resume.
    let (
        initial_stumps,
        initial_rounds,
        initial_dart_weights_arg,
        initial_ema_stats_arg,
        baselines,
    ) = if let Some(ws) = warm_start {
        if ws.baselines.len() != n_outputs {
            return Err(format!(
                "warm-start baselines length {} != n_outputs {n_outputs}",
                ws.baselines.len()
            ));
        }
        if let Some(ema) = ws.initial_ema_stats.as_ref()
            && ema.len() != n_outputs
        {
            return Err(format!(
                "warm-start initial_ema_stats length {} != n_outputs {n_outputs}",
                ema.len()
            ));
        }
        (
            ws.stumps,
            ws.initial_rounds_completed,
            ws.initial_dart_tree_weights,
            ws.initial_ema_stats,
            ws.baselines,
        )
    } else {
        let cold_baselines: Vec<f32> = per_output_objective
            .iter()
            .zip(effective_targets.iter())
            .map(|(obj, targets)| obj.initial_prediction(targets))
            .collect();
        (Vec::new(), 0, None, None, cold_baselines)
    };

    // Per-output prediction vectors, seeded from baselines.
    let mut predictions: Vec<Vec<f32>> = baselines.iter().map(|&b| vec![b; n_rows]).collect();

    let learning_rate = params.learning_rate;
    let mut all_stumps: Vec<TrainedStump> = initial_stumps;
    let mut rounds_completed: usize = 0;

    // v0.10.4: MorphBoost runtime state. Active when `params.morph_config`
    // is `Some`. `total_iterations` covers warm-start prefix + new rounds so
    // the LR schedule + depth-penalty curve see the full horizon, mirroring
    // the single-output multiclass path. EMA snapshot from a prior
    // MorphBoost fit (passed via `JointWarmStartState.initial_ema_stats`)
    // re-seeds `MorphState::ema_stats` so a warm-resumed N+M fit matches
    // a fresh N+M fit byte-for-byte.
    let total_iterations_u32 = (initial_rounds + n_estimators) as u32;
    let mut morph_state: Option<crate::MorphState> = params
        .morph_config
        .map(|cfg| crate::MorphState::new(cfg, n_outputs, total_iterations_u32, learning_rate));
    if let (Some(ms), Some(snapshot)) = (morph_state.as_mut(), initial_ema_stats_arg.as_ref()) {
        for (i, stat) in snapshot.iter().take(ms.ema_stats.len()).enumerate() {
            ms.ema_stats[i] = *stat;
        }
    }

    // v0.10.3: joint DART state. `dart_state.tree_weights[r]` is the
    // weight applied to round-r's tree at predict time;
    // `dart_round_start_offsets[r]` + `dart_round_counts[r]` track the
    // stump range in `all_stumps` for that round (one tree per round on
    // the joint trainer, but the stump *count* varies under leaf-wise
    // growth).
    let dart_params = match params.boosting_mode {
        alloygbm_core::BoostingMode::Dart {
            drop_rate,
            max_drop,
            normalize_type,
            sample_type,
        } => Some((drop_rate, max_drop, normalize_type, sample_type)),
        _ => None,
    };
    let mut dart_state = crate::DartState::default();
    let mut dart_round_start_offsets: Vec<usize> = Vec::new();
    let mut dart_round_counts: Vec<usize> = Vec::new();

    // v0.10.3: warm-start — replay prior-stump contributions onto
    // `predictions` so the new round's gradients see the correct
    // residual. Group prior stumps by `tree_id` and walk each tree at
    // its DART weight (1.0 for non-DART warm-starts).
    if !all_stumps.is_empty() {
        // PR #36 review (C2): validate that every prior stump's
        // `feature_index` is `< feature_count` BEFORE replay. Without
        // this check `walk_tree_into_predictions` indexes
        // `binned_matrix.bins[row * feature_count + feature]` which
        // panics across the PyO3 boundary if the prior fit was trained
        // on more features than the current one. Surface a clean
        // validation error instead.
        for (idx, s) in all_stumps.iter().enumerate() {
            let fi = s.split.feature_index as usize;
            if fi >= feature_count {
                return Err(format!(
                    "warm-start: prior stump {idx} references feature_index {fi} \
                     which is >= the current feature_count {feature_count}. The \
                     init_model appears to have been trained on a wider feature \
                     set than the current X; either pad X to match the prior \
                     schema or fit fresh without `warm_start=True`."
                ));
            }
        }
        let mut grouped: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (idx, s) in all_stumps.iter().enumerate() {
            grouped
                .entry(s.split.node_id / TREE_NODE_STRIDE)
                .or_default()
                .push(idx);
        }
        // For DART warm-start, derive per-tree weights either from the
        // explicit `initial_dart_tree_weights` arg or from the first
        // stump's `tree_weight` (mirrors `apply_dart_tree_weights`).
        let derived_dart_weights: Option<Vec<f32>> = if dart_params.is_some() {
            if let Some(dw) = initial_dart_weights_arg.as_ref() {
                if dw.len() != initial_rounds {
                    return Err(format!(
                        "warm_start.initial_dart_tree_weights length {} != initial_rounds_completed {initial_rounds}",
                        dw.len()
                    ));
                }
                Some(dw.clone())
            } else {
                let mut reconstructed: Vec<f32> = vec![1.0; initial_rounds];
                for (tid, indices) in &grouped {
                    if let Some(&first) = indices.first()
                        && (*tid as usize) < reconstructed.len()
                    {
                        reconstructed[*tid as usize] = all_stumps[first].tree_weight;
                    }
                }
                Some(reconstructed)
            }
        } else {
            None
        };
        for (tree_idx, stump_indices) in &grouped {
            // Materialize this tree's stumps as owned clones so we can
            // hand a contiguous slice to the walker without aliasing
            // `all_stumps` (which we'd otherwise need to borrow
            // immutably while `predictions` is mutably borrowed below).
            let tree_stumps: Vec<TrainedStump> = stump_indices
                .iter()
                .map(|&i| all_stumps[i].clone())
                .collect();
            let scale = if let Some(dw) = derived_dart_weights.as_ref() {
                dw.get(*tree_idx as usize).copied().unwrap_or(1.0)
            } else {
                1.0
            };
            walk_tree_into_predictions(
                &tree_stumps,
                binned_matrix,
                feature_count,
                n_rows,
                n_outputs,
                &mut predictions,
                1.0,
                scale,
            );
        }

        // Seed DART bookkeeping from the prior fit so new rounds can
        // dropout/restore prior trees correctly.
        if dart_params.is_some() {
            dart_state.tree_weights =
                derived_dart_weights.unwrap_or_else(|| vec![1.0; initial_rounds]);
            for r in 0..initial_rounds {
                let r_u32 = r as u32;
                if let Some(indices) = grouped.get(&r_u32) {
                    let start = *indices.iter().min().unwrap();
                    dart_round_start_offsets.push(start);
                    dart_round_counts.push(indices.len());
                } else {
                    // Round r contributed no stumps (degenerate).
                    dart_round_start_offsets.push(0);
                    dart_round_counts.push(0);
                }
            }
        }
    }

    // v0.10.6: per_round_gradient neutralization — build the projector once,
    // then project each per-output gradient buffer in place every round.
    // Mirrors the single-output multiclass path (lib.rs:3581+).
    let gradient_projector: Option<crate::FactorProjector> =
        match (effective_neutralization_config(params), factor_exposures) {
            (Some(cfg), Some(exposures))
                if cfg.kind == alloygbm_core::NeutralizationKind::PerRoundGradient =>
            {
                Some(
                    crate::FactorProjector::new(exposures, None, cfg.ridge_lambda)
                        .map_err(|err| format!("per_round_gradient projector: {err:?}"))?,
                )
            }
            (Some(cfg), None)
                if cfg.kind == alloygbm_core::NeutralizationKind::PerRoundGradient =>
            {
                return Err(
                    "factor_exposures are required when neutralization='per_round_gradient'"
                        .to_string(),
                );
            }
            _ => None,
        };

    for round in 0..n_estimators {
        // v0.10.3 warm-start: when continuing from a prior fit, all
        // per-round seeds, dropout indices, and `node_id` encodings
        // mix `global_round = round + initial_rounds` so a warm-resumed
        // N+M fit produces the same RNG draws on round M..N+M as a
        // fresh N+M fit.
        let global_round = round + initial_rounds;

        // v0.10.3 joint DART: drop a random subset of previously-trained
        // trees before computing gradients. Subtract their (currently
        // weighted) contributions from `predictions` so the new tree
        // fits on residuals of the dropped-out ensemble (mirrors the
        // single-output DART flow at crates/engine/src/lib.rs:4895).
        let dropped_tree_ids: Vec<usize> =
            if let Some((drop_rate, max_drop, _normalize_type, sample_type)) = dart_params {
                if dart_state.tree_weights.is_empty() {
                    Vec::new()
                } else {
                    let drops = crate::select_dropouts(
                        dart_state.tree_weights.len(),
                        drop_rate,
                        max_drop,
                        sample_type,
                        &dart_state.tree_weights,
                        params.seed,
                        global_round,
                    );
                    for &tree_id in &drops {
                        let w_old = dart_state.tree_weights[tree_id];
                        let start = dart_round_start_offsets[tree_id];
                        let count = dart_round_counts[tree_id];
                        if count > 0 {
                            walk_tree_into_predictions(
                                &all_stumps[start..start + count],
                                binned_matrix,
                                feature_count,
                                n_rows,
                                n_outputs,
                                &mut predictions,
                                -1.0,
                                w_old,
                            );
                        }
                    }
                    drops
                }
            } else {
                Vec::new()
            };

        // Compute per-output gradients on current predictions.
        let mut grads_per_output: Vec<Vec<GradientPair>> = Vec::with_capacity(n_outputs);
        for k in 0..n_outputs {
            let g = per_output_objective[k].compute_gradients(
                &predictions[k],
                effective_targets[k],
                group_id,
            )?;
            grads_per_output.push(g);
        }

        // v0.10.6: per_round_gradient — project each per-output gradient
        // buffer through the factor projector. Applied BEFORE row sampling
        // so the joint GOSS scorer (which mutates `grads_per_output`) and
        // the histogram builder both see projected residuals.
        if let Some(projector) = &gradient_projector {
            let mut projection_scratch: Vec<f32> = Vec::with_capacity(n_rows);
            for buf in grads_per_output.iter_mut() {
                projector
                    .project_gradient_pairs_in_place_with_scratch(buf, &mut projection_scratch)
                    .map_err(|err| format!("per_round_gradient projection: {err:?}"))?;
            }
        }

        // v0.10.4: update per-output EMA stats BEFORE tree-building so
        // morph split selection sees the latest mean/std. Mirrors the
        // single-output multiclass path
        // (crates/engine/src/lib.rs:3974 `update_ema_from_gradient_pairs`).
        if let Some(ms) = morph_state.as_mut() {
            for (k, g) in grads_per_output.iter().enumerate() {
                ms.update_ema_from_gradient_pairs(g, k);
            }
        }
        let morph_ctx: Option<JointMorphContext> = morph_state
            .as_ref()
            .map(|ms| JointMorphContext::from_state(ms, global_round as u32, total_iterations_u32));

        // v0.10.3: joint GOSS. When `boosting_mode = Goss`, score rows by
        // `Σₖ |g_{i,k}|`, keep top_rate fraction, sample other_rate
        // uniformly from the rest, and amplify the sampled-low rows'
        // gradients so per-output histogram statistics remain unbiased
        // estimators of the full-data gradient sums. Mutates
        // `grads_per_output` in place. Falls back to the row_subsample
        // path under Standard / DART.
        //
        // row_subsample (v0.10.2, fixed v0.10.2.1): seeded Bernoulli row
        // mask. When row_subsample < 1.0, the round's tree builder works
        // on the SAMPLED rows only — `sampled_rows: Vec<u32>` is passed
        // as the root's `row_indices`. This means `min_data_in_leaf`
        // operates on the sampled set, matching single-output semantics
        // (v0.10.2 zeroed gradients but left rows in the index, so a
        // split could create a leaf with enough rows but too few
        // sampled rows).
        //
        // The post-build prediction-update walk below operates on all
        // `n_rows` via the BinnedMatrix tree walk (not row_indices), so
        // un-sampled rows still receive their tree-walked leaf delta —
        // matching LightGBM's `bagging_fraction` semantics where every
        // row's predictions are updated each round, but the tree itself
        // is fit on the sampled subset.
        let sampled_rows_opt: Option<Vec<u32>> = if let Some(rows) =
            select_joint_row_indices_for_round(
                params.boosting_mode,
                n_rows,
                params.seed,
                global_round as u64,
                &mut grads_per_output,
            ) {
            Some(rows)
        } else if params.row_subsample < 1.0 {
            let mut rng_state: u64 = params.seed.wrapping_mul(0x9E3779B97F4A7C15)
                ^ ((global_round as u64).wrapping_mul(0xBF58476D1CE4E5B9));
            if rng_state == 0 {
                rng_state = 0xDEADBEEFCAFEBABE;
            }
            let rate = params.row_subsample;
            let mut sampled: Vec<u32> = Vec::with_capacity(n_rows / 2 + 1);
            for row in 0..n_rows {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let u01 = ((rng_state >> 11) & ((1u64 << 24) - 1)) as f32 / ((1u64 << 24) as f32);
                if u01 < rate {
                    sampled.push(row as u32);
                }
            }
            // Edge case: nothing sampled. Fall back to all-rows for this
            // round so the trainer doesn't produce a degenerate empty tree.
            if sampled.is_empty() {
                None
            } else {
                Some(sampled)
            }
        } else {
            None
        };
        // Gradients pass through unchanged; the trainer indexes them by
        // row id from the (potentially filtered) row_indices list.
        let active_grads: &[Vec<GradientPair>] = grads_per_output.as_slice();

        // Build one shared tree. Dispatch on tree_growth: leaf-wise uses
        // the priority-queue best-first builder gated by `max_leaves`;
        // level-wise uses the BFS depth-bounded builder.
        let mut round_result = if params.tree_growth == TreeGrowth::Leaf {
            let max_leaves = params.max_leaves.ok_or_else(|| {
                "tree_growth='leaf' requires max_leaves to be set on the joint trainer".to_string()
            })?;
            build_joint_round_leafwise(
                params,
                binned_matrix,
                active_grads,
                n_outputs,
                max_leaves,
                categorical_features,
                global_round,
                sampled_rows_opt.as_deref(),
                morph_ctx.as_ref(),
                factor_exposures,
            )?
        } else {
            build_joint_round_inner(
                params,
                binned_matrix,
                active_grads,
                n_outputs,
                categorical_features,
                global_round,
                sampled_rows_opt.as_deref(),
                morph_ctx.as_ref(),
                factor_exposures,
            )?
        };
        if round_result.stumps.is_empty() {
            break;
        }

        crate::refine_joint_quantile_leaves(
            &mut round_result.stumps,
            binned_matrix,
            &predictions,
            &effective_targets,
            per_output_objective,
        )
        .map_err(|e| e.to_string())?;

        rounds_completed += 1;

        // v0.10.0 review fix (Comment 3) + v0.10.4 MorphBoost: pre-scale the
        // per-leaf K-output deltas before persisting so the artifact already
        // encodes the LR-scaled contribution. JointPredictor adds the leaf
        // values directly without re-applying `learning_rate`.
        //
        // Standard scale: `learning_rate` (unchanged from v0.10.0).
        // Morph scale: `morph_lr * leaf_shrink * depth_penalty` where
        //   morph_lr = MorphState::lr_for_iter(round)
        //   leaf_shrink = max(0, 1 - morph_rate * round/total)
        //   depth_penalty = depth_penalty_base ^ (depth/3), depth derived from
        //     local_node_id via `(id+1).ilog2()`.
        // Depth-penalty applies per-stump because non-root leaves have larger
        // depth than root leaves.
        let standard_scale_active =
            morph_state.is_none() && (learning_rate - 1.0).abs() > f32::EPSILON;
        let morph_active = morph_state.is_some();
        if standard_scale_active || morph_active {
            // Pre-compute morph scalars that are stump-independent.
            let (morph_lr, leaf_shrink) = if let Some(ms) = morph_state.as_ref() {
                let lr = ms.lr_for_iter(global_round);
                let t = global_round as f32;
                let total_t = total_iterations_u32.max(1) as f32;
                let shrink = (1.0 - ms.config.morph_rate * (t / total_t)).max(0.0);
                (lr, shrink)
            } else {
                (learning_rate, 1.0_f32)
            };
            let depth_penalty_base = morph_state
                .as_ref()
                .map(|ms| ms.config.depth_penalty_base)
                .unwrap_or(1.0);
            for stump in round_result.stumps.iter_mut() {
                // local_node_id is the pre-encode id (re-encode happens
                // later). depth = floor(log2(id + 1)).
                let local_id = stump.split.node_id;
                let depth = (local_id + 1).ilog2();
                let depth_penalty = if morph_active {
                    depth_penalty_base.powf(depth as f32 / 3.0)
                } else {
                    1.0
                };
                let scale = if morph_active {
                    morph_lr * leaf_shrink * depth_penalty
                } else {
                    learning_rate
                };
                if let Some((left_k, right_k)) = stump.multi_output_leaf_values.as_mut() {
                    for v in left_k.iter_mut() {
                        *v *= scale;
                    }
                    for v in right_k.iter_mut() {
                        *v *= scale;
                    }
                }
                // Keep the placeholder scalar consistent for any scalar code
                // path that accidentally reads it.
                if let Some((left_k, _)) = stump.multi_output_leaf_values.as_ref() {
                    stump.left_leaf_value = LeafValue::Scalar(left_k[0]);
                }
                if let Some((_, right_k)) = stump.multi_output_leaf_values.as_ref() {
                    stump.right_leaf_value = LeafValue::Scalar(right_k[0]);
                }
            }
        }

        // v0.10.0 review fix (Comment 2) — refactored in v0.10.3:
        // update training-time predictions via a per-row tree walk over
        // THIS round's stumps. Previously we applied every stump's
        // delta to every row, which is correct only when max_depth == 1
        // (each row reaches every stump). For max_depth > 1, non-root
        // stumps must only affect rows that reach them — which is
        // exactly what JointPredictor does at predict time. The walk
        // is now factored into `walk_tree_into_predictions`, shared
        // with DART dropout subtraction and warm-start replay.
        //
        // `round_result.stumps` are still pre-encode at this point
        // (local node IDs); the helper extracts local IDs via
        // `node_id % TREE_NODE_STRIDE` which works under both encodings.
        walk_tree_into_predictions(
            &round_result.stumps,
            binned_matrix,
            feature_count,
            n_rows,
            n_outputs,
            &mut predictions,
            1.0,
            1.0,
        );

        // Re-encode node_id to be globally unique across rounds (joint
        // trainer outputs one tree per round; local_node_id stays the
        // same, tree_index = round). Track the round's stump range so
        // DART can subtract / re-add this tree on later rounds.
        let round_start = all_stumps.len();
        let global_round = round + initial_rounds;
        for mut stump in round_result.stumps.into_iter() {
            let local_node_id = stump.split.node_id;
            stump.split.node_id = encode_tree_node_id(global_round, local_node_id)
                .map_err(|e| format!("encode_tree_node_id: {e:?}"))?;
            all_stumps.push(stump);
        }
        let round_count = all_stumps.len() - round_start;
        dart_round_start_offsets.push(round_start);
        dart_round_counts.push(round_count);

        // v0.10.3 joint DART finalize: rescale the new tree from 1.0
        // down to `new_w = 1 / (K + 1)`, and re-add each dropped tree
        // at its rescaled weight. Mirrors the single-output DART
        // finalize block in crates/engine/src/lib.rs:5118.
        if let Some((_, _, normalize_type, _)) = dart_params {
            let k = dropped_tree_ids.len() as f32;
            let new_w = 1.0 / (k + 1.0);
            let drop_factor = match normalize_type {
                alloygbm_core::DartNormalize::Tree => k / (k + 1.0),
                alloygbm_core::DartNormalize::Forest => 1.0 / (k + 1.0),
            };
            // 1. Scale new tree from 1.0 down to new_w: subtract
            //    (1.0 - new_w) of the new tree's contribution.
            let delta_scale = 1.0_f32 - new_w;
            if delta_scale.abs() > f32::EPSILON && round_count > 0 {
                walk_tree_into_predictions(
                    &all_stumps[round_start..round_start + round_count],
                    binned_matrix,
                    feature_count,
                    n_rows,
                    n_outputs,
                    &mut predictions,
                    -1.0,
                    delta_scale,
                );
            }
            // 2. Re-add each dropped tree at its post-normalize weight
            //    `w_new = w_old * drop_factor`.
            for &tree_id in &dropped_tree_ids {
                let w_old = dart_state.tree_weights[tree_id];
                let w_new = w_old * drop_factor;
                let start = dart_round_start_offsets[tree_id];
                let count = dart_round_counts[tree_id];
                if count > 0 {
                    walk_tree_into_predictions(
                        &all_stumps[start..start + count],
                        binned_matrix,
                        feature_count,
                        n_rows,
                        n_outputs,
                        &mut predictions,
                        1.0,
                        w_new,
                    );
                }
            }
            // 3. Push placeholder weight for the new tree, then run
            //    `apply_normalization` which rescales dropped trees in
            //    place AND sets the new-tree weight to `new_w`.
            dart_state.tree_weights.push(1.0);
            let new_tree_index = dart_state.tree_weights.len() - 1;
            crate::apply_normalization(
                &mut dart_state.tree_weights,
                &dropped_tree_ids,
                normalize_type,
                new_tree_index,
            );
        }
    }

    // v0.10.3 joint DART: stamp the per-tree `tree_weight` onto every
    // stump in that tree so the artifact's `DartTreeWeights` section
    // round-trips correctly. Mirrors the multiclass DART stamping
    // step at the end of `fit_multiclass_iterations_impl`.
    if dart_params.is_some() {
        for (round_idx, &w) in dart_state.tree_weights.iter().enumerate() {
            let start = dart_round_start_offsets[round_idx];
            let count = dart_round_counts[round_idx];
            for s in all_stumps.iter_mut().skip(start).take(count) {
                s.tree_weight = w;
            }
        }
    }

    // v0.10.4: persist MorphBoost EMA snapshot into the artifact so
    // warm-resume can re-seed `MorphState::ema_stats`. Mirrors the
    // single-output regressor path (lib.rs:4413).
    let morph_metadata = morph_state
        .as_ref()
        .map(|ms| alloygbm_core::MorphMetadataPayload {
            config: ms.config,
            // `final_iteration` captures where training ended (last completed
            // global round); `final_total` mirrors the full horizon used by
            // the LR schedule + leaf-shrink curve so warm-resume can recompute
            // them consistently.
            final_iteration: (initial_rounds + rounds_completed).saturating_sub(1) as u32,
            final_total: total_iterations_u32,
            ema_stats: ms.ema_stats.clone(),
        });

    let model = TrainedModel {
        baseline_prediction: 0.0, // Joint model uses per-output baselines (see summary)
        feature_count,
        stumps: all_stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: format!("joint_multi_output[{n_outputs}]"),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata,
        dro_metadata: effective_dro_config(params).map(|cfg| DroMetadataPayload { config: *cfg }),
        feature_baseline: None,
        neutralization_metadata: effective_neutralization_config(params)
            .map(|cfg| alloygbm_core::NeutralizationMetadataPayload { config: *cfg }),
    };

    // v0.10.3 warm-start: report TOTAL rounds completed (prior + new)
    // so a downstream consumer can decode "total tree count" from a
    // single integer.
    let total_rounds_completed = initial_rounds + rounds_completed;

    Ok(JointTrainingSummary {
        baselines,
        model,
        per_output_objective_names: per_output_objective
            .iter()
            .map(|o| o.name().to_string())
            .collect(),
        rounds_completed: total_rounds_completed,
    })
}

/// Build `ModelMetadata` for a joint-trained artifact. The Python wrapper
/// passes feature names + per-output baselines to round-trip cleanly.
pub fn build_joint_metadata(
    feature_names: Vec<String>,
    per_output_objective_names: &[String],
    baselines: &[f32],
) -> ModelMetadata {
    let baseline_summary = baselines
        .iter()
        .map(|b| format!("{b:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let objective_label = format!(
        "joint_multi_output:{}|baselines={}",
        per_output_objective_names.join("+"),
        baseline_summary,
    );
    ModelMetadata {
        format_version: alloygbm_core::MODEL_FORMAT_V1,
        feature_names,
        trained_device: Device::Cpu,
        objective: objective_label,
        num_classes: None,
    }
}
