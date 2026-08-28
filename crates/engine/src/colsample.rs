//! Per-node feature subsampling (`colsample_bynode`). Stateless: the kept set
//! at a node is a pure, deterministic function of `(seed, node_id, feature)`,
//! so it needs no per-node bookkeeping. Composes with interaction constraints
//! at the same per-node histogram filter used by the split search.

use alloygbm_core::HistogramBundle;

use crate::sampling::mixed_hash;
use crate::trainer::InteractionConstraintIndex;
use crate::trainer::filter_histogram_bundle_by_features;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Active per-node colsample configuration. Absent (`None` at call sites) when
/// `rate >= 1.0`, in which case no filtering happens.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ColsampleBynode {
    pub rate: f32,
    pub seed: u64,
}

/// Deterministic per-node keep decision for one feature. `true` (always kept)
/// when `rate >= 1.0`.
pub(crate) fn colsample_bynode_keeps_feature(
    seed: u64,
    node_id: u64,
    feature: u32,
    rate: f32,
) -> bool {
    if rate >= 1.0 {
        return true;
    }
    let hashed = mixed_hash(seed ^ node_id.wrapping_mul(GOLDEN) ^ (feature as u64).wrapping_add(1));
    // Uniform in [0, 1): take the top 53 bits as an f64 mantissa.
    let unit = (hashed >> 11) as f64 / ((1u64 << 53) as f64);
    unit < rate as f64
}

/// Build the per-node candidate histogram bundle, composing interaction
/// constraints and colsample_bynode. Returns `None` when neither filter is
/// active (caller uses the unfiltered bundle unchanged). When the composed
/// keep-set would be empty, colsample is dropped for that node (interaction
/// set retained) so a node never has zero candidate features.
pub(crate) fn filter_histograms_for_node(
    histograms: &HistogramBundle,
    constraint_index: Option<&InteractionConstraintIndex>,
    active_groups: Option<u64>,
    colsample: Option<ColsampleBynode>,
    node_id: u64,
) -> Option<HistogramBundle> {
    let interaction_active = matches!((constraint_index, active_groups), (Some(_), Some(_)));
    let colsample_active = colsample.is_some();
    if !interaction_active && !colsample_active {
        return None;
    }
    let interaction_allows = |feature: u32| match (constraint_index, active_groups) {
        (Some(index), Some(groups)) => index.feature_allowed(groups, feature),
        _ => true,
    };
    let colsample_keeps = |feature: u32| match colsample {
        Some(cs) => colsample_bynode_keeps_feature(cs.seed, node_id, feature, cs.rate),
        None => true,
    };
    // First pass: interaction AND colsample.
    let composed = filter_histogram_bundle_by_features(histograms, |f| {
        interaction_allows(f) && colsample_keeps(f)
    });
    if composed.feature_count() > 0 {
        return Some(composed);
    }
    // Fallback: colsample emptied the node — keep the interaction-only set.
    if interaction_active {
        return Some(filter_histogram_bundle_by_features(histograms, |f| {
            interaction_allows(f)
        }));
    }
    // Colsample-only and it emptied everything: keep all features for this node.
    None
}
