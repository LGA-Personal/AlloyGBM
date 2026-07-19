//! Interaction-constraint bookkeeping helpers used by the level-wise and
//! leaf-wise tree builders.

use alloygbm_core::HistogramBundle;

use crate::error::{EngineError, EngineResult};

/// Precomputed lookup driving interaction-constraint enforcement during
/// tree growth.  Built once per fit from `TrainParams.interaction_constraints`.
///
/// Each feature carries a `u64` bitset of the constraint groups it belongs
/// to.  Features outside every group ("unconstrained") have a `0` bitset and
/// are always allowed at any node.  Constrained features are allowed only
/// when at least one of their containing groups is still in the per-node
/// `active_groups` bitset; once a path commits to a group, the active set
/// narrows to the intersection of that feature's groups.
#[derive(Debug, Clone)]
pub(crate) struct InteractionConstraintIndex {
    /// `feature_groups[f]` — bitset of constraint-group indices that contain
    /// feature `f`.  `0` means the feature is unconstrained.
    feature_groups: Vec<u64>,
    /// Number of declared constraint groups (`≤ 64`).
    group_count: u32,
}

impl InteractionConstraintIndex {
    /// Build the lookup from `TrainParams.interaction_constraints`.  Returns
    /// `None` when no constraints are configured so callers can skip the
    /// per-node bookkeeping entirely.  Validation of group count / indices
    /// happens earlier in `core::validate_train_params`; this routine just
    /// re-checks bounds defensively (the engine doesn't always know the
    /// feature count at param-validation time).
    pub(crate) fn from_constraints(
        constraints: &[Vec<u32>],
        feature_count: usize,
    ) -> EngineResult<Option<Self>> {
        if constraints.is_empty() {
            return Ok(None);
        }
        if constraints.len() > 64 {
            return Err(EngineError::InvalidConfig(format!(
                "interaction_constraints supports at most 64 groups (got {})",
                constraints.len()
            )));
        }
        let mut feature_groups = vec![0u64; feature_count];
        for (gi, group) in constraints.iter().enumerate() {
            let bit = 1u64 << gi;
            for &f in group {
                let fi = f as usize;
                if fi >= feature_count {
                    return Err(EngineError::InvalidConfig(format!(
                        "interaction_constraints group {gi} references feature {f} which exceeds feature_count {feature_count}"
                    )));
                }
                feature_groups[fi] |= bit;
            }
        }
        Ok(Some(Self {
            feature_groups,
            group_count: constraints.len() as u32,
        }))
    }

    /// Bitset of all groups marked active at the tree root.  All groups
    /// start active and are intersected as a path descends through
    /// constrained features.
    pub(crate) fn root_active_groups(&self) -> u64 {
        if self.group_count == 0 {
            0
        } else if self.group_count >= 64 {
            u64::MAX
        } else {
            (1u64 << self.group_count) - 1
        }
    }

    /// Compute the `active_groups` bitset for a child node when the parent
    /// splits on `split_feature`.  Splitting on an unconstrained feature
    /// leaves the active set unchanged; splitting on a constrained feature
    /// narrows the set to groups that *also* contain that feature.
    #[inline]
    pub(crate) fn descend(&self, active_groups: u64, split_feature: u32) -> u64 {
        let f = split_feature as usize;
        if f >= self.feature_groups.len() {
            return active_groups;
        }
        let fg = self.feature_groups[f];
        if fg == 0 {
            active_groups
        } else {
            active_groups & fg
        }
    }

    /// Whether `feature` is allowed at a node whose ancestors imply
    /// `active_groups`.  Unconstrained features are always allowed; a
    /// constrained feature is allowed iff some group containing it is still
    /// active.
    #[inline]
    pub(crate) fn feature_allowed(&self, active_groups: u64, feature: u32) -> bool {
        let f = feature as usize;
        if f >= self.feature_groups.len() {
            return true;
        }
        let fg = self.feature_groups[f];
        fg == 0 || (active_groups & fg) != 0
    }
}

/// Clone a [`HistogramBundle`] keeping only the per-feature histograms whose
/// feature index satisfies `is_allowed`.  Used as a per-node filter for
/// interaction constraints — child histograms are still built with the
/// parent's tiles (so the subtraction trick keeps working), but the split
/// search at a constrained node ignores feature columns that aren't allowed
/// on this path.
pub(crate) fn filter_histogram_bundle_by_features(
    bundle: &HistogramBundle,
    is_allowed: impl Fn(u32) -> bool,
) -> HistogramBundle {
    bundle.filtered(is_allowed)
}
