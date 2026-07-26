use std::collections::HashMap;

use alloygbm_core::LeafValue;

use crate::tree_node::{decode_tree_node_id, left_child_node_id, right_child_node_id};
use crate::{EngineError, EngineResult, TrainedStump};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MonotoneBounds {
    pub(crate) lower: f32,
    pub(crate) upper: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundedChildren {
    pub(crate) left_output: f32,
    pub(crate) right_output: f32,
    pub(crate) left_bounds: MonotoneBounds,
    pub(crate) right_bounds: MonotoneBounds,
}

pub(crate) fn has_active_monotone_constraints(constraints: &[i8]) -> bool {
    constraints.iter().any(|&constraint| constraint != 0)
}

pub(crate) fn monotone_constraint_for_feature(constraints: &[i8], feature_index: u32) -> i8 {
    constraints
        .get(feature_index as usize)
        .copied()
        .unwrap_or(0)
}

impl MonotoneBounds {
    pub(crate) fn root(max_abs_leaf_value: f32) -> EngineResult<Self> {
        if !max_abs_leaf_value.is_finite() || max_abs_leaf_value < 0.0 {
            return Err(EngineError::InvalidConfig(format!(
                "max_abs_leaf_value must be finite and non-negative, got {max_abs_leaf_value}"
            )));
        }
        Ok(Self {
            lower: -max_abs_leaf_value,
            upper: max_abs_leaf_value,
        })
    }

    pub(crate) fn bound_children(
        self,
        constraint: i8,
        left_raw: f32,
        right_raw: f32,
    ) -> EngineResult<BoundedChildren> {
        if !self.lower.is_finite() || !self.upper.is_finite() || self.lower > self.upper {
            return Err(EngineError::InvalidConfig(format!(
                "monotone bounds must be finite and ordered, got [{}, {}]",
                self.lower, self.upper
            )));
        }
        if !left_raw.is_finite() || !right_raw.is_finite() {
            return Err(EngineError::InvalidConfig(format!(
                "monotone child outputs must be finite, got left={left_raw}, right={right_raw}"
            )));
        }

        let left = left_raw.clamp(self.lower, self.upper);
        let right = right_raw.clamp(self.lower, self.upper);
        let bounded = match constraint {
            0 => BoundedChildren {
                left_output: left,
                right_output: right,
                left_bounds: self,
                right_bounds: self,
            },
            1 | -1 => {
                let midpoint = ((f64::from(left) + f64::from(right)) * 0.5)
                    .clamp(f64::from(self.lower), f64::from(self.upper))
                    as f32;
                let (left_bounds, right_bounds) = if constraint == 1 {
                    (
                        MonotoneBounds {
                            lower: self.lower,
                            upper: midpoint,
                        },
                        MonotoneBounds {
                            lower: midpoint,
                            upper: self.upper,
                        },
                    )
                } else {
                    (
                        MonotoneBounds {
                            lower: midpoint,
                            upper: self.upper,
                        },
                        MonotoneBounds {
                            lower: self.lower,
                            upper: midpoint,
                        },
                    )
                };
                BoundedChildren {
                    left_output: left.clamp(left_bounds.lower, left_bounds.upper),
                    right_output: right.clamp(right_bounds.lower, right_bounds.upper),
                    left_bounds,
                    right_bounds,
                }
            }
            _ => {
                return Err(EngineError::InvalidConfig(format!(
                    "monotone constraint must be -1, 0, or 1, got {constraint}"
                )));
            }
        };

        if !bounded.left_output.is_finite() || !bounded.right_output.is_finite() {
            return Err(EngineError::InvalidConfig(
                "bounded monotone child outputs must be finite".to_string(),
            ));
        }
        Ok(bounded)
    }
}

pub(crate) fn project_monotone_tree(
    stumps: &mut [TrainedStump],
    constraints: &[i8],
    max_abs_leaf_value: f32,
) -> EngineResult<()> {
    if !has_active_monotone_constraints(constraints) {
        return Ok(());
    }

    let mut stumps_by_local = HashMap::with_capacity(stumps.len());
    let mut tree_id = None;
    for (index, stump) in stumps.iter().enumerate() {
        let (stump_tree_id, local_node_id) = decode_tree_node_id(stump.split.node_id);
        if let Some(expected_tree_id) = tree_id {
            if stump_tree_id != expected_tree_id {
                return Err(EngineError::InvalidConfig(format!(
                    "monotone tree contains mixed tree ids {expected_tree_id} and {stump_tree_id}"
                )));
            }
        } else {
            tree_id = Some(stump_tree_id);
        }
        if stumps_by_local.insert(local_node_id, index).is_some() {
            return Err(EngineError::InvalidConfig(format!(
                "monotone tree contains duplicate local node id {local_node_id}"
            )));
        }
        scalar_leaf_value(&stump.left_leaf_value, local_node_id, "left")?;
        scalar_leaf_value(&stump.right_leaf_value, local_node_id, "right")?;
    }

    if !stumps_by_local.contains_key(&0) {
        return Err(EngineError::InvalidConfig(
            "monotone tree is missing local root node id 0".to_string(),
        ));
    }

    let root_bounds = MonotoneBounds::root(max_abs_leaf_value)?;
    let mut visited = vec![false; stumps.len()];
    project_monotone_node(
        0,
        0.0,
        root_bounds,
        stumps,
        &stumps_by_local,
        constraints,
        &mut visited,
    )?;

    if let Some((index, _)) = visited.iter().enumerate().find(|(_, seen)| !**seen) {
        let (_, local_node_id) = decode_tree_node_id(stumps[index].split.node_id);
        return Err(EngineError::InvalidConfig(format!(
            "monotone tree contains disconnected local node id {local_node_id}"
        )));
    }
    Ok(())
}

pub(crate) fn project_monotone_forest(
    stumps: &mut [TrainedStump],
    stumps_per_round: &[usize],
    constraints: &[i8],
    max_abs_leaf_value: f32,
) -> EngineResult<()> {
    if !has_active_monotone_constraints(constraints) {
        return Ok(());
    }
    MonotoneBounds::root(max_abs_leaf_value)?;

    let covered_stumps = stumps_per_round.iter().try_fold(0_usize, |total, &count| {
        total.checked_add(count).ok_or_else(|| {
            EngineError::InvalidConfig("monotone round stump counts overflow usize".to_string())
        })
    })?;
    if covered_stumps != stumps.len() {
        return Err(EngineError::InvalidConfig(format!(
            "monotone round stump counts cover {covered_stumps} stumps but forest contains {}",
            stumps.len()
        )));
    }

    let mut cursor = 0_usize;
    for &round_stump_count in stumps_per_round {
        let round_end = cursor + round_stump_count;
        if round_stump_count != 0 {
            project_monotone_tree(
                &mut stumps[cursor..round_end],
                constraints,
                max_abs_leaf_value,
            )?;
        }
        cursor = round_end;
    }
    Ok(())
}

pub(crate) fn validate_monotone_forest(
    stumps: &[TrainedStump],
    stumps_per_round: &[usize],
    constraints: &[i8],
    max_abs_leaf_value: f32,
) -> EngineResult<()> {
    if !has_active_monotone_constraints(constraints) {
        return Ok(());
    }

    let mut projected = stumps.to_vec();
    project_monotone_forest(
        &mut projected,
        stumps_per_round,
        constraints,
        max_abs_leaf_value,
    )?;

    for (original, bounded) in stumps.iter().zip(&projected) {
        if !stump_matches_projection(original, bounded) {
            let (tree_id, local_node_id) = decode_tree_node_id(original.split.node_id);
            return Err(EngineError::InvalidConfig(format!(
                "warm-start tree {tree_id} local node {local_node_id} violates the requested monotone contract"
            )));
        }
    }
    Ok(())
}

fn scalar_leaf_value(leaf: &LeafValue, local_node_id: u32, side: &str) -> EngineResult<f32> {
    match leaf {
        LeafValue::Scalar(value) if value.is_finite() => Ok(*value),
        LeafValue::Scalar(value) => Err(EngineError::InvalidConfig(format!(
            "monotone tree local node {local_node_id} has non-finite {side} scalar value {value}"
        ))),
        LeafValue::Linear(_) => Err(EngineError::InvalidConfig(format!(
            "active monotone constraints do not support linear leaves at local node {local_node_id}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn project_monotone_node(
    local_node_id: u32,
    parent_absolute: f32,
    bounds: MonotoneBounds,
    stumps: &mut [TrainedStump],
    stumps_by_local: &HashMap<u32, usize>,
    constraints: &[i8],
    visited: &mut [bool],
) -> EngineResult<()> {
    let index = stumps_by_local[&local_node_id];
    if visited[index] {
        return Err(EngineError::InvalidConfig(format!(
            "monotone tree revisited local node id {local_node_id}"
        )));
    }

    let feature_index = stumps[index].split.feature_index;
    let left_delta = scalar_leaf_value(&stumps[index].left_leaf_value, local_node_id, "left")?;
    let right_delta = scalar_leaf_value(&stumps[index].right_leaf_value, local_node_id, "right")?;
    let left_raw = parent_absolute + left_delta;
    let right_raw = parent_absolute + right_delta;
    let bounded = bounds.bound_children(
        monotone_constraint_for_feature(constraints, feature_index),
        left_raw,
        right_raw,
    )?;
    let bounded_left_delta = if bounded.left_output.to_bits() == left_raw.to_bits() {
        left_delta
    } else {
        bounded.left_output - parent_absolute
    };
    let bounded_right_delta = if bounded.right_output.to_bits() == right_raw.to_bits() {
        right_delta
    } else {
        bounded.right_output - parent_absolute
    };
    if !bounded_left_delta.is_finite() || !bounded_right_delta.is_finite() {
        return Err(EngineError::InvalidConfig(format!(
            "monotone tree local node {local_node_id} produced non-finite parent-relative deltas"
        )));
    }

    stumps[index].left_leaf_value = LeafValue::Scalar(bounded_left_delta);
    stumps[index].right_leaf_value = LeafValue::Scalar(bounded_right_delta);
    visited[index] = true;

    let left_local_node_id = left_child_node_id(local_node_id).map_err(|error| {
        EngineError::InvalidConfig(format!(
            "invalid left child for monotone local node {local_node_id}: {error}"
        ))
    })?;
    let right_local_node_id = right_child_node_id(local_node_id).map_err(|error| {
        EngineError::InvalidConfig(format!(
            "invalid right child for monotone local node {local_node_id}: {error}"
        ))
    })?;
    if stumps_by_local.contains_key(&left_local_node_id) {
        project_monotone_node(
            left_local_node_id,
            bounded.left_output,
            bounded.left_bounds,
            stumps,
            stumps_by_local,
            constraints,
            visited,
        )?;
    }
    if stumps_by_local.contains_key(&right_local_node_id) {
        project_monotone_node(
            right_local_node_id,
            bounded.right_output,
            bounded.right_bounds,
            stumps,
            stumps_by_local,
            constraints,
            visited,
        )?;
    }
    Ok(())
}

fn stump_matches_projection(original: &TrainedStump, bounded: &TrainedStump) -> bool {
    original.split == bounded.split
        && original.tree_weight == bounded.tree_weight
        && original.multi_output_leaf_values == bounded.multi_output_leaf_values
        && scalar_leaf_bits_match(&original.left_leaf_value, &bounded.left_leaf_value)
        && scalar_leaf_bits_match(&original.right_leaf_value, &bounded.right_leaf_value)
}

fn scalar_leaf_bits_match(original: &LeafValue, bounded: &LeafValue) -> bool {
    match (original, bounded) {
        (LeafValue::Scalar(original), LeafValue::Scalar(bounded)) => {
            original.to_bits() == bounded.to_bits()
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineError, TrainedStump, encode_tree_node_id};
    use alloygbm_core::{LeafValue, LinearLeaf, NodeStats, SplitCandidate};

    fn stump(
        tree_id: usize,
        local_node_id: u32,
        feature_index: u32,
        left: LeafValue,
        right: LeafValue,
    ) -> TrainedStump {
        TrainedStump::new_unweighted(
            SplitCandidate {
                node_id: encode_tree_node_id(tree_id, local_node_id)
                    .expect("test node id must encode"),
                feature_index,
                threshold_bin: 0,
                gain: 1.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 1.0,
                    grad_sq_sum: 0.0,
                    row_count: 1,
                },
                right_stats: NodeStats {
                    grad_sum: 0.0,
                    hess_sum: 1.0,
                    grad_sq_sum: 0.0,
                    row_count: 1,
                },
            },
            left,
            right,
        )
    }

    fn scalar_stump(
        tree_id: usize,
        local_node_id: u32,
        feature_index: u32,
        left: f32,
        right: f32,
    ) -> TrainedStump {
        stump(
            tree_id,
            local_node_id,
            feature_index,
            LeafValue::Scalar(left),
            LeafValue::Scalar(right),
        )
    }

    fn scalar_bits(stumps: &[TrainedStump]) -> Vec<(u32, u32)> {
        stumps
            .iter()
            .map(|stump| {
                let LeafValue::Scalar(left) = stump.left_leaf_value else {
                    panic!("test expected scalar left leaf");
                };
                let LeafValue::Scalar(right) = stump.right_leaf_value else {
                    panic!("test expected scalar right leaf");
                };
                (left.to_bits(), right.to_bits())
            })
            .collect()
    }

    #[test]
    fn increasing_reversed_children_meet_at_midpoint() {
        let bounded = MonotoneBounds {
            lower: -4.0,
            upper: 6.0,
        }
        .bound_children(1, 5.0, -1.0)
        .expect("valid bounds");
        assert_eq!(bounded.left_output, 2.0);
        assert_eq!(bounded.right_output, 2.0);
        assert_eq!(
            bounded.left_bounds,
            MonotoneBounds {
                lower: -4.0,
                upper: 2.0
            }
        );
        assert_eq!(
            bounded.right_bounds,
            MonotoneBounds {
                lower: 2.0,
                upper: 6.0
            }
        );
    }

    #[test]
    fn decreasing_reversed_children_meet_at_midpoint() {
        let bounded = MonotoneBounds {
            lower: -4.0,
            upper: 6.0,
        }
        .bound_children(-1, -2.0, 4.0)
        .expect("valid bounds");
        assert_eq!(bounded.left_output, 1.0);
        assert_eq!(bounded.right_output, 1.0);
    }

    #[test]
    fn ordered_children_keep_their_scalar_bits() {
        let left = f32::from_bits(0x3eaaaaab);
        let right = f32::from_bits(0x3f2aaaab);
        let bounded = MonotoneBounds {
            lower: -1.0,
            upper: 1.0,
        }
        .bound_children(1, left, right)
        .expect("valid bounds");
        assert_eq!(bounded.left_output.to_bits(), left.to_bits());
        assert_eq!(bounded.right_output.to_bits(), right.to_bits());
    }

    #[test]
    fn unconstrained_children_inherit_parent_bounds() {
        let parent = MonotoneBounds {
            lower: -3.0,
            upper: 7.0,
        };
        let bounded = parent.bound_children(0, -2.0, 4.0).expect("valid bounds");
        assert_eq!(bounded.left_output, -2.0);
        assert_eq!(bounded.right_output, 4.0);
        assert_eq!(bounded.left_bounds, parent);
        assert_eq!(bounded.right_bounds, parent);
    }

    #[test]
    fn child_outputs_are_clamped_to_parent_bounds() {
        let bounded = MonotoneBounds {
            lower: -2.0,
            upper: 3.0,
        }
        .bound_children(0, -5.0, 9.0)
        .expect("valid bounds");
        assert_eq!(bounded.left_output, -2.0);
        assert_eq!(bounded.right_output, 3.0);
    }

    #[test]
    fn constrained_child_bounds_split_at_midpoint_by_direction() {
        let parent = MonotoneBounds {
            lower: -4.0,
            upper: 6.0,
        };
        let increasing = parent
            .bound_children(1, -2.0, 4.0)
            .expect("valid increasing bounds");
        assert_eq!(
            increasing.left_bounds,
            MonotoneBounds {
                lower: -4.0,
                upper: 1.0
            }
        );
        assert_eq!(
            increasing.right_bounds,
            MonotoneBounds {
                lower: 1.0,
                upper: 6.0
            }
        );

        let decreasing = parent
            .bound_children(-1, 4.0, -2.0)
            .expect("valid decreasing bounds");
        assert_eq!(
            decreasing.left_bounds,
            MonotoneBounds {
                lower: 1.0,
                upper: 6.0
            }
        );
        assert_eq!(
            decreasing.right_bounds,
            MonotoneBounds {
                lower: -4.0,
                upper: 1.0
            }
        );
    }

    #[test]
    fn invalid_direction_is_rejected() {
        let error = MonotoneBounds {
            lower: -1.0,
            upper: 1.0,
        }
        .bound_children(2, 0.0, 0.0)
        .expect_err("unsupported direction must fail");
        assert!(matches!(error, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn invalid_parent_bounds_are_rejected() {
        for bounds in [
            MonotoneBounds {
                lower: 1.0,
                upper: -1.0,
            },
            MonotoneBounds {
                lower: f32::NEG_INFINITY,
                upper: 1.0,
            },
            MonotoneBounds {
                lower: -1.0,
                upper: f32::NAN,
            },
        ] {
            let error = bounds
                .bound_children(0, 0.0, 0.0)
                .expect_err("invalid parent bounds must fail");
            assert!(matches!(error, EngineError::InvalidConfig(_)));
        }
    }

    #[test]
    fn non_finite_child_values_are_rejected() {
        let bounds = MonotoneBounds {
            lower: -1.0,
            upper: 1.0,
        };
        for (left, right) in [
            (f32::NAN, 0.0),
            (0.0, f32::NAN),
            (f32::INFINITY, 0.0),
            (0.0, f32::NEG_INFINITY),
        ] {
            let error = bounds
                .bound_children(0, left, right)
                .expect_err("non-finite child must fail");
            assert!(matches!(error, EngineError::InvalidConfig(_)));
        }
    }

    #[test]
    fn root_rejects_invalid_caps() {
        for cap in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            let error = MonotoneBounds::root(cap).expect_err("invalid root cap must fail");
            assert!(matches!(error, EngineError::InvalidConfig(_)));
        }
    }

    #[test]
    fn root_accepts_zero_and_finite_positive_caps() {
        assert_eq!(
            MonotoneBounds::root(0.0).expect("zero cap is valid"),
            MonotoneBounds {
                lower: -0.0,
                upper: 0.0
            }
        );
        assert_eq!(
            MonotoneBounds::root(12.5).expect("positive cap is valid"),
            MonotoneBounds {
                lower: -12.5,
                upper: 12.5
            }
        );
    }

    #[test]
    fn midpoint_uses_f64_to_resist_f32_overflow() {
        let bounded = MonotoneBounds {
            lower: 0.0,
            upper: f32::MAX,
        }
        .bound_children(1, f32::MAX, f32::MAX)
        .expect("finite midpoint");
        assert_eq!(bounded.left_output, f32::MAX);
        assert_eq!(bounded.right_output, f32::MAX);
        assert!(bounded.left_bounds.upper.is_finite());
        assert!(bounded.right_bounds.lower.is_finite());
    }

    #[test]
    fn increasing_root_projection_collapses_reversed_absolute_children() {
        let mut stumps = vec![scalar_stump(0, 0, 0, 5.0, -1.0)];
        project_monotone_tree(&mut stumps, &[1], 10.0).expect("projection succeeds");
        assert_eq!(
            scalar_bits(&stumps),
            vec![(2.0_f32.to_bits(), 2.0_f32.to_bits())]
        );
    }

    #[test]
    fn descendant_projection_cannot_escape_its_inherited_interval() {
        let mut stumps = vec![
            scalar_stump(0, 0, 0, -2.0, 2.0),
            scalar_stump(0, 1, 0, 5.0, 6.0),
        ];
        project_monotone_tree(&mut stumps, &[1], 10.0).expect("projection succeeds");
        assert_eq!(
            scalar_bits(&stumps),
            vec![
                ((-2.0_f32).to_bits(), 2.0_f32.to_bits()),
                (2.0_f32.to_bits(), 2.0_f32.to_bits()),
            ]
        );
    }

    #[test]
    fn decreasing_root_projection_mirrors_increasing_order() {
        let mut stumps = vec![scalar_stump(0, 0, 0, -2.0, 4.0)];
        project_monotone_tree(&mut stumps, &[-1], 10.0).expect("projection succeeds");
        assert_eq!(
            scalar_bits(&stumps),
            vec![(1.0_f32.to_bits(), 1.0_f32.to_bits())]
        );
    }

    #[test]
    fn all_zero_constraints_leave_every_scalar_bit_unchanged() {
        let left = f32::from_bits(0x80000000);
        let right = f32::from_bits(0x3eaaaaab);
        let mut stumps = vec![
            scalar_stump(0, 0, 0, left, right),
            scalar_stump(0, 1, 1, f32::MAX, -f32::MAX),
        ];
        let before = scalar_bits(&stumps);
        project_monotone_forest(&mut stumps, &[2], &[0, 0], 0.0)
            .expect("unconstrained projection is a no-op");
        assert_eq!(scalar_bits(&stumps), before);
    }

    #[test]
    fn all_zero_constraints_leave_linear_leaves_untouched() {
        let linear = LinearLeaf::identity_scaled(1.0, vec![0.5], vec![0]);
        let mut stumps = vec![stump(
            0,
            0,
            0,
            LeafValue::Linear(linear.clone()),
            LeafValue::Linear(linear),
        )];
        let before = stumps.clone();
        project_monotone_tree(&mut stumps, &[0], 0.0).expect("unconstrained projection is a no-op");
        assert_eq!(stumps, before);
    }

    #[test]
    fn compliant_constrained_tree_keeps_every_scalar_bit() {
        let mut stumps = vec![
            scalar_stump(0, 0, 0, -2.0, 2.0),
            scalar_stump(
                0,
                1,
                1,
                f32::from_bits(0xbeaaaaab),
                f32::from_bits(0x3eaaaaab),
            ),
            scalar_stump(
                0,
                2,
                1,
                f32::from_bits(0xbe4ccccd),
                f32::from_bits(0x3e4ccccd),
            ),
        ];
        let before = scalar_bits(&stumps);
        project_monotone_tree(&mut stumps, &[1, 0], 10.0).expect("projection succeeds");
        assert_eq!(scalar_bits(&stumps), before);
    }

    #[test]
    fn active_constraints_reject_linear_leaves() {
        let linear = LinearLeaf::identity_scaled(1.0, vec![0.5], vec![0]);
        let mut stumps = vec![stump(
            0,
            0,
            0,
            LeafValue::Linear(linear),
            LeafValue::Scalar(2.0),
        )];
        let error = project_monotone_tree(&mut stumps, &[1], 10.0)
            .expect_err("linear leaves cannot be projected");
        assert!(matches!(error, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn active_constraints_reject_non_finite_scalar_leaves() {
        let mut stumps = vec![scalar_stump(0, 0, 0, f32::NAN, 1.0)];
        let error = project_monotone_tree(&mut stumps, &[1], 10.0)
            .expect_err("non-finite scalar leaves cannot be projected");
        assert!(matches!(error, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn duplicate_local_node_ids_are_rejected() {
        let mut stumps = vec![
            scalar_stump(0, 0, 0, -1.0, 1.0),
            scalar_stump(0, 0, 0, -2.0, 2.0),
        ];
        let error = project_monotone_tree(&mut stumps, &[1], 10.0)
            .expect_err("duplicate local ids must fail");
        assert!(matches!(error, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn missing_local_root_is_rejected() {
        let mut stumps = vec![scalar_stump(0, 1, 0, -1.0, 1.0)];
        let error = project_monotone_tree(&mut stumps, &[1], 10.0)
            .expect_err("missing local root must fail");
        assert!(matches!(error, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn disconnected_local_nodes_are_rejected() {
        let mut stumps = vec![
            scalar_stump(0, 0, 0, -1.0, 1.0),
            scalar_stump(0, 3, 0, -1.0, 1.0),
        ];
        let error = project_monotone_tree(&mut stumps, &[1], 10.0)
            .expect_err("disconnected local ids must fail");
        assert!(matches!(error, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn mixed_tree_ids_are_rejected() {
        let mut stumps = vec![
            scalar_stump(0, 0, 0, -1.0, 1.0),
            scalar_stump(1, 1, 0, -1.0, 1.0),
        ];
        let error =
            project_monotone_tree(&mut stumps, &[1], 10.0).expect_err("mixed tree ids must fail");
        assert!(matches!(error, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn malformed_round_counts_are_rejected() {
        let stump = scalar_stump(0, 0, 0, -1.0, 1.0);
        for counts in [vec![2], vec![0]] {
            let mut stumps = vec![stump.clone()];
            let error = project_monotone_forest(&mut stumps, &counts, &[1], 10.0)
                .expect_err("round counts must exactly partition stumps");
            assert!(matches!(error, EngineError::InvalidConfig(_)));
        }
    }

    #[test]
    fn forest_projection_accepts_zero_count_rounds() {
        let mut stumps = vec![scalar_stump(1, 0, 0, 5.0, -1.0)];
        project_monotone_forest(&mut stumps, &[0, 1], &[1], 10.0)
            .expect("zero-count round is valid");
        assert_eq!(
            scalar_bits(&stumps),
            vec![(2.0_f32.to_bits(), 2.0_f32.to_bits())]
        );
    }

    #[test]
    fn warm_start_validation_accepts_a_compliant_forest() {
        let stumps = vec![
            scalar_stump(0, 0, 0, -2.0, 2.0),
            scalar_stump(1, 0, 0, -1.0, 3.0),
        ];
        validate_monotone_forest(&stumps, &[1, 1], &[1], 10.0)
            .expect("compliant warm-start forest is valid");
    }

    #[test]
    fn warm_start_validation_rejects_a_tree_projection_would_modify() {
        let stumps = vec![scalar_stump(0, 0, 0, 5.0, -1.0)];
        let error = validate_monotone_forest(&stumps, &[1], &[1], 10.0)
            .expect_err("violating warm-start forest must fail");
        assert!(matches!(
            error,
            EngineError::InvalidConfig(message)
                if message.contains("warm-start tree")
                    && message.contains("monotone contract")
        ));
    }
}
