use std::env;
use std::fs;
use std::path::PathBuf;

use alloygbm_core::{LeafValue, NodeStats, SplitCandidate};
use alloygbm_engine::{TrainedModel, TrainedStump};

const TREE_NODE_STRIDE: u32 = 1 << 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let output = PathBuf::from(args.next().ok_or("missing output path")?);
    let tree_count: usize = args
        .next()
        .ok_or("missing tree count")?
        .to_string_lossy()
        .parse()?;
    let shape = args
        .next()
        .unwrap_or_else(|| "sparse".into())
        .to_string_lossy()
        .into_owned();
    if args.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    if shape != "sparse" && shape != "balanced" {
        return Err(format!("unknown shape {shape:?}").into());
    }

    let empty_stats = NodeStats {
        grad_sum: 0.0,
        hess_sum: 1.0,
        grad_sq_sum: 0.0,
        row_count: 1,
    };
    let nodes_per_tree = if shape == "sparse" { 16 } else { 7 };
    let mut stumps = Vec::with_capacity(tree_count * nodes_per_tree);
    for tree_index in 0..tree_count {
        let local_node_ids: Vec<u32> = if shape == "sparse" {
            let mut ids = Vec::with_capacity(16);
            let mut node_id = 0_u32;
            for _ in 0..16 {
                ids.push(node_id);
                node_id = node_id * 2 + 2;
            }
            ids
        } else {
            (0..7).collect()
        };
        for local_node_id in local_node_ids.iter().copied() {
            let split = SplitCandidate {
                node_id: tree_index as u32 * TREE_NODE_STRIDE + local_node_id,
                feature_index: 0,
                threshold_bin: 0,
                gain: 1.0,
                default_left: false,
                is_categorical: false,
                categorical_bitset: None,
                left_stats: empty_stats.clone(),
                right_stats: empty_stats.clone(),
            };
            stumps.push(TrainedStump::new_unweighted(
                split,
                LeafValue::Scalar(-0.01),
                LeafValue::Scalar(0.01),
            ));
        }
        if shape == "sparse" {
            debug_assert_eq!(local_node_ids.last(), Some(&65_534));
        }
    }
    let model = TrainedModel {
        baseline_prediction: 0.0,
        feature_count: 1,
        stumps,
        categorical_state: None,
        node_debug_stats: None,
        objective: "squared_error".to_string(),
        native_categorical_feature_indices: Vec::new(),
        morph_metadata: None,
        dro_metadata: None,
        feature_baseline: None,
        neutralization_metadata: None,
    };
    fs::write(output, model.to_artifact_bytes()?)?;
    Ok(())
}
