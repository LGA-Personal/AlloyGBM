use std::cell::RefCell;

use rayon::prelude::*;

use crate::error::{EngineError, EngineResult};

/// Row count from which per-row loss terms are computed on several threads.
const PARALLEL_LOSS_MIN_ROWS: usize = 8_192;
const PARALLEL_LOSS_CHUNK_ROWS: usize = 4_096;

thread_local! {
    /// Reused per-row loss terms, so the parallel path allocates once per
    /// thread rather than once per round.
    static LOSS_TERMS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn squared_error_loss(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<f32> {
    if predictions.len() != targets.len() {
        return Err(EngineError::ContractViolation(format!(
            "predictions length {} does not match targets length {}",
            predictions.len(),
            targets.len()
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }

    Ok(squared_error_loss_unchecked(
        predictions,
        targets,
        sample_weights,
    ))
}

fn squared_error_loss_unchecked(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> f32 {
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }
    let sum = if let Some(weights) = sample_weights {
        let mut total = 0.0_f32;
        for index in 0..n {
            let residual = predictions[index] - targets[index];
            total += residual * residual * weights[index];
        }
        total
    } else {
        // Unrolled 4-wide accumulation for auto-vectorization
        let mut sum0 = 0.0_f32;
        let mut sum1 = 0.0_f32;
        let mut sum2 = 0.0_f32;
        let mut sum3 = 0.0_f32;
        let chunks = n / 4;
        for i in 0..chunks {
            let base = i * 4;
            let r0 = predictions[base] - targets[base];
            let r1 = predictions[base + 1] - targets[base + 1];
            let r2 = predictions[base + 2] - targets[base + 2];
            let r3 = predictions[base + 3] - targets[base + 3];
            sum0 += r0 * r0;
            sum1 += r1 * r1;
            sum2 += r2 * r2;
            sum3 += r3 * r3;
        }
        let mut total = sum0 + sum1 + sum2 + sum3;
        for index in (chunks * 4)..n {
            let residual = predictions[index] - targets[index];
            total += residual * residual;
        }
        total
    };
    // Return mean squared error (not sum) for scale-independent loss values.
    sum / n as f32
}

pub(crate) fn binary_crossentropy_loss(
    predictions: &[f32],
    targets: &[f32],
    sample_weights: Option<&[f32]>,
) -> EngineResult<f32> {
    if predictions.len() != targets.len() {
        return Err(EngineError::ContractViolation(format!(
            "predictions length {} does not match targets length {}",
            predictions.len(),
            targets.len()
        )));
    }
    if let Some(weights) = sample_weights
        && weights.len() != targets.len()
    {
        return Err(EngineError::ContractViolation(format!(
            "weights length {} does not match targets length {}",
            weights.len(),
            targets.len()
        )));
    }
    // Numerically stable log-loss: -[y*log(p) + (1-y)*log(1-p)]
    // where p = sigmoid(prediction) and prediction is in logit space.
    // Stable formulation: max(pred,0) - pred*y + log(1 + exp(-|pred|))
    let n = predictions.len();
    if n == 0 {
        return Ok(0.0);
    }
    let row_term = |index: usize| {
        let pred = predictions[index];
        let y = targets[index];
        let weight = sample_weights.map_or(1.0, |w| w[index]);
        let loss = pred.max(0.0) - pred * y + (1.0 + (-pred.abs()).exp()).ln();
        loss * weight
    };
    let total = if n >= PARALLEL_LOSS_MIN_ROWS && rayon::current_num_threads() > 1 {
        // The exp/ln per row dominates, so the row terms are computed in
        // parallel and then summed in index order: the same additions in the
        // same order as the serial loop, so the loss is bit-identical.
        LOSS_TERMS.with(|cell| {
            let mut terms = cell.borrow_mut();
            terms.clear();
            terms.resize(n, 0.0);
            terms
                .par_chunks_mut(PARALLEL_LOSS_CHUNK_ROWS)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    let start = chunk_index * PARALLEL_LOSS_CHUNK_ROWS;
                    for (offset, term) in chunk.iter_mut().enumerate() {
                        *term = row_term(start + offset);
                    }
                });
            let mut total = 0.0_f32;
            for &term in terms.iter() {
                total += term;
            }
            total
        })
    } else {
        let mut total = 0.0_f32;
        for index in 0..n {
            total += row_term(index);
        }
        total
    };
    // Return mean log-loss (not sum) for scale-independent loss values.
    Ok(total / n as f32)
}
