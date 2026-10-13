mod binary;
mod glm;
mod multiclass;
mod quantile;
mod ranking;
mod squared;

pub use binary::BinaryCrossEntropyObjective;
pub(crate) use binary::sigmoid;
pub use glm::{GammaObjective, PoissonObjective, TweedieObjective};
pub use multiclass::MultiClassSoftmaxObjective;
pub use quantile::QuantileObjective;
pub(crate) use quantile::{resolve_boundaries_for_len, weighted_quantile};
pub use ranking::{
    LambdaMARTObjective, PairwiseRankingObjective, QueryRMSEObjective, XeNDCGObjective,
    YetiRankObjective, compute_group_boundaries,
};
pub use squared::SquaredErrorObjective;

use alloygbm_core::GradientPair;
use rayon::prelude::*;

/// Rows per parallel chunk when filling row-wise gradients.
const ROWWISE_GRADIENT_CHUNK_ROWS: usize = 4_096;

/// Fill `buffer` with `row(index)` for every index in `0..len`.
///
/// Each row's gradient depends only on that row, so filling chunks on
/// several threads writes exactly the values the serial loop would: the
/// split affects speed only, never results.
pub(crate) fn fill_rowwise_gradients(
    buffer: &mut Vec<GradientPair>,
    len: usize,
    row: impl Fn(usize) -> GradientPair + Sync,
) {
    buffer.clear();
    if len >= 2 * ROWWISE_GRADIENT_CHUNK_ROWS && rayon::current_num_threads() > 1 {
        buffer.resize(
            len,
            GradientPair {
                grad: 0.0,
                hess: 0.0,
            },
        );
        buffer
            .par_chunks_mut(ROWWISE_GRADIENT_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * ROWWISE_GRADIENT_CHUNK_ROWS;
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    *slot = row(start + offset);
                }
            });
    } else {
        buffer.extend((0..len).map(row));
    }
}
