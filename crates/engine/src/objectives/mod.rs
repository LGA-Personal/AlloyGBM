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
