mod binary;
mod glm;
mod quantile;
mod ranking;
mod squared;

pub use binary::BinaryCrossEntropyObjective;
pub use glm::{GammaObjective, PoissonObjective, TweedieObjective};
pub use quantile::QuantileObjective;
pub use ranking::{
    LambdaMARTObjective, PairwiseRankingObjective, QueryRMSEObjective, XeNDCGObjective,
    YetiRankObjective, compute_group_boundaries,
};
pub use squared::SquaredErrorObjective;
pub(crate) use binary::sigmoid;
pub(crate) use quantile::{resolve_boundaries_for_len, weighted_quantile};
#[allow(unused_imports)]
pub(crate) use ranking::log_sum_exp;
