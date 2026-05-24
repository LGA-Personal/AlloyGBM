mod binary;
mod glm;
mod quantile;
mod squared;

pub use binary::BinaryCrossEntropyObjective;
pub use glm::{GammaObjective, PoissonObjective, TweedieObjective};
pub use quantile::QuantileObjective;
pub use squared::SquaredErrorObjective;
pub(crate) use binary::sigmoid;
pub(crate) use quantile::{resolve_boundaries_for_len, weighted_quantile};
