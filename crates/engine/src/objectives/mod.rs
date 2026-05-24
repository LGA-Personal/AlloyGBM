mod binary;
mod glm;
mod squared;

pub use binary::BinaryCrossEntropyObjective;
pub use glm::{GammaObjective, PoissonObjective, TweedieObjective};
pub use squared::SquaredErrorObjective;
pub(crate) use binary::sigmoid;
