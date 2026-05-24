mod binary;
mod squared;

pub use binary::BinaryCrossEntropyObjective;
pub use squared::SquaredErrorObjective;
pub(crate) use binary::sigmoid;
