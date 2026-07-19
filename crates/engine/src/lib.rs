use alloygbm_core::{
    BinnedMatrix, BoostingMode, CategoricalStatePayloadV1, DroMetadataPayload, FeatureTile,
    GradientPair, LeafModelKind, MorphMetadataPayload, NodeSlice, TrainParams, TrainingDataset,
    TreeGrowth, validate_train_params, validate_training_dataset,
};
#[cfg(test)]
use alloygbm_core::{
    FactorExposureMatrix, HistogramBundle, LeafValue, PartitionResult, SplitCandidate,
};
#[cfg(test)]
use alloygbm_core::{MODEL_FORMAT_V1, ModelSectionKind, NodeStats};
#[cfg(test)]
use alloygbm_core::{ModelMetadata, deserialize_model_artifact_v1, serialize_model_artifact_v1};

mod error;
pub use error::{EngineError, EngineResult};

mod env;
mod tree_node;
pub(crate) use tree_node::*;

use env::{
    experiment_force_manual_policy_enabled, experiment_leaf_refinement_enabled,
    split_selection_options_from_env,
};

pub mod dart;
pub use dart::{DartState, apply_normalization, select_dropouts};

pub mod shared_histogram;
pub use shared_histogram::{
    HistComponent, MultiOutputHistogram, build_multi_output_histogram_inplace,
    compute_multi_output_split_gain, subtract_multi_output_histogram,
};

pub mod joint;
pub use joint::{JointObjective, JointRoundResult, JointWarmStartState, build_joint_round};

mod morph_state;
pub(crate) use morph_state::MorphTreeContext;
pub use morph_state::{MorphState, resolve_lr_schedule};

mod factor;
pub(crate) use factor::FactorProjector;

mod split_options;
pub use split_options::{
    CategoricalFeatureInfo, FactorSplitContext, LinearContext, MorphContext, SplitSelectionOptions,
};

mod traits;
pub use traits::{BackendOps, HistogramExecution, ObjectiveOps, PerRoundMetricCallback};

mod objectives;
pub use objectives::{
    BinaryCrossEntropyObjective, GammaObjective, LambdaMARTObjective, MultiClassSoftmaxObjective,
    PairwiseRankingObjective, PoissonObjective, QuantileObjective, QueryRMSEObjective,
    SquaredErrorObjective, TweedieObjective, XeNDCGObjective, YetiRankObjective,
    compute_group_boundaries,
};
mod multiclass_model;
pub use multiclass_model::{MultiClassIterationRunSummary, MultiClassTrainedModel};

mod types;
mod warm_start;
pub use types::{
    ArtifactCompatibilityMode, ArtifactCompatibilityReport, CategoricalTargetEncodingSpec,
    FitContractEvaluation, IterationControls, IterationDiagnostics, IterationRunSummary,
    IterationStopReason, NodeDebugStats, TrainRoundSummary, TrainedStump, TrainingPolicyMode,
    ValidationDatasetRef,
};
pub(crate) use types::{IterationExecutionContext, PolicyFitRequest, gradient_l2_norm_only};
pub use warm_start::{MultiClassWarmStartState, WarmStartState};

mod trained_model;
pub use trained_model::TrainedModel;

mod artifact;

mod loss;
pub(crate) use loss::{binary_crossentropy_loss, squared_error_loss};

mod sampling;
pub(crate) use sampling::*;

mod tiling;
pub(crate) use tiling::*;

mod round;
pub(crate) use round::*;

mod leaf_refinement;
pub(crate) use leaf_refinement::*;

mod trainer;
pub use trainer::Trainer;
pub(crate) use trainer::*;

#[cfg(test)]
mod tests;
