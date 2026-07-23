pub mod artifact_format;
pub mod binned;
pub mod config;
pub mod dataset;
pub mod dro;
pub mod error;
pub mod feature_bundling;
pub mod histogram;
pub mod leaf;
pub mod linear_histogram;
pub mod neutralization;
pub mod simd;
pub mod training_mode;
pub mod validation;

pub use artifact_format::{
    CATEGORICAL_STATE_FORMAT_V1, CategoricalStatePayloadV1, DartTreeWeightsPayload,
    DroMetadataPayload, FeatureBaselinePayload, LinearLeafCoefficientsPayload, LinearLeafEntry,
    MODEL_BINARY_HEADER_LEN, MODEL_BINARY_MAGIC, MODEL_FORMAT_V1, MODEL_SECTION_DESCRIPTOR_LEN,
    ModelArtifactSection, ModelBinaryHeader, ModelIoContractV1, ModelMetadata,
    ModelSectionDescriptor, ModelSectionKind, MorphMetadataPayload, MultiOutputLeafValuesPayload,
    NativeCategoricalSplitsPayload, NeutralizationMetadataPayload, ParsedModelArtifactV1,
    RequiredSectionCompatibilityReport, decode_categorical_state_payload_v1,
    decode_dart_tree_weights_payload, decode_dro_metadata_payload, decode_feature_baseline_payload,
    decode_linear_leaf_coefficients_payload, decode_multi_output_leaf_values_payload,
    decode_native_categorical_splits_payload, decode_neutralization_metadata_payload,
    decode_optional_categorical_state_section_v1, decode_optional_dart_tree_weights_section,
    decode_optional_dro_metadata_artifact_section, decode_optional_feature_baseline_section,
    decode_optional_linear_leaf_coefficients_section,
    decode_optional_morph_metadata_artifact_section, decode_optional_morph_metadata_section,
    decode_optional_multi_output_leaf_values_section,
    decode_optional_native_categorical_splits_section,
    decode_optional_neutralization_metadata_artifact_section, deserialize_metadata_json,
    deserialize_model_artifact_v1, encode_categorical_state_payload_v1,
    encode_dart_tree_weights_payload, encode_dro_metadata_payload, encode_feature_baseline_payload,
    encode_linear_leaf_coefficients_payload, encode_morph_metadata_payload,
    encode_multi_output_leaf_values_payload, encode_native_categorical_splits_payload,
    encode_neutralization_metadata_payload, format_required_section_auto_mode_error,
    format_required_section_mode_error, optional_single_section,
    required_section_compatibility_report, serialize_metadata_json, serialize_model_artifact_v1,
    validate_categorical_state_payload_v1,
};
pub use binned::{BinStorage, BinnedLayout, BinnedMatrix, MISSING_BIN_U8, MISSING_BIN_U16};
pub use config::{
    BoostingMode, DartNormalize, DartSampleType, Device, LeafModelKind, LeafSolverKind,
    MAX_TREE_NODE_SLOTS, TrainParams, TreeGrowth,
};
pub use dataset::{
    ColumnarMatrixColumnView, ColumnarMatrixView, DatasetMatrix, DatasetSchema, DenseMatrixView,
    TrainingDataset,
};
pub use dro::{DroConfig, DroMetric};
pub use error::{CoreError, CoreResult};
pub use feature_bundling::{
    FeatureBundleAssignment, FeatureBundleMap, count_exact_feature_bundle_conflicts,
    discover_exact_feature_bundles,
};
pub use histogram::{
    FeatureHistogram, FeatureTile, GradientPair, HistogramBin, HistogramBundle,
    HistogramFeatureView, NodeSlice, NodeStats, leaf_effective_gradient, leaf_gain_term,
};
pub use leaf::{LeafValue, LinearLeaf, PartitionResult, SplitCandidate};
pub use linear_histogram::{
    LinearFeatureHistogram, LinearFeatureScaler, LinearHistogramBin, LinearHistogramBundle,
    MAX_PL_MATRIX_ENTRIES, MAX_PL_REGRESSORS, pl_matrix_index, subtract_linear_histogram_bundle,
};
pub use neutralization::{FactorExposureMatrix, FactorNeutralizationConfig, NeutralizationKind};
pub use training_mode::{
    GradientEmaStats, LrSchedule, MorphConfig, MorphPrecomputed, TrainingMode,
};
pub use validation::{
    validate_binned_matrix, validate_columnar_matrix_view, validate_dataset_matrix,
    validate_dataset_schema, validate_dense_matrix_view, validate_model_contract_v1,
    validate_train_params, validate_training_dataset,
};

#[cfg(test)]
mod tests;
