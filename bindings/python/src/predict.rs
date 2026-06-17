use crate::MAX_CONTINUOUS_QUANTIZED_BIN_U8;
use crate::errors::predictor_error_to_pyerr;
use crate::quantization::{
    quantize_dense_values_linear_inplace_wide, quantize_dense_values_linear_rank_inplace_wide,
    quantize_linear_value,
};
use alloygbm_engine::{ArtifactCompatibilityMode, TrainedModel};
use alloygbm_predictor::{Predictor, PredictorError};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// -- Piece A: NativePredictorHandle ------------------------------------------

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct NativePredictorHandle {
    predictor: Predictor,
}

#[pymethods]
impl NativePredictorHandle {
    #[new]
    #[pyo3(signature = (artifact_bytes, strict=true))]
    fn new(artifact_bytes: &[u8], strict: bool) -> PyResult<Self> {
        let predictor = load_predictor_from_artifact_impl(artifact_bytes, strict)
            .map_err(predictor_error_to_pyerr)?;
        Ok(Self { predictor })
    }

    fn predict_batch(&self, rows: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch(&rows)
            .map_err(predictor_error_to_pyerr)
    }

    fn predict_dense(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
    ) -> PyResult<Vec<f32>> {
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &values,
        )
        .map_err(predictor_error_to_pyerr)
    }

    /// Predict from a numpy array (zero-copy). Requires float thresholds converted.
    fn predict_numpy(&self, array: PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
        let shape = array.shape();
        let row_count = shape[0];
        let feature_count = shape[1];
        let array_view = array.as_array();
        // Access the underlying contiguous slice (zero-copy)
        let values = array_view.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("numpy array must be C-contiguous")
        })?;
        self.predictor
            .predict_batch_dense(values, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }

    /// Predict from raw f32 bytes — zero Python-to-Rust list overhead.
    /// Requires float thresholds to be converted first (convert_thresholds_to_float).
    fn predict_dense_float_bytes(
        &self,
        values_bytes: &[u8],
        row_count: usize,
        feature_count: usize,
    ) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch_dense_bytes(values_bytes, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }

    /// Quantize raw f32 bytes to bins using linear scaling, then predict.
    /// Single-pass: fuses bytes→f32 conversion with quantization (one allocation).
    fn predict_dense_quantized_linear_bytes(
        &self,
        values_bytes: &[u8],
        row_count: usize,
        feature_count: usize,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
        if !values_bytes.len().is_multiple_of(4) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "values_bytes length must be a multiple of 4 (f32)",
            ));
        }
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_mins/feature_maxs length must match feature_count",
            ));
        }
        // Fused bytes→f32+quantize (single allocation, parallel), then predict.
        let total = row_count * feature_count;
        let mut quantized = vec![0.0_f32; total];
        let chunk_size = 4096.max(row_count / rayon::current_num_threads().max(1));
        quantized
            .par_chunks_mut(chunk_size * feature_count)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let rows_in_chunk = out_chunk.len() / feature_count;
                for local_row in 0..rows_in_chunk {
                    let row_index = row_start + local_row;
                    let byte_base = row_index * feature_count * 4;
                    let out_base = local_row * feature_count;
                    for fi in 0..feature_count {
                        let bi = byte_base + fi * 4;
                        let value = f32::from_ne_bytes([
                            values_bytes[bi],
                            values_bytes[bi + 1],
                            values_bytes[bi + 2],
                            values_bytes[bi + 3],
                        ]);
                        // v0.9.0 Limitation 4 fix: preserve NaN through the
                        // f32 cast so the predictor's `is_nan` check fires
                        // and routes through `default_left`.
                        out_chunk[out_base + fi] = if value.is_nan() {
                            f32::NAN
                        } else {
                            quantize_linear_value(value, feature_mins[fi], feature_maxs[fi]) as f32
                        };
                    }
                }
            });
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &quantized,
        )
        .map_err(predictor_error_to_pyerr)
    }

    /// Quantize raw float values to bins using linear scaling, then predict.
    /// Avoids the Python-side quantization loop (1.95B iterations for 2.5M×780).
    #[pyo3(signature = (values, row_count, feature_count, feature_mins, feature_maxs, max_data_bin=None))]
    fn predict_dense_quantized_linear(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        max_data_bin: Option<u16>,
    ) -> PyResult<Vec<f32>> {
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_mins/feature_maxs length must match feature_count",
            ));
        }
        let mdb = max_data_bin.unwrap_or(MAX_CONTINUOUS_QUANTIZED_BIN_U8);
        let quantized = quantize_dense_values_linear_inplace_wide(
            &values,
            row_count,
            feature_count,
            &feature_mins,
            &feature_maxs,
            mdb,
        );
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &quantized,
        )
        .map_err(predictor_error_to_pyerr)
    }

    /// Convert bin-index thresholds to float thresholds using per-feature min/max.
    /// After calling this, predict_dense works directly on raw floats — no quantization needed.
    /// `max_data_bin` is the maximum data bin index (e.g. 254 for 256 bins, 510 for 512 bins).
    fn convert_thresholds_to_float(
        &mut self,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        max_data_bin: u16,
    ) -> PyResult<()> {
        self.predictor
            .convert_bin_thresholds_to_float(&feature_mins, &feature_maxs, max_data_bin)
            .map_err(predictor_error_to_pyerr)
    }

    /// Convert bin-index thresholds to float thresholds using per-feature quantile cuts.
    fn convert_thresholds_to_float_quantile(
        &mut self,
        feature_cuts: Vec<Vec<f32>>,
    ) -> PyResult<()> {
        self.predictor
            .convert_bin_thresholds_to_float_quantile(&feature_cuts)
            .map_err(predictor_error_to_pyerr)
    }

    /// Convert bin-index thresholds to float thresholds for pre-binned integer data.
    fn convert_thresholds_to_float_prebinned(&mut self) -> PyResult<()> {
        self.predictor
            .convert_bin_thresholds_to_float_prebinned()
            .map_err(predictor_error_to_pyerr)
    }

    /// Quantize raw float values using selective rank (linear + rank fallback), then predict.
    #[pyo3(signature = (values, row_count, feature_count, feature_mins, feature_maxs, rank_flags, feature_sorted_values, max_data_bin=None))]
    fn predict_dense_quantized_linear_rank(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
        feature_mins: Vec<f32>,
        feature_maxs: Vec<f32>,
        rank_flags: Vec<bool>,
        feature_sorted_values: Vec<Vec<f32>>,
        max_data_bin: Option<u16>,
    ) -> PyResult<Vec<f32>> {
        if feature_mins.len() != feature_count || feature_maxs.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature_mins/feature_maxs length must match feature_count",
            ));
        }
        if rank_flags.len() != feature_count || feature_sorted_values.len() != feature_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rank_flags/feature_sorted_values length must match feature_count",
            ));
        }
        let mdb = max_data_bin.unwrap_or(MAX_CONTINUOUS_QUANTIZED_BIN_U8);
        let quantized = quantize_dense_values_linear_rank_inplace_wide(
            &values,
            row_count,
            feature_count,
            &feature_mins,
            &feature_maxs,
            &rank_flags,
            &feature_sorted_values,
            mdb,
        );
        predictor_predict_batch_dense_with_predictor(
            &self.predictor,
            row_count,
            feature_count,
            &quantized,
        )
        .map_err(predictor_error_to_pyerr)
    }

    // -- Multi-class prediction -----------------------------------------------

    fn is_multiclass(&self) -> bool {
        self.predictor.is_multiclass()
    }

    fn num_classes(&self) -> Option<usize> {
        self.predictor.num_classes()
    }

    /// Multi-class prediction returning flat Vec of length n_rows * K.
    fn predict_multiclass(&self, rows: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch_multiclass(&rows)
            .map_err(predictor_error_to_pyerr)
    }

    /// Multi-class prediction from dense flat array.
    fn predict_dense_multiclass(
        &self,
        values: Vec<f32>,
        row_count: usize,
        feature_count: usize,
    ) -> PyResult<Vec<f32>> {
        self.predictor
            .predict_batch_dense_multiclass(&values, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }

    /// Multi-class prediction from a numpy array (zero-copy).
    fn predict_numpy_multiclass(&self, array: PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
        let shape = array.shape();
        let row_count = shape[0];
        let feature_count = shape[1];
        let array_view = array.as_array();
        let values = array_view.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("numpy array must be C-contiguous")
        })?;
        self.predictor
            .predict_batch_dense_multiclass(values, row_count, feature_count)
            .map_err(predictor_error_to_pyerr)
    }
}

// -- Piece B: predict/shap implementation functions --------------------------

fn load_predictor_from_artifact_impl(
    artifact_bytes: &[u8],
    strict: bool,
) -> Result<Predictor, PredictorError> {
    if strict {
        TrainedModel::from_artifact_bytes_with_mode(
            artifact_bytes,
            ArtifactCompatibilityMode::Strict,
        )
        .map_err(|error| {
            PredictorError::ContractViolation(format!(
                "canonical predictor path requires strict dual-section artifact: {error}"
            ))
        })?;
    }
    Predictor::from_artifact_bytes(artifact_bytes)
}

pub(crate) fn predictor_predict_batch_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, false)?;
    predictor.predict_batch(rows)
}

fn predictor_predict_batch_dense_with_predictor(
    predictor: &Predictor,
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<Vec<f32>, PredictorError> {
    predictor.predict_batch_dense(values, row_count, feature_count)
}

fn predictor_predict_batch_dense_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, false)?;
    predictor_predict_batch_dense_with_predictor(&predictor, row_count, feature_count, values)
}

pub(crate) fn predictor_predict_batch_canonical_impl(
    artifact_bytes: &[u8],
    rows: &[Vec<f32>],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, true)?;
    predictor.predict_batch(rows)
}

fn predictor_predict_batch_canonical_dense_impl(
    artifact_bytes: &[u8],
    row_count: usize,
    feature_count: usize,
    values: &[f32],
) -> Result<Vec<f32>, PredictorError> {
    let predictor = load_predictor_from_artifact_impl(artifact_bytes, true)?;
    predictor_predict_batch_dense_with_predictor(&predictor, row_count, feature_count, values)
}

#[pyfunction]
pub(crate) fn predictor_predict_batch(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    predictor_predict_batch_impl(artifact_bytes, &rows).map_err(predictor_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn predictor_predict_batch_dense(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<Vec<f32>> {
    predictor_predict_batch_dense_impl(artifact_bytes, row_count, feature_count, &values)
        .map_err(predictor_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn predictor_predict_batch_canonical(
    artifact_bytes: &[u8],
    rows: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    predictor_predict_batch_canonical_impl(artifact_bytes, &rows).map_err(predictor_error_to_pyerr)
}

#[pyfunction]
pub(crate) fn predictor_predict_batch_canonical_dense(
    artifact_bytes: &[u8],
    values: Vec<f32>,
    row_count: usize,
    feature_count: usize,
) -> PyResult<Vec<f32>> {
    predictor_predict_batch_canonical_dense_impl(artifact_bytes, row_count, feature_count, &values)
        .map_err(predictor_error_to_pyerr)
}
