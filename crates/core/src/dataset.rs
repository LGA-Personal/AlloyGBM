use crate::error::{CoreError, CoreResult};
use crate::neutralization::FactorExposureMatrix;
use crate::{validate_columnar_matrix_view, validate_dataset_matrix, validate_dense_matrix_view};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSchema {
    pub feature_count: usize,
    pub has_time_index: bool,
    pub has_group_id: bool,
    pub categorical_feature_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasetMatrix {
    pub row_count: usize,
    pub feature_count: usize,
    pub values: Vec<f32>,
}

impl DatasetMatrix {
    pub fn new(row_count: usize, feature_count: usize, values: Vec<f32>) -> CoreResult<Self> {
        let matrix = Self {
            row_count,
            feature_count,
            values,
        };
        validate_dataset_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Create a lightweight matrix that only stores row/feature dimensions.
    /// Values are not populated — only use when the training path does not
    /// need dense float values (i.e. no categorical target encoding).
    pub fn new_metadata_only(row_count: usize, feature_count: usize) -> CoreResult<Self> {
        if row_count == 0 {
            return Err(CoreError::Validation(
                "row_count must be greater than 0".to_string(),
            ));
        }
        if feature_count == 0 {
            return Err(CoreError::Validation(
                "feature_count must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            row_count,
            feature_count,
            values: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseMatrixView<'a> {
    pub row_count: usize,
    pub feature_count: usize,
    pub values: &'a [f32],
}

impl<'a> DenseMatrixView<'a> {
    pub fn new(row_count: usize, feature_count: usize, values: &'a [f32]) -> CoreResult<Self> {
        let view = Self {
            row_count,
            feature_count,
            values,
        };
        validate_dense_matrix_view(&view)?;
        Ok(view)
    }

    pub fn row(&self, row_index: usize) -> CoreResult<&'a [f32]> {
        if row_index >= self.row_count {
            return Err(CoreError::Validation(format!(
                "row index {row_index} is out of bounds for row_count {}",
                self.row_count
            )));
        }
        let start = row_index * self.feature_count;
        let end = start + self.feature_count;
        Ok(&self.values[start..end])
    }

    pub fn value_at(&self, row_index: usize, feature_index: usize) -> CoreResult<f32> {
        if feature_index >= self.feature_count {
            return Err(CoreError::Validation(format!(
                "feature index {feature_index} is out of bounds for feature_count {}",
                self.feature_count
            )));
        }
        Ok(self.row(row_index)?[feature_index])
    }

    pub fn to_dataset_matrix(&self) -> CoreResult<DatasetMatrix> {
        DatasetMatrix::new(self.row_count, self.feature_count, self.values.to_vec())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnarMatrixColumnView<'a> {
    pub values: &'a [f32],
    pub validity: Option<&'a [bool]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnarMatrixView<'a> {
    pub row_count: usize,
    pub columns: Vec<ColumnarMatrixColumnView<'a>>,
}

impl<'a> ColumnarMatrixView<'a> {
    pub fn new(row_count: usize, columns: Vec<ColumnarMatrixColumnView<'a>>) -> CoreResult<Self> {
        let view = Self { row_count, columns };
        validate_columnar_matrix_view(&view)?;
        Ok(view)
    }

    pub fn feature_count(&self) -> usize {
        self.columns.len()
    }

    pub fn value_at(&self, row_index: usize, feature_index: usize) -> CoreResult<Option<f32>> {
        if row_index >= self.row_count {
            return Err(CoreError::Validation(format!(
                "row index {row_index} is out of bounds for row_count {}",
                self.row_count
            )));
        }
        let column = self.columns.get(feature_index).ok_or_else(|| {
            CoreError::Validation(format!(
                "feature index {feature_index} is out of bounds for feature_count {}",
                self.feature_count()
            ))
        })?;
        if column.validity.is_some_and(|mask| !mask[row_index]) {
            return Ok(None);
        }
        Ok(Some(column.values[row_index]))
    }

    pub fn to_dataset_matrix(&self, null_fill_value: f32) -> CoreResult<DatasetMatrix> {
        let mut values = Vec::with_capacity(self.row_count * self.feature_count());
        for row_index in 0..self.row_count {
            for feature_index in 0..self.feature_count() {
                values.push(
                    self.value_at(row_index, feature_index)?
                        .unwrap_or(null_fill_value),
                );
            }
        }
        DatasetMatrix::new(self.row_count, self.feature_count(), values)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingDataset {
    pub matrix: DatasetMatrix,
    pub targets: Vec<f32>,
    pub sample_weights: Option<Vec<f32>>,
    pub time_index: Option<Vec<i64>>,
    pub group_id: Option<Vec<u32>>,
    pub factor_exposures: Option<FactorExposureMatrix>,
}

impl TrainingDataset {
    pub fn row_count(&self) -> usize {
        self.matrix.row_count
    }
}
