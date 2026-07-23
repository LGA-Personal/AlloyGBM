use crate::error::CoreResult;
use crate::validate_binned_matrix;
use crate::{CoreError, FeatureBundleMap};

pub const MISSING_BIN_U8: u8 = 255;
pub const MISSING_BIN_U16: u16 = 65535;

/// Adaptive bin storage: u8 for <=255 max bins, u16 for >255.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinStorage {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl BinStorage {
    /// Get the bin value at the given index as a u16.
    #[inline]
    pub fn get(&self, index: usize) -> u16 {
        match self {
            Self::U8(bins) => u16::from(bins[index]),
            Self::U16(bins) => bins[index],
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(bins) => bins.len(),
            Self::U16(bins) => bins.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sentinel value used for missing/NaN bins.
    pub fn missing_bin(&self) -> u16 {
        match self {
            Self::U8(_) => u16::from(MISSING_BIN_U8),
            Self::U16(_) => MISSING_BIN_U16,
        }
    }

    /// Whether this storage uses u8 bins.
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::U8(_))
    }

    #[inline]
    fn set(&mut self, index: usize, value: u16) {
        match self {
            Self::U8(bins) => bins[index] = value.min(u16::from(u8::MAX)) as u8,
            Self::U16(bins) => bins[index] = value,
        }
    }

    fn storage_bytes(&self) -> usize {
        match self {
            Self::U8(bins) => bins.len(),
            Self::U16(bins) => bins.len() * size_of::<u16>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinnedLayout {
    ColumnMajor,
    Dual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinnedMatrix {
    pub row_count: usize,
    pub feature_count: usize,
    pub max_bin: u16,
    /// The bin index used for NaN/missing values.
    /// For u8 mode: always 255 (MISSING_BIN_U8).
    /// For u16 mode: max_data_bin + 1 (dynamic, avoids wasteful 65535 sentinel).
    pub nan_bin_index: u16,
    pub(crate) storage_feature_count: usize,
    pub(crate) storage_max_bin: u16,
    pub(crate) feature_bundle_map: Option<FeatureBundleMap>,
    /// Optional row-major adaptive storage. Empty for column-major-only matrices.
    pub(crate) bins_adaptive: BinStorage,
    /// Required column-major adaptive storage used by histogram construction.
    pub(crate) bins_col_adaptive: BinStorage,
}

impl BinnedMatrix {
    /// Create a BinnedMatrix from u8 bins (max_bin <= 254).
    pub fn new(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        bins: Vec<u8>,
    ) -> CoreResult<Self> {
        Self::new_with_layout(row_count, feature_count, max_bin, bins, BinnedLayout::Dual)
    }

    pub fn new_with_layout(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        bins: Vec<u8>,
        layout: BinnedLayout,
    ) -> CoreResult<Self> {
        let bins_col = transpose_bins_to_column_major_u8(&bins, row_count, feature_count);
        let bins_adaptive = match layout {
            BinnedLayout::ColumnMajor => BinStorage::U8(Vec::new()),
            BinnedLayout::Dual => BinStorage::U8(bins),
        };
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index: MISSING_BIN_U8 as u16,
            storage_feature_count: feature_count,
            storage_max_bin: max_bin,
            feature_bundle_map: None,
            bins_adaptive,
            bins_col_adaptive: BinStorage::U8(bins_col),
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Create a column-major-only matrix from bins already ordered by feature, then row.
    pub fn new_from_column_major(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        bins_col: Vec<u8>,
    ) -> CoreResult<Self> {
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index: MISSING_BIN_U8 as u16,
            storage_feature_count: feature_count,
            storage_max_bin: max_bin,
            feature_bundle_map: None,
            bins_adaptive: BinStorage::U8(Vec::new()),
            bins_col_adaptive: BinStorage::U8(bins_col),
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Create a BinnedMatrix from u16 bins (max_bin > 254).
    /// `nan_bin_index` is the bin value used for NaN/missing data (typically max_data_bin + 1).
    pub fn new_u16(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        nan_bin_index: u16,
        bins_u16: Vec<u16>,
    ) -> CoreResult<Self> {
        Self::new_u16_with_layout(
            row_count,
            feature_count,
            max_bin,
            nan_bin_index,
            bins_u16,
            BinnedLayout::Dual,
        )
    }

    pub fn new_u16_with_layout(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        nan_bin_index: u16,
        bins_u16: Vec<u16>,
        layout: BinnedLayout,
    ) -> CoreResult<Self> {
        let bins_col_u16 = transpose_bins_to_column_major_u16(&bins_u16, row_count, feature_count);
        let bins_adaptive = match layout {
            BinnedLayout::ColumnMajor => BinStorage::U16(Vec::new()),
            BinnedLayout::Dual => BinStorage::U16(bins_u16),
        };
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index,
            storage_feature_count: feature_count,
            storage_max_bin: max_bin,
            feature_bundle_map: None,
            bins_adaptive,
            bins_col_adaptive: BinStorage::U16(bins_col_u16),
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Create a column-major-only wide matrix from bins ordered by feature, then row.
    pub fn new_u16_from_column_major(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        nan_bin_index: u16,
        bins_col_u16: Vec<u16>,
    ) -> CoreResult<Self> {
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index,
            storage_feature_count: feature_count,
            storage_max_bin: max_bin,
            feature_bundle_map: None,
            bins_adaptive: BinStorage::U16(Vec::new()),
            bins_col_adaptive: BinStorage::U16(bins_col_u16),
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Whether this matrix uses u16 bin storage.
    pub fn is_wide_bins(&self) -> bool {
        matches!(self.bins_col_adaptive, BinStorage::U16(_))
    }

    /// Read a bin value from column-major adaptive storage.
    /// `index` is the flat offset (feature * row_count + row).
    #[inline]
    pub fn col_bin(&self, index: usize) -> u16 {
        if self.feature_bundle_map.is_none() {
            return self.bins_col_adaptive.get(index);
        }
        let feature = index / self.row_count;
        let row = index % self.row_count;
        self.bin_at(row, feature)
    }

    /// Read a bin value from row-major adaptive storage.
    /// `index` is the flat offset (row * feature_count + feature).
    #[inline]
    pub fn row_bin(&self, index: usize) -> u16 {
        if self.has_row_major() {
            self.bins_adaptive.get(index)
        } else {
            let row = index / self.feature_count;
            let feature = index % self.feature_count;
            self.bin_at(row, feature)
        }
    }

    /// Read a bin by row and feature without converting between flat layouts.
    #[inline]
    pub fn bin_at(&self, row: usize, feature: usize) -> u16 {
        if let Some(map) = &self.feature_bundle_map {
            let assignment = map
                .assignment(feature)
                .expect("logical feature index is validated by callers");
            let storage_index = assignment.storage_feature * self.row_count + row;
            let storage_bin = self.bins_col_adaptive.get(storage_index);
            return assignment.decode(storage_bin, self.nan_bin_index, map.logical_missing_bin());
        }
        if self.has_row_major() {
            self.bins_adaptive.get(row * self.feature_count + feature)
        } else {
            self.bins_col_adaptive.get(feature * self.row_count + row)
        }
    }

    /// The sentinel value used for missing/NaN bins in this matrix.
    #[inline]
    pub fn missing_bin(&self) -> u16 {
        self.feature_bundle_map
            .as_ref()
            .map_or(self.nan_bin_index, FeatureBundleMap::logical_missing_bin)
    }

    /// Whether column-major adaptive storage is available (non-empty).
    #[inline]
    pub fn has_col_major(&self) -> bool {
        !self.bins_col_adaptive.is_empty()
    }

    #[inline]
    pub fn has_row_major(&self) -> bool {
        !self.bins_adaptive.is_empty()
    }

    pub fn layout(&self) -> BinnedLayout {
        if self.has_row_major() {
            BinnedLayout::Dual
        } else {
            BinnedLayout::ColumnMajor
        }
    }

    pub fn storage_bytes(&self) -> usize {
        self.bins_adaptive.storage_bytes() + self.bins_col_adaptive.storage_bytes()
    }

    pub fn cell_count(&self) -> usize {
        self.row_count * self.feature_count
    }

    /// Set the bin value at (row, feature) in all storage arrays.
    /// Used for re-mapping native categorical feature columns after binning.
    pub fn set_bin(&mut self, row: usize, feature: usize, value: u16) {
        assert!(
            self.feature_bundle_map.is_none(),
            "cannot mutate logical bins after feature bundling"
        );
        let row_idx = row * self.feature_count + feature;
        let col_idx = feature * self.row_count + row;
        if self.has_row_major() {
            self.bins_adaptive.set(row_idx, value);
        }
        self.bins_col_adaptive.set(col_idx, value);
        if value != self.nan_bin_index {
            self.storage_max_bin = self.storage_max_bin.max(value);
        }
    }

    pub fn effective_feature_count(&self) -> usize {
        self.storage_feature_count
    }

    pub fn feature_bundle_map(&self) -> Option<&FeatureBundleMap> {
        self.feature_bundle_map.as_ref()
    }

    #[inline]
    pub fn storage_col_bin(&self, storage_feature: usize, row: usize) -> u16 {
        self.bins_col_adaptive
            .get(storage_feature * self.row_count + row)
    }

    pub fn with_exact_feature_bundles(self, map: FeatureBundleMap) -> CoreResult<Self> {
        if self.feature_bundle_map.is_some() {
            return Err(CoreError::Validation(
                "binned matrix already has a feature bundle map".to_string(),
            ));
        }
        if map.original_feature_count() != self.feature_count {
            return Err(CoreError::Validation(format!(
                "bundle map feature count {} does not match matrix feature_count {}",
                map.original_feature_count(),
                self.feature_count
            )));
        }
        if map.bundle_count() == 0 {
            return Ok(self);
        }

        let storage_feature_count = map.effective_feature_count();
        let storage_max_bin = map.storage_max_bin();
        let use_u8 = storage_max_bin <= u16::from(u8::MAX - 1);
        let nan_bin_index = if use_u8 {
            u16::from(MISSING_BIN_U8)
        } else {
            storage_max_bin + 1
        };
        let mut encoded = vec![0_u16; self.row_count * storage_feature_count];
        for feature in 0..self.feature_count {
            let assignment = map
                .assignment(feature)
                .expect("bundle map covers every original feature");
            let source_start = feature * self.row_count;
            for row in 0..self.row_count {
                let original_bin = self.bins_col_adaptive.get(source_start + row);
                let encoded_bin = if original_bin == self.missing_bin() {
                    nan_bin_index
                } else if assignment.is_bundled() {
                    if original_bin == 0 {
                        0
                    } else {
                        if original_bin > assignment.bin_span {
                            return Err(CoreError::Validation(format!(
                                "feature {feature} bin {original_bin} exceeds trained bundle span {}",
                                assignment.bin_span
                            )));
                        }
                        assignment.bin_offset + original_bin - 1
                    }
                } else {
                    original_bin
                };
                let target = assignment.storage_feature * self.row_count + row;
                if assignment.is_bundled() && encoded_bin != 0 && encoded[target] != 0 {
                    return Err(CoreError::Validation(format!(
                        "bundle map conflict at row {row}, storage feature {}",
                        assignment.storage_feature
                    )));
                }
                if encoded_bin != 0 {
                    encoded[target] = encoded_bin;
                }
            }
        }
        let bins_col_adaptive = if use_u8 {
            BinStorage::U8(encoded.into_iter().map(|bin| bin as u8).collect())
        } else {
            BinStorage::U16(encoded)
        };
        let bundled = Self {
            row_count: self.row_count,
            feature_count: self.feature_count,
            max_bin: self.max_bin,
            nan_bin_index,
            storage_feature_count,
            storage_max_bin,
            feature_bundle_map: Some(map),
            bins_adaptive: if use_u8 {
                BinStorage::U8(Vec::new())
            } else {
                BinStorage::U16(Vec::new())
            },
            bins_col_adaptive,
        };
        validate_binned_matrix(&bundled)?;
        Ok(bundled)
    }
}

/// Transpose row-major bins to column-major for cache-friendly per-feature access.
fn transpose_bins_to_column_major_u8(
    bins: &[u8],
    row_count: usize,
    feature_count: usize,
) -> Vec<u8> {
    let total = row_count * feature_count;
    if total == 0 || bins.len() != total {
        return Vec::new();
    }
    let mut col_major = vec![0u8; total];
    for row in 0..row_count {
        let row_base = row * feature_count;
        for feature in 0..feature_count {
            col_major[feature * row_count + row] = bins[row_base + feature];
        }
    }
    col_major
}

fn transpose_bins_to_column_major_u16(
    bins: &[u16],
    row_count: usize,
    feature_count: usize,
) -> Vec<u16> {
    let total = row_count * feature_count;
    if total == 0 || bins.len() != total {
        return Vec::new();
    }
    let mut col_major = vec![0u16; total];
    for row in 0..row_count {
        let row_base = row * feature_count;
        for feature in 0..feature_count {
            col_major[feature * row_count + row] = bins[row_base + feature];
        }
    }
    col_major
}
