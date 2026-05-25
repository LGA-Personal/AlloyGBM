use crate::error::CoreResult;
use crate::validate_binned_matrix;

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
    /// Row-major: bins[row * feature_count + feature]
    pub bins: Vec<u8>,
    /// Column-major: bins_col[feature * row_count + row] — for cache-friendly histogram building.
    pub bins_col: Vec<u8>,
    /// Row-major adaptive storage (mirrors `bins` but supports u16).
    pub bins_adaptive: BinStorage,
    /// Column-major adaptive storage (mirrors `bins_col` but supports u16).
    pub bins_col_adaptive: BinStorage,
}

impl BinnedMatrix {
    /// Create a BinnedMatrix from u8 bins (max_bin <= 254).
    pub fn new(
        row_count: usize,
        feature_count: usize,
        max_bin: u16,
        bins: Vec<u8>,
    ) -> CoreResult<Self> {
        let bins_col = transpose_bins_to_column_major_u8(&bins, row_count, feature_count);
        let bins_adaptive = BinStorage::U8(bins.clone());
        let bins_col_adaptive = BinStorage::U8(bins_col.clone());
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index: MISSING_BIN_U8 as u16,
            bins,
            bins_col,
            bins_adaptive,
            bins_col_adaptive,
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
        // For backward compatibility, also create u8 vecs (clamped) for legacy code paths.
        let bins_u8: Vec<u8> = bins_u16
            .iter()
            .map(|&b| if b >= 255 { 255 } else { b as u8 })
            .collect();
        let bins_col_u8 = transpose_bins_to_column_major_u8(&bins_u8, row_count, feature_count);
        let bins_col_u16 = transpose_bins_to_column_major_u16(&bins_u16, row_count, feature_count);
        let bins_adaptive = BinStorage::U16(bins_u16);
        let bins_col_adaptive = BinStorage::U16(bins_col_u16);
        let matrix = Self {
            row_count,
            feature_count,
            max_bin,
            nan_bin_index,
            bins: bins_u8,
            bins_col: bins_col_u8,
            bins_adaptive,
            bins_col_adaptive,
        };
        validate_binned_matrix(&matrix)?;
        Ok(matrix)
    }

    /// Whether this matrix uses u16 bin storage.
    pub fn is_wide_bins(&self) -> bool {
        matches!(self.bins_adaptive, BinStorage::U16(_))
    }

    /// Read a bin value from column-major adaptive storage.
    /// `index` is the flat offset (feature * row_count + row).
    #[inline]
    pub fn col_bin(&self, index: usize) -> u16 {
        self.bins_col_adaptive.get(index)
    }

    /// Read a bin value from row-major adaptive storage.
    /// `index` is the flat offset (row * feature_count + feature).
    #[inline]
    pub fn row_bin(&self, index: usize) -> u16 {
        self.bins_adaptive.get(index)
    }

    /// The sentinel value used for missing/NaN bins in this matrix.
    #[inline]
    pub fn missing_bin(&self) -> u16 {
        self.nan_bin_index
    }

    /// Whether column-major adaptive storage is available (non-empty).
    #[inline]
    pub fn has_col_major(&self) -> bool {
        !self.bins_col_adaptive.is_empty()
    }

    /// Set the bin value at (row, feature) in all storage arrays.
    /// Used for re-mapping native categorical feature columns after binning.
    pub fn set_bin(&mut self, row: usize, feature: usize, value: u16) {
        let row_idx = row * self.feature_count + feature;
        let col_idx = feature * self.row_count + row;
        let val_u8 = if value >= 255 { 255u8 } else { value as u8 };

        if row_idx < self.bins.len() {
            self.bins[row_idx] = val_u8;
        }
        if col_idx < self.bins_col.len() {
            self.bins_col[col_idx] = val_u8;
        }
        match &mut self.bins_adaptive {
            BinStorage::U8(v) => {
                if row_idx < v.len() {
                    v[row_idx] = val_u8;
                }
            }
            BinStorage::U16(v) => {
                if row_idx < v.len() {
                    v[row_idx] = value;
                }
            }
        }
        match &mut self.bins_col_adaptive {
            BinStorage::U8(v) => {
                if col_idx < v.len() {
                    v[col_idx] = val_u8;
                }
            }
            BinStorage::U16(v) => {
                if col_idx < v.len() {
                    v[col_idx] = value;
                }
            }
        }
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
