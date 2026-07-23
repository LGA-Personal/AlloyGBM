use crate::{BinnedMatrix, CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureBundleAssignment {
    pub original_feature: usize,
    pub storage_feature: usize,
    pub bin_offset: u16,
    pub bin_span: u16,
    bundled: bool,
}

impl FeatureBundleAssignment {
    pub fn is_bundled(self) -> bool {
        self.bundled
    }

    pub(crate) fn decode(self, storage_bin: u16, missing_bin: u16) -> u16 {
        if storage_bin == missing_bin {
            return missing_bin;
        }
        if !self.bundled {
            return storage_bin;
        }
        let end = self.bin_offset + self.bin_span;
        if (self.bin_offset..end).contains(&storage_bin) {
            storage_bin - self.bin_offset + 1
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureBundleMap {
    original_feature_count: usize,
    effective_feature_count: usize,
    assignments: Vec<FeatureBundleAssignment>,
    bundles: Vec<Vec<usize>>,
    skipped_feature_count: usize,
    observed_conflict_count: usize,
    storage_max_bin: u16,
}

impl FeatureBundleMap {
    pub fn original_feature_count(&self) -> usize {
        self.original_feature_count
    }

    pub fn effective_feature_count(&self) -> usize {
        self.effective_feature_count
    }

    pub fn bundle_count(&self) -> usize {
        self.bundles.len()
    }

    pub fn bundled_feature_count(&self) -> usize {
        self.bundles.iter().map(Vec::len).sum()
    }

    pub fn skipped_feature_count(&self) -> usize {
        self.skipped_feature_count
    }

    pub fn observed_conflict_count(&self) -> usize {
        self.observed_conflict_count
    }

    pub fn assignment(&self, feature: usize) -> Option<FeatureBundleAssignment> {
        self.assignments.get(feature).copied()
    }

    pub fn bundle_members(&self, bundle: usize) -> Option<&[usize]> {
        self.bundles.get(bundle).map(Vec::as_slice)
    }

    pub(crate) fn storage_max_bin(&self) -> u16 {
        self.storage_max_bin
    }
}

#[derive(Debug)]
struct CandidateFeature {
    feature: usize,
    rows: Vec<u64>,
    nonzero_count: usize,
    max_bin: u16,
}

#[derive(Debug)]
struct PendingBundle {
    members: Vec<usize>,
    occupied_rows: Vec<u64>,
    bin_span: u32,
}

fn conflicts(left: &[u64], right: &[u64]) -> bool {
    left.iter()
        .zip(right)
        .any(|(&left_word, &right_word)| left_word & right_word != 0)
}

pub fn discover_exact_feature_bundles(
    matrix: &BinnedMatrix,
    excluded_features: &[bool],
) -> CoreResult<FeatureBundleMap> {
    if excluded_features.len() != matrix.feature_count {
        return Err(CoreError::Validation(format!(
            "excluded feature mask length {} does not match feature_count {}",
            excluded_features.len(),
            matrix.feature_count
        )));
    }

    let word_count = matrix.row_count.div_ceil(64);
    let missing = matrix.missing_bin();
    let mut feature_max_bins = vec![0_u16; matrix.feature_count];
    let mut candidates = Vec::new();
    let mut skipped_feature_count = 0;

    for (feature, &excluded) in excluded_features.iter().enumerate() {
        if excluded {
            skipped_feature_count += 1;
            continue;
        }
        let mut rows = vec![0_u64; word_count];
        let mut nonzero_count = 0;
        let mut max_bin = 0_u16;
        let mut has_missing = false;
        for row in 0..matrix.row_count {
            let bin = matrix.bin_at(row, feature);
            if bin == missing {
                has_missing = true;
                break;
            }
            max_bin = max_bin.max(bin);
            if bin != 0 {
                rows[row / 64] |= 1_u64 << (row % 64);
                nonzero_count += 1;
            }
        }
        feature_max_bins[feature] = max_bin;
        if has_missing || nonzero_count == 0 {
            skipped_feature_count += 1;
            continue;
        }
        candidates.push(CandidateFeature {
            feature,
            rows,
            nonzero_count,
            max_bin,
        });
    }

    candidates.sort_unstable_by(|left, right| {
        right
            .nonzero_count
            .cmp(&left.nonzero_count)
            .then_with(|| left.feature.cmp(&right.feature))
    });

    let mut pending = Vec::<PendingBundle>::new();
    let mut observed_conflict_count = 0;
    for candidate in candidates {
        let mut selected = None;
        for (bundle_index, bundle) in pending.iter().enumerate() {
            let has_conflict = conflicts(&bundle.occupied_rows, &candidate.rows);
            if !has_conflict
                && bundle.bin_span + u32::from(candidate.max_bin) <= u32::from(u16::MAX - 1)
            {
                selected = Some(bundle_index);
                break;
            }
            if has_conflict {
                observed_conflict_count += 1;
            }
        }
        if let Some(bundle_index) = selected {
            let bundle = &mut pending[bundle_index];
            for (occupied, candidate_rows) in bundle.occupied_rows.iter_mut().zip(&candidate.rows) {
                *occupied |= *candidate_rows;
            }
            bundle.bin_span += u32::from(candidate.max_bin);
            bundle.members.push(candidate.feature);
        } else {
            pending.push(PendingBundle {
                members: vec![candidate.feature],
                occupied_rows: candidate.rows,
                bin_span: u32::from(candidate.max_bin),
            });
        }
    }

    let bundles = pending
        .into_iter()
        .filter(|bundle| bundle.members.len() >= 2)
        .map(|bundle| bundle.members)
        .collect::<Vec<_>>();
    let mut assignments = vec![
        FeatureBundleAssignment {
            original_feature: 0,
            storage_feature: 0,
            bin_offset: 0,
            bin_span: 0,
            bundled: false,
        };
        matrix.feature_count
    ];
    let mut bundled = vec![false; matrix.feature_count];
    let mut storage_max_bin = matrix.max_bin;
    for (storage_feature, members) in bundles.iter().enumerate() {
        let mut offset = 1_u16;
        for &feature in members {
            let span = feature_max_bins[feature];
            assignments[feature] = FeatureBundleAssignment {
                original_feature: feature,
                storage_feature,
                bin_offset: offset,
                bin_span: span,
                bundled: true,
            };
            bundled[feature] = true;
            offset += span;
        }
        storage_max_bin = storage_max_bin.max(offset.saturating_sub(1));
    }

    let mut next_storage_feature = bundles.len();
    for feature in 0..matrix.feature_count {
        if bundled[feature] {
            continue;
        }
        assignments[feature] = FeatureBundleAssignment {
            original_feature: feature,
            storage_feature: next_storage_feature,
            bin_offset: 0,
            bin_span: feature_max_bins[feature],
            bundled: false,
        };
        next_storage_feature += 1;
    }

    Ok(FeatureBundleMap {
        original_feature_count: matrix.feature_count,
        effective_feature_count: next_storage_feature,
        assignments,
        bundles,
        skipped_feature_count,
        observed_conflict_count,
        storage_max_bin,
    })
}
