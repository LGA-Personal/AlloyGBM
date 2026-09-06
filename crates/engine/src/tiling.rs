//! Feature-tile helpers used by the histogram split-finding loop.
//!
//! Tiles partition the (sub-)sampled feature index list into
//! contiguous u32 ranges that fit in L2 cache and parallelize cleanly
//! across rayon workers. The training loop builds one
//! `Vec<FeatureTile>` per round via [`sampled_feature_tiles`] and
//! hands it to the backend's per-tile histogram kernels.

use alloygbm_core::FeatureTile;

use crate::error::{EngineError, EngineResult};
use crate::sampling::sampled_indices;

/// Maximum features per tile. Keeps the histogram arena small enough to fit in
/// L2 cache (64 features × 256 bins × 12 bytes ≈ 192 KB) and creates enough
/// tiles for rayon to parallelize across cores.
pub(crate) const MAX_TILE_FEATURE_WIDTH: usize = 64;

/// Smallest tile width when splitting features for parallelism.
///
/// This used to be 16, which silently capped histogram parallelism at
/// `ceil(feature_count / 16)` workers: a 40-feature fit produced three tiles
/// and could never use more than three cores however large `n_jobs` was.
/// Measured speedup tracked the tile count almost exactly. Narrower tiles also
/// mean *smaller* per-tile arenas, so cache residency improves rather than
/// degrades; 4 was the best of {1, 2, 4, 8, 16} across 40-, 320-, and
/// 16-feature workloads.
const MIN_TILE_FEATURE_WIDTH: usize = 4;

/// Compute a tile size that keeps each thread busy with enough work but
/// produces enough tiles to amortize parallelism overhead. Aim for roughly
/// 2 tiles per thread so straggling threads can steal work.
///
/// Tile boundaries never affect the trained model: each tile accumulates its
/// own features' histograms independently, and the per-feature bin sums are
/// identical however the features are grouped.
pub(crate) fn compute_optimal_tile_size(feature_count: usize, n_threads: usize) -> usize {
    if n_threads <= 1 || feature_count <= MIN_TILE_FEATURE_WIDTH {
        return feature_count.clamp(1, MAX_TILE_FEATURE_WIDTH);
    }
    let target_tiles = n_threads.saturating_mul(2);
    let raw_tile = feature_count.div_ceil(target_tiles);
    let preferred = raw_tile.clamp(MIN_TILE_FEATURE_WIDTH, MAX_TILE_FEATURE_WIDTH);

    // The minimum width is a cache preference, not a correctness rule, so it
    // yields when honouring it would leave threads with no tile at all.
    //
    // Without this the floor re-creates the very ceiling it replaced, just
    // further out: it caps the tile count at `feature_count / 4`, so a
    // 40-feature fit can never exceed 10 tiles however many cores the host
    // has. That is invisible when tuning on a 10-core machine and severe on a
    // 64-core one -- exactly the kind of constant that makes a library fast on
    // its author's laptop and mediocre everywhere else.
    if feature_count.div_ceil(preferred) >= n_threads {
        preferred
    } else {
        raw_tile.clamp(1, MAX_TILE_FEATURE_WIDTH)
    }
}

pub(crate) fn feature_tiles_from_sorted_indices(
    indices: &[usize],
) -> EngineResult<Vec<FeatureTile>> {
    if indices.is_empty() {
        return Err(EngineError::ContractViolation(
            "feature subsampling produced no feature indices".to_string(),
        ));
    }

    let n_threads = rayon::current_num_threads();
    let tile_width = compute_optimal_tile_size(indices.len(), n_threads);

    let mut tiles = Vec::new();
    let mut run_start = indices[0];
    let mut previous = indices[0];
    for &current in indices.iter().skip(1) {
        if current == previous + 1 && (current - run_start) < tile_width {
            previous = current;
            continue;
        }
        tiles.push(FeatureTile::new(run_start as u32, (previous + 1) as u32)?);
        run_start = current;
        previous = current;
    }
    tiles.push(FeatureTile::new(run_start as u32, (previous + 1) as u32)?);
    Ok(tiles)
}

pub(crate) fn sampled_feature_tiles(
    feature_count: usize,
    col_subsample: f32,
    seed_base: u64,
    round_index: u64,
) -> EngineResult<(Vec<FeatureTile>, usize)> {
    let selected = sampled_indices(feature_count, col_subsample, seed_base, round_index);
    let coverage_count = selected.len();
    let tiles = feature_tiles_from_sorted_indices(&selected)?;
    Ok((tiles, coverage_count))
}
