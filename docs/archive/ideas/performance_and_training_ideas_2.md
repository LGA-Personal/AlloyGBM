# AlloyGBM Performance & Training Ideas v2 (From LightGBM/XGBoost/CatBoost Deep Dive)

## Purpose
Capture architecture-level ideas from three highly optimized GBDT systems and convert them into actionable candidates for AlloyGBM v0.9.7+.

This document emphasizes:
- Histogram construction/subtraction
- Split search
- Data layout/binning
- Threading/execution model
- Prediction/inference path

Focus is on regression-relevant fundamentals; optional variants are listed only when they are structurally meaningful.

## Repositories Reviewed (local snapshots)
- LightGBM `569f89a`
- XGBoost `07cad5e`
- CatBoost `623e0b48`

---

## 1) LightGBM: Architecture Findings

### 1.1 Histogram lifecycle and subtraction
- Uses a bounded histogram pool sized from memory budget (`histogram_pool_size`) and total bin footprint.
- Builds histogram for the smaller child and derives sibling via subtraction from parent when possible.
- Supports quantized grad/hess histogram paths (16-bit/32-bit internal formats) with adaptive handling.

Observed in:
- `tmp/upstream_refs/LightGBM/src/treelearner/serial_tree_learner.cpp` (init/cache sizing, small-child strategy, subtract path)
- `tmp/upstream_refs/LightGBM/src/treelearner/serial_tree_learner.h` (aligned ordered grad/hess buffers + histogram pool)

### 1.2 Split search behavior
- Feature-parallel split scoring with per-thread best split buffers and final reduction.
- Histogram fix-up for most-frequent bin before gain evaluation (prevents needing to explicitly accumulate all bins in scan path).

Observed in:
- `tmp/upstream_refs/LightGBM/src/treelearner/serial_tree_learner.cpp`

### 1.3 Data layout/binning
- Feature-group abstraction packs bins with explicit per-subfeature offsets.
- Chooses dense vs sparse bin storage based on feature sparsity and grouped-feature characteristics.
- Separate handling for dense multi-value vs sparse/grouped structures.

Observed in:
- `tmp/upstream_refs/LightGBM/include/LightGBM/feature_group.h`
- `tmp/upstream_refs/LightGBM/src/io/train_share_states.cpp`

### 1.4 Threading model
- Runtime auto-test chooses row-wise vs col-wise histogram construction path.
- Both paths are initialized/tested and the faster path is selected for training.

Observed in:
- `tmp/upstream_refs/LightGBM/src/io/dataset.cpp`

### 1.5 Prediction path
- Specialized prediction loops/macros with block/thread traversal and precomputed split metadata (`default_bins`, `max_bins`).
- Fast constant-leaf short-circuit for trivial trees.

Observed in:
- `tmp/upstream_refs/LightGBM/src/io/tree.cpp`

---

## 2) XGBoost: Architecture Findings

### 2.1 Histogram lifecycle and subtraction
- Child assignment explicitly favors building the lower-hessian side and deriving sibling by subtraction.
- Uses `BoundedHistCollection` with contiguous per-node histogram storage and controlled overflow behavior.
- Rearrangement logic preserves subtraction trick where parent histogram still exists; otherwise falls back to full build.

Observed in:
- `tmp/upstream_refs/xgboost/src/tree/hist/histogram.cc`
- `tmp/upstream_refs/xgboost/src/tree/hist/hist_cache.h`
- `tmp/upstream_refs/xgboost/src/tree/hist/histogram.h`

### 2.2 Split search behavior
- Per-feature cumulative histogram scans (forward and backward) with explicit missing-value direction handling.
- Thread-local candidate accumulation followed by reduction.
- For categorical splits, supports one-hot or partition-based evaluation.

Observed in:
- `tmp/upstream_refs/xgboost/src/tree/hist/evaluate_splits.h`

### 2.3 Data layout/binning
- Quantile-sketch-based cut matrix (`HistogramCuts`) with per-feature cut pointers.
- Compressed bin index storage (`uint8/uint16/uint32`) with per-feature offsets to reduce index width.

Observed in:
- `tmp/upstream_refs/xgboost/src/common/hist_util.h`
- `tmp/upstream_refs/xgboost/src/common/hist_util.cc`

### 2.4 Threading model
- 2D blocked parallelism across (nodes × row ranges) for histogram build/reduce.
- Cache-aware block size estimation from L1/L2/L3 to choose a practical work partition.
- Row-wise vs column-wise histogram kernel chosen from data density/cache fit.

Observed in:
- `tmp/upstream_refs/xgboost/src/tree/hist/histogram.h`

### 2.5 Prediction path
- Density-based switch to block prediction.
- Array-layout unrolling of top tree levels for better locality when block size > 1.
- Thread-local feature vector buffers and per-block tree-depth-aware traversal.

Observed in:
- `tmp/upstream_refs/xgboost/src/predictor/cpu_predictor.cc`
- `tmp/upstream_refs/xgboost/src/predictor/array_tree_layout.h`

---

## 3) CatBoost: Architecture Findings

### 3.1 Histogram/scoring lifecycle and subtraction
- Explicit subtraction-trick plumbing for leafwise scoring via `{parent, sibling} -> current` stats derivation.
- Uses smallest/paired-leaf strategy in depthwise search to maximize subtraction reuse.
- Tree-level score-stat caching and cache refresh policy to avoid recomputing full stats every level.

Observed in:
- `tmp/upstream_refs/catboost/private/libs/algo/leafwise_scoring.h`
- `tmp/upstream_refs/catboost/private/libs/algo/leafwise_scoring.cpp`
- `tmp/upstream_refs/catboost/private/libs/algo/greedy_tensor_search.cpp`
- `tmp/upstream_refs/catboost/private/libs/algo/calc_score_cache.cpp`

### 3.2 Split search behavior
- Split-ensemble abstraction (single feature, packed binary splits, exclusive bundles, feature groups).
- Scoring loops are organized around bucket stats and split-ensemble traversal, not only raw feature loops.

Observed in:
- `tmp/upstream_refs/catboost/private/libs/algo/leafwise_scoring.h`
- `tmp/upstream_refs/catboost/private/libs/algo/leafwise_scoring.cpp`

### 3.3 Data layout/binning
- Evaluation-time quantization in fixed-size blocks (`FORMULA_EVALUATION_BLOCK_SIZE=128`) into compact `ui8` bins.
- SIMD/non-SIMD binarization paths; explicit NaN substitution policy during quantization.
- Pre-quantized path support for preprocessed pools.

Observed in:
- `tmp/upstream_refs/catboost/libs/model/cpu/quantization.h`

### 3.4 Threading/execution model
- Extensive blocked execution via local executor (`ExecRange`) and explicitly chosen block sizes in training paths.
- Candidate-parallel score calculation using local executor for sub-candidate loops.

Observed in:
- `tmp/upstream_refs/catboost/private/libs/algo/leafwise_scoring.cpp`
- `tmp/upstream_refs/catboost/private/libs/algo/tensor_search_helpers.cpp`

### 3.5 Prediction path
- Template-dispatched predictor path across key booleans (oblivious/non-oblivious, single-doc vs blocked, single-class vs multi-class, xor-mask usage).
- Shallow-tree SSE fast paths with gather/add leaf accumulation and prefetching.
- Blocked and single-doc evaluators selected at runtime.

Observed in:
- `tmp/upstream_refs/catboost/libs/model/cpu/evaluator_impl.cpp`
- `tmp/upstream_refs/catboost/libs/model/cpu/evaluator.h`
- `tmp/upstream_refs/catboost/libs/model/cpu/formula_evaluator.cpp`

---

## Cross-System Architectural Best Practices (Common Patterns)

1. Build-smaller / derive-larger histogram strategy
- Shared by all three in different forms.
- Key impact: significant reduction in histogram construction work.

2. Bounded histogram memory with explicit cache policy
- Prevents uncontrolled memory blow-ups and keeps hot histograms contiguous.

3. Histogram precision compression where safe
- Narrow formats (`u8/u16` bins, quantized grad/hess paths) reduce bandwidth and improve cache behavior.

4. Runtime kernel selection by data/hardware profile
- Row-wise vs col-wise or blocked-vs-nonblocked selection is often benchmarked/decided at runtime.

5. Cache-aware blocking over rows/nodes
- Work partitioning is tied to L1/L2/L3 and feature density, not fixed constants alone.

6. Predictor path specialization is first-class
- Inference has dedicated architecture (template dispatch, shallow-tree fast paths, block traversal), not a generic afterthought.

7. Split evaluation is cumulative and branch-aware
- Prefix/suffix scans over histogram bins with explicit missing/categorical handling.

---

## Candidate Backlog for AlloyGBM (Architecture-Level)

| Priority | Idea | Type | Why This Is Architectural | Expected Effect | Notes |
| --- | --- | --- | --- | --- | --- |
| P0 | **Global histogram cache manager with bounded contiguous storage + overflow policy** | Fundamental | Defines core training memory lifecycle and histogram reuse semantics. | Faster training; lower allocator overhead; more predictable memory. | Incorporates parent/sibling availability tracking for subtraction path validity. |
| P0 | **Canonical build-smaller/derive-sibling pipeline across all growth policies** | Fundamental | Changes baseline split-eval workflow at each node expansion. | Large histogram-build savings on deep trees. | Must be robust when parent hist is evicted. |
| P0 | **2D blocked histogram executor (node x row-range) with thread-local buffers + reduction** | Fundamental | Replaces simple per-feature/per-row loops with cache-oriented execution model. | Better multicore scaling; lower contention/false sharing. | Align/reduce buffers explicitly; avoid shared hot counters. |
| P0 | **Runtime histogram kernel chooser (row-wise vs col-wise, density-aware)** | Fundamental | Chooses core training kernel path dynamically for each dataset profile. | Consistent speedups across diverse datasets/hardware. | Keep forced override flags only for debugging/benchmarking. |
| P1 | **Compressed bin-index representation with per-feature offsets** | Fundamental | Changes dataset/bin representation and all histogram consumers. | Lower RAM, better cache locality, faster scans. | Use `u8/u16/u32` auto-selection from max bin id. |
| P1 | **Quantile-cut builder with persisted cut matrix and deterministic mapping** | Fundamental | Defines training/predict binning contract end-to-end. | Better speed/quality tradeoff than exact split enumeration. | Aligns with existing quantile ideas from v1 doc. |
| P1 | **Histogram scan engine with explicit forward/backward missing-direction evaluation** | Fundamental | Reworks split scoring core to cumulative scans + missing-aware evaluation. | Accuracy stability + robust missing handling with low overhead. | Especially useful when missingness is nontrivial. |
| P1 | **Tree-level stats cache for split scoring (dirty/refresh policy)** | Fundamental | Makes scoring cache state part of training architecture. | Lower recomputation cost between adjacent expansions/levels. | Inspired by CatBoost tree-level cache behavior. |
| P2 | **Prediction-time dual path: blocked batch kernel + single-row low-overhead kernel** | Fundamental | Creates explicit inference architecture instead of one-size-fits-all path. | Better p50/p99 inference across online and batch workloads. | Route by row count and density. |
| P2 | **Top-of-tree compact array layout for batch inference** | Fundamental | Alters predictor traversal representation for top levels. | Higher SIMD/cache efficiency in batch predict. | Keep fallback to existing tree traversal for deep remainder. |
| P2 | **Shallow-tree fused evaluation path (multi-tree chunking)** | Fundamental | Changes predictor scheduling across trees, not just micro-ops. | Lower instruction overhead; better throughput for many shallow trees. | CatBoost-style 4-tree chunking is a reference shape. |
| P3 | **Micro-architectural prefetch tuning in selected hot loops** | Variant | Kernel-level tuning on top of above architecture. | Can improve throughput in memory-bound loops; can regress if overused. | Gate behind compile/runtime flag and keep benchmark-guarded. |
| P3 | **ISA-specific inference kernels (SSE/AVX2/NEON) behind shared kernel interface** | Variant (platform specialization) | Not a new algorithm, but meaningful architecture for per-ISA execution backends. | Better per-platform peak inference/training kernels. | Keep one canonical scalar path and strict parity tests. |

---

## How This Extends v1 Ideas

Adds depth beyond `performance_and_training_ideas.md` by specifying:
- Memory-lifecycle architecture for histogram caching/overflow.
- Explicit 2D blocked execution model for training kernels.
- Predictor architecture split (single-row vs blocked batch) with top-tree array layout.
- Tree-level scoring cache lifecycle and subtraction-validity logic.

It also reinforces prior high-leverage ideas already identified in v1:
- quantile/histogram foundation
- subtraction/reuse as a baseline pattern
- inference-path specialization as a first-class optimization target

---

## Recommended Test Order (to keep iteration tight)

1. Global histogram cache manager + canonical subtraction pipeline (P0)
2. 2D blocked histogram executor (P0)
3. Runtime row/col kernel chooser (P0)
4. Split-scan engine with missing-direction handling (P1)
5. Predictor dual path + top-tree array layout (P2)

The order above prioritizes expected speed impact while preserving RMSE stability.

---

## Round 2 Addendum (Deeper Source Pass)

This addendum captures additional architecture-level findings from a second pass through the same three codebases. The focus is still on fundamentals, with optional variants clearly labeled.

### LightGBM: Additional Findings

1. Adaptive inner histogram precision during block build
- In sparse/multival histogram construction, LightGBM can temporarily downshift to 8-bit inner histogram bins when block-size and quant-bin cardinality make overflow impossible, then merge/move back to the requested outer precision.
- This is a memory-bandwidth optimization at the kernel level without changing external histogram contracts.

Observed in:
- `tmp/upstream_refs/LightGBM/include/LightGBM/train_share_states.h`

2. Quantized gradient pipeline with global synchronization and dynamic bin-width assignment
- Gradient/hessian discretization computes global max absolute values (including distributed max sync), applies scaling, and supports stochastic rounding.
- Histogram precision for nodes/leaves is chosen dynamically (8/16/32-bit) based on per-leaf potential max statistic ranges.

Observed in:
- `tmp/upstream_refs/LightGBM/src/treelearner/gradient_discretizer.cpp`

3. Dense and sparse histogram kernels are intentionally different algorithms
- Dense bins use templated kernels with optional prefetch and packed integer accumulation paths.
- Sparse bins use delta-coded index traversal with intersection-style scans between leaf row indices and sparse non-zero index streams.

Observed in:
- `tmp/upstream_refs/LightGBM/src/io/dense_bin.hpp`
- `tmp/upstream_refs/LightGBM/src/io/sparse_bin.hpp`

4. EFB grouping is multi-pass and heuristic, not one-shot
- Grouping tries at least two orderings (native order and non-zero-count order), compares resulting group counts, then chooses the better grouping and shuffles groups.
- A second-round dense-threshold branch (`dense_threshold = 0.4`) separates dense-compatible groups from sparse remainder logic.

Observed in:
- `tmp/upstream_refs/LightGBM/src/io/dataset.cpp`

5. Col-wise histogram path explicitly splits dense groups vs multival group
- Dense groups are processed in a parallel pass with reordered gradients/hessians when indices are present.
- Multi-val group is handled separately, with an ordered-data fast path only when dense groups were already materialized.

Observed in:
- `tmp/upstream_refs/LightGBM/src/io/dataset.cpp`

6. Data partition is a first-class runtime structure
- Leaf partitions are contiguous slices over a single index array.
- Splits use a partition runner over block ranges, keeping left/right children contiguous and cheap to access.

Observed in:
- `tmp/upstream_refs/LightGBM/src/treelearner/data_partition.hpp`

7. Data-parallel training is feature-sharded + reduce-scatter histogram synchronization
- Features are assigned to machines by approximate bin-load balancing.
- Histogram copy/convert + reduce-scatter + local restore/fix are explicit stages before split evaluation and subtraction.

Observed in:
- `tmp/upstream_refs/LightGBM/src/treelearner/data_parallel_tree_learner.cpp`

### XGBoost: Additional Findings

1. Row partitioning uses a staged execution pipeline
- Per split batch: compute split conditions, build blocked 2D work space, allocate per-task partition buffers, partition rows, compute row offsets, merge back, then commit split metadata to row sets.
- This makes row-position update architecture explicit and parallel-safe.

Observed in:
- `tmp/upstream_refs/xgboost/src/tree/common_row_partitioner.h`

2. Node-expansion policy is abstracted behind a driver
- Lossguide mode pops one highest-gain node.
- Depthwise mode pops same-depth batches up to a configurable batch size.
- Shared validity checks enforce max depth/max leaves centrally.

Observed in:
- `tmp/upstream_refs/xgboost/src/tree/driver.h`

3. Updater loop is tightly structured around split-apply and position maintenance
- Core sequence is: apply split, update row positions, build histograms, evaluate child splits, push/pop from expansion driver.
- This gives a clean place to integrate additional caching/planning logic.

Observed in:
- `tmp/upstream_refs/xgboost/src/tree/updater_quantile_hist.cc`

4. Prediction cache is incremental and versioned
- Tree predictions are accumulated with layer/version tracking; cache is invalidated on incompatible layer requests.
- There is explicit update logic to add only the newest tree contribution from stored node positions.

Observed in:
- `tmp/upstream_refs/xgboost/src/tree/hist/evaluate_splits.h`
- `tmp/upstream_refs/xgboost/src/tree/updater_quantile_hist.cc`
- `tmp/upstream_refs/xgboost/src/gbm/gbtree.cc`

5. Bin-index storage is dynamically compressed and grown in place
- Dense index matrices auto-select `uint8` / `uint16` / `uint32` representation by max bin cardinality.
- Buffer growth is done via malloc-resource resize to avoid re-copying previous page state when appending batches.

Observed in:
- `tmp/upstream_refs/xgboost/src/data/gradient_index.cc`
- `tmp/upstream_refs/xgboost/src/data/gradient_index.h`

6. Histogram kernel prefetch is conditional on row-index contiguity
- Dispatch detects contiguous row blocks and disables software prefetch when hardware prefetch is likely enough.
- Non-contiguous tails use split prefetch/no-prefetch spans.

Observed in:
- `tmp/upstream_refs/xgboost/src/common/hist_util.cc`

### CatBoost: Additional Findings

1. Sampling/side-selection is implemented as a reusable control-mask substrate
- A boolean control vector drives compaction, sampling, and side selection with dense and sparse copy kernels.
- Dense path uses unrolled writes; sparse path uses fast search for next active item.

Observed in:
- `tmp/upstream_refs/catboost/catboost/private/libs/algo/calc_score_cache.cpp`

2. Smallest-side-first is explicit and cheap
- For depthwise splits, CatBoost counts split-bit population per block, then chooses the smaller side for control-mask materialization.
- This directly minimizes downstream data movement.

Observed in:
- `tmp/upstream_refs/catboost/catboost/private/libs/algo/calc_score_cache.cpp`

3. Query-aware slicing is native in fold/block builders
- Block creation can be driven by queries and control masks, with pairwise competitor remapping when needed.
- This avoids generic row-level assumptions in ranking-style paths.

Observed in:
- `tmp/upstream_refs/catboost/catboost/private/libs/algo/calc_score_cache.cpp`

4. Split-stat caching uses split-ensemble keys + memory pool GC
- Cached bucket stats are keyed by split ensemble and allocated from a pool.
- Garbage collection is triggered by memory waste threshold relative to initial reservation.

Observed in:
- `tmp/upstream_refs/catboost/catboost/private/libs/algo/calc_score_cache.h`
- `tmp/upstream_refs/catboost/catboost/private/libs/algo/calc_score_cache.cpp`
- `tmp/upstream_refs/catboost/catboost/private/libs/algo/scoring.cpp`

5. Index updates are callback-fused across split set
- Split-specific update callbacks are prepared once (including bundle/pack/group feature cases) and then applied per row block.
- This reduces repeated control overhead for deep trees.

Observed in:
- `tmp/upstream_refs/catboost/catboost/private/libs/algo/index_calcer.cpp`

6. Repacked split encoding is central to fast apply
- Apply path uses a compact 4-byte repacked split (`featureIndex`, `xorMask`, `splitIdx`) for branch checks.
- One-hot logic is encoded through xor-mask and sentinel split index rather than separate branch-heavy code.

Observed in:
- `tmp/upstream_refs/catboost/catboost/libs/model/model.h`
- `tmp/upstream_refs/catboost/catboost/libs/model/model.cpp`

7. Apply-time runtime metadata is precomputed, not derived per call
- Tree first-leaf offsets, used feature counts, and minimal vector sizes are precomputed into apply metadata.
- This minimizes per-inference setup and bounds checks.

Observed in:
- `tmp/upstream_refs/catboost/catboost/libs/model/model.h`
- `tmp/upstream_refs/catboost/catboost/libs/model/model.cpp`

8. Predictor function selection is compile-time specialized via runtime booleans
- CatBoost chooses among specialized kernels based on: oblivious/non-oblivious, single-doc/block, single-class/multi-class, xor-mask need, and leaf-index-only mode.
- Blocked quantized path validates stride assumptions and runs directly on quantized blocks.

Observed in:
- `tmp/upstream_refs/catboost/catboost/libs/model/cpu/evaluator_impl.cpp`
- `tmp/upstream_refs/catboost/catboost/libs/model/cpu/evaluator.h`
- `tmp/upstream_refs/catboost/catboost/libs/model/cpu/formula_evaluator.cpp`

---

## New Cross-System Best Practices (Round 2)

1. Make row partitioning a staged subsystem, not an inlined side effect.
2. Treat precision as local and adaptive (inner kernel precision can differ from external precision).
3. Use control masks as a general compaction primitive across sampling, splitting, and query handling.
4. Version prediction caches and define strict invalidation semantics.
5. Keep split metadata repacked for branch-light inference.
6. Separate data movement planning from arithmetic kernels (offset prep, callback prep, then execution).
7. Prefer per-task temporary buffers with explicit merge phases over shared mutable histograms.
8. Gate software prefetching by contiguity heuristics.

---

## Candidate Backlog Expansion (Round 2)

| Priority | Idea | Type | Why This Is Architectural | Expected Effect | Notes |
| --- | --- | --- | --- | --- | --- |
| P0 | **Adaptive histogram precision ladder (inner 8/16/32 with safe merge-up)** | Fundamental | Changes core histogram construction contract and memory traffic profile. | Lower bandwidth, better cache residency, improved training speed. | Must enforce overflow-safe thresholds and deterministic accumulation. |
| P0 | **Row partition engine with staged commit (split-conditions -> partition buffers -> merge)** | Fundamental | Replaces ad-hoc position updates with a dedicated subsystem. | Better threading scalability and cleaner split/update lifecycle. | Enables later async expansion work. |
| P0 | **Prediction cache manager with versioning + incremental tree delta updates** | Fundamental | Changes inference/training interaction and cache lifecycle semantics. | Significant reduction in repeated prediction work during training/eval loops. | Needs strict invalidation rules when layer range changes. |
| P1 | **Control-mask compaction API shared by sampling and split-side extraction** | Fundamental | Unifies multiple data-movement paths under one optimized primitive. | Less duplicate logic, faster sampling/compaction, better SIMD opportunities. | Include dense and sparse control-density modes. |
| P1 | **Repacked split descriptor for inference (`feature`, `mask`, `threshold`)** | Fundamental | Alters predictor data representation and traversal inner loop. | Lower branch cost and improved instruction/cache efficiency. | Keep scalar reference kernel for parity testing. |
| P1 | **Callback-fused index update planner for depth expansion** | Fundamental | Restructures split-index update into plan+execute phases. | Lower control overhead for deep trees and many active splits. | Especially useful with packed/bundled feature storage. |
| P1 | **Score-stat cache with memory pool + garbage-collection threshold** | Fundamental | Makes split-stat memory lifecycle explicit in architecture. | Reduced allocator churn and repeated score-stat recomputation. | Pair with split-ensemble cache keys. |
| P2 | **Contiguity-aware prefetch policy for histogram/index kernels** | Variant | Micro-architectural layer on top of core kernels. | Throughput gain on sparse/non-contiguous access patterns. | Must be benchmark-guarded; can regress on some CPUs. |
| P2 | **Query-aware block slicing path (group/ranking-safe compaction)** | Variant (task-specific) | Adds specialized dataflow for grouped objectives. | Better scaling for query-structured datasets. | Keep optional for regression-only default path. |
| P3 | **Distributed-ready histogram transport abstraction (copy/convert/reduce/restore)** | Fundamental (future distributed mode) | Establishes clear interfaces for networked histogram synchronization. | Easier evolution to data-parallel training without rework. | Not needed for single-node release path now. |

---

## Suggested Round 2 Test Order

1. Adaptive histogram precision ladder (P0)
2. Prediction cache manager with versioning and incremental updates (P0)
3. Row partition engine with staged commit (P0)
4. Repacked split descriptor + scalar parity harness (P1)
5. Control-mask compaction API (P1)

---

## Execution Status (v0.9.7 Sprint Snapshot)

- Upstream reference repos are available locally at:
  - `tmp/upstream_refs/LightGBM`
  - `tmp/upstream_refs/xgboost`
  - `tmp/upstream_refs/catboost`
- Candidate execution outcomes (kept/rejected, with benchmark deltas and retry guidance) are tracked in:
  - `docs/architecture/ideas/v0_9_7_candidate_log.md`
- Current direction: prioritize split quality and data representation improvements for RMSE/MAE/R2 before pursuing additional pure speed variants.
