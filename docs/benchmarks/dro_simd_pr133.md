# PR #133 DRO SIMD Performance Evidence

Status: `DONE`

This report records the implementation and evidence for PR #133. The change reduces
the cost of the opt-in scalar DRO leaf solver. It does not claim that DRO improves
predictive quality, that DRO is always faster than standard leaves, or that the
default radius is portable across objectives and target scales.

## Scope

The active numeric DRO scanner is exhaustive: it evaluates every valid numeric
threshold and both learned missing-value directions. The f64 variance and radius
calculation is evaluated in safe `wide::f64x4` lanes; f32 Newton gain and validity
masks remain in the existing arithmetic domain. The scalar scanner remains the
oracle and is retained for:

- native categorical splits;
- factor-penalized scans;
- Morph+DRO scans;
- joint multi-output DRO, whose shared split selection remains standard and whose
  DRO behavior is leaf-only.

`dro_config=None` and `dro_radius=0.0` continue to use the standard SIMD scanner.
No public parameter, formula, default, artifact section, metadata field, dependency,
or unsafe code was added.

## Provenance

| Item | Value |
| --- | --- |
| Production base | `2b2e3ef` |
| Baseline benchmark source | production `2b2e3ef` plus benchmark-only harness changes in a temporary worktree |
| Baseline JSON git head | `2b2e3efb9952ee0766a939f77364879a29af5282` |
| Candidate benchmark commit | `bde1ce1150d52de2255043d73e26ec61f5ccf72d` |
| Hardware | Apple M4, `Mac16,12` |
| OS | macOS 26.5.2, Darwin 25.5.0 |
| Architecture | arm64 / aarch64 |
| Rust | `rustc 1.92.0`, Cargo `1.92.0` |
| Python | `3.13.5` |
| AlloyGBM | `0.12.10` |
| NumPy | `2.5.0` |
| scikit-learn | `1.9.0` |
| SIMD runtime | AVX2 disabled; AVX2 override unset |

## Scanner Benchmark

Each arm used seven release-mode repetitions. The DRO benchmark rows use the
existing 16-, 64-, and 255-bin histogram fixture, with `dro_radius=0.05`,
`dro_metric="wasserstein"`, `lambda_l1=0.1`, and `lambda_l2=1.0`.

| Case | Baseline median (ns/iter) | Candidate median (ns/iter) | Speedup |
| --- | ---: | ---: | ---: |
| 16 bins | 23,089.33 | 21,956.25 | 1.052x |
| 64 bins | 24,864.33 | 22,663.75 | 1.097x |
| 255 bins | 186,385.50 | 123,459.58 | 1.510x |

The scanner gate requires at least 1.5x at both 64 and 255 bins. The 255-bin case
passes; the 64-bin case remains below the threshold, so acceptance uses the
declared end-to-end fallback. The final candidate includes only the declared
behavior-preserving sequence: invariant SIMD broadcasts were hoisted, both missing
directions reuse one loaded prefix set, and two four-lane groups are loaded per
eight-bin block. The rejected prototype timings are listed below.

## End-to-End Matrix

The matrix used both `standard` and `dro` arms, deterministic single-threaded
training, five seeds (`0, 1, 2, 3, 4`), and the fixed DRO settings above. The
quality comparison uses the exact keyed records, with absolute and relative
tolerances of `1e-7`. The candidate has 90/90 exact keys, and the maximum primary
and secondary quality deltas are both `0.0`.

The table reports the candidate/baseline median total-fit ratio for each case.
Ratios below `1.0` are faster. Shapes and budgets are the full fixture values.

| Case | Task | Shape | Rows x features | Rounds | Primary metric | DRO ratio | Standard ratio |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| `reg-small-narrow` | regression | small-narrow | 640 x 8 | 80 | RMSE | 0.736756 | 1.011813 |
| `reg-small-wide` | regression | small-wide | 640 x 128 | 80 | RMSE | 0.663743 | 1.009563 |
| `reg-tall-narrow` | regression | tall-narrow | 8192 x 16 | 80 | RMSE | 0.729810 | 1.003166 |
| `reg-tall-wide` | regression | tall-wide | 8192 x 128 | 60 | RMSE | 0.686767 | 1.006951 |
| `reg-noisy` | regression | medium | 2048 x 32 | 100 | RMSE | 0.775521 | 1.006283 |
| `binary-imbalanced` | binary | medium | 4096 x 32 | 100 | log loss | 0.773869 | 0.992632 |
| `multiclass-wide` | multiclass | small-wide | 2048 x 96 | 80 | log loss | 0.702460 | 0.999336 |
| `rank-small-query` | ranking | tall-narrow | 2400 x 24, 120 queries | 60 | NDCG@10 | 0.732061 | 0.993394 |
| `rank-large-query` | ranking | tall-narrow | 4096 x 24, 8 queries | 60 | NDCG@10 | 0.846193 | 0.978227 |

The comparison aggregates DRO ratios by shape as follows:

| Shape | Median DRO ratio |
| --- | ---: |
| medium | 0.774887 |
| small-narrow | 0.736756 |
| small-wide | 0.681625 |
| tall-narrow | 0.745845 |
| tall-wide | 0.686767 |

The aggregate shape-median ratio is `0.736756`, or a `26.3244%` DRO fit-time
improvement. The median native-training stage ratio across the 45 DRO records is
`0.730394`; the corresponding direct per-record fit ratio is `0.732061`.
Input adaptation and native bridge preparation medians were `1.021012` and
`1.000662`, respectively. Prediction was timed separately; its median ratio was
`0.970258`. The comparison's declared shape-median aggregate is the acceptance
value.

The 5% shape-regression guard passed for every DRO shape. The 15% end-to-end DRO
fallback passed because `26.3244% >= 15%`. Standard timing is grouped by dataset
across the five seeds: the machine-readable `standard_case_time_ratios` field
contains the per-case medians, and `worst_standard_case_ratio` is `1.011813`,
inside the allowed `1.03`. The descriptive `worst_standard_record_ratio` is
`1.127643`; it is retained in the comparison output but does not gate acceptance.
The machine-readable comparison records `passed: true` with exact quality
equivalence and exact key coverage.

## Robustness Sentinel

The required quick sentinel used seeds `7,13`, 12% contaminated labels, 100 trees,
depth 4, learning rate `0.06`, and `lambda_l2=1.0`. Values below are unchanged
within the report's printed precision.

| Path | Solver | Clean-fit RMSE | Contaminated-fit RMSE | Corruption penalty |
| --- | --- | ---: | ---: | ---: |
| Scalar regressor | standard | 0.79635 | 1.10978 | +0.31344 |
| Scalar regressor | dro | 0.80591 | 1.09732 | +0.29141 |
| Joint multi-label | standard | 0.66976 | 0.93823 | +0.26848 |
| Joint multi-label | dro | 0.67662 | 0.93793 | +0.26131 |

This is a regression sentinel, not a claim that the default radius improves
robustness on every workload. Joint DRO remains leaf-only.

## Rejected Prototypes

The comparison artifact records timings for prototypes rejected under the declared
gates. The initial four-lane implementation had scanner medians of `22,363.92`,
`23,755.75`, and `175,467.25` ns/iter for 16, 64, and 255 bins, with a 7.26%
median DRO fit improvement. The invariant-hoisting prototype measured `22,284.17`,
`23,753.08`, and `164,909.50` ns/iter, with an 8.09% fit improvement. Neither
reached the scanner gate or the 15% fit fallback; neither changed behavior.

## Reproduction

Run from the repository root on the same host class:

```bash
git worktree add --detach /tmp/alloygbm-pr133-dro-baseline-round1 2b2e3ef
cp benchmarks/dro_performance.py \
  /tmp/alloygbm-pr133-dro-baseline-round1/benchmarks/dro_performance.py
cp crates/backend_cpu/benches/histogram_kernels.rs \
  /tmp/alloygbm-pr133-dro-baseline-round1/crates/backend_cpu/benches/histogram_kernels.rs
git -C /tmp/alloygbm-pr133-dro-baseline-round1 status --short
git -C /tmp/alloygbm-pr133-dro-baseline-round1 diff -- \
  crates/core crates/engine crates/backend_cpu/src bindings/python/alloygbm

cd /tmp/alloygbm-pr133-dro-baseline-round1
for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr133_dro_split_baseline.txt

VIRTUAL_ENV=/path/to/repo/.venv PATH=/path/to/repo/.venv/bin:$PATH \
  /path/to/repo/.venv/bin/maturin develop --release

VIRTUAL_ENV=/path/to/repo/.venv PATH=/path/to/repo/.venv/bin:$PATH \
  /path/to/repo/.venv/bin/python benchmarks/dro_performance.py run \
  --arms standard dro --seeds 0 1 2 3 4 \
  --output /path/to/repo/benchmarks/results/pr133_dro_fit_baseline.json

cd /path/to/repo
maturin develop --release

for run in 1 2 3 4 5 6 7; do
  cargo bench -p alloygbm-backend-cpu --bench histogram_kernels
done | tee benchmarks/results/pr133_dro_split_simd.txt

/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dro_performance.py run \
  --arms standard dro --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr133_dro_fit_simd.json

/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dro_performance.py compare \
  benchmarks/results/pr133_dro_fit_baseline.json \
  benchmarks/results/pr133_dro_fit_simd.json \
  --output benchmarks/results/pr133_dro_comparison.json

/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dro_robustness.py \
  --seeds 7,13 --quick --output /tmp/pr133_dro_robustness.md

/Users/lashby/Projects/AlloyGBM/.venv/bin/python -m pytest \
  bindings/python/tests/test_dro_leaf_solver.py \
  benchmarks/tests/test_dro_robustness.py \
  benchmarks/tests/test_dro_performance.py -q
```

The baseline command was run in a detached temporary worktree at production
`2b2e3ef`, after copying only the benchmark harness and benchmark fixture changes;
`git status` showed only those benchmark files and the production-source diff was
empty. The committed baseline JSON identifies the production-base HEAD; the
candidate JSON identifies `bde1ce1`. `fit_seconds` starts after estimator
construction and stops immediately after `fit`; `predict_seconds` is a separate
field and cannot affect fit timing.
