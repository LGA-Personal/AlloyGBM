# Joint-DRO Robust Split-Gain Evidence

This records the evidence collected for the opt-in `dro_robust_split` flag
(`MultiLabelGBMRanker(multi_label_mode="joint", leaf_solver="dro",
dro_robust_split=True)`, mapping to `TrainParams.joint_dro_robust_split_gain`
in the Rust engine), which closes the "decide with data" half of the
[`docs/reviews/2026-07-02-v0.12.10-special-modes-resolutions.md`](../reviews/2026-07-02-v0.12.10-special-modes-resolutions.md)
§3 DRO-leaves gap: joint DRO previously made only leaf *values* robust; split
*selection* used standard Newton gain. This PR adds a parallel per-bin
`grad_sq` buffer (allocated only when the flag is on) so numeric-threshold
split selection can also route through the DRO effective-gradient formula
(`alloygbm_core::leaf_gain_term`), while keeping the flag off by default
(byte-identical to prior behavior) and leaving native-categorical and
MorphBoost joint splits on their existing gain regardless of the flag.

**Decision (reviewer): shipped as an experimental, off-by-default opt-in**
(`dro_robust_split`), NOT promoted to a default. The numbers below show no
reliable quality benefit, so it is not a default; it is retained (byte-identical
when off, trivial added memory) for evaluation on real heteroscedastic
downstream workloads that synthetic fixtures under-represent. This document
states the measured numbers plainly.

## Method

- Benchmark script: [`benchmarks/joint_dro_robust_split_benchmark.py`](../../benchmarks/joint_dro_robust_split_benchmark.py).
- Three fixed-seed (`seed=13`) synthetic fixtures, each fit twice (same seed,
  `dro_radius=0.4`, `leaf_solver="dro"`) with `dro_robust_split=False` vs.
  `True`:
  - **`homoscedastic_control`** — 2-output regression, constant-variance
    Gaussian noise everywhere. A control where robust split-gain has little
    reason to help (no region has disproportionate gradient variance to
    detect).
  - **`heteroscedastic_regions`** — 2-output regression, two feature regions
    with a 40x noise-standard-deviation ratio (`X[:, 0] > 0` "turbulent" vs.
    `X[:, 0] <= 0` "calm"). The scenario robust split-gain is meant to help:
    a high-variance region can otherwise mislead standard Newton gain into
    an over-confident split.
  - **`heteroscedastic_outliers_ranking`** — 2-output `rank:ndcg` joint
    ranking fixture with query groups; a fraction of rows (`X[:, 0] > 0.5`)
    carry heavy-tailed relevance-label noise (20x the baseline noise scale).
- Quality metric: mean RMSE across the 2 outputs on a 25% held-out split for
  the regression fixtures; mean NDCG across the 2 outputs for the ranking
  fixture.
- Each `(fixture, flag)` combination is fit in its **own subprocess** so the
  peak-RSS reading (`resource.getrusage(...).ru_maxrss`) reflects only that
  one fit + predict + eval, not the parent process or a prior combination
  run in-process.
- Two invocations were run: the required `--quick` pass (`n_estimators=12`,
  ~576–3,072 rows depending on fixture) and a larger `--full` pass
  (`n_estimators=60`, ~3,840–5,120 rows) to check whether the quick-mode
  deltas were an artifact of the small fixture size.
- Host: macOS 26.5.2 (Darwin 25.5.0, arm64), Python 3.13.5, numpy 2.5.0.
  Source commit for this evidence: `46c1dca602976769275cecb57eb0094dee7d4bd9`.

## Results — `--quick`

| Fixture | Metric | Off | On | Delta (on − off) | Fit s (off) | Fit s (on) | Time ratio | Peak RSS MiB (off) | Peak RSS MiB (on) | RSS ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| homoscedastic_control | mean_rmse | 0.608455 | 0.613031 | +0.004577 | 0.1061 | 0.1341 | 1.264x | 160.59 | 160.88 | 1.002x |
| heteroscedastic_regions | mean_rmse | 6.394254 | 6.409024 | +0.014769 | 0.0798 | 0.0857 | 1.075x | 160.89 | 160.52 | 0.998x |
| heteroscedastic_outliers_ranking | mean_ndcg | 0.934416 | 0.929012 | −0.005404 | 0.0978 | 0.1157 | 1.183x | 160.86 | 160.75 | 0.999x |

## Results — `--full`

| Fixture | Metric | Off | On | Delta (on − off) | Fit s (off) | Fit s (on) | Time ratio | Peak RSS MiB (off) | Peak RSS MiB (on) | RSS ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| homoscedastic_control | mean_rmse | 0.249717 | 0.250574 | +0.000857 | 0.5464 | 0.6542 | 1.197x | 161.83 | 161.61 | 0.999x |
| heteroscedastic_regions | mean_rmse | 5.723177 | 5.710844 | −0.012333 | 0.4987 | 0.5823 | 1.168x | 162.03 | 161.73 | 0.998x |
| heteroscedastic_outliers_ranking | mean_ndcg | 0.947836 | 0.941470 | −0.006365 | 0.5892 | 0.6820 | 1.158x | 162.77 | 162.77 | 1.000x |

All predictions were finite (`True`/`True`) in every combination in both
passes.

## Observations

**Quality delta.** Across both passes and all three fixtures, the RMSE/NDCG
delta between `dro_robust_split=False` and `=True` is small (at most
~0.015 absolute on `heteroscedastic_regions`, ~0.006 on the ranking fixture)
and **inconsistent in sign**: robust split-gain is very slightly worse on
`homoscedastic_control` in both passes (as expected — there is no
heteroscedasticity for it to exploit, and it adds noise to split selection
instead), very slightly better on `heteroscedastic_regions` and the ranking
fixture in the `--full` pass, but very slightly worse on
`heteroscedastic_regions` in the `--quick` pass. Given the deltas are all
within roughly one order of magnitude of run-to-run noise for these fixture
sizes, this benchmark does **not** show a clear, consistent quality
improvement from making joint-DRO split selection robust — including on the
`heteroscedastic_regions` fixture that was specifically designed to favor it.

**Fit-time cost.** `dro_robust_split=True` consistently costs roughly
1.08x–1.26x the fit wall time of the flag-off path across every fixture and
both passes — the extra `grad_sq` accumulation sweep per node is a real,
measurable cost every time it's enabled (regardless of whether it changes
quality).

**Memory cost — instrument limitation.** The measured RSS ratio is
~0.998x–1.002x in every row: at these fixture sizes (max 4 features, ~5,120
rows, default `max_bin`), the joint per-node histogram plus its `grad_sq`
buffer is on the order of tens of kilobytes, while the whole-process peak
RSS is dominated by the ~160 MiB Python/NumPy/Rust-extension baseline. Put
plainly: **whole-process peak RSS, as measured here, is the wrong instrument
to detect the hypothesized ~1.5x joint-histogram memory delta** — a delta of
that magnitude on a tens-of-KB buffer is many orders of magnitude below the
process's own noise floor. This benchmark's RSS numbers should be read as
"no signal, not a null result" for the memory question, not as evidence that
the ~1.5x hypothesis is wrong. Confirming or refuting the ~1.5x claim would
need either much larger row/feature/bin counts (to grow the buffer itself
into a visible fraction of process RSS) or a Rust-level allocation
instrument that isolates just the per-round histogram/`grad_sq` buffers from
the rest of the process.

## Scope reminder

This flag only affects **numeric threshold** splits; native-categorical and
MorphBoost joint splits keep their existing (non-robust) gain even when
`dro_robust_split=True` (deferred scope, matching v0.10.5's leaf-only DRO
precedent for those paths). It is a no-op unless `leaf_solver="dro"` and
`dro_radius > 0` are also active. `dro_robust_split=False` (the default) is
byte-identical to the pre-existing DRO-leaf-only path — pinned by the cargo
test `joint_robust_split_flag_off_matches_leaf_only_byte_for_byte` and the
pytest `test_joint_dro_robust_split_default_off_byte_identical`.
