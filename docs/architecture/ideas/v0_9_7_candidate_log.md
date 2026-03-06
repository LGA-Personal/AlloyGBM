# AlloyGBM v0.9.7 Candidate Execution Log

## Purpose
Capture what has already been tried in the v0.9.7 optimization sprint so future iterations avoid repeating weak standalone ideas and focus on higher-leverage coordinated upgrades.

This log is intentionally outcome-oriented:
- `kept`: merged to `main` as a v0.9.7 part.
- `rejected`: code reverted after A/B.
- `variant-kept`: preserved as configurable or env-gated path, not promoted as universal default.

## Upstream References (Local, Non-Product Dependencies)
- `tmp/upstream_refs/LightGBM` (source snapshot used for architectural study)
- `tmp/upstream_refs/xgboost`
- `tmp/upstream_refs/catboost`

These repos are local reference material and should not be included in commits.

## Current Accepted Baseline Themes

### Training/inference throughput improvements already merged
- Parallel histogram tiles.
- Predictor tree traversal rewrite.
- Cached parsed predictor handle per fitted model.
- Predictor batch row parallelization.
- Histogram subtraction in depth expansion.
- Fused split partition + stats accumulation.
- Deterministic subsampling fast path (`nth-select`).
- Reduced release-build gradient validation overhead.
- Round prediction buffer reuse.

### Quality-oriented upgrades already merged
- Tree-semantics delta updates per node (`candidate29`).
- Backend-integrated L2 split/leaf regularization plumbing (`candidate32`).
- L1+L2 split/leaf scoring (`candidate33`) with strong deep-low-lr fit-time tradeoff.

### Newly kept in this step
- `candidate36`: selective tail-rank fallback for `linear` binning (env-gated).

## Candidate Outcomes (Recent Quality-Focused Wave)

| Candidate | Idea | Status | Core A/B Outcome | Guidance |
| --- | --- | --- | --- | --- |
| 27 | High-resolution bins (`max_bins=1024`) for linear/rank/quantile | Rejected | `fit +2621.92%`, `predict +217.66%`, `RMSE -1.00%`, `MAE -1.28%`, `R2 +0.00417` | Accuracy gain is real but cost is unusable standalone. Revisit only with a fundamentally cheaper representation. |
| 28 | Leafwise best-first growth (env-gated) | Rejected | `fit +84.51%`, no quality gain (`RMSE/MAE/R2` unchanged) | Do not retry alone. If revisited, must be paired with strong sampling/computation control (for example GOSS/MVS). |
| 29 | Tree-semantics delta updates per node | Kept | `fit +3.32%`, `RMSE -5.03%`, `MAE -4.80%`, `R2 +0.01994` | Strong quality lift for modest runtime cost; remains a core v0.9.7 quality improvement. |
| 31 | Engine-side regularized split re-score (env-gated, high overhead path) | Rejected | `fit +39.14%`, quality improved (`RMSE -1.32%`, `R2 +0.00534`) | Quality direction good, but architecture too expensive; superseded by candidate32 backend-integrated path. |
| 32 | Backend-integrated regularized split scoring (L2/min-child-hessian) | Kept | `fit +0.80%`, `RMSE -0.26%`, `MAE -0.06%`, `R2 +0.00119` | Keep as low-overhead regularization foundation. |
| 33 | L1 + L2 regularized split/leaf scoring | Kept | Focused: `RMSE -0.88%`, `MAE -0.29%`, `R2 +0.00419`; Deep low-lr: `fit -20.57%` with near-flat quality | Accepted tradeoff, especially strong for deep low-lr low-SNR workloads. |
| 34 | Coarse-to-fine line-search threshold refinement | Rejected | `fit +5.66%`, `predict +2.13%`, `RMSE +1.06%`, `MAE +0.48%`, `R2 -0.02912` | Clear negative. Do not retry in current form. |
| 35 | Linear winsorized quantization (env-gated) | Rejected | `fit +3.68%`, `predict +2.82%`, quality unchanged (`RMSE/MAE/R2` flat) | No value. Keep out unless combined with a materially different binning/search redesign. |
| 36 | Selective tail-rank fallback inside linear binning (env-gated) | Kept | `fit -0.39%`, `predict +1.34%`, `RMSE -2.14%`, `MAE -4.72%`, `R2 +0.06526`, wins `9/18` | Promising quality boost at near-flat fit cost; strongest on `panel_time_series`, neutral on `dow_jones_financial`. |

## Scenario Notes for Candidate36
- `panel_time_series` (`n=9`): `RMSE -6.57%`, `MAE -14.43%`, `R2 +0.19109`.
- `dow_jones_financial` (`n=9`): essentially neutral quality deltas.
- Profile medians:
  - `deep_low_lr`: quality improved with slight fit improvement.
  - `mid_balanced`: quality improved with slight fit improvement.
  - `shallow_high_lr`: quality improved with slight fit regression.

## Do-Not-Retry Standalone (Unless Architecture Changes)
- High-resolution global bins as a direct knob (`candidate27` shape).
- Leafwise expansion without sampling/control coupling (`candidate28` shape).
- Coarse-to-fine line-search split refinement (`candidate34` shape).
- Linear winsorization-only quantization (`candidate35` shape).

## Likely Synergy Directions (Worth Coordinated Packages)
1. Backend-regularized split scoring (`candidate32/33`) + selective tail-aware quantization (`candidate36`) under one controlled package.
2. Data representation upgrades from `performance_and_training_ideas_2.md`:
   - adaptive precision histograms,
   - staged row partition engine,
   - repacked split descriptors.
3. Quality-first split-search improvements with strict fit-time guardrails and deep-low-lr included in every acceptance run.

## Benchmarking Protocol Reminder (Current)
- Always include deep-low-lr slice in A/B:
  - `shallow_high_lr:0.2:4:200`
  - `mid_balanced:0.1:6:400`
  - `deep_low_lr:0.01:8:5000`
- Seeds: `7,17,29`.
- Focus scenarios used recently for quality direction:
  - `panel_time_series`
  - `dow_jones_financial`
- Keep full-matrix checks for promotion decisions.
