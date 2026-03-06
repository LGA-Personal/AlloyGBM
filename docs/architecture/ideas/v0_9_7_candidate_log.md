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
- `candidate43`: split leaf-magnitude filter in backend split scoring (env-gated).
- `candidate44`: `candidate36 + candidate43` coordinated env preset.
- `candidate45`: `candidate33 + candidate43` coordinated env preset.
- `candidate46`: `candidate36 + candidate33 + candidate43` coordinated env preset.

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
| 37 | Piecewise heavy-tail hybrid inside linear binning (env-gated) | Rejected | `fit +16.47%`, `predict +15.28%`, `RMSE -2.14%`, `MAE -5.04%`, `R2 +0.06549`, wins `9/18` | Same directional quality as candidate36 but with much worse runtime. Full rank fallback already captures the useful gain more cheaply. |
| 38 | Candidate33 + candidate36 combined env bundle | Rejected | vs baseline: `fit +19.45%`, `predict +17.86%`, `RMSE -4.26%`, `MAE -4.89%`, `R2 +0.11855`; vs candidate36: `fit +16.23%`, `predict +9.64%`, `RMSE -0.71%`, `MAE -1.63%`, `R2 +0.01906` | Small extra time-series lift is offset by finance regression and too much added runtime over candidate36 alone. |
| 39 | Repeated-extreme endpoint bucket inside linear binning (env-gated) | Rejected | vs baseline: `fit -1.95%`, `predict -0.43%`, `RMSE -1.94%`, `MAE -4.65%`, `R2 +0.05954`; vs candidate36: `fit -3.98%`, `predict -6.52%`, quality roughly flat-to-slightly worse | Cheap surrogate for candidate36, but accuracy-first review favors candidate36’s slightly stronger panel quality. |
| 40 | Lower-tail-only selective rank fallback inside linear binning (env-gated) | Rejected | vs baseline: `fit -1.54%`, `predict +0.70%`, `RMSE -1.94%`, `MAE -4.65%`, `R2 +0.05954`; vs candidate36: `fit -3.94%`, `predict -2.97%`, quality flat-to-slightly worse | Another cheaper candidate36 surrogate; keeping rank on only the dominant lower tail did not improve the panel slice enough to justify replacing full selective tail rank. |
| 41 | Soft split-balance penalty in backend split scoring (env-gated) | Rejected | vs baseline: `fit +11.97%`, `predict +6.29%`, `RMSE -0.93%`, `MAE -0.88%`, `R2 +0.02540` | Panel quality did improve, but finance RMSE/R2 regressed and the runtime penalty was too large for the modest overall gain. |
| 42 | Early min-child-row pruning in backend split scan (env-gated) | Rejected | seed-7 tuning vs baseline: `fit -8.37%`, `predict -0.57%`, `RMSE +1.18%`, `MAE +3.61%`, `R2 -0.03100` | Faster, but quality regressed immediately on both focus scenarios. Reject without running the full focused matrix. |
| 43 | Split leaf-magnitude filter in backend split scoring (env-gated) | Kept | vs baseline: `fit -21.82%`, `predict -22.52%`, `RMSE -0.02%`, `MAE -0.05%`, `R2 +0.00068` | Strong deep-low-lr speed win with effectively flat focused-slice quality; neutral on `panel_time_series`, mildly positive on `dow_jones_financial`. |
| 44 | Candidate36 + candidate43 coordinated env preset | Kept | vs baseline: `fit -33.39%`, `predict -25.47%`, `RMSE -2.34%`, `MAE -4.89%`, `R2 +0.07006`; vs candidate36: `fit -19.20%`, `predict -14.91%`, quality essentially flat-to-slightly better | This package preserves candidate36’s panel-quality gains while inheriting candidate43’s deep-low-lr/finance speed win. Strongest coordinated preset so far. |
| 45 | Candidate33 + candidate43 coordinated env preset | Kept | vs baseline: `fit -26.84%`, `predict -6.16%`, `RMSE -0.64%`, `MAE -2.27%`, `R2 +0.01896`; vs candidate33: `fit -16.56%`, `predict -7.86%`, quality essentially flat | This package improves candidate33’s runtime materially while keeping its focused-slice quality profile nearly unchanged. Useful low-signal/deep-low-lr preset. |
| 46 | Candidate36 + Candidate33 + Candidate43 coordinated env preset | Kept | vs baseline: `fit -32.36%`, `predict -20.23%`, `RMSE -4.32%`, `MAE -4.89%`, `R2 +0.12042`; vs candidate44: `fit -3.52%`, `predict -0.23%`, `RMSE -0.75%`, `MAE -1.23%`, `R2 +0.01969` | Best focused-slice quality package so far. Finance gives back a small amount of quality relative to candidate44, but the overall accuracy gain is materially larger while runtime remains much better than baseline. |

## Scenario Notes for Candidate36
- `panel_time_series` (`n=9`): `RMSE -6.57%`, `MAE -14.43%`, `R2 +0.19109`.
- `dow_jones_financial` (`n=9`): essentially neutral quality deltas.
- Profile medians:
  - `deep_low_lr`: quality improved with slight fit improvement.
  - `mid_balanced`: quality improved with slight fit improvement.
  - `shallow_high_lr`: quality improved with slight fit regression.

## Scenario Notes for Candidate43
- `panel_time_series` (`n=9`): median quality unchanged (`RMSE 0.00%`, `MAE 0.00%`, `R2 0.00000`) with slight speed improvement.
- `dow_jones_financial` (`n=9`): `fit -42.51%`, `predict -38.04%`, `RMSE -0.20%`, `MAE -0.06%`, `R2 +0.00487`.
- The win is concentrated in deep-low-lr / low-signal behavior rather than the panel heavy-tail slice.

## Scenario Notes for Candidate44
- `panel_time_series` (`n=9`): `fit -10.06%`, `predict -5.99%`, `RMSE -6.59%`, `MAE -14.38%`, `R2 +0.19023`.
- `dow_jones_financial` (`n=9`): `fit -50.78%`, `predict -46.78%`, `RMSE -0.20%`, `MAE -0.06%`, `R2 +0.00487`.
- Relative to `candidate36` alone, the package keeps panel quality effectively unchanged while adding a large finance/deep-low-lr speed win.

## Scenario Notes for Candidate45
- `panel_time_series` (`n=9`): `fit -6.30%`, `predict -1.20%`, `RMSE -1.38%`, `MAE -4.85%`, `R2 +0.04372`.
- `dow_jones_financial` (`n=9`): `fit -46.77%`, `predict -43.12%`, `RMSE +0.27%`, `MAE +0.47%`, `R2 -0.00622`.
- Relative to `candidate33` alone, the package keeps panel quality unchanged and slightly improves finance quality while materially reducing runtime.

## Scenario Notes for Candidate46
- `panel_time_series` (`n=9`): `fit -13.39%`, `predict -4.00%`, `RMSE -7.18%`, `MAE -17.13%`, `R2 +0.22581`.
- `dow_jones_financial` (`n=9`): `fit -49.72%`, `predict -45.48%`, `RMSE +0.27%`, `MAE +0.47%`, `R2 -0.00622`.
- Relative to `candidate44`, the package gives a further panel-quality lift while keeping large baseline-relative speed gains; the extra regularization mainly costs a small amount of finance quality rather than runtime.

## Do-Not-Retry Standalone (Unless Architecture Changes)
- High-resolution global bins as a direct knob (`candidate27` shape).
- Leafwise expansion without sampling/control coupling (`candidate28` shape).
- Coarse-to-fine line-search split refinement (`candidate34` shape).
- Linear winsorization-only quantization (`candidate35` shape).
- Piecewise heavy-tail hybrid remap on top of selective tail rank (`candidate37` shape).
- Candidate33 regularization + candidate36 tail-rank as a blanket combined preset (`candidate38` shape).
- Repeated-extreme endpoint bucket as a replacement for selective tail rank (`candidate39` shape).
- Lower-tail-only selective rank fallback as a replacement for selective tail rank (`candidate40` shape).
- Soft split-balance penalty as a standalone split-search quality knob (`candidate41` shape).
- Early min-child-row pruning as a standalone split-search quality knob (`candidate42` shape).

## Likely Synergy Directions (Worth Coordinated Packages)
1. Backend-regularized split scoring (`candidate32/33`) + selective tail-aware quantization (`candidate36`) under one controlled package.
2. Data representation upgrades from `performance_and_training_ideas_2.md`:
   - adaptive precision histograms,
   - staged row partition engine,
   - repacked split descriptors.
3. Quality-first split-search improvements with strict fit-time guardrails and deep-low-lr included in every acceptance run.
4. Backend-side split-search quality work beyond the now-accepted tail-rank/regularization/filter stack, rather than more linear-tail quantization variants.

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
