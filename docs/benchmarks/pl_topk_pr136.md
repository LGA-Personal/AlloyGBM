# PR #136: Top-k PL Histogram Construction

## Decision

PR #136 implements bounded piecewise-linear (PL) split rescoring behind the
public `pl_split_candidates` parameter. The production default is `0`, which
uses the existing standard split criterion and fits linear child models only
after selecting a split. Positive values are experimental opt-ins.

The originally proposed default of `8` was rejected. It reduced the cost of
exhaustive PL rescoring on wide data, but it was substantially slower than the
production path and its held-out quality was inconsistent. Keeping `0` as the
default preserves behavior while making the bounded architecture available for
explicit evaluation.

## Method

The baseline is production commit `ea4df36`; the candidate evidence was captured
at `69ef04e`. Every fit ran in a fresh subprocess on macOS 26.5.2 arm64 with
Python 3.13.5, AlloyGBM 0.12.10, NumPy 2.5.0, and scikit-learn 1.9.0.

The matrix contains five seeds and 75 fixture/checkpoint identities spanning
small/narrow, small/wide, tall/narrow, tall/wide, binary, multiclass, ranking,
local-linear, nonlinear, sparse-signal, and raw-scale data. Candidate arms are:

- `default`: omit the parameter;
- `k0`: explicitly set `pl_split_candidates=0`;
- `k8`: shortlist eight numeric features;
- `all`: set a value above the feature count for exhaustive eligible-feature
  rescoring.

Each record captures primary and secondary held-out metrics, fit time, peak RSS,
prediction digest, artifact digest, and completed rounds. The machine-readable
inputs and comparison are committed under `benchmarks/results/`.

## Results

| Check | Result |
|---|---:|
| Default artifact parity vs production | exact on all 75 records |
| Default prediction parity vs production | exact on all 75 records |
| Explicit `k=0` artifact/prediction parity | exact on all 75 records |
| Median `k=8 / k=0` fit-time ratio | 8.09x |
| Worst `k=8 / k=0` fit-time ratio | 15.05x |
| Wide `k=8 / all` fit-time ratio | 13.96–20.66% (median 16.49%) |
| Worst observed RSS growth vs `k=0` | 560 KiB |

Quality was mixed. At the final checkpoint, median `k=8` improved binary log
loss (0.3050 to 0.2956), small/narrow regression RMSE (0.8740 to 0.8223), and
small/wide regression RMSE (0.8425 to 0.8239). It regressed local-linear RMSE
(0.3675 to 0.6063), raw-scale RMSE (0.4457 to 0.5514), multiclass log loss
(0.8860 to 0.9205), and tall/wide regression RMSE (0.6536 to 0.7124). Ranking
NDCG@10 was effectively neutral (0.9398 to 0.9405).

These measurements support the bounded architecture but do not support a
quality or convergence claim for `k=8`. Opt-in quality and fixed-round cost are
therefore observations rather than acceptance gates.

## Architecture

Standard scalar histograms select the best overall split and rank one numeric
candidate per eligible feature. Each shortlisted feature is then exhaustively
rescored over all thresholds and both missing directions with its own split-path
regressor set. Rayon workers reuse thread-local one-feature PL histogram scratch,
and final winner reduction follows deterministic shortlist order. MorphBoost,
constant leaves, `k=0`, and native-categorical winners keep their existing paths.

The implementation does not add an artifact section or change prediction.

## Rejected Trials

1. The historical matrix-gain formulation with default `8` reached a
   representative local-linear RMSE ratio of 1.762 and fixed-round time ratio of
   8.66x versus production.
2. An uncommitted joint intercept/slope gain formulation still produced a 1.539
   representative RMSE ratio and approximately 20x fixed-round cost.

Both trials were discarded rather than weakening the predeclared default gates.

## Reproduction

```bash
python benchmarks/pl_topk_performance.py run \
  --arms default k0 k8 all --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr136_pl_topk_candidate.json

python benchmarks/pl_topk_performance.py compare \
  benchmarks/results/pr136_pl_topk_baseline.json \
  benchmarks/results/pr136_pl_topk_candidate.json \
  --output benchmarks/results/pr136_pl_topk_comparison.json
```

The comparison must report `passed: true`, exact default and `k=0` digest
parity, bounded RSS, and a worst wide `k=8 / all` time ratio no greater than
0.50. The report intentionally preserves all opt-in quality observations.
