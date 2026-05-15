# Examples

Runnable scripts demonstrating AlloyGBM's public API. Each example is
self-contained, uses only standard scikit-learn datasets (no network
access required), and ends with a clear printed summary so it doubles
as an end-to-end smoke test.

## Running

After installing AlloyGBM (`pip install alloygbm` or `maturin develop
--release` from a clone):

```bash
python examples/01_regression_basics.py
python examples/02_binary_classification.py
python examples/03_ranking_with_groups.py
python examples/04_multi_label_ranking.py
python examples/05_factor_neutral_boosting.py
python examples/06_interaction_constraints.py
python examples/07_warm_start_continuation.py
python examples/08_shap_explanations.py
```

Each example also includes a one-line docstring at the top of the file
describing what it demonstrates, so they're greppable.

## What's Covered

| File | Demonstrates |
|---|---|
| `01_regression_basics.py` | `GBMRegressor`, train/eval split, RMSE, SHAP |
| `02_binary_classification.py` | `GBMClassifier`, `predict_proba`, accuracy, log-loss |
| `03_ranking_with_groups.py` | `GBMRanker` with `rank:ndcg`, group IDs, NDCG@k |
| `04_multi_label_ranking.py` | `MultiLabelGBMRanker`, per-label `ranking_objective` lists |
| `05_factor_neutral_boosting.py` | `neutralization="per_round_gradient"` with `factor_exposures` |
| `06_interaction_constraints.py` | LightGBM-compatible `interaction_constraints=[[...]]` |
| `07_warm_start_continuation.py` | `warm_start=True` / `init_model` resumption |
| `08_shap_explanations.py` | TreeSHAP via `shap_values()` + global `feature_importances()` |

If you add a new example, add it to this table.

## Adding A New Example

Style guide:

1. **Self-contained.** No external file dependencies. Use sklearn
   datasets or generate synthetic data with a fixed seed.
2. **Print a summary.** End the script with one or two `print()` lines
   showing the key metric. Examples are smoke tests; if a feature
   regresses, the print output changes.
3. **Comment the *why*, not the *what*.** Readers can see what the
   API call does; explain why this specific configuration is the
   right choice for the demonstrated scenario.
4. **Fixed seeds.** Every example uses `seed=7` so runs are
   reproducible.
