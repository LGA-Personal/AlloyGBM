# PR #137 DART Expected-Drop Calibration

Status: implemented and revision-hardened on `codex/dart-drop-calibration`; the public default is `5`.

## Decision

The fixed five-seed matrix selected `dart_max_drop=5`, the largest candidate that
passed all eight predeclared gates. Candidate `2` also passed. Candidates `10`
and `20` were rejected by the stress pressure and stress fit-time gates. The
incumbent `50` remains available as an explicit compatibility override.

The matrix contains 10 fixtures x 5 seeds x 5 explicit caps = 250 records.
Every arm passed an explicit cap, so the matrix decision is independent of the
installed Python default.

## Method

The harness is `benchmarks/dart_policy_calibration.py`. Each fit runs in a fresh
subprocess and records completed rounds, finite held-out metrics, fit time,
normalized peak RSS, configured dropout pressure, prediction SHA-256, artifact
SHA-256, and the capture source commit. The fixed commands were:

```bash
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dart_policy_calibration.py run \
  --caps 2 5 10 20 50 --seeds 0 1 2 3 4 \
  --output benchmarks/results/pr137_dart_policy_matrix.json
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dart_policy_calibration.py run-compat \
  --arms default cap50 --seeds 0 1 2 \
  --output benchmarks/results/pr137_dart_policy_production_compat.json
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dart_policy_calibration.py run-compat \
  --arms default cap-selected cap50 --selected-cap 5 --seeds 0 1 2 \
  --output benchmarks/results/pr137_dart_policy_candidate_compat.json
/Users/lashby/Projects/AlloyGBM/.venv/bin/python benchmarks/dart_policy_calibration.py compare \
  benchmarks/results/pr137_dart_policy_matrix.json \
  --production-compat benchmarks/results/pr137_dart_policy_production_compat.json \
  --candidate-compat benchmarks/results/pr137_dart_policy_candidate_compat.json \
  --output /tmp/pr137-dart-policy-comparison.json
```

The explicit-cap matrix was regenerated from committed harness `14cd42a`; its
records do not depend on the installed default. The production compatibility
capture remains the pre-default-change capture from source `92a4964`. The
candidate compatibility capture was regenerated after comparator hardening
from committed state `9074faf`.

The host was macOS 26.5.2 arm64 on an Apple arm64 machine with 10 logical CPUs
and 24 GiB RAM. The shared environment used Python 3.13.5, NumPy 2.5.0,
scikit-learn 1.9.0, and AlloyGBM 0.12.10. Timing is a same-host policy gate,
not a universal speed claim across machines or process environments.

## Fixed Contract

Uniform dropout pressure is computed as:

```text
sum(min(dart_max_drop, max(1, dart_drop_rate * existing_tree_count)))
```

The positive-rate forced-one rule contributes one selected tree when sampling
would otherwise select none. Multiclass pressure advances `existing_tree_count`
by the four class trees committed per logical round; it is not treated as one
tree per round. Ranking train/test rows are split by whole contiguous groups:
64 groups for training and 16 groups for held-out NDCG.

The cap grid, incumbent, seeds, fixtures, metric orientation, and thresholds
were fixed before selection:

| Gate | Threshold | 2 | 5 | 10 | 20 |
|---|---:|:---:|:---:|:---:|:---:|
| Complete and finite | every record | pass | pass | pass | pass |
| Five-seed median primary quality | <= 1.02 | pass | pass | pass | pass |
| Individual-seed primary quality | <= 1.10 | pass | pass | pass | pass |
| Accuracy / NDCG loss | <= 0.02 / 0.01 | pass | pass | pass | pass |
| Stress configured pressure ratio | <= 0.50 | pass | pass | **fail** | **fail** |
| Stress fit-time ratio | <= 0.85 | pass | pass | **fail** | **fail** |
| Peak RSS | max(15%, 32 MiB) over incumbent | pass | pass | pass | pass |
| Compatibility and determinism | all required checks | pass | pass | pass | pass |

Lower-is-better metrics use `candidate / incumbent`; NDCG uses
`incumbent / candidate`. The machine comparator validates the exact
predeclared matrix catalog, caps, seeds, complete unique keys, finite values,
positive resources, fixture metadata, recomputed configured pressure,
four-tree multiclass pressure, all eight gates, and largest-passing selection
without rounding. It also validates the fixed compatibility capture catalogs,
actual arm caps, source-commit uniformity, stored-check consistency, and
cross-capture cap-50 hashes.

## Quality Ratios

Values are five-seed median primary-quality ratios against cap `50`; the
incumbent reference is `1.000000`.

| Fixture | Cap 2 | Cap 5 | Cap 10 | Cap 20 |
|---|---:|---:|---:|---:|
| `reg-small-narrow` | 0.936121 | 0.996482 | 1.000000 | 1.000000 |
| `reg-small-wide` | 0.937803 | 0.985150 | 0.999034 | 1.000000 |
| `reg-tall-narrow` | 0.666693 | 0.843672 | 0.956706 | 0.998884 |
| `reg-tall-wide-leaf` | 0.850520 | 0.937087 | 0.982476 | 0.999550 |
| `reg-long-stress` | 0.618469 | 0.764947 | 0.866946 | 0.947327 |
| `binary-medium` | 0.599699 | 0.837823 | 0.972980 | 0.999887 |
| `multiclass-four` | 0.731057 | 0.853913 | 0.937221 | 0.985013 |
| `ranking-groups` | 0.989367 | 0.994712 | 1.000000 | 1.000000 |
| `reg-weighted` | 0.726138 | 0.873260 | 0.961848 | 0.998185 |
| `reg-forest` | 0.677843 | 0.889119 | 0.968215 | 0.996229 |

The largest individual-seed quality ratios were `1.001169`, `1.011631`,
`1.011631`, and `1.000006` for caps `2`, `5`, `10`, and `20` respectively.
Median accuracy deltas for binary/multiclass were respectively
`(+0.004878, +0.078125)`, `(0.000000, +0.040625)`,
`(0.000000, +0.012500)`, and `(0.000000, +0.006250)`. Median NDCG@10
deltas (candidate minus incumbent; positive is better) were `+0.009739`,
`+0.004729`, `0.000000`, and `0.000000`.

## Work, Time, and RSS

Stress aggregates use the five stress fixtures with 200 or 300 rounds and 25
records per cap. The 100-round multiclass fixture remains in the 250-record
matrix for quality and accuracy coverage but is excluded from pressure/time
aggregation. Ratios are against the cap-50 median aggregate. Peak RSS is the
maximum across the full 250-record matrix; incumbent maximum RSS was
`179486720` bytes and the allowed ceiling was `213041152` bytes.

| Cap | Median pressure | Pressure ratio | Median fit seconds | Time ratio | Max RSS bytes | RSS ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 383.5 | 0.192279 | 0.218926 | 0.461466 | 178782208 | 0.996075 |
| 5 | 877.0 | 0.439709 | 0.297684 | 0.627477 | 179339264 | 0.999178 |
| 10 | 1499.5 | 0.751817 | 0.396034 | 0.834786 | 178864128 | 0.996531 |
| 20 | 1994.5 | 1.000000 | 0.472432 | 0.995822 | 178667520 | 0.995436 |
| 50 | 1994.5 | 1.000000 | 0.474414 | 1.000000 | 179486720 | 1.000000 |

Rejected candidates retain their exact reasons in the comparison JSON:

- Cap `10`: stress pressure ratio `0.75181749812 > 0.5`.
- Cap `20`: stress pressure ratio `1.0 > 0.5`; stress fit-time ratio
  `0.995821779481 > 0.85`.

## Compatibility Hashes

Production compatibility contains 24 records across four fixtures, three seeds,
and `default`/`cap50` arms. All 12 default-versus-cap50 prediction and artifact
SHA-256 pairs match exactly. Candidate compatibility contains 36 records across
`default`, `cap-selected`, and `cap50`; all 12 default-versus-selected-cap-5
pairs match exactly, and all 12 candidate-cap50 pairs match the production
cap50 capture. Each compatibility fit was repeated in the subprocess; all
repeated prediction and artifact hashes matched.

The complete 64-character lowercase SHA-256 values remain in the committed
JSON evidence files. The machine comparator consumes capture metadata, actual
caps, determinism repeats, and hash parity. The separate regression suite
also pins exact artifact and prediction equality for selected-cap `n_jobs=1`
versus `n_jobs=2`, repeated explicit-50 regression/multiclass/ranking fits,
selected-default level/leaf fits, and selected-default warm-start predictions
within the existing `1e-5` relative / `1e-6` absolute tolerance; warm-start and
`n_jobs` are not JSON compatibility records or comparator inputs. Independent
multilabel mode inherits cap `5`; joint mode forwards explicit cap `50`
unchanged.

## Implementation

Only the two public Python constructor literals changed, both in
`bindings/python/alloygbm/_regressor/_core.py`. Rust/native training,
dropout RNG, sampling, forced-one behavior, truncation order, normalization,
artifacts, and predictor code are unchanged. Users who need the previous
default can pass `dart_max_drop=50` explicitly; explicit values retain their
existing meaning.
