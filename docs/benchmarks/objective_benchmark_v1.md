# Objective Benchmark Pack

Deterministic, offline validation for the reviewed large-query LambdaMART and
skewed-count GLM paths. It measures held-out synthetic data only; it is a
regression and calibration check, not a cross-library comparison.

## Configuration

- Seeds: `7, 13, 29`
- Ranking: four training and two held-out query groups, `512` rows each
- Models: `50` trees, depth `4`, learning rate `0.06`, `lambda_l2=1.0`
- Ranking metric: mean held-out NDCG@10
- GLM metric: held-out mean deviance against a train-mean baseline

Command:

```bash
.venv/bin/python benchmarks/objective_benchmark.py --gate
```

## LambdaMART Large-Query A/B

`full` evaluates all pairwise candidates. `top_10` uses
`lambdarank_truncation_level=10`. Timing is descriptive because host load and
Rayon scheduling make it unsuitable as a hard gate.

| Arm | Held-out NDCG@10 | Median fit (s) |
| --- | ---: | ---: |
| `full` | 0.97065 | 0.049 |
| `top_10` | 0.96297 | 0.032 |

The truncated arm loses `0.00768` absolute NDCG@10 on this fixture while
reducing median fit time by about 35%. The benchmark gate permits at most
`0.10` absolute NDCG@10 loss; it does not gate timing.

## Skewed-Count GLM Validation

The Poisson, Gamma, and Tweedie fixtures each use targets in their supported
domain. The two Poisson rows isolate the default
`poisson_max_delta_step=0.7` stabilizer from the legacy zero-step setting.
Lower deviance is better.

| Objective | Held-out deviance | Train-mean baseline | Median fit (s) |
| --- | ---: | ---: | ---: |
| `poisson_default` | 1.51352 | 2.82301 | 0.011 |
| `poisson_no_stabilizer` | 1.39007 | 2.82301 | 0.011 |
| `gamma` | 1.16029 | 1.62547 | 0.011 |
| `tweedie` | 1.87065 | 3.06581 | 0.013 |

The stabilized Poisson path is finite and improves the held-out fixture by
46% relative to the train-mean baseline. The zero-step arm happens to score
better here, so this report intentionally does not claim that a nonzero
maximum delta step is universally superior; its purpose is to validate stable,
useful behavior on skewed counts.
