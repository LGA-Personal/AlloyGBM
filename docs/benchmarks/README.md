# Benchmark Notes

Benchmark execution lives in `benchmarks/`.

Use that directory for:

- dataset preparation
- cross-library model comparison
- result artifacts under `benchmarks/results/`

Benchmark documentation in `docs/` should stay focused on:

- how to interpret the benchmark suite
- which scenarios are representative
- what the benchmark results say about AlloyGBM's current strengths and gaps
- what the stage timing columns say about Python adaptation versus native training cost

Current benchmark entry points:

- cross-library runner guide: `benchmarks/README.md`
- MorphBoost-focused harnesses (`morph_report.py`, `morph_ablation.py`,
  `numerai_benchmark.py`): see `benchmarks/README.md`
- deterministic DRO clean-holdout harness: `benchmarks/dro_robustness.py`
  with report at [dro_robustness_v1.md](dro_robustness_v1.md)
- deterministic large-query LambdaMART and skewed-count GLM harness:
  `benchmarks/objective_benchmark.py` with report at
  [objective_benchmark_v1.md](objective_benchmark_v1.md)
- deterministic July-review evidence harness:
  `benchmarks/review_guardrails.py` with report at
  [review_guardrails_v1.md](review_guardrails_v1.md). Its DART profile keeps
  the 200-round `0.20 / 50` stress arm visible; only explicit default-like
  profiles (`drop_rate <= 0.10`) use the 1.50x RMSE quality gate, and timing
  remains descriptive.
- deterministic scalar monotone-constraint acceptance harness:
  `benchmarks/monotone_constraints_benchmark.py` with report at
  [monotone_constraints_v1.md](monotone_constraints_v1.md). It checks finite
  numeric sweeps for zero scalar-prediction violations; timing is descriptive.
- deterministic fit-thread and multiclass class-tree scaling harness:
  `benchmarks/multiclass_parallelism_benchmark.py` with report at
  [multiclass_parallelism_v1.md](multiclass_parallelism_v1.md). It requires
  exact serial/parallel artifact and prediction hashes, finite probabilities,
  completed rounds, and prior-beating log loss across row/feature shapes,
  class counts, and both growth strategies.
- isolated histogram and partition allocation-reuse harness:
  `benchmarks/allocation_reuse_benchmark.py` with report at
  [allocation_reuse_v1.md](allocation_reuse_v1.md). It compares manifest-attested
  baseline and candidate native extensions across four matrix shapes and both
  growth strategies, requiring exact artifact and prediction digests before
  applying aggregate native-time and RSS gates.
- isolated sampled-prediction-delta harness:
  `benchmarks/sampled_prediction_delta_benchmark.py`, with Task 6 evidence
  at [sampled_prediction_delta_v1.md](sampled_prediction_delta_v1.md) and
  [sampled_prediction_delta_v1.json](sampled_prediction_delta_v1.json).
  It pairs scalar and multiclass subsample/GOSS cases exactly, keeps DART and
  quantile as explicit full-replay sentinels, and reserves timing/RSS claims
  for five-repetition runs from distinct manifest-attested commits.
- descriptive high-class-count, low-drop-cap multiclass DART scratch harness:
  `benchmarks/multiclass_dart_scratch_benchmark.py` with report at
  [multiclass_dart_scratch_v1.md](multiclass_dart_scratch_v1.md). It has no
  timing gate.
- isolated baseline/candidate architecture harness for the six July-review
  projects: `benchmarks/architectural_backlog/` with methodology and baseline
  at [architectural_backlog_v1.md](architectural_backlog_v1.md)
- comparative inspiration and follow-ups: `docs/plans/perpetual_inspiration_for_alloygbm.md`
- older benchmark writeups: `docs/archive/benchmarks/`

The cross-library runner registers two MorphBoost variants of AlloyGBM by
default — `alloygbm_morph` and `alloygbm_morph_cosine` — alongside the
standard `alloygbm` arm. Use the runner's `--models` flag to filter which
arms run; see [user/morphboost.md](../user/morphboost.md) for parameter
semantics and the [paper](https://arxiv.org/pdf/2511.13234) for the
formulation.

## Auto-policy calibration

`auto_policy_benchmark.py` exercises independent row/column shape strata
across regression, classification, and ranking objectives. The compact CI
sentinel is:

```bash
python benchmarks/auto_policy_benchmark.py --quick --gate
```

The full three-seed evidence and selection decision are recorded in
[auto_policy_calibration_v1.md](auto_policy_calibration_v1.md).

## Monotone-constraint acceptance

The compact CI sentinel and full committed-evidence capture are:

```bash
python3 benchmarks/monotone_constraints_benchmark.py --quick --gate
python3 benchmarks/monotone_constraints_benchmark.py \
  --gate \
  --output docs/benchmarks/monotone_constraints_v1.md
```

The full report covers regression and binary scalar models across training-row
counts, feature widths, both constraint directions, both tree-growth
strategies, and three seeds. It requires finite sweep values, zero violations,
completed rounds, bounded quality degradation, and improvement over a constant
predictor. Missing values are not ordered by this numeric sweep; they follow
the model's learned missing branch.
