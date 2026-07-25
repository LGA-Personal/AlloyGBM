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
