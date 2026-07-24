# Review Evidence Guardrails Design

## Purpose

The July 2026 special-modes review has three evidence items that should be
resolved before adding more training behavior:

- compare the current L2-proxy quantile split criterion with smoothed pinball;
- calibrate the exposed GOSS `top_rate` / `other_rate` surface;
- profile how DART dropout work scales with fit horizon, drop rate, and cap.

This PR adds deterministic, offline experiments and a small CI guard. It does
not add estimator parameters, alter objective math, optimize DART, or change
trained artifacts. Timing is descriptive because shared CI runners cannot
provide a stable wall-clock contract.

## Repository Layout

```text
benchmarks/review_guardrails.py
benchmarks/tests/test_review_guardrails.py
docs/benchmarks/review_guardrails_v1.md
docs/benchmarks/review_evidence_guardrails_design.md
.github/workflows/ci.yml
```

The benchmark remains one module because the three experiments share result
types, aggregation, report rendering, gate evaluation, seed parsing, and the
quick/full command contract. Domain-specific fixture and scoring helpers stay
separate within the module.

## Command Contract

Full evidence run:

```bash
.venv/bin/python benchmarks/review_guardrails.py --gate
```

Development and CI run:

```bash
.venv/bin/python benchmarks/review_guardrails.py --quick --gate
```

The CLI accepts `--seeds`, `--output`, and repeatable `--section` values from
`quantile`, `goss`, and `dart`. Without `--output`, the report is written to
stdout. Gate failures print every failed contract before returning nonzero.

## Smoothed-Pinball Split Experiment

This experiment deliberately evaluates split selection rather than fitting a
custom-objective model. The Python custom-objective path does not run
quantile's empirical leaf refinement, so comparing it with
`objective="quantile"` would conflate the split criterion and leaf solver.

For each seed and `alpha` in `0.1`, `0.5`, and `0.9`, a deterministic
heteroscedastic fixture supplies one training node and an independent holdout.
The current arm uses the quantile objective's constant proxy Hessian. Smoothed
arms use asymmetric Huberized pinball gradients and Hessians with transition
widths proportional to the training residual MAD. Every valid threshold is
scored with the same regularized Newton gain formula.

After selecting a threshold, both child predictions are replaced by their
weighted empirical training quantiles. Held-out pinball loss therefore
compares only the consequence of split selection while preserving AlloyGBM's
actual leaf-value contract. The no-split empirical quantile is recorded as the
baseline.

Quality gates require:

- deterministic fixtures and selections for a fixed seed;
- finite gains and held-out losses;
- non-empty child partitions;
- every arm to remain within 10% of the no-split loss on the aggregate;
- the report to identify the best median smoothing width without claiming it
  should become a production default.

The result decides whether a later production smoothed-pinball PR is justified.
It does not select or expose a public smoothing value.

## GOSS Rate Sweep

The GOSS fixture is held-out nonlinear regression with independent train/test
noise. Each seed compares:

- full-row standard boosting;
- uniform row subsampling at the same retained fraction as each GOSS arm;
- GOSS `(top_rate, other_rate)` values `(0.1, 0.1)`, `(0.2, 0.1)`,
  `(0.2, 0.2)`, and `(0.3, 0.1)`.

Models use manual policy, deterministic quantile binning, fixed depth and
learning rate, and all requested rounds. Results include held-out RMSE,
train-mean baseline RMSE, retained fraction, completed rounds, and fit time.

Quality gates require finite predictions, all requested rounds, every GOSS arm
to beat the mean baseline, and aggregate GOSS RMSE to stay within 1.35x of its
matched uniform-subsample control. Timings are reported only. The report may
recommend rate regions for later auto-policy work but cannot change defaults.

## DART Dropout Profile

The DART profile uses the same held-out regression signal and compares standard
boosting with DART configurations spanning:

- fit horizons of 50, 100, and 200 rounds;
- drop rates of `0.05`, `0.1`, and `0.2`;
- `dart_max_drop` values of `5`, `20`, and `50`.

The full matrix uses representative combinations rather than the Cartesian
product: uncapped-pressure arms vary horizon and drop rate, while cap-isolation
arms hold horizon and rate fixed. Quick mode uses shorter horizons and one
seed.

Results include held-out RMSE, completed rounds, fit time, time per round,
standard-mode time ratio, and a deterministic configured dropout-pressure
estimate. Actual historical dropout sets are intentionally not added to the
public model API or artifact solely for benchmarking.

Quality gates require finite predictions, all requested rounds, and DART RMSE
within 1.50x of the matching standard model. No timing ratio is blocking. The
full report determines whether the next DART PR should cap expected drops or
target another measured hotspot.

## CI Integration

The existing `python-smoke` matrix already builds and installs a release wheel.
Only its Ubuntu/Python 3.13 leg will additionally run:

```bash
python -m pytest benchmarks/tests/test_review_guardrails.py -q
python benchmarks/review_guardrails.py --quick --gate
```

This avoids another wheel build and prevents six duplicate benchmark runs.
Required CI checks cover deterministic data contracts, finite results,
completion, and bounded quality. Wall-clock fields must be finite and positive
but are never compared with thresholds.

## Tests

Contract tests are written before implementation and cover:

- deterministic fixtures and valid quantile/GOSS/DART domains;
- weighted empirical quantiles and pinball loss on hand-computed examples;
- smoothed gradient continuity and non-negative Hessians;
- split selection and report rendering on small data;
- gate rejection for non-finite values, incomplete fits, invalid partitions,
  and material quality regressions;
- one small real-model GOSS/DART run through the installed extension;
- CLI section selection and output behavior.

The full repository verification remains `cargo test --workspace`,
`cargo clippy --workspace --all-targets -- -D warnings`, formatting checks, the
complete Python suite, benchmark contract tests, and the full benchmark gate.

## Documentation And Resolution Tracking

The committed report records configuration, per-arm medians, quality
comparisons, and descriptive timings. Benchmark indexes link to it. The
special-modes resolution document marks the GOSS-rate evidence complete,
records the smoothed-pinball decision, and narrows DART's remaining item to the
specific optimization supported by the profile. It must not mark an
optimization fixed merely because profiling is complete.

