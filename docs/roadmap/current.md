# Current Roadmap

## Direction

AlloyGBM is a Rust-first gradient boosting system with Python bindings, aimed at strong practical performance on structured regression workloads, with a particular interest in financial and time-aware tabular problems.

The current priorities are:

1. Keep the CPU training and inference path competitive on real-world tabular workloads.
2. Improve benchmark coverage so comparative claims are easy to defend.
3. Keep the Python API stable and low-friction while moving hot paths into native code.
4. Build toward a clean long-term roadmap for ranking, richer constraints, and future accelerator backends.

## Near-Term Work

- Close the remaining performance gap on broad tabular regression datasets such as California Housing and Bike Sharing.
- Continue tuning dataset-aware training policy without making the public API harder to reason about.
- Improve benchmark reporting so AlloyGBM's strongest and weakest regimes are both easy to see.
- Keep predictor, artifact, and explanation paths compatible as training internals evolve.

## Longer-Term Themes

- Ranking and finance-native evaluation support.
- Better operational diagnostics and model introspection.
- Accelerator roadmap work after the CPU baseline is solid enough to preserve as a reference implementation.

## Planning Style

The project no longer uses the old version-layer planning hierarchy as the active documentation model.

Going forward:

- current intent lives in `docs/roadmap/`
- research notes live in `docs/ideas/`
- benchmark framing lives in `docs/benchmarks/` and `benchmarks/`
- older layered planning notes remain in `docs/architecture/` as archive material
