# Historical benchmark archive

**Draft cutover: do not merge the retirement PR until the full standard Lite acceptance audit passes.** No full Lite acceptance is claimed by the presence of these files.

This preserves the entire bespoke benchmark tree from `ecbafa1fb06130d5fe8e481e87d6aaa9fbb79a09`, plus the prior CI definition and replaced benchmark documentation/test file. [manifest.json](manifest.json) records every original path, archive destination and SHA-256 hash. Relative source layouts are retained; archived scripts are not supported entry points and are not promised to execute from here.

The replacement is [standard TabArena](../../../../benchmarks/tabarena/README.md). Its official tasks/splits and upstream model processing replace the old parameter-matching comparisons, custom quality/speed gates, frontier/development suites, bespoke selection and summaries. Historical reports remain under [docs/benchmarks](../../../benchmarks/README.md) with explicit archive labels. Their numbers were not obtained under TabArena and must not be mixed with new standard scores.

All 155 tracked legacy benchmark files were archived, including benchmark-only tests and fixtures. The old CI benchmark steps and observational comparator job were retired; the active CI still runs Rust and Python library correctness suites, plus the isolated TabArena integration job.

The library test audit found two dependencies of the old harness, in library tests. In `test_morph_helpers.py`, its three calibration-gate tests concern the retired harness. They are preserved in the archived original file. Its eight Morph library helper tests remain in active library tests. Existing library tests retain native artifact/determinism, monotonicity, sampling/DART, multiclass parallelism, ranking, objectives and special-mode correctness; no benchmark performance threshold is presented as a correctness invariant. Ranking/temporal/special-mode comparative performance is a documented gap in the new suite.

`test_dart.py` also dynamically loaded the old calibration selector. Its six constructor/clone/pickle/artifact/determinism cases remain active, asserting the declared cap of 5 directly; they no longer execute a retired benchmark to decide the expected public API value. The unmodified prior test file is included in the hash manifest.
