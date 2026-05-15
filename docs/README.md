# AlloyGBM Docs

This documentation is split into two layers:

- user-facing package documentation in `docs/user/`
- internal project documentation in the rest of `docs/`
- a Read the Docs style Sphinx site source tree in `docs/site/`

## User Docs

Start here if you want to install or use AlloyGBM:

1. `docs/user/installation.md`
2. `docs/user/quickstart.md`
3. `docs/user/gbmregressor.md`
4. `docs/user/gbmclassifier.md`
5. `docs/user/gbmranker.md`
6. `docs/user/validation.md`
7. `docs/user/explanations.md`
8. `docs/user/benchmarks.md`

## Internal Docs

These sections are primarily for project planning, benchmarking methodology,
and repository evolution:

- `roadmap/`
  - Current project direction and medium-term priorities.
- `benchmarks/`
  - Benchmark framing and repo-level evaluation notes.
- `ideas/`
  - Research notes, inspiration, and candidate follow-ups.
- `reference/`
  - Stable pointers into the codebase and developer-facing entry points.
- `archive/`
  - Legacy material from the previous planning/documentation system.
  - Includes `v0.1_plans/` with pre-v0.2.0 limitation analysis and implementation plans.
- `limitations.md`
  - Current limitations, the resolved-since-v0.1 list, and v0.7.2+ follow-ups.

If you are trying to understand the project as a maintainer, start here:

1. `docs/roadmap/current.md`
2. `benchmarks/README.md`
3. `docs/reference/README.md`

If you are working on the hosted documentation site itself, start here:

1. `.readthedocs.yaml`
2. `docs/site/source/index.rst`
3. `docs/site/source/conf.py`
