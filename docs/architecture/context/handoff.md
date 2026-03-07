# Handoff

## Current Layer and Status
- Parent milestone: `docs/architecture/v1.0/v0.9` (in progress).
- Active target: `docs/architecture/v1.0/v0.9/v0.9.7`.
- Current state: focused competitiveness tuning is paused after a long candidate wave; best accepted preset is `candidate46`.

## Completed This Session
- Committed accepted focused-slice preset history:
  - `dbf87e8` `v0.9.7 (part): keep split leaf-magnitude filter`
  - `63e2ffa` `v0.9.7 (part): keep tail-rank leaf-filter preset`
  - `d8185ba` `v0.9.7 (part): keep reg leaf-filter preset`
  - `6638b55` `v0.9.7 (part): keep full quality preset`
  - `f82844a` `v0.9.7 (part): record rejected tuning follow-ups`
- Accepted preset to treat as current focused baseline:
  - `candidate46` = `ALLOYGBM_EXPERIMENT_LINEAR_TAIL_RANK=1`
  - `candidate46` = `ALLOYGBM_EXPERIMENT_SPLIT_L2=1`
  - `candidate46` = `ALLOYGBM_EXPERIMENT_SPLIT_L1=0.1`
  - `candidate46` = `ALLOYGBM_EXPERIMENT_MIN_CHILD_HESS=0`
  - `candidate46` = `ALLOYGBM_EXPERIMENT_SPLIT_MIN_LEAF_MAGNITUDE=0.02`
- Rejected follow-up candidates and logged evidence for:
  - `candidate47` lower leaf-magnitude threshold
  - `candidate48` depth-gated leaf-magnitude filter
  - `candidate49` parent-relative split gain floor
  - `candidate50` stronger tail-asymmetry activation gate
- Added optional CatBoost support to `benchmarks/run_model_comparison.py`.
  - Harness now auto-detects CatBoost and benchmarks it when installed.
  - When missing, runs continue and record `catboost_runtime` as unavailable.
- Added benchmark harness regression test:
  - `benchmarks/tests/test_run_model_comparison.py`

## Validation Evidence
- `cargo test -p alloygbm-engine -p alloygbm-backend-cpu` -> PASS
- `PYTHONPATH=bindings/python python3 -m unittest discover -s bindings/python/tests -p 'test_*.py'` -> PASS (`84` tests after candidate50 revert)
- `python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'` -> PASS (`9` tests)
- `PYTHONPATH=bindings/python python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile smoke:0.2:4:20 --profile-seeds 7 --scenarios dense_numeric --output-dir benchmarks/results/catboost_harness_smoke` -> PASS
  - runtime note: `catboost runtime: unavailable (ModuleNotFoundError: No module named 'catboost')`

## Competitiveness Snapshot
- Broad historical matrix still favors peers:
  - full `default`: `xgboost` best RMSE in `11/12` cells, `lightgbm` in `1/12`, `alloygbm` in `0/12`
  - fastest fit: `lightgbm` in `12/12`
  - fastest predict: `xgboost` in `12/12`
- Best current focused preset is `candidate46` on `panel_time_series` + `dow_jones_financial` across:
  - `shallow_high_lr:0.2:4:200`
  - `mid_balanced:0.1:6:400`
  - `deep_low_lr:0.01:8:5000`
  - seeds `7,17,29`
- `candidate46` vs prior focused baseline:
  - fit `-32.36%`
  - predict `-20.23%`
  - RMSE `-4.32%`
  - MAE `-4.89%`
  - R2 `+0.12042`
- Focused peer picture is still behind:
  - vs `lightgbm`: quality is closer, but Alloy fit/predict remain far slower
  - vs `xgboost`: Alloy still trails on both quality and speed

## Exact Unfinished Tasks
1. Decide whether to pause `v0.9.7` or do one more competitiveness pass with a genuinely different representation-side idea.
2. Install CatBoost locally and run a real cross-library matrix to replace the current LightGBM/XGBoost-only picture.
3. If resuming tuning, avoid more split-threshold and tail-activation retunes; they are converging to dominated tradeoffs.
4. Write `docs/architecture/v1.0/v0.9/v0.9.7/implementation_notes.md`.
5. Write `docs/architecture/v1.0/v0.9/v0.9.7/verification_report.md`.

## Strongest Next Directions
1. Add CatBoost to the actual benchmark comparison environment:
   - install package first, then rerun focused and broad matrices
2. Representation-side candidate, not another threshold tweak:
   - hybrid tail mapping for `candidate36`-flagged features where dense core stays linear and only extremes use rank
3. If broad competitiveness is the goal, rerun full `default` and `default_ultra` on the accepted stack before more micro-tuning to confirm where the remaining gap really is.

## Known Risks and Gotchas
- Current worktree is intentionally dirty outside this task:
  - `docs/architecture/context/session_brief.md`
  - `docs/architecture/v1.0/v0.9/v0.9.7/plan.md`
  - `tmp/`
- Do not revert user edits in those paths.
- Recent benchmark artifacts use two runtime styles:
  - older focused accepted-package runs used `/tmp/.../site-packages`
  - latest tuning smoke/rejection runs used `PYTHONPATH=bindings/python`
  - keep comparisons within the same runtime snapshot
- CatBoost is wired into the harness but not installed in this environment yet.

## Exact Next Command
`cd /Users/lashby/Projects/AlloyGBM && python3 -m pip install catboost && PYTHONPATH=bindings/python python3 -B benchmarks/run_model_comparison.py --profile-grid none --profile shallow_high_lr:0.2:4:200 --profile mid_balanced:0.1:6:400 --profile deep_low_lr:0.01:8:5000 --profile-seeds 7,17,29 --scenarios panel_time_series dow_jones_financial --output-dir benchmarks/results/v097_catboost_focus`
