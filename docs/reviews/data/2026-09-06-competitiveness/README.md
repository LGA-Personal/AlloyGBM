# Evidence for the 2026-09-06 competitiveness review

These are diagnostic experiments using the existing benchmark generator,
factories and policy-arm helper. They are **not** performance-gate runs.
No implementation was changed to collect them.

| File | Contents |
|---|---|
| [provenance.json](provenance.json) | Reviewed commit, release-build command, wheel/extension hashes and runtime versions |
| [accuracy_probe.jsonl](accuracy_probe.jsonl) | Six sequential fits: original auto baseline, gain-floor removal, manual defaults, linear 256, quantile 1024, baseline repeat |
| [confirm_accuracy.jsonl](confirm_accuracy.jsonl) | Five model configurations × three seeds, including actual CatBoost resolved parameters and tail/central RMSE |
| [quantile_budget_probe.jsonl](quantile_budget_probe.jsonl) | Current externally generated bin grid, corrected grid at 1 thread, corrected grid at 10 threads; artifact/prediction hashes |
| [accuracy_probe.py](accuracy_probe.py) | Exact executed first driver |
| [confirm_accuracy.py](confirm_accuracy.py) | Exact executed three-seed driver |
| [quantile_budget_probe.py](quantile_budget_probe.py) | Exact executed missing-bin budget driver |

The Python drivers are retained exactly as executed, including their temporary
paths and shared import preamble. The commands below relocate those paths into
a fresh directory for a repeat run. Run from the repository root, at the commit
in `provenance.json`, with the listed peer versions installed. `maturin` and the
project `.venv` are prerequisites. No existing evidence is overwritten.

```bash
REVIEW_RUN_DIR=$(mktemp -d /tmp/alloygbm-competitiveness-reproduce.XXXXXX)
export REVIEW_RUN_DIR
maturin build --release --interpreter .venv/bin/python --out "$REVIEW_RUN_DIR/wheels"
.venv/bin/python -m pip install --no-deps --target "$REVIEW_RUN_DIR/runtime" "$REVIEW_RUN_DIR"/wheels/*.whl
.venv/bin/python - <<'PY'
import os
from pathlib import Path

source = Path('docs/reviews/data/2026-09-06-competitiveness')
destination = Path(os.environ['REVIEW_RUN_DIR'])
for name in ('accuracy_probe.py', 'confirm_accuracy.py', 'quantile_budget_probe.py'):
    text = (source / name).read_text()
    text = text.replace('/tmp/alloygbm-review-20260906/', str(destination) + '/')
    (destination / name).write_text(text)
PY
PYTHONPATH="$REVIEW_RUN_DIR/runtime:$PWD" .venv/bin/python "$REVIEW_RUN_DIR/accuracy_probe.py"
PYTHONPATH="$REVIEW_RUN_DIR/runtime:$PWD" .venv/bin/python "$REVIEW_RUN_DIR/confirm_accuracy.py"
PYTHONPATH="$REVIEW_RUN_DIR/runtime:$PWD" .venv/bin/python "$REVIEW_RUN_DIR/quantile_budget_probe.py"
```

The original run also supplied the runtime directory first in `PYTHONPATH`.
All drivers pin OpenMP/BLAS/Rayon environment variables to one before imports,
clear `ALLOYGBM_*` experiment overrides, and use a fit-local `n_jobs` setting.
The corrected-grid ten-thread fit leaves unrelated BLAS threads at one; it is
a determinism probe, not a cross-library speed comparison.

The baseline generator seed is `20260902`; confirmation adds `20260903` and
`20260904`. All use 500,000 rows × 40 features and the same 80/20 split.
The extra seeds confirm the same distribution; no new real-data suite was run.
Changing binning was selected after inspecting the first seed, and competitors
were not given equivalent tuning. Treat the comparison as an explanation of
the existing benchmark result, not a tuned competitive ranking.

`quantile_budget_probe.py` externally constructs bins and then uses AlloyGBM's
existing pre-binned path. Its fit times exclude that preprocessing. The
current-grid control exactly matches the raw-feature baseline's predictions
and tree artifact hash, establishing that the grid change is isolated for
this fixture. Native implementation, full mode coverage and compatibility
tests remain follow-up work.
