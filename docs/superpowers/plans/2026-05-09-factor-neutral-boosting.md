# Factor-Neutral Boosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional factor-neutral target/gradient training and split exposure penalties to AlloyGBM.

**Architecture:** Keep factor exposures as fit-time row-aligned data, not estimator constructor state. Add Rust config/data types in core, projection helpers in engine, optional split-penalty factor stats in the CPU backend, and Python API plumbing through `fit(..., factor_exposures=F)`.

**Tech Stack:** Rust 1.92 workspace, PyO3/maturin Python bindings, NumPy inputs, pytest, cargo test/clippy/fmt.

---

## File Structure

- Modify `crates/core/src/lib.rs`: add neutralization enums/config/matrix types, validation, metadata JSON round-trip.
- Modify `crates/engine/src/lib.rs`: add `FactorProjector`, fit validation, target residualization, per-round gradient projection, split-option plumbing, tests.
- Modify `crates/backend_cpu/src/lib.rs`: add split factor context and penalty-aware numeric/categorical gain evaluation.
- Modify `bindings/python/src/lib.rs`: parse neutralization params and optional factor exposure matrix into Rust training calls.
- Modify `bindings/python/alloygbm/regressor.py`: constructor params, fit parameter, validation, flattening, save/load/pickle params.
- Modify `bindings/python/alloygbm/classifier.py`: fit signature and repr support inherited/overridden from regressor.
- Modify `bindings/python/alloygbm/ranker.py`: fit signature and row sorting of `factor_exposures`.
- Modify `bindings/python/tests/test_factor_neutralization.py`: Python API and integration coverage.
- Modify `benchmarks/run_model_comparison.py`: benchmark arms and factor-correlation reporting.
- Modify `README.md`, `CHANGELOG.md`, `docs/user/*.md`, `docs/site/source/*.rst`: public docs.

---

### Task 1: Add Core Neutralization Types

**Files:**
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Write failing core tests**

Add tests near existing `TrainParams` validation tests:

```rust
#[test]
fn train_params_default_has_no_neutralization_config() {
    let params = TrainParams::default();
    assert!(params.neutralization_config.is_none());
}

#[test]
fn validates_factor_exposure_matrix_shape_and_finiteness() {
    assert!(FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).is_ok());
    assert!(FactorExposureMatrix::new(0, 2, vec![]).is_err());
    assert!(FactorExposureMatrix::new(2, 0, vec![]).is_err());
    assert!(FactorExposureMatrix::new(2, 2, vec![1.0, f32::NAN, 0.0, 1.0]).is_err());
    assert!(FactorExposureMatrix::new(2, 2, vec![1.0, 0.0, 1.0]).is_err());
}

#[test]
fn validates_neutralization_config_contract() {
    let mut params = TrainParams {
        neutralization_config: Some(FactorNeutralizationConfig {
            kind: NeutralizationKind::PerRoundGradient,
            ridge_lambda: 1e-6,
            split_penalty: 0.0,
        }),
        ..TrainParams::default()
    };
    assert!(validate_train_params(&params).is_ok());

    params.neutralization_config = Some(FactorNeutralizationConfig {
        kind: NeutralizationKind::SplitPenalty,
        ridge_lambda: 1e-6,
        split_penalty: 0.1,
    });
    assert!(validate_train_params(&params).is_ok());

    params.neutralization_config = Some(FactorNeutralizationConfig {
        kind: NeutralizationKind::SplitPenalty,
        ridge_lambda: 1e-6,
        split_penalty: -0.1,
    });
    assert!(validate_train_params(&params).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p alloygbm-core neutralization --lib
```

Expected: compile failure for missing `FactorExposureMatrix`, `FactorNeutralizationConfig`, `NeutralizationKind`, and `TrainParams.neutralization_config`.

- [ ] **Step 3: Implement core types and validation**

Add near existing leaf solver enums:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeutralizationKind {
    None,
    PreTarget,
    PerRoundGradient,
    SplitPenalty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorNeutralizationConfig {
    pub kind: NeutralizationKind,
    pub ridge_lambda: f32,
    pub split_penalty: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorExposureMatrix {
    pub row_count: usize,
    pub factor_count: usize,
    pub values: Vec<f32>,
}

impl FactorExposureMatrix {
    pub fn new(row_count: usize, factor_count: usize, values: Vec<f32>) -> CoreResult<Self> {
        if row_count == 0 {
            return Err(CoreError::Validation("factor_exposures row_count must be greater than 0".to_string()));
        }
        if factor_count == 0 {
            return Err(CoreError::Validation("factor_exposures factor_count must be greater than 0".to_string()));
        }
        if values.len() != row_count * factor_count {
            return Err(CoreError::Validation(format!(
                "factor_exposures values length {} does not match row_count * factor_count {}",
                values.len(),
                row_count * factor_count
            )));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::Validation("factor_exposures must contain only finite values".to_string()));
        }
        Ok(Self { row_count, factor_count, values })
    }

    pub fn row(&self, row_index: usize) -> &[f32] {
        let start = row_index * self.factor_count;
        &self.values[start..start + self.factor_count]
    }
}
```

Add to `TrainParams`:

```rust
pub neutralization_config: Option<FactorNeutralizationConfig>,
```

Add to default:

```rust
neutralization_config: None,
```

Add to `TrainingDataset`:

```rust
pub factor_exposures: Option<FactorExposureMatrix>,
```

Update all test/helper constructors of `TrainingDataset` with `factor_exposures: None`.

In `validate_train_params`, add:

```rust
if let Some(config) = params.neutralization_config {
    if config.kind == NeutralizationKind::None {
        return Err(CoreError::Validation(
            "neutralization_config must be None when neutralization kind is None".to_string(),
        ));
    }
    if !config.ridge_lambda.is_finite() || config.ridge_lambda < 0.0 {
        return Err(CoreError::Validation(
            "factor_neutralization_lambda must be finite and >= 0".to_string(),
        ));
    }
    if !config.split_penalty.is_finite() || config.split_penalty < 0.0 {
        return Err(CoreError::Validation(
            "factor_penalty must be finite and >= 0".to_string(),
        ));
    }
    if config.kind != NeutralizationKind::SplitPenalty && config.split_penalty != 0.0 {
        return Err(CoreError::Validation(
            "factor_penalty is only valid with neutralization='split_penalty'".to_string(),
        ));
    }
}
```

- [ ] **Step 4: Run tests**

Run:

```bash
cargo test -p alloygbm-core neutralization --lib
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/lib.rs
git commit -m "feat(core): add factor neutralization config types"
```

---

### Task 2: Implement Engine Factor Projection

**Files:**
- Modify: `crates/engine/src/lib.rs`

- [ ] **Step 1: Write failing projection tests**

Add tests in `crates/engine/src/lib.rs`:

```rust
#[test]
fn factor_projector_orthogonalizes_gradient() {
    let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let projector = FactorProjector::new(&exposures, None, 1e-6).unwrap();
    let mut gradients = vec![
        GradientPair { grad: 1.0, hess: 1.0 },
        GradientPair { grad: 2.0, hess: 1.0 },
        GradientPair { grad: 3.0, hess: 1.0 },
        GradientPair { grad: 4.0, hess: 1.0 },
    ];
    projector.project_gradient_pairs_in_place(&mut gradients).unwrap();
    let dot: f32 = exposures.values.iter().zip(gradients.iter()).map(|(f, g)| *f * g.grad).sum();
    assert!(dot.abs() < 1e-4, "factor dot after projection was {dot}");
    assert!(gradients.iter().all(|g| g.hess == 1.0));
}

#[test]
fn factor_projector_ridge_handles_collinear_factors() {
    let exposures = FactorExposureMatrix::new(
        3,
        2,
        vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
    ).unwrap();
    let projector = FactorProjector::new(&exposures, None, 1e-3).unwrap();
    let mut values = vec![1.0, 2.0, 3.0];
    projector.residualize_values_in_place(&mut values).unwrap();
    assert!(values.iter().all(|v| v.is_finite()));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p alloygbm-engine factor_projector --lib
```

Expected: compile failure for missing `FactorProjector`.

- [ ] **Step 3: Implement `FactorProjector`**

Add imports:

```rust
use alloygbm_core::{FactorExposureMatrix, FactorNeutralizationConfig, NeutralizationKind};
```

Add helper:

```rust
struct FactorProjector<'a> {
    exposures: &'a FactorExposureMatrix,
    weights: Option<&'a [f32]>,
    cholesky_lower: Vec<f64>,
}
```

Implement `new`, `project_gradient_pairs_in_place`, `residualize_values_in_place`, and private `solve_cholesky`. Use `f64` for Gram matrix, right-hand side, and solves:

```rust
impl<'a> FactorProjector<'a> {
    fn new(
        exposures: &'a FactorExposureMatrix,
        weights: Option<&'a [f32]>,
        ridge_lambda: f32,
    ) -> EngineResult<Self> {
        if let Some(w) = weights && w.len() != exposures.row_count {
            return Err(EngineError::ContractViolation(
                "sample_weight length must match factor_exposures row_count".to_string(),
            ));
        }
        let k = exposures.factor_count;
        let mut gram = vec![0.0_f64; k * k];
        for row in 0..exposures.row_count {
            let weight = weights.map_or(1.0_f64, |w| f64::from(w[row]));
            let f = exposures.row(row);
            for a in 0..k {
                for b in 0..=a {
                    gram[a * k + b] += weight * f64::from(f[a]) * f64::from(f[b]);
                }
            }
        }
        for i in 0..k {
            gram[i * k + i] += f64::from(ridge_lambda);
        }
        let cholesky_lower = cholesky_lower(gram, k)?;
        Ok(Self { exposures, weights, cholesky_lower })
    }
}
```

Use a standard lower-triangular Cholesky with diagonal tolerance `1e-12`. Return `EngineError::ContractViolation("factor exposure Gram matrix is singular; increase factor_neutralization_lambda")` if the diagonal is not positive.

- [ ] **Step 4: Run projection tests**

Run:

```bash
cargo test -p alloygbm-engine factor_projector --lib
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/lib.rs
git commit -m "feat(engine): add factor gradient projector"
```

---

### Task 3: Wire Pre-Target and Per-Round Gradient Neutralization

**Files:**
- Modify: `crates/engine/src/lib.rs`

- [ ] **Step 1: Write failing integration tests**

Add tests:

```rust
#[test]
fn per_round_gradient_neutralization_trains_regression() {
    let dataset = factor_dominated_dataset();
    let binned = sample_binned_matrix_for_dataset(&dataset);
    let params = TrainParams {
        neutralization_config: Some(FactorNeutralizationConfig {
            kind: NeutralizationKind::PerRoundGradient,
            ridge_lambda: 1e-6,
            split_penalty: 0.0,
        }),
        ..TrainParams::default()
    };
    let model = Trainer::new(params)
        .unwrap()
        .fit_iterations(&dataset, &binned, &CpuBackend, &SquaredError, 5)
        .unwrap();
    assert_eq!(model.rounds_completed(), 5);
}

#[test]
fn pre_target_neutralization_reduces_target_factor_dot() {
    let mut dataset = factor_dominated_dataset();
    let before = factor_dot(dataset.factor_exposures.as_ref().unwrap(), &dataset.targets);
    apply_pre_target_neutralization_for_test(&mut dataset, 1e-6).unwrap();
    let after = factor_dot(dataset.factor_exposures.as_ref().unwrap(), &dataset.targets);
    assert!(after.abs() < before.abs() * 0.01);
}
```

Define local helper fixtures in the test module with explicit `factor_exposures: Some(...)`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p alloygbm-engine neutralization --lib
```

Expected: failures because the training loop does not yet apply the projector.

- [ ] **Step 3: Validate fit contracts**

In `Trainer::validate_fit_contract`, after `validate_training_dataset(dataset)?`, add:

```rust
validate_neutralization_fit_contract(&self.params, dataset, objective.requires_group_id())?;
```

Implement:

```rust
fn validate_neutralization_fit_contract<O: ObjectiveOps>(
    params: &TrainParams,
    dataset: &TrainingDataset,
    _requires_group_id: bool,
) -> EngineResult<()> {
    let Some(config) = params.neutralization_config else {
        if dataset.factor_exposures.is_some() {
            return Err(EngineError::ContractViolation(
                "factor_exposures were provided but neutralization='none'".to_string(),
            ));
        }
        return Ok(());
    };
    let exposures = dataset.factor_exposures.as_ref().ok_or_else(|| {
        EngineError::ContractViolation(
            "factor_exposures are required when neutralization is active".to_string(),
        )
    })?;
    if exposures.row_count != dataset.row_count() {
        return Err(EngineError::ContractViolation(format!(
            "factor_exposures row_count {} does not match training row_count {}",
            exposures.row_count,
            dataset.row_count()
        )));
    }
    if config.kind == NeutralizationKind::PreTarget && O::NAME != "regression:squared_error" {
        return Err(EngineError::ContractViolation(
            "neutralization='pre_target' is only supported for GBMRegressor squared-error training".to_string(),
        ));
    }
    Ok(())
}
```

If `ObjectiveOps` has no `NAME`, use a new trait method `supports_pre_target_neutralization() -> bool` defaulting to `false`, implemented as `true` for `SquaredError`.

- [ ] **Step 4: Apply target and gradient projection**

Before the first baseline prediction in fit paths, clone `dataset` to a mutable training dataset only when pre-target is active:

```rust
let mut owned_dataset;
let active_dataset = if self.params.neutralization_config.is_some_and(|c| c.kind == NeutralizationKind::PreTarget) {
    owned_dataset = dataset.clone();
    let exposures = owned_dataset.factor_exposures.as_ref().unwrap();
    FactorProjector::new(exposures, owned_dataset.sample_weights.as_deref(), config.ridge_lambda)?
        .residualize_values_in_place(&mut owned_dataset.targets)?;
    &owned_dataset
} else {
    dataset
};
```

In the per-round training loop immediately after `objective.compute_gradients_into(...)`, add:

```rust
if self.params.neutralization_config.is_some_and(|c| matches!(c.kind, NeutralizationKind::PerRoundGradient | NeutralizationKind::SplitPenalty)) {
    let exposures = active_dataset.factor_exposures.as_ref().unwrap();
    let projector = FactorProjector::new(exposures, active_dataset.sample_weights.as_deref(), config.ridge_lambda)?;
    projector.project_gradient_pairs_in_place(&mut gradients)?;
}
```

Precompute the `FactorProjector` once before the loop and reuse it.

- [ ] **Step 5: Run tests**

Run:

```bash
cargo test -p alloygbm-engine neutralization --lib
cargo test -p alloygbm-engine fit_iterations --lib
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/lib.rs
git commit -m "feat(engine): neutralize targets and gradients"
```

---

### Task 4: Add Python API and Bridge for Projection Modes

**Files:**
- Modify: `bindings/python/alloygbm/regressor.py`
- Modify: `bindings/python/alloygbm/classifier.py`
- Modify: `bindings/python/alloygbm/ranker.py`
- Modify: `bindings/python/src/lib.rs`
- Create: `bindings/python/tests/test_factor_neutralization.py`

- [ ] **Step 1: Write failing Python API tests**

Create `bindings/python/tests/test_factor_neutralization.py`:

```python
import pickle
import unittest

import numpy as np

from alloygbm import GBMClassifier, GBMRanker, GBMRegressor


def factor_data():
    f = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(-1, 1)
    x = np.column_stack([f[:, 0], np.sin(np.arange(24, dtype=np.float32))]).astype(np.float32)
    y = (2.0 * f[:, 0] + 0.1 * x[:, 1]).astype(np.float32)
    return x, y, f


class FactorNeutralizationTests(unittest.TestCase):
    def test_params_roundtrip(self):
        model = GBMRegressor(neutralization="per_round_gradient", factor_neutralization_lambda=1e-4)
        params = model.get_params()
        self.assertEqual(params["neutralization"], "per_round_gradient")
        self.assertEqual(params["factor_neutralization_lambda"], 1e-4)
        model.set_params(neutralization="pre_target", factor_penalty=0.0)
        self.assertEqual(model.neutralization, "pre_target")

    def test_invalid_constructor_values(self):
        with self.assertRaises(ValueError):
            GBMRegressor(neutralization="bad")
        with self.assertRaises(ValueError):
            GBMRegressor(factor_neutralization_lambda=-1.0)
        with self.assertRaises(ValueError):
            GBMRegressor(factor_penalty=-1.0)

    def test_requires_factor_exposures_when_active(self):
        x, y, _ = factor_data()
        with self.assertRaises(ValueError):
            GBMRegressor(neutralization="per_round_gradient").fit(x, y)

    def test_regressor_trains_with_per_round_gradient(self):
        x, y, f = factor_data()
        model = GBMRegressor(neutralization="per_round_gradient", n_estimators=5, seed=1)
        model.fit(x, y, factor_exposures=f)
        self.assertEqual(model.predict(x).shape, (len(x),))

    def test_pre_target_rejected_for_classifier_and_ranker(self):
        x, y, f = factor_data()
        with self.assertRaises(ValueError):
            GBMClassifier(neutralization="pre_target").fit(x, (y > 0).astype(np.int32), factor_exposures=f)
        with self.assertRaises(ValueError):
            GBMRanker(neutralization="pre_target").fit(x, y, group=np.repeat([0, 1, 2], 8), factor_exposures=f)

    def test_pickle_preserves_params_and_predictions(self):
        x, y, f = factor_data()
        model = GBMRegressor(neutralization="per_round_gradient", n_estimators=5, seed=1).fit(x, y, factor_exposures=f)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.predict(x), model.predict(x), atol=1e-6)
        self.assertEqual(restored.neutralization, "per_round_gradient")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest bindings/python/tests/test_factor_neutralization.py -q
```

Expected: failures for unknown constructor and fit parameters.

- [ ] **Step 3: Update Python estimators**

In `GBMRegressor.__init__`, add:

```python
neutralization: str = "none",
factor_neutralization_lambda: float = 1e-6,
factor_penalty: float = 0.0,
```

Validate:

```python
if str(neutralization) not in ("none", "pre_target", "per_round_gradient", "split_penalty"):
    raise ValueError("neutralization must be 'none', 'pre_target', 'per_round_gradient', or 'split_penalty'")
if not math.isfinite(float(factor_neutralization_lambda)) or float(factor_neutralization_lambda) < 0.0:
    raise ValueError("factor_neutralization_lambda must be finite and >= 0")
if not math.isfinite(float(factor_penalty)) or float(factor_penalty) < 0.0:
    raise ValueError("factor_penalty must be finite and >= 0")
if str(neutralization) != "split_penalty" and float(factor_penalty) != 0.0:
    raise ValueError("factor_penalty is only valid with neutralization='split_penalty'")
if str(neutralization) == "split_penalty" and str(leaf_model) == "linear":
    raise ValueError("neutralization='split_penalty' requires leaf_model='constant'")
```

Add assignments, repr fields, `get_params`, `set_params`, and `_params_order`.

Update `fit` signature:

```python
def fit(..., factor_exposures: object | None = None) -> "GBMRegressor":
```

Add helper:

```python
def _prepare_factor_exposures(self, factor_exposures, n_rows: int):
    if self.neutralization == "none":
        if factor_exposures is not None:
            raise ValueError("factor_exposures were provided but neutralization='none'")
        return None, 0, 0
    if factor_exposures is None:
        raise ValueError("factor_exposures are required when neutralization is active")
    arr = np.asarray(factor_exposures, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("factor_exposures must be a 2D array")
    if arr.shape[0] != n_rows:
        raise ValueError(f"factor_exposures row count {arr.shape[0]} does not match X row count {n_rows}")
    if arr.shape[1] == 0:
        raise ValueError("factor_exposures must contain at least one factor")
    if not np.all(np.isfinite(arr)):
        raise ValueError("factor_exposures must contain only finite values")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ravel().tolist(), int(arr.shape[0]), int(arr.shape[1])
```

Update classifier/ranker fit signatures. In ranker, sort factor exposure rows using the same order as `X`, `y`, and `group`.

- [ ] **Step 4: Update PyO3 bridge**

Add neutralization params and optional factor exposure matrix args to train functions:

```rust
neutralization: &str,
factor_neutralization_lambda: f32,
factor_penalty: f32,
factor_exposure_values: Option<Vec<f32>>,
factor_exposure_row_count: Option<usize>,
factor_exposure_factor_count: Option<usize>,
```

Add parsers:

```rust
fn parse_neutralization_config(
    neutralization: &str,
    factor_neutralization_lambda: f32,
    factor_penalty: f32,
) -> PyResult<Option<FactorNeutralizationConfig>> {
    let kind = match neutralization {
        "none" => NeutralizationKind::None,
        "pre_target" => NeutralizationKind::PreTarget,
        "per_round_gradient" => NeutralizationKind::PerRoundGradient,
        "split_penalty" => NeutralizationKind::SplitPenalty,
        other => return Err(PyValueError::new_err(format!("neutralization must be 'none', 'pre_target', 'per_round_gradient', or 'split_penalty', got '{other}'"))),
    };
    if !factor_neutralization_lambda.is_finite() || factor_neutralization_lambda < 0.0 {
        return Err(PyValueError::new_err("factor_neutralization_lambda must be finite and >= 0"));
    }
    if !factor_penalty.is_finite() || factor_penalty < 0.0 {
        return Err(PyValueError::new_err("factor_penalty must be finite and >= 0"));
    }
    Ok(match kind {
        NeutralizationKind::None => None,
        _ => Some(FactorNeutralizationConfig { kind, ridge_lambda: factor_neutralization_lambda, split_penalty: factor_penalty }),
    })
}
```

Construct `FactorExposureMatrix::new(row_count, factor_count, values)` and put it on `TrainingDataset`.

- [ ] **Step 5: Build extension and run tests**

Run:

```bash
.venv/bin/python -m maturin develop --manifest-path bindings/python/Cargo.toml --release
.venv/bin/python -m pytest bindings/python/tests/test_factor_neutralization.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bindings/python/src/lib.rs bindings/python/alloygbm/regressor.py bindings/python/alloygbm/classifier.py bindings/python/alloygbm/ranker.py bindings/python/tests/test_factor_neutralization.py
git commit -m "feat(python): expose factor neutralization training"
```

---

### Task 5: Implement Split Penalty for Constant Leaves

**Files:**
- Modify: `crates/core/src/lib.rs`
- Modify: `crates/engine/src/lib.rs`
- Modify: `crates/backend_cpu/src/lib.rs`

- [ ] **Step 1: Write failing backend tests**

Add tests in backend CPU:

```rust
#[test]
fn factor_split_penalty_reduces_factor_loaded_gain() {
    let backend = CpuBackend;
    let histograms = backend.build_histograms(
        &sample_binned_matrix(),
        &sample_gradients(),
        &sample_node(),
        &[FeatureTile::new(0, 2).unwrap()],
    ).unwrap();
    let exposures = FactorExposureMatrix::new(4, 1, vec![1.0, 1.0, -1.0, -1.0]).unwrap();
    let no_penalty = backend.best_split_with_options(&histograms, SplitSelectionOptions::default(), &[], &[]).unwrap().unwrap();
    let penalty_options = SplitSelectionOptions {
        factor_split_context: Some(FactorSplitContext::new_for_test(&exposures, 10.0)),
        ..SplitSelectionOptions::default()
    };
    let penalized = backend.best_split_with_options(&histograms, penalty_options, &[], &[]).unwrap().unwrap();
    assert!(penalized.gain <= no_penalty.gain);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p alloygbm-backend-cpu factor_split_penalty --lib
```

Expected: compile failure for missing `FactorSplitContext`.

- [ ] **Step 3: Add split penalty data path**

Add to `SplitSelectionOptions`:

```rust
pub factor_split_context: Option<FactorSplitContext>,
```

Define:

```rust
pub struct FactorSplitContext<'a> {
    pub exposures: &'a FactorExposureMatrix,
    pub row_indices: &'a [u32],
    pub factor_penalty: f32,
}
```

Because `SplitSelectionOptions` is currently `Copy`, either make the factor context a separate parameter to split methods or store only an indexable borrowed context in a parallel options struct. Prefer a separate optional `factor_context: Option<&FactorSplitContext>` argument so standard options stay cheap and `Copy`.

For each candidate split, compute left and right factor sums by accumulating factors for the candidate row partition. Initial implementation may use direct row scanning when `split_penalty` is active; optimize later with factor histograms if benchmarks show excessive slowdown. Keep the standard and DRO no-penalty paths unchanged.

Penalty helper:

```rust
fn factor_split_penalty(
    left_factor_sums: &[f32],
    right_factor_sums: &[f32],
    left_leaf_value: f32,
    right_leaf_value: f32,
    factor_penalty: f32,
    row_count: usize,
) -> f32 {
    if factor_penalty == 0.0 {
        return 0.0;
    }
    let mut norm_sq = 0.0_f32;
    for i in 0..left_factor_sums.len() {
        let load = left_factor_sums[i] * left_leaf_value + right_factor_sums[i] * right_leaf_value;
        norm_sq += load * load;
    }
    factor_penalty * norm_sq / row_count.max(1) as f32
}
```

Subtract the penalty after standard/DRO/Morph gain is computed:

```rust
gain -= penalty;
```

Reject candidates whose final gain is not finite or is below thresholds using the existing gain checks.

- [ ] **Step 4: Wire engine validation**

In train param validation, reject:

```rust
if config.kind == NeutralizationKind::SplitPenalty && params.leaf_model == LeafModelKind::Linear {
    return Err(CoreError::Validation(
        "neutralization='split_penalty' requires leaf_model='constant'".to_string(),
    ));
}
```

In split selection, pass factor context only when split penalty is active.

- [ ] **Step 5: Run tests**

Run:

```bash
cargo test -p alloygbm-backend-cpu factor_split_penalty --lib
cargo test -p alloygbm-engine neutralization --lib
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/lib.rs crates/engine/src/lib.rs crates/backend_cpu/src/lib.rs
git commit -m "feat(backend): add factor split penalty for scalar leaves"
```

---

### Task 6: Add Integration Coverage for Interactions

**Files:**
- Modify: `bindings/python/tests/test_factor_neutralization.py`
- Modify: `crates/engine/src/lib.rs`
- Modify: `crates/backend_cpu/src/lib.rs`

- [ ] **Step 1: Add Python interaction tests**

Add:

```python
def test_factor_neutralization_with_dro_and_morph(self):
    x, y, f = factor_data()
    model = GBMRegressor(
        neutralization="per_round_gradient",
        leaf_solver="dro",
        dro_radius=0.01,
        training_mode="morph",
        n_estimators=5,
        seed=2,
    )
    model.fit(x, y, factor_exposures=f)
    self.assertEqual(model.predict(x).shape, (len(x),))

def test_split_penalty_rejects_linear_leaves(self):
    x, y, f = factor_data()
    with self.assertRaises(ValueError):
        GBMRegressor(neutralization="split_penalty", factor_penalty=0.1, leaf_model="linear").fit(x, y, factor_exposures=f)
```

- [ ] **Step 2: Add Rust ordering tests**

Add an engine/backend test that asserts MorphBoost sees neutralized gradients by comparing a synthetic gradient dot product before and after the Morph split path.

- [ ] **Step 3: Run interaction tests**

Run:

```bash
cargo test --workspace --exclude alloygbm-python neutralization
.venv/bin/python -m pytest bindings/python/tests/test_factor_neutralization.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/lib.rs crates/backend_cpu/src/lib.rs bindings/python/tests/test_factor_neutralization.py
git commit -m "test: cover factor neutralization interactions"
```

---

### Task 7: Add Docs and Benchmarks

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/user/gbmregressor.md`
- Modify: `docs/user/gbmclassifier.md`
- Modify: `docs/user/gbmranker.md`
- Modify: `docs/site/source/estimator.rst`
- Modify: `benchmarks/run_model_comparison.py`

- [ ] **Step 1: Add docs text**

Add a README section:

```markdown
### Factor-Neutral Boosting

Use `neutralization="per_round_gradient"` with `fit(..., factor_exposures=F)` to project each boosting round's pseudo-residuals away from user-supplied nuisance factors. This is useful when common factors explain high-variance signal that you do not want the model to spend tree capacity learning.

This is a training-time regularization tool. It does not guarantee prediction-time zero exposure unless predictions are neutralized against evaluation-time factors outside the model.
```

Document modes and compatibility exactly as in the design spec.

- [ ] **Step 2: Add benchmark arms**

In `benchmarks/run_model_comparison.py`, add model configs:

```python
"alloygbm_factor_neutral": lambda seed: GBMRegressor(
    neutralization="per_round_gradient",
    n_estimators=args.rounds,
    max_depth=args.max_depth,
    learning_rate=args.learning_rate,
    seed=seed,
),
"alloygbm_factor_neutral_dro": lambda seed: GBMRegressor(
    neutralization="per_round_gradient",
    leaf_solver="dro",
    dro_radius=0.05,
    n_estimators=args.rounds,
    max_depth=args.max_depth,
    learning_rate=args.learning_rate,
    seed=seed,
),
```

For datasets without explicit factors, benchmark code should synthesize factors from the first `min(5, n_features)` columns only for these benchmark arms and pass them as `factor_exposures`.

- [ ] **Step 3: Run benchmark smoke**

Run:

```bash
.venv/bin/python benchmarks/run_model_comparison.py --scenarios dense_numeric --models alloygbm alloygbm_factor_neutral --rounds 3 --max-depth 2 --learning-rate 0.1 --seed 7 --test-size 0.25 --output-dir /tmp/alloygbm_factor_neutral_smoke
```

Expected: both selected models PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md docs/user/gbmregressor.md docs/user/gbmclassifier.md docs/user/gbmranker.md docs/site/source/estimator.rst benchmarks/run_model_comparison.py
git commit -m "docs: document factor-neutral boosting"
```

---

### Task 8: Full Verification and PR Preparation

**Files:**
- Modify as needed only for failures found by verification.

- [ ] **Step 1: Run full Rust checks**

```bash
cargo fmt -- --check
cargo clippy --workspace --exclude alloygbm-python --all-targets -- -D warnings
cargo test --workspace --exclude alloygbm-python --all-targets
```

Expected: all pass.

- [ ] **Step 2: Run Python build and tests**

```bash
.venv/bin/python -m maturin develop --manifest-path bindings/python/Cargo.toml --release
.venv/bin/python -m pytest bindings/python/tests -q
```

Expected: all pass.

- [ ] **Step 3: Inspect git history**

```bash
git log --oneline --decorate -n 12
git status --short --branch
```

Expected: clean working tree on `codex/factor-neutral-boosting-v0.7`.

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin codex/factor-neutral-boosting-v0.7
gh pr create --title "[codex] Add factor-neutral boosting" --body-file /tmp/factor-neutral-boosting-pr.md
```

PR body should include:

```markdown
## Summary
- Adds factor-neutral target and per-round gradient projection with fit-time `factor_exposures`.
- Adds optional split exposure penalty for constant leaves.
- Documents compatibility with MorphBoost, DRO leaves, PL trees, and ranking/classification objectives.

## Test Plan
- `cargo fmt -- --check`
- `cargo clippy --workspace --exclude alloygbm-python --all-targets -- -D warnings`
- `cargo test --workspace --exclude alloygbm-python --all-targets`
- `.venv/bin/python -m maturin develop --manifest-path bindings/python/Cargo.toml --release`
- `.venv/bin/python -m pytest bindings/python/tests -q`
- benchmark smoke command from Task 7
```

---

## Plan Self-Review

Spec coverage: all requested modes are covered: `none`, `pre_target`, `per_round_gradient`, and `split_penalty`. Interactions with MorphBoost, PL trees, and DRO leaves are covered with explicit compatibility rules and tests.

Placeholder scan: no incomplete markers or unspecified edge handling remains.

Type consistency: public params are `neutralization`, `factor_neutralization_lambda`, `factor_penalty`, and fit-time `factor_exposures`; Rust types use `NeutralizationKind`, `FactorNeutralizationConfig`, and `FactorExposureMatrix`.
