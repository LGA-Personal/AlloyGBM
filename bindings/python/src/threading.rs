use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;
use rayon::ThreadPoolBuilder;

#[derive(Clone, Copy)]
pub(crate) struct FitThreadCount(isize);

impl FitThreadCount {
    pub(crate) fn into_inner(self) -> isize {
        self.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FitThreadCount {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(
                "n_jobs must be None, -1, or a positive integer",
            ));
        }
        obj.extract::<isize>()
            .map(Self)
            .map_err(|_| PyValueError::new_err("n_jobs must be None, -1, or a positive integer"))
    }
}

fn available_fit_threads() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

pub(crate) fn resolve_fit_thread_count(n_jobs: Option<isize>) -> PyResult<usize> {
    match n_jobs {
        None | Some(-1) => Ok(available_fit_threads()),
        Some(value) if value > 0 => usize::try_from(value)
            .map_err(|_| PyValueError::new_err("n_jobs must be None, -1, or a positive integer")),
        Some(_) => Err(PyValueError::new_err(
            "n_jobs must be None, -1, or a positive integer",
        )),
    }
}

pub(crate) fn install_in_fit_pool<T, F>(n_jobs: Option<isize>, operation: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> PyResult<T> + Send,
{
    let worker_count = resolve_fit_thread_count(n_jobs)?;
    let pool = ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|error| {
            PyRuntimeError::new_err(format!(
                "failed to create the n_jobs={worker_count} fit thread pool: {error}"
            ))
        })?;
    pool.install(operation)
}

#[cfg(test)]
mod tests {
    use super::{install_in_fit_pool, resolve_fit_thread_count};
    use pyo3::Python;
    use rayon::prelude::*;

    #[test]
    fn resolve_fit_thread_count_accepts_supported_values() {
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);

        assert_eq!(
            resolve_fit_thread_count(None).expect("None should use available threads"),
            available
        );
        assert_eq!(
            resolve_fit_thread_count(Some(-1)).expect("-1 should use available threads"),
            available
        );
        assert_eq!(
            resolve_fit_thread_count(Some(1)).expect("one worker should be accepted"),
            1
        );
        assert_eq!(
            resolve_fit_thread_count(Some(3)).expect("positive workers should be accepted"),
            3
        );
    }

    #[test]
    fn resolve_fit_thread_count_rejects_invalid_ranges() {
        Python::initialize();

        assert!(
            resolve_fit_thread_count(Some(0))
                .expect_err("zero must fail")
                .to_string()
                .contains("n_jobs")
        );
        assert!(
            resolve_fit_thread_count(Some(-2))
                .expect_err("values below -1 must fail")
                .to_string()
                .contains("n_jobs")
        );
    }

    #[test]
    fn private_pool_bounds_nested_rayon_work() {
        let (thread_count, worker_indices) = install_in_fit_pool(Some(2), || {
            Ok((
                rayon::current_num_threads(),
                (0..128usize)
                    .into_par_iter()
                    .flat_map(|_| {
                        (0..8usize)
                            .into_par_iter()
                            .map(|_| rayon::current_thread_index().expect("private pool worker"))
                    })
                    .collect::<Vec<_>>(),
            ))
        })
        .expect("private pool should execute");

        assert_eq!(thread_count, 2);
        assert!(!worker_indices.is_empty());
        assert!(worker_indices.iter().all(|index| *index < 2));
    }

    #[test]
    fn sequential_private_pools_keep_independent_sizes() {
        let one = install_in_fit_pool(Some(1), || Ok(rayon::current_num_threads()))
            .expect("one-worker pool should execute");
        let three = install_in_fit_pool(Some(3), || Ok(rayon::current_num_threads()))
            .expect("three-worker pool should execute");

        assert_eq!(one, 1);
        assert_eq!(three, 3);
    }
}
