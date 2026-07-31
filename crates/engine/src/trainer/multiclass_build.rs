use crate::{EngineResult, IterationDiagnostics, TrainedStump};

pub(super) const MIN_MULTICLASS_PARALLEL_WORK: usize = 16_384;

pub(super) struct MulticlassTreeBuildOutcome {
    pub(super) class_index: usize,
    pub(super) diagnostics: IterationDiagnostics,
    pub(super) round_stumps: Vec<TrainedStump>,
}

pub(super) fn should_parallelize_multiclass_trees(
    class_count: usize,
    sampled_row_count: usize,
    sampled_feature_count: usize,
) -> bool {
    if class_count < 2 || rayon::current_num_threads() < 2 {
        return false;
    }
    class_count
        .saturating_mul(sampled_row_count)
        .saturating_mul(sampled_feature_count.max(1))
        >= MIN_MULTICLASS_PARALLEL_WORK
}

pub(super) fn collect_ordered_class_results<T>(
    results: Vec<EngineResult<T>>,
) -> EngineResult<Vec<T>> {
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::{
        MIN_MULTICLASS_PARALLEL_WORK, collect_ordered_class_results,
        should_parallelize_multiclass_trees,
    };
    use crate::{EngineError, EngineResult};
    use rayon::prelude::*;
    use std::time::Duration;

    #[test]
    fn multiclass_parallel_policy_requires_threads_and_enough_work() {
        let one_worker = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-worker pool");
        assert!(!one_worker.install(|| { should_parallelize_multiclass_trees(12, 1_024, 8) }));

        let four_workers = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-worker pool");
        assert!(!four_workers.install(|| { should_parallelize_multiclass_trees(1, 16_384, 8) }));
        assert!(!four_workers.install(|| { should_parallelize_multiclass_trees(12, 32, 8) }));
        assert!(four_workers.install(|| { should_parallelize_multiclass_trees(2, 1_024, 8) }));
        assert_eq!(MIN_MULTICLASS_PARALLEL_WORK, 16_384);
    }

    #[test]
    fn multiclass_parallel_policy_uses_saturating_work_arithmetic() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two-worker pool");
        assert!(pool.install(|| {
            should_parallelize_multiclass_trees(usize::MAX, usize::MAX, usize::MAX)
        }));
    }

    #[test]
    fn ordered_class_results_return_the_lowest_class_error() {
        let results: Vec<EngineResult<usize>> = vec![
            Ok(0),
            Err(EngineError::BackendUnavailable("class 1".to_string())),
            Err(EngineError::BackendUnavailable("class 2".to_string())),
        ];

        let error = collect_ordered_class_results(results).expect_err("two classes fail");
        assert_eq!(
            error,
            EngineError::BackendUnavailable("class 1".to_string())
        );
    }

    #[test]
    fn ordered_class_results_preserve_class_order() {
        let results = vec![Ok(0), Ok(1), Ok(2), Ok(3)];
        assert_eq!(
            collect_ordered_class_results(results).expect("all classes succeed"),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn indexed_parallel_collection_preserves_class_order() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-worker pool");
        let results = pool.install(|| {
            (0..4usize)
                .into_par_iter()
                .map(|class_index| {
                    std::thread::sleep(Duration::from_millis(
                        u64::try_from(4 - class_index).expect("small delay"),
                    ));
                    Ok(class_index)
                })
                .collect::<Vec<EngineResult<usize>>>()
        });

        assert_eq!(
            collect_ordered_class_results(results).expect("all classes succeed"),
            vec![0, 1, 2, 3]
        );
    }
}
