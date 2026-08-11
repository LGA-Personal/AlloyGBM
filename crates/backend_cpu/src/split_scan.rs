use std::cell::RefCell;

thread_local! {
    static THREAD_SPLIT_SCAN_SCRATCH: RefCell<SplitScanScratch> =
        RefCell::new(SplitScanScratch::default());
}

#[derive(Debug, Default)]
struct SplitScanScratch {
    cumulative_grad: Vec<f32>,
    cumulative_hess: Vec<f32>,
    cumulative_grad_sq: Vec<f32>,
    cumulative_count: Vec<u32>,
}

impl SplitScanScratch {
    fn resize_standard(&mut self, len: usize) {
        self.cumulative_grad.resize(len, 0.0);
        self.cumulative_hess.resize(len, 0.0);
        self.cumulative_count.resize(len, 0);
    }

    fn resize_dro(&mut self, len: usize) {
        self.resize_standard(len);
        self.cumulative_grad_sq.resize(len, 0.0);
    }
}

pub(super) fn with_split_scan_scratch<R>(
    scan_limit: usize,
    f: impl FnOnce(&mut [f32], &mut [f32], &mut [u32]) -> R,
) -> R {
    THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.resize_standard(scan_limit);
        let SplitScanScratch {
            cumulative_grad,
            cumulative_hess,
            cumulative_count,
            ..
        } = &mut *scratch;
        f(
            cumulative_grad.as_mut_slice(),
            cumulative_hess.as_mut_slice(),
            cumulative_count.as_mut_slice(),
        )
    })
}

pub(super) fn with_dro_split_scan_scratch<R>(
    scan_limit: usize,
    f: impl FnOnce(&mut [f32], &mut [f32], &mut [f32], &mut [u32]) -> R,
) -> R {
    THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.resize_dro(scan_limit);
        let SplitScanScratch {
            cumulative_grad,
            cumulative_hess,
            cumulative_grad_sq,
            cumulative_count,
        } = &mut *scratch;
        f(
            cumulative_grad.as_mut_slice(),
            cumulative_hess.as_mut_slice(),
            cumulative_grad_sq.as_mut_slice(),
            cumulative_count.as_mut_slice(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn split_scan_scratch_reuses_buffers_and_initializes_active_slices() {
        let first = with_split_scan_scratch(
            257,
            |cumulative_grad: &mut [f32],
             cumulative_hess: &mut [f32],
             cumulative_count: &mut [u32]| {
                assert_eq!(cumulative_grad.len(), 257);
                assert_eq!(cumulative_hess.len(), 257);
                assert_eq!(cumulative_count.len(), 257);
                (
                    cumulative_grad.as_ptr(),
                    cumulative_hess.as_ptr(),
                    cumulative_count.as_ptr(),
                )
            },
        );
        let first_capacities = THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
            let scratch = cell.borrow();
            (
                scratch.cumulative_grad.capacity(),
                scratch.cumulative_hess.capacity(),
                scratch.cumulative_count.capacity(),
            )
        });

        let second = with_split_scan_scratch(
            17,
            |cumulative_grad: &mut [f32],
             cumulative_hess: &mut [f32],
             cumulative_count: &mut [u32]| {
                assert_eq!(cumulative_grad.len(), 17);
                assert_eq!(cumulative_hess.len(), 17);
                assert_eq!(cumulative_count.len(), 17);
                (
                    cumulative_grad.as_ptr(),
                    cumulative_hess.as_ptr(),
                    cumulative_count.as_ptr(),
                )
            },
        );
        let second_capacities = THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
            let scratch = cell.borrow();
            (
                scratch.cumulative_grad.capacity(),
                scratch.cumulative_hess.capacity(),
                scratch.cumulative_count.capacity(),
            )
        });

        assert_eq!(first, second);
        assert_eq!(first_capacities, second_capacities);

        with_split_scan_scratch(513, |cumulative_grad, cumulative_hess, cumulative_count| {
            assert_eq!(cumulative_grad.len(), 513);
            assert_eq!(cumulative_hess.len(), 513);
            assert_eq!(cumulative_count.len(), 513);
        });
    }

    #[test]
    fn standard_split_scan_scratch_never_allocates_gradient_square() {
        let capacities = std::thread::spawn(|| {
            let initial =
                THREAD_SPLIT_SCAN_SCRATCH.with(|cell| cell.borrow().cumulative_grad_sq.capacity());

            with_split_scan_scratch(257, |_, _, _| {});
            let after_first_standard =
                THREAD_SPLIT_SCAN_SCRATCH.with(|cell| cell.borrow().cumulative_grad_sq.capacity());

            with_split_scan_scratch(17, |_, _, _| {});
            let after_second_standard =
                THREAD_SPLIT_SCAN_SCRATCH.with(|cell| cell.borrow().cumulative_grad_sq.capacity());

            with_dro_split_scan_scratch(19, |_, _, _, _| {});
            let after_dro =
                THREAD_SPLIT_SCAN_SCRATCH.with(|cell| cell.borrow().cumulative_grad_sq.capacity());

            with_split_scan_scratch(513, |_, _, _| {});
            let after_final_standard =
                THREAD_SPLIT_SCAN_SCRATCH.with(|cell| cell.borrow().cumulative_grad_sq.capacity());

            (
                initial,
                after_first_standard,
                after_second_standard,
                after_dro,
                after_final_standard,
            )
        })
        .join()
        .expect("scratch capacity probe thread");

        assert_eq!(capacities.0, 0);
        assert_eq!(capacities.1, 0);
        assert_eq!(capacities.2, 0);
        assert!(capacities.3 >= 19);
        assert_eq!(capacities.4, capacities.3);
    }

    #[test]
    fn split_scan_scratch_recovers_after_callback_panic() {
        let panic_result = std::panic::catch_unwind(|| {
            with_split_scan_scratch(31, |_, _, _| panic!("intentional scratch callback panic"));
        });
        assert!(panic_result.is_err());

        with_split_scan_scratch(9, |cumulative_grad, cumulative_hess, cumulative_count| {
            assert_eq!(cumulative_grad.len(), 9);
            assert_eq!(cumulative_hess.len(), 9);
            assert_eq!(cumulative_count.len(), 9);
            cumulative_grad.fill(1.0);
            cumulative_hess.fill(2.0);
            cumulative_count.fill(3);
        });
    }

    #[test]
    fn dro_split_scan_scratch_reuses_gradient_square_and_isolates_workers() {
        let first = with_dro_split_scan_scratch(
            19,
            |cumulative_grad: &mut [f32],
             cumulative_hess: &mut [f32],
             cumulative_grad_sq: &mut [f32],
             cumulative_count: &mut [u32]| {
                cumulative_grad.fill(1.0);
                cumulative_hess.fill(2.0);
                cumulative_grad_sq.fill(3.0);
                cumulative_count.fill(4);
                (
                    cumulative_grad.as_ptr(),
                    cumulative_hess.as_ptr(),
                    cumulative_grad_sq.as_ptr(),
                    cumulative_count.as_ptr(),
                )
            },
        );
        let first_capacities = THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
            let scratch = cell.borrow();
            (
                scratch.cumulative_grad.capacity(),
                scratch.cumulative_hess.capacity(),
                scratch.cumulative_grad_sq.capacity(),
                scratch.cumulative_count.capacity(),
            )
        });

        let second = with_dro_split_scan_scratch(
            7,
            |cumulative_grad: &mut [f32],
             cumulative_hess: &mut [f32],
             cumulative_grad_sq: &mut [f32],
             cumulative_count: &mut [u32]| {
                for index in 0..cumulative_grad.len() {
                    cumulative_grad[index] = index as f32;
                    cumulative_hess[index] = (index * 2) as f32;
                    cumulative_grad_sq[index] = (index * 3) as f32;
                    cumulative_count[index] = index as u32;
                }
                assert_eq!(cumulative_grad_sq, &[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
                (
                    cumulative_grad.as_ptr(),
                    cumulative_hess.as_ptr(),
                    cumulative_grad_sq.as_ptr(),
                    cumulative_count.as_ptr(),
                )
            },
        );
        let second_capacities = THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
            let scratch = cell.borrow();
            (
                scratch.cumulative_grad.capacity(),
                scratch.cumulative_hess.capacity(),
                scratch.cumulative_grad_sq.capacity(),
                scratch.cumulative_count.capacity(),
            )
        });

        assert_eq!(first, second);
        assert_eq!(first_capacities, second_capacities);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("scratch isolation pool");
        let values = pool.install(|| {
            (0..8_usize)
                .into_par_iter()
                .map(|worker| {
                    with_dro_split_scan_scratch(5, |_, _, cumulative_grad_sq, _| {
                        cumulative_grad_sq.fill(worker as f32);
                        cumulative_grad_sq.to_vec()
                    })
                })
                .collect::<Vec<_>>()
        });
        for (worker, values) in values.into_iter().enumerate() {
            assert_eq!(values, vec![worker as f32; 5]);
        }
    }

    #[test]
    fn dro_split_scan_scratch_recovers_after_callback_panic() {
        let panic_result = std::panic::catch_unwind(|| {
            with_dro_split_scan_scratch(31, |_, _, _, _| {
                panic!("intentional DRO scratch callback panic")
            });
        });
        assert!(panic_result.is_err());

        with_dro_split_scan_scratch(
            9,
            |cumulative_grad, cumulative_hess, cumulative_grad_sq, cumulative_count| {
                assert_eq!(cumulative_grad.len(), 9);
                assert_eq!(cumulative_hess.len(), 9);
                assert_eq!(cumulative_grad_sq.len(), 9);
                assert_eq!(cumulative_count.len(), 9);
            },
        );
    }
}
