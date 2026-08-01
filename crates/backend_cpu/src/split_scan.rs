use std::cell::RefCell;

thread_local! {
    static THREAD_SPLIT_SCAN_SCRATCH: RefCell<SplitScanScratch> =
        RefCell::new(SplitScanScratch::default());
}

#[derive(Debug, Default)]
struct SplitScanScratch {
    cumulative_grad: Vec<f32>,
    cumulative_hess: Vec<f32>,
    cumulative_count: Vec<u32>,
}

impl SplitScanScratch {
    fn resize(&mut self, len: usize) {
        self.cumulative_grad.resize(len, 0.0);
        self.cumulative_hess.resize(len, 0.0);
        self.cumulative_count.resize(len, 0);
    }
}

pub(super) fn with_split_scan_scratch<R>(
    scan_limit: usize,
    f: impl FnOnce(&mut [f32], &mut [f32], &mut [u32]) -> R,
) -> R {
    THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.resize(scan_limit);
        let SplitScanScratch {
            cumulative_grad,
            cumulative_hess,
            cumulative_count,
        } = &mut *scratch;
        f(
            cumulative_grad.as_mut_slice(),
            cumulative_hess.as_mut_slice(),
            cumulative_count.as_mut_slice(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
