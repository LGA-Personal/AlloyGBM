use std::cell::RefCell;

thread_local! {
    static THREAD_SPLIT_SCAN_SCRATCH: RefCell<SplitScanScratch> =
        RefCell::new(SplitScanScratch::default());
}

#[derive(Debug, Default)]
pub(crate) struct SplitScanScratch {
    pub(crate) cumulative_grad: Vec<f32>,
    pub(crate) cumulative_hess: Vec<f32>,
    pub(crate) cumulative_count: Vec<u32>,
}

impl SplitScanScratch {
    fn resize(&mut self, len: usize) {
        self.cumulative_grad.resize(len, 0.0);
        self.cumulative_hess.resize(len, 0.0);
        self.cumulative_count.resize(len, 0);
    }
}

pub(crate) fn with_split_scan_scratch<R>(
    scan_limit: usize,
    f: impl FnOnce(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<u32>) -> R,
) -> R {
    THREAD_SPLIT_SCAN_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.resize(scan_limit);
        let SplitScanScratch {
            cumulative_grad,
            cumulative_hess,
            cumulative_count,
        } = &mut *scratch;
        f(cumulative_grad, cumulative_hess, cumulative_count)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_scan_scratch_reuses_buffers_and_initializes_active_slices() {
        let first =
            with_split_scan_scratch(257, |cumulative_grad, cumulative_hess, cumulative_count| {
                assert_eq!(cumulative_grad.len(), 257);
                assert_eq!(cumulative_hess.len(), 257);
                assert_eq!(cumulative_count.len(), 257);
                (
                    (cumulative_grad.as_ptr(), cumulative_grad.capacity()),
                    (cumulative_hess.as_ptr(), cumulative_hess.capacity()),
                    (cumulative_count.as_ptr(), cumulative_count.capacity()),
                )
            });

        let second =
            with_split_scan_scratch(17, |cumulative_grad, cumulative_hess, cumulative_count| {
                assert_eq!(cumulative_grad.len(), 17);
                assert_eq!(cumulative_hess.len(), 17);
                assert_eq!(cumulative_count.len(), 17);
                (
                    (cumulative_grad.as_ptr(), cumulative_grad.capacity()),
                    (cumulative_hess.as_ptr(), cumulative_hess.capacity()),
                    (cumulative_count.as_ptr(), cumulative_count.capacity()),
                )
            });

        assert_eq!(first.0.0, second.0.0);
        assert_eq!(first.1.0, second.1.0);
        assert_eq!(first.2.0, second.2.0);
        assert_eq!(first.0.1, second.0.1);
        assert_eq!(first.1.1, second.1.1);
        assert_eq!(first.2.1, second.2.1);

        with_split_scan_scratch(513, |cumulative_grad, cumulative_hess, cumulative_count| {
            assert_eq!(cumulative_grad.len(), 513);
            assert_eq!(cumulative_hess.len(), 513);
            assert_eq!(cumulative_count.len(), 513);
        });
    }
}
