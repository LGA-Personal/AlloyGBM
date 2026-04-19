//! Metal GPU backend for AlloyGBM on Apple Silicon.
//!
//! The crate compiles as a stub on non-macOS targets so `cargo check
//! --workspace` stays green cross-platform; the real implementation is
//! gated by `cfg(target_os = "macos")`.
//!
//! Stage 1 scope is tracked in `docs/metal-backend/STATUS.md`.

#[cfg(target_os = "macos")]
mod device;

#[cfg(target_os = "macos")]
pub use device::{MetalCapabilities, MetalDevice};

pub mod kernels;

#[cfg(target_os = "macos")]
pub struct MetalBackend {
    pub metal_device: MetalDevice,
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    /// Probe the system Metal device and build a backend handle. Returns
    /// an error when Metal is unavailable — callers (the PyO3 layer) are
    /// expected to warn-and-fall-back to `CpuBackend`.
    pub fn new() -> Result<Self, String> {
        let metal_device = MetalDevice::probe()?;
        if !metal_device.capabilities.apple7 {
            return Err(format!(
                "Metal backend requires GPU family Apple7 or later; \
                 device '{}' does not support it",
                metal_device.capabilities.device_name
            ));
        }
        Ok(Self { metal_device })
    }

    /// Read-only capability snapshot.
    pub fn capabilities(&self) -> &MetalCapabilities {
        &self.metal_device.capabilities
    }
}

#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        Err("Metal backend is only available on macOS".to_string())
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #![allow(unsafe_code)]

    use super::*;
    use objc2_foundation::NSString;
    use objc2_metal::MTLDevice;

    #[test]
    fn probe_default_device() {
        match MetalBackend::new() {
            Ok(backend) => {
                let caps = backend.capabilities();
                assert!(caps.apple7, "expected Apple7+ on the CI/dev machine");
                assert!(!caps.device_name.is_empty());
            }
            Err(_) => {
                // Headless runner without a Metal device — not a failure.
            }
        }
    }

    #[test]
    fn histogram_shader_compiles() {
        let Ok(backend) = MetalBackend::new() else {
            return; // no Metal device available — skip.
        };

        let source = NSString::from_str(kernels::histogram::HISTOGRAM_SHADER_SOURCE);
        let result = backend
            .metal_device
            .device
            .newLibraryWithSource_options_error(&source, None);
        match result {
            Ok(_library) => {}
            Err(err) => panic!(
                "histogram.metal failed to compile: {}",
                err.localizedDescription()
            ),
        }
    }
}
