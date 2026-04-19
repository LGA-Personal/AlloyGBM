//! Default-device probe + capability flags.
//!
//! `MetalDevice::probe()` is called exactly once per `MetalBackend::new()`.
//! It locates the system Metal device, opens a serial command queue, and
//! records the capability flags that gate Metal 3 baseline vs. Metal 4
//! fast-path dispatch.

#![allow(unsafe_code)]

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};

/// Capability snapshot captured at probe time. Stable across the lifetime
/// of the owning `MetalDevice` (a Metal device does not hot-swap GPUs).
#[derive(Debug, Clone)]
pub struct MetalCapabilities {
    /// `MTLGPUFamilyApple7` (M1 and later). Baseline requirement for this
    /// backend — below Apple7 the kernels are not supported.
    pub apple7: bool,
    /// `MTLGPUFamilyMetal4` (macOS 26 Tahoe+). Gates the ICB, residency-set,
    /// and pipeline-harvesting fast paths. Probed via a raw selector to
    /// stay forward-compatible with `objc2-metal` versions that predate
    /// the Metal 4 variant in their enum bindings.
    pub metal4: bool,
    /// GPU marketing name (e.g. `"Apple M2 Pro"`).
    pub device_name: String,
}

/// Owned Metal device + its serial command queue + capability flags.
pub struct MetalDevice {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub capabilities: MetalCapabilities,
}

impl MetalDevice {
    /// Locate the default Metal device, open a command queue, read
    /// capability flags. Returns an error when Metal is unavailable
    /// (headless VM, non-GPU mac) or when queue allocation fails.
    pub fn probe() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "no Metal-capable GPU found on this system".to_string())?;

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "failed to create Metal command queue".to_string())?;

        let apple7 = device.supportsFamily(MTLGPUFamily::Apple7);

        // `MTLGPUFamilyMetal4 = 5002` per Apple's `MTLDevice.h`. If the
        // binding exposes the variant we could call `supportsFamily`
        // directly; using `msg_send!` with the raw `NSInteger` is
        // binding-version-agnostic and still correct.
        // SAFETY: `supportsFamily:` is read-only on the device, safe
        // to invoke from any thread, and returns `NO` for unknown
        // family values (so the cast is inherently robust).
        let metal4: bool = unsafe { msg_send![&*device, supportsFamily: 5002_isize] };

        let device_name = device.name().to_string();

        Ok(Self {
            device,
            queue,
            capabilities: MetalCapabilities {
                apple7,
                metal4,
                device_name,
            },
        })
    }
}
