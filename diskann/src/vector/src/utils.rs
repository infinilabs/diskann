#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub fn prefetch_vector(ptr: *const u8) {
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn prefetch_vector(_ptr: *const u8) {
    // No-op for non-x86_64 architectures
} 