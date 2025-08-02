#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub fn distance_l2_vector_f16(a: &[f16], b: &[f16]) -> f32 {
    // SIMD implementation for x86_64
    let mut sum = 0.0f32;
    let len = a.len();

    // Use AVX2 if available
    if is_x86_feature_detected!("avx2") {
        // AVX2 implementation
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let va = _mm256_loadu_ps(&a[i] as *const f16 as *const f32);
                let vb = _mm256_loadu_ps(&b[i] as *const f16 as *const f32);
                let diff = _mm256_sub_ps(va, vb);
                let squared = _mm256_mul_ps(diff, diff);
                let sum_vec = _mm256_add_ps(sum_vec, squared);
                sum += _mm256_reduce_add_ps(sum_vec);
            }
            i += 8;
        }
    }

    // Handle remaining elements
    for i in i..len {
        let diff = a[i] as f32 - b[i] as f32;
        sum += diff * diff;
    }

    sum.sqrt()
}

#[cfg(not(target_arch = "x86_64"))]
pub fn distance_l2_vector_f16(a: &[f16], b: &[f16]) -> f32 {
    // Scalar fallback for non-x86_64 architectures
    let mut sum = 0.0f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let diff = *ai as f32 - *bi as f32;
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
pub fn distance_l2_vector_f32(a: &[f32], b: &[f32]) -> f32 {
    // SIMD implementation for x86_64
    let mut sum = 0.0f32;
    let len = a.len();

    // Use AVX2 if available
    if is_x86_feature_detected!("avx2") {
        // AVX2 implementation
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let va = _mm256_loadu_ps(&a[i]);
                let vb = _mm256_loadu_ps(&b[i]);
                let diff = _mm256_sub_ps(va, vb);
                let squared = _mm256_mul_ps(diff, diff);
                let sum_vec = _mm256_add_ps(sum_vec, squared);
                sum += _mm256_reduce_add_ps(sum_vec);
            }
            i += 8;
        }
    }

    // Handle remaining elements
    for i in i..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }

    sum.sqrt()
}

#[cfg(not(target_arch = "x86_64"))]
pub fn distance_l2_vector_f32(a: &[f32], b: &[f32]) -> f32 {
    // Scalar fallback for non-x86_64 architectures
    let mut sum = 0.0f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let diff = ai - bi;
        sum += diff * diff;
    }
    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
pub fn distance_cosine_vector_f32(a: &[f32], b: &[f32]) -> f32 {
    // SIMD implementation for x86_64
    let mut dot_product = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    let len = a.len();

    // Use AVX2 if available
    if is_x86_feature_detected!("avx2") {
        // AVX2 implementation
        let mut i = 0;
        while i + 8 <= len {
            unsafe {
                let va = _mm256_loadu_ps(&a[i]);
                let vb = _mm256_loadu_ps(&b[i]);
                let dot = _mm256_mul_ps(va, vb);
                let norm_a_vec = _mm256_mul_ps(va, va);
                let norm_b_vec = _mm256_mul_ps(vb, vb);
                dot_product += _mm256_reduce_add_ps(dot);
                norm_a += _mm256_reduce_add_ps(norm_a_vec);
                norm_b += _mm256_reduce_add_ps(norm_b_vec);
            }
            i += 8;
        }
    }

    // Handle remaining elements
    for i in i..len {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denominator = (norm_a * norm_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        1.0 - (dot_product / denominator)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn distance_cosine_vector_f32(a: &[f32], b: &[f32]) -> f32 {
    // Scalar fallback for non-x86_64 architectures
    let mut dot_product = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (ai, bi) in a.iter().zip(b.iter()) {
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denominator = (norm_a * norm_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        1.0 - (dot_product / denominator)
    }
}
