/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Distance calculation for L2 Metric

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::Half;

/// Calculate the distance by vector arithmetic
#[inline(never)]
pub fn distance_l2_vector_f16<const N: usize>(a: &[Half; N], b: &[Half; N]) -> f32 {
    debug_assert_eq!(N % 8, 0);

    #[cfg(target_arch = "x86_64")]
    {
        // make sure the addresses are bytes aligned
        debug_assert_eq!(a.as_ptr().align_offset(32), 0);
        debug_assert_eq!(b.as_ptr().align_offset(32), 0);

        unsafe {
            let mut sum = _mm256_setzero_ps();
            let a_ptr = a.as_ptr() as *const __m128i;
            let b_ptr = b.as_ptr() as *const __m128i;

            // Iterate over the elements in steps of 8
            for i in (0..N).step_by(8) {
                let a_vec = _mm256_cvtph_ps(_mm_load_si128(a_ptr.add(i / 8)));
                let b_vec = _mm256_cvtph_ps(_mm_load_si128(b_ptr.add(i / 8)));

                let diff = _mm256_sub_ps(a_vec, b_vec);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
            /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
            let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
            /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
            let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
            /* Conversion to float is a no-op on x86-64 */
            _mm_cvtss_f32(x32)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Scalar fallback for non-x86_64 architectures
        let mut sum = 0.0f32;
        for i in 0..N {
            let diff = f32::from(a[i]) - f32::from(b[i]);
            sum += diff * diff;
        }
        sum
    }
}

/// Calculate the distance by vector arithmetic
#[inline(never)]
pub fn distance_l2_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 8, 0);

    #[cfg(target_arch = "x86_64")]
    {
        // make sure the addresses are bytes aligned
        debug_assert_eq!(a.as_ptr().align_offset(32), 0);
        debug_assert_eq!(b.as_ptr().align_offset(32), 0);

        unsafe {
            let mut sum = _mm256_setzero_ps();

            // Iterate over the elements in steps of 8
            for i in (0..N).step_by(8) {
                let a_vec = _mm256_load_ps(&a[i]);
                let b_vec = _mm256_load_ps(&b[i]);
                let diff = _mm256_sub_ps(a_vec, b_vec);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
            /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
            let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
            /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
            let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
            /* Conversion to float is a no-op on x86-64 */
            _mm_cvtss_f32(x32)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Scalar fallback for non-x86_64 architectures
        let mut sum = 0.0f32;
        for i in 0..N {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }
}

#[inline(never)]
pub fn distance_cosine_vector_f32<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    debug_assert_eq!(N % 8, 0);

    #[cfg(target_arch = "x86_64")]
    {
        debug_assert_eq!(a.as_ptr().align_offset(32), 0);
        debug_assert_eq!(b.as_ptr().align_offset(32), 0);

        unsafe {
            let mut dot = _mm256_setzero_ps();
            let mut norm_a = _mm256_setzero_ps();
            let mut norm_b = _mm256_setzero_ps();

            for i in (0..N).step_by(8) {
                let a_vec = _mm256_load_ps(&a[i]);
                let b_vec = _mm256_load_ps(&b[i]);

                // calculate point product
                dot = _mm256_fmadd_ps(a_vec, b_vec, dot);

                // Compute squared norm
                norm_a = _mm256_fmadd_ps(a_vec, a_vec, norm_a);
                norm_b = _mm256_fmadd_ps(b_vec, b_vec, norm_b);
            }

            let dot_sum = hsum256_ps(dot);
            let norm_a_sum = hsum256_ps(norm_a);
            let norm_b_sum = hsum256_ps(norm_b);

            let norm_product = (norm_a_sum * norm_b_sum).sqrt();
            if norm_product == 0.0 {
                0.0
            } else {
                1.0 - (dot_sum / norm_product)
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Scalar fallback for non-x86_64 architectures
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..N {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let norm_product = (norm_a * norm_b).sqrt();
        if norm_product == 0.0 {
            0.0
        } else {
            1.0 - (dot / norm_product)
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}
