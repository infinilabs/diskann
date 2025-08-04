/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use num_traits::Num;
use std::alloc::{alloc, dealloc, Layout};

use std::sync::Arc;
use std::sync::Mutex;

use crate::disk_search::pq_flash_index::Distance;

/// Non recursive mutex
pub struct NonRecursiveMutex<T> {
    inner: Mutex<T>,
}

impl<T> NonRecursiveMutex<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
        }
    }

    pub fn lock(&self) -> std::sync::MutexGuard<T> {
        self.inner.lock().unwrap()
    }
}

/// Round up X to the nearest multiple of Y
#[inline]
pub fn round_up<T>(x: T, y: T) -> T
where
    T: Num + Copy,
{
    div_round_up(x, y) * y
}

/// Rounded-up division
#[inline]
pub fn div_round_up<T>(x: T, y: T) -> T
where
    T: Num + Copy,
{
    (x / y)
        + if x % y != T::zero() {
            T::one()
        } else {
            T::zero()
        }
}

/// Round down X to the nearest multiple of Y
#[inline]
pub fn round_down<T>(x: T, y: T) -> T
where
    T: Num + Copy,
{
    (x / y) * y
}

/// Is aligned
#[inline]
pub fn is_aligned<T>(x: T, y: T) -> bool
where
    T: Num + Copy,
{
    x % y == T::zero()
}

#[inline]
pub fn is_512_aligned(x: u64) -> bool {
    is_aligned(x, 512)
}

#[inline]
pub fn is_4096_aligned(x: u64) -> bool {
    is_aligned(x, 4096)
}

/// all metadata of individual sub-component files is written in first 4KB for unified files
pub const METADATA_SIZE: usize = 4096;

pub const BUFFER_SIZE_FOR_CACHED_IO: usize = 1024 * 1048576;

pub const PBSTR: &str = "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";

pub const PBWIDTH: usize = 60;

/// Allocate aligned memory
#[inline]
pub fn alloc_aligned(size: usize, alignment: usize) -> *mut std::ffi::c_void {
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, alignment);
        alloc(layout) as *mut std::ffi::c_void
    }
}

/// Free aligned memory
#[inline]
pub fn aligned_free(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        unsafe {
            // Note: This is a simplified implementation. In a real implementation,
            // you'd need to track the layout used for allocation.
            let layout = Layout::from_size_align_unchecked(1, 1);
            dealloc(ptr as *mut u8, layout);
        }
    }
}

/// Get distance function based on metric
pub fn get_distance_function<T>(_metric: diskann_vector::Metric) -> Arc<Distance<T>> {
    // This is a placeholder implementation
    // In a real implementation, you'd return the appropriate distance function
    Arc::new(Distance::<T>::new())
}

/// Get binary file metadata
pub fn get_bin_metadata(
    _file_path: &str,
    _metadata_size: usize,
) -> std::io::Result<(usize, usize)> {
    // This is a simplified implementation
    // In a real implementation, you'd read the actual metadata
    Ok((0, 0))
}

/// Load aligned binary file
pub fn load_aligned_bin<T>(_file_path: &str) -> std::io::Result<(*mut T, usize, usize)> {
    // Placeholder implementation
    Ok((std::ptr::null_mut(), 0, 0))
}

/// Aggregate coordinates for PQ computation
pub fn aggregate_coords(
    _ids: &[u32],
    _n_ids: u64,
    _data: *mut u8,
    _n_chunks: u64,
    _pq_coord_scratch: &mut [u8],
) {
    // Placeholder implementation
}

/// PQ distance lookup
pub fn pq_dist_lookup(
    _pq_coord_scratch: &[u8],
    _n_ids: u64,
    _n_chunks: u64,
    _pq_dists: &[f32],
    _dists_out: &mut [f32],
) {
    // Placeholder implementation
}

macro_rules! convert_types {
    ($name:ident, $intput_type:ty, $output_type:ty) => {
        /// Write data into file
        pub fn $name(srcmat: &[$intput_type], npts: usize, dim: usize) -> Vec<$output_type> {
            let mut destmat: Vec<$output_type> = Vec::new();
            for i in 0..npts {
                for j in 0..dim {
                    destmat.push(srcmat[i * dim + j] as $output_type);
                }
            }
            destmat
        }
    };
}
convert_types!(convert_types_usize_u8, usize, u8);
convert_types!(convert_types_usize_u32, usize, u32);
convert_types!(convert_types_usize_u64, usize, u64);
convert_types!(convert_types_u64_usize, u64, usize);
convert_types!(convert_types_u32_usize, u32, usize);
convert_types!(convert_types_u64_u32, u64, u32);

#[cfg(test)]
mod file_util_test {
    use super::*;
    use std::any::type_name;

    #[test]
    fn round_up_test() {
        assert_eq!(round_up(252, 8), 256);
        assert_eq!(round_up(256, 8), 256);
    }

    #[test]
    fn div_round_up_test() {
        assert_eq!(div_round_up(252, 8), 32);
        assert_eq!(div_round_up(256, 8), 32);
    }

    #[test]
    fn round_down_test() {
        assert_eq!(round_down(252, 8), 248);
        assert_eq!(round_down(256, 8), 256);
    }

    #[test]
    fn is_aligned_test() {
        assert!(!is_aligned(252, 8));
        assert!(is_aligned(256, 8));
    }

    #[test]
    fn is_512_aligned_test() {
        assert!(!is_512_aligned(520));
        assert!(is_512_aligned(512));
    }

    #[test]
    fn is_4096_aligned_test() {
        assert!(!is_4096_aligned(4090));
        assert!(is_4096_aligned(4096));
    }

    #[test]
    fn convert_types_test() {
        let data = vec![0u64, 1u64, 2u64];
        let output = convert_types_u64_usize(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "usize");
        assert_eq!(output[0], 0usize);

        let data = vec![0usize, 1usize, 2usize];
        let output = convert_types_usize_u8(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "u8");
        assert_eq!(output[0], 0u8);

        let data = vec![0usize, 1usize, 2usize];
        let output = convert_types_usize_u64(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "u64");
        assert_eq!(output[0], 0u64);

        let data = vec![0u32, 1u32, 2u32];
        let output = convert_types_u32_usize(&data, 3, 1);
        assert_eq!(output.len(), 3);
        assert_eq!(type_of(output[0]), "usize");
        assert_eq!(output[0], 0usize);
    }

    fn type_of<T>(_: T) -> &'static str {
        type_name::<T>()
    }
}
