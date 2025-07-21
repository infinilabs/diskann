/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! ANN disk index abstraction

use vector::FullPrecisionDistance;

use crate::model::{IndexConfiguration, DiskIndexBuildParameters};
use crate::storage::DiskIndexStorage;
use crate::model::vertex::{DIM_104, DIM_128, DIM_256, DIM_512};

use crate::common::{ANNResult, ANNError};

use super::DiskIndex;

/// ANN disk index abstraction for custom <T, N>
pub trait ANNDiskIndex<T> : Sync + Send
where T : Default + Copy + Sync + Send + Into<f32>
 {
    /// Build index
    fn build(&mut self, codebook_prefix: &str) -> ANNResult<()>;

    /// Search the index for K nearest neighbors of query using given L value, for benchmarking purposes
    fn search(&self, query : &[T], k_value : usize, l_value : u32, indices : &mut[u32]) -> ANNResult<u32>;

    /// Search the index for K nearest neighbors of query using given L value, for benchmarking purposes
    fn search_with_distance(&self, query : &[T], k_value : usize, l_value : u32, indices : &mut[u32], distances: &mut[f32]) -> ANNResult<u32>;
}

/// Create Index<T, N> based on configuration
pub fn create_disk_index<'a, T>(
    disk_build_param: Option<DiskIndexBuildParameters>, 
    config: IndexConfiguration, 
    storage: DiskIndexStorage<T>,
) -> ANNResult<Box<dyn ANNDiskIndex<T> + 'a>> 
where
    T: Default + Copy + Sync + Send + Into<f32> + 'a,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_512]: FullPrecisionDistance<T, DIM_512>,
{
    match config.aligned_dim {
        DIM_104 => {
            let index = Box::new(DiskIndex::<T, DIM_104>::new(disk_build_param, config, storage));
            Ok(index as Box<dyn ANNDiskIndex<T>>)
        },
        DIM_128 => {
            let index = Box::new(DiskIndex::<T, DIM_128>::new(disk_build_param, config, storage));
            Ok(index as Box<dyn ANNDiskIndex<T>>)
        },
        DIM_256 => {
            let index = Box::new(DiskIndex::<T, DIM_256>::new(disk_build_param, config, storage));
            Ok(index as Box<dyn ANNDiskIndex<T>>)
        },
        DIM_512 => {
            let index = Box::new(DiskIndex::<T, DIM_512>::new(disk_build_param, config, storage));
            Ok(index as Box<dyn ANNDiskIndex<T>>)
        },
        _ => Err(ANNError::log_index_error(format!("Invalid dimension: {}", config.aligned_dim))),
    }
}
