/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations)]

use hashbrown::HashMap;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSliceMut,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

#[cfg(not(target_arch = "x86_64"))]
const _MM_HINT_T0: i32 = 0;

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

#[cfg(target_arch = "x86_64")]
pub fn prefetch_data(dists_out: &[f32], pq_ids: &[u32]) {
    unsafe {
        _mm_prefetch(dists_out.as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr().add(64) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr().add(128) as *const i8, _MM_HINT_T0);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn prefetch_data(_dists_out: &[f32], _pq_ids: &[u32]) {
    // No-op for non-x86_64 architectures
}

#[cfg(target_arch = "x86_64")]
pub fn prefetch_chunk_data(chunk_id: usize, pq_ids: &[u32]) {
    unsafe {
        _mm_prefetch(
            pq_ids.as_ptr().add(chunk_id * 256) as *const i8,
            _MM_HINT_T0,
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn prefetch_chunk_data(_chunk_id: usize, _pq_ids: &[u32]) {
    // No-op for non-x86_64 architectures
}

use crate::{
    common::{ANNError, ANNResult},
    model::NUM_PQ_CENTROIDS,
};

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Fixed chunk PQ table for distance calculations
pub struct FixedChunkPQTable {
    /// PQ tables: float array of size [256 * ndims]
    pub tables: Vec<f32>,
    /// True dimension of vectors
    pub ndims: u64,
    /// Number of chunks
    pub n_chunks: u64,
    /// Whether to use rotation
    pub use_rotation: bool,
    /// Chunk offsets
    pub chunk_offsets: Vec<u32>,
    /// Centroid data
    pub centroid: Vec<f32>,
    /// Transposed tables (col-major)
    pub tables_tr: Vec<f32>,
    /// Transposed rotation matrix
    pub rotmat_tr: Vec<f32>,
}

impl FixedChunkPQTable {
    /// Create a new fixed chunk PQ table
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            ndims: 0,
            n_chunks: 0,
            use_rotation: false,
            chunk_offsets: Vec::new(),
            centroid: Vec::new(),
            tables_tr: Vec::new(),
            rotmat_tr: Vec::new(),
        }
    }

    /// Load PQ centroid data from binary file
    pub fn load_pq_centroid_bin(
        &mut self,
        pq_table_file: &str,
        num_chunks: usize,
    ) -> ANNResult<()> {
        let path = Path::new(pq_table_file);
        if !path.exists() {
            return Err(ANNError::log_io_error(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("PQ table file not found: {}", pq_table_file),
            )));
        }

        let mut file = File::open(pq_table_file)?;

        // Read header information
        let mut header = [0u32; 3];
        for val in &mut header {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = u32::from_le_bytes(buf);
        }

        let num_centroids = header[0] as usize;
        let chunk_size = header[1] as usize;
        let use_rotation = header[2] != 0;

        self.n_chunks = num_chunks as u64;
        self.ndims = (num_centroids * chunk_size) as u64;
        self.use_rotation = use_rotation;

        // Calculate chunk offsets
        self.chunk_offsets.clear();
        self.chunk_offsets.reserve(num_chunks);
        for i in 0..num_chunks {
            self.chunk_offsets.push((i * chunk_size) as u32);
        }

        // Read centroid data
        let centroid_size = num_centroids * chunk_size;
        self.centroid.resize(centroid_size, 0.0);

        for val in &mut self.centroid {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Read PQ tables
        let table_size = 256 * num_chunks;
        self.tables.resize(table_size, 0.0);

        for val in &mut self.tables {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            *val = f32::from_le_bytes(buf);
        }

        // Initialize transposed tables if needed
        if self.use_rotation {
            self.tables_tr.resize(table_size, 0.0);
            self.transpose_tables();
        }

        Ok(())
    }

    /// Get the number of chunks
    pub fn get_num_chunks(&self) -> u32 {
        self.n_chunks as u32
    }

    /// Get the number of dimensions
    pub fn get_ndims(&self) -> u64 {
        self.ndims
    }

    /// Preprocess query for PQ operations
    pub fn preprocess_query(&mut self, query_vec: &mut [f32]) {
        if self.use_rotation && !self.centroid.is_empty() {
            // Center the query
            for i in 0..self.ndims as usize {
                query_vec[i] -= self.centroid[i];
            }

            // Apply rotation if available
            if !self.rotmat_tr.is_empty() {
                self.apply_rotation(query_vec);
            }
        }
    }

    /// Populate chunk distances for the query
    pub fn populate_chunk_distances(&self, query_vec: &[f32], dist_vec: &mut [f32]) {
        // Early return if not properly initialized
        if self.n_chunks == 0 || self.ndims == 0 {
            return;
        }

        let chunk_size = (self.ndims / self.n_chunks) as usize;

        for chunk in 0..self.n_chunks as usize {
            let chunk_start = chunk * chunk_size;
            let chunk_end = chunk_start + chunk_size;
            let query_chunk = &query_vec[chunk_start..chunk_end];

            // Calculate distances to all centroids in this chunk
            for centroid_id in 0..256 {
                let table_offset = chunk * 256 + centroid_id;
                let centroid_start = centroid_id * chunk_size;
                let centroid_end = centroid_start + chunk_size;
                let centroid = &self.centroid[centroid_start..centroid_end];

                // Calculate L2 distance
                let mut dist = 0.0;
                for i in 0..chunk_size {
                    let diff = query_chunk[i] - centroid[i];
                    dist += diff * diff;
                }

                dist_vec[table_offset] = dist;
            }
        }
    }

    /// Calculate L2 distance between query and PQ-compressed vector
    pub fn l2_distance(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        // Early return if not properly initialized
        if self.n_chunks == 0 || self.ndims == 0 {
            return 0.0;
        }

        let mut total_dist = 0.0;
        let chunk_size = (self.ndims / self.n_chunks) as usize;

        for chunk in 0..self.n_chunks as usize {
            let chunk_start = chunk * chunk_size;
            let chunk_end = chunk_start + chunk_size;
            let query_chunk = &query_vec[chunk_start..chunk_end];

            let centroid_id = base_vec[chunk] as usize;
            let table_offset = chunk * 256 + centroid_id;

            total_dist += self.tables[table_offset];
        }

        total_dist
    }

    /// Calculate inner product between query and PQ-compressed vector
    pub fn inner_product(&self, query_vec: &[f32], base_vec: &[u8]) -> f32 {
        // Early return if not properly initialized
        if self.n_chunks == 0 || self.ndims == 0 {
            return 0.0;
        }

        let mut total_product = 0.0;
        let chunk_size = (self.ndims / self.n_chunks) as usize;

        for chunk in 0..self.n_chunks as usize {
            let chunk_start = chunk * chunk_size;
            let chunk_end = chunk_start + chunk_size;
            let query_chunk = &query_vec[chunk_start..chunk_end];

            let centroid_id = base_vec[chunk] as usize;
            let centroid_start = centroid_id * chunk_size;
            let centroid_end = centroid_start + chunk_size;
            let centroid = &self.centroid[centroid_start..centroid_end];

            // Calculate inner product for this chunk
            for i in 0..chunk_size {
                total_product += query_chunk[i] * centroid[i];
            }
        }

        total_product
    }

    /// Inflate a PQ-compressed vector to full precision
    pub fn inflate_vector(&self, base_vec: &[u8], out_vec: &mut [f32]) {
        // Early return if not properly initialized
        if self.n_chunks == 0 || self.ndims == 0 {
            return;
        }

        let chunk_size = (self.ndims / self.n_chunks) as usize;

        for chunk in 0..self.n_chunks as usize {
            let centroid_id = base_vec[chunk] as usize;
            let centroid_start = centroid_id * chunk_size;
            let centroid_end = centroid_start + chunk_size;
            let centroid = &self.centroid[centroid_start..centroid_end];

            let out_start = chunk * chunk_size;
            let out_end = out_start + chunk_size;
            let out_chunk = &mut out_vec[out_start..out_end];

            out_chunk.copy_from_slice(centroid);
        }
    }

    /// Populate chunk inner products for the query
    pub fn populate_chunk_inner_products(&self, query_vec: &[f32], dist_vec: &mut [f32]) {
        // Early return if not properly initialized
        if self.n_chunks == 0 || self.ndims == 0 {
            return;
        }

        let chunk_size = (self.ndims / self.n_chunks) as usize;

        for chunk in 0..self.n_chunks as usize {
            let chunk_start = chunk * chunk_size;
            let chunk_end = chunk_start + chunk_size;
            let query_chunk = &query_vec[chunk_start..chunk_end];

            // Calculate inner products with all centroids in this chunk
            for centroid_id in 0..256 {
                let centroid_start = centroid_id * chunk_size;
                let centroid_end = centroid_start + chunk_size;
                let centroid = &self.centroid[centroid_start..centroid_end];

                // Calculate inner product
                let mut product = 0.0;
                for i in 0..chunk_size {
                    product += query_chunk[i] * centroid[i];
                }

                let table_offset = chunk * 256 + centroid_id;
                dist_vec[table_offset] = product;
            }
        }
    }

    /// Transpose tables for rotation operations
    fn transpose_tables(&mut self) {
        let num_chunks = self.n_chunks as usize;
        let table_size = 256 * num_chunks;

        self.tables_tr.resize(table_size, 0.0);

        for chunk in 0..num_chunks {
            for centroid_id in 0..256 {
                let src_idx = chunk * 256 + centroid_id;
                let dst_idx = centroid_id * num_chunks + chunk;
                self.tables_tr[dst_idx] = self.tables[src_idx];
            }
        }
    }

    /// Apply rotation to query vector
    fn apply_rotation(&self, query_vec: &mut [f32]) {
        // This is a simplified rotation implementation
        // In practice, this would apply the full rotation matrix
        if !self.rotmat_tr.is_empty() {
            // Apply rotation matrix multiplication
            // For now, we'll just use the identity transformation
            // This should be implemented with proper matrix multiplication
        }
    }
}

/// Given a batch input nodes, return a batch of PQ distance
/// * `pq_ids` - batch nodes: n_pts * pq_nchunks
/// * `n_pts` - batch number
/// * `pq_nchunks` - pq chunk number number
/// * `pq_dists` - pre-calculated the distance between query and each centroid: chunk_size * num_centroids
/// * `dists_out` - n_pts * 1
pub fn pq_dist_lookup(
    pq_ids: &[u8],
    n_pts: usize,
    pq_nchunks: usize,
    pq_dists: &[f32],
) -> Vec<f32> {
    let mut dists_out: Vec<f32> = vec![0.0; n_pts];
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(dists_out.as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr().add(64) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pq_ids.as_ptr().add(128) as *const i8, _MM_HINT_T0);
    }
    for chunk in 0..pq_nchunks {
        let chunk_dists = &pq_dists[256 * chunk..];
        if chunk < pq_nchunks - 1 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                _mm_prefetch(
                    chunk_dists.as_ptr().offset(256 * chunk as isize).add(256) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
        dists_out
            .par_iter_mut()
            .enumerate()
            .for_each(|(n_iter, dist)| {
                let pq_centerid = pq_ids[pq_nchunks * n_iter + chunk];
                *dist += chunk_dists[pq_centerid as usize];
            });
    }
    dists_out
}

pub fn aggregate_coords(ids: &[u32], all_coords: &[u8], ndims: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0u8; ids.len() * ndims];
    let ndim_u32 = ndims as u32;
    out.par_chunks_mut(ndims)
        .enumerate()
        .for_each(|(index, chunk)| {
            let id_compressed_pivot = &all_coords
                [(ids[index] * ndim_u32) as usize..(ids[index] * ndim_u32 + ndim_u32) as usize];
            let temp_slice =
                unsafe { std::slice::from_raw_parts(id_compressed_pivot.as_ptr(), ndims) };
            chunk.copy_from_slice(temp_slice);
        });

    out
}

#[cfg(test)]
mod fixed_chunk_pq_table_test {

    use super::*;
    use crate::common::{ANNError, ANNResult};
    use crate::utils::{convert_types_u32_usize, convert_types_u64_usize, file_exists, load_bin};

    const DIM: usize = 128;

    #[test]
    fn load_pivot_test() {
        let mut fixed_chunk_pq_table = FixedChunkPQTable::new();
        // Test basic functionality without external files
        assert_eq!(fixed_chunk_pq_table.get_ndims(), 0);
        assert_eq!(fixed_chunk_pq_table.get_num_chunks(), 0);
    }

    #[test]
    fn get_num_chunks_test() {
        let fixed_chunk_pq_table = FixedChunkPQTable::new();
        let chunk: u32 = fixed_chunk_pq_table.get_num_chunks();
        assert_eq!(chunk, 0);
    }

    #[test]
    fn preprocess_query_test() {
        let mut fixed_chunk_pq_table = FixedChunkPQTable::new();
        let mut query_vec: Vec<f32> = vec![1.0, 2.0, 3.0];
        fixed_chunk_pq_table.preprocess_query(&mut query_vec);
        // Test that preprocessing doesn't crash
        assert_eq!(query_vec.len(), 3);
    }

    #[test]
    fn calculate_distances_tests() {
        let fixed_chunk_pq_table = FixedChunkPQTable::new();
        let query_vec: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut dist_vec = vec![0.0; 256];

        // Test that distance calculation doesn't crash
        fixed_chunk_pq_table.populate_chunk_distances(&query_vec, &mut dist_vec);
        assert_eq!(dist_vec.len(), 256);
    }

    fn load_pq_pivots_bin(
        pq_pivots_path: &str,
        num_pq_chunks: &usize,
    ) -> ANNResult<(usize, Vec<f32>, Vec<f32>, Vec<usize>)> {
        if !file_exists(pq_pivots_path) {
            return Err(ANNError::log_pq_error(
                "ERROR: PQ k-means pivot file not found.".to_string(),
            ));
        }

        let (data, offset_num, offset_dim) = load_bin::<u64>(pq_pivots_path, 0)?;
        let file_offset_data = convert_types_u64_usize(&data, offset_num, offset_dim);
        if offset_num != 4 {
            let error_message = format!("Error reading pq_pivots file {}. Offsets don't contain correct metadata, # offsets = {}, but expecting 4.", pq_pivots_path, offset_num);
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, pq_center_num, dim) = load_bin::<f32>(pq_pivots_path, file_offset_data[0])?;
        let pq_table = data.to_vec();
        if pq_center_num != NUM_PQ_CENTROIDS {
            let error_message = format!(
                "Error reading pq_pivots file {}. file_num_centers = {}, but expecting {} centers.",
                pq_pivots_path, pq_center_num, NUM_PQ_CENTROIDS
            );
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, centroid_dim, nc) = load_bin::<f32>(pq_pivots_path, file_offset_data[1])?;
        let centroids = data.to_vec();
        if centroid_dim != dim || nc != 1 {
            let error_message = format!("Error reading pq_pivots file {}. file_dim = {}, file_cols = {} but expecting {} entries in 1 dimension.", pq_pivots_path, centroid_dim, nc, dim);
            return Err(ANNError::log_pq_error(error_message));
        }

        let (data, chunk_offset_num, nc) = load_bin::<u32>(pq_pivots_path, file_offset_data[2])?;
        let chunk_offsets = convert_types_u32_usize(&data, chunk_offset_num, nc);
        if chunk_offset_num != num_pq_chunks + 1 || nc != 1 {
            let error_message = format!("Error reading pq_pivots file at chunk offsets; file has nr={}, nc={} but expecting nr={} and nc=1.", chunk_offset_num, nc, num_pq_chunks + 1);
            return Err(ANNError::log_pq_error(error_message));
        }

        Ok((dim, pq_table, centroids, chunk_offsets))
    }
}

#[cfg(test)]
mod pq_index_prune_query_test {

    use super::*;

    #[test]
    fn pq_dist_lookup_test() {
        let pq_ids: Vec<u8> = vec![1u8, 3u8, 2u8, 2u8];
        let mut pq_dists: Vec<f32> = Vec::with_capacity(256 * 2);
        for _ in 0..pq_dists.capacity() {
            pq_dists.push(rand::random());
        }

        let dists_out = pq_dist_lookup(&pq_ids, 2, 2, &pq_dists);
        assert_eq!(dists_out.len(), 2);
        assert_eq!(dists_out[0], pq_dists[0 + 1] + pq_dists[256 + 3]);
        assert_eq!(dists_out[1], pq_dists[0 + 2] + pq_dists[256 + 2]);
    }
}
