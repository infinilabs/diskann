/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use std::mem::size_of;

use crate::common::{ANNResult, AlignedBoxWithSlice};

const MAX_PQ_CHUNKS: usize = 512;

#[derive(Debug)]
/// PQ scratch
pub struct PQScratch {
    /// Aligned pq table dist scratch, must be at least [256 * NCHUNKS]
    pub aligned_pqtable_dist_scratch: AlignedBoxWithSlice<f32>,
    /// Aligned dist scratch, must be at least diskann MAX_DEGREE
    pub aligned_dist_scratch: AlignedBoxWithSlice<f32>,
    /// Aligned pq coord scratch, must be at least [N_CHUNKS * MAX_DEGREE]
    pub aligned_pq_coord_scratch: AlignedBoxWithSlice<u8>,
    /// Rotated query
    pub rotated_query: AlignedBoxWithSlice<f32>,
    /// Aligned query float
    pub aligned_query_float: AlignedBoxWithSlice<f32>,
}

impl PQScratch {
    const ALIGNED_ALLOC_256: usize = 256;

    /// Create a new pq scratch
    pub fn new(graph_degree: usize, aligned_dim: usize) -> ANNResult<Self> {
        let aligned_pq_coord_scratch =
            AlignedBoxWithSlice::new(graph_degree * MAX_PQ_CHUNKS, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_pqtable_dist_scratch =
            AlignedBoxWithSlice::new(256 * MAX_PQ_CHUNKS, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_dist_scratch =
            AlignedBoxWithSlice::new(graph_degree, PQScratch::ALIGNED_ALLOC_256)?;
        let aligned_query_float = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;
        let rotated_query = AlignedBoxWithSlice::new(aligned_dim, 8 * size_of::<f32>())?;

        Ok(Self {
            aligned_pqtable_dist_scratch,
            aligned_dist_scratch,
            aligned_pq_coord_scratch,
            rotated_query,
            aligned_query_float,
        })
    }

    /// Set rotated_query and aligned_query_float values
    pub fn set<T>(&mut self, dim: usize, query: &[T], norm: f32)
    where
        T: Into<f32> + Copy,
    {
        for (d, item) in query.iter().enumerate().take(dim) {
            let query_val: f32 = (*item).into();
            if (norm - 1.0).abs() > f32::EPSILON {
                self.rotated_query[d] = query_val / norm;
                self.aligned_query_float[d] = query_val / norm;
            } else {
                self.rotated_query[d] = query_val;
                self.aligned_query_float[d] = query_val;
            }
        }
    }
}

/// PQ query scratch space for distance calculations
pub struct PQQueryScratch {
    /// Aligned query vector in float format
    pub aligned_query_float: Vec<f32>,
    /// Rotated query vector for PQ operations
    pub rotated_query: Vec<f32>,
    /// PQ table distance scratch space
    pub aligned_pqtable_dist_scratch: Vec<f32>,
    /// Aligned distance scratch space
    pub aligned_dist_scratch: Vec<f32>,
    /// Aligned PQ coordinate scratch space
    pub aligned_pq_coord_scratch: Vec<u8>,
    /// Data dimension
    pub data_dim: u64,
    /// Number of PQ chunks
    pub n_chunks: u64,
}

impl PQQueryScratch {
    /// Create a new PQ query scratch space
    pub fn new(data_dim: u64, n_chunks: u64) -> Self {
        let aligned_dim = data_dim as usize;
        let chunk_size = (data_dim / n_chunks) as usize;
        
        Self {
            aligned_query_float: vec![0.0f32; aligned_dim],
            rotated_query: vec![0.0f32; aligned_dim],
            aligned_pqtable_dist_scratch: vec![0.0f32; 256 * n_chunks as usize],
            aligned_dist_scratch: vec![0.0f32; 1024], // Reasonable default
            aligned_pq_coord_scratch: vec![0u8; chunk_size * 1024], // Reasonable default
            data_dim,
            n_chunks,
        }
    }

    /// Initialize the scratch space with query data
    pub fn initialize<T>(&mut self, data_dim: u64, aligned_query: &[T]) 
    where
        T: Copy + Into<f32>,
    {
        self.data_dim = data_dim;
        let dim = data_dim as usize;
        
        // Ensure buffers are large enough
        if self.aligned_query_float.len() < dim {
            self.aligned_query_float.resize(dim, 0.0);
        }
        if self.rotated_query.len() < dim {
            self.rotated_query.resize(dim, 0.0);
        }
        
        // Copy and convert query data
        for i in 0..dim {
            self.aligned_query_float[i] = aligned_query[i].into();
            self.rotated_query[i] = aligned_query[i].into();
        }
    }

    /// Get mutable reference to aligned query float buffer
    pub fn aligned_query_float_mut(&mut self) -> &mut [f32] {
        &mut self.aligned_query_float
    }

    /// Get reference to aligned query float buffer
    pub fn aligned_query_float(&self) -> &[f32] {
        &self.aligned_query_float
    }

    /// Get mutable reference to rotated query buffer
    pub fn rotated_query_mut(&mut self) -> &mut [f32] {
        &mut self.rotated_query
    }

    /// Get reference to rotated query buffer
    pub fn rotated_query(&self) -> &[f32] {
        &self.rotated_query
    }

    /// Get mutable reference to PQ table distance scratch
    pub fn aligned_pqtable_dist_scratch_mut(&mut self) -> &mut [f32] {
        &mut self.aligned_pqtable_dist_scratch
    }

    /// Get reference to PQ table distance scratch
    pub fn aligned_pqtable_dist_scratch(&self) -> &[f32] {
        &self.aligned_pqtable_dist_scratch
    }

    /// Get mutable reference to aligned distance scratch
    pub fn aligned_dist_scratch_mut(&mut self) -> &mut [f32] {
        &mut self.aligned_dist_scratch
    }

    /// Get reference to aligned distance scratch
    pub fn aligned_dist_scratch(&self) -> &[f32] {
        &self.aligned_dist_scratch
    }

    /// Get mutable reference to aligned PQ coordinate scratch
    pub fn aligned_pq_coord_scratch_mut(&mut self) -> &mut [u8] {
        &mut self.aligned_pq_coord_scratch
    }

    /// Get reference to aligned PQ coordinate scratch
    pub fn aligned_pq_coord_scratch(&self) -> &[u8] {
        &self.aligned_pq_coord_scratch
    }

    /// Resize distance scratch buffer if needed
    pub fn ensure_dist_scratch_size(&mut self, size: usize) {
        if self.aligned_dist_scratch.len() < size {
            self.aligned_dist_scratch.resize(size, 0.0);
        }
    }

    /// Resize PQ coordinate scratch buffer if needed
    pub fn ensure_pq_coord_scratch_size(&mut self, size: usize) {
        if self.aligned_pq_coord_scratch.len() < size {
            self.aligned_pq_coord_scratch.resize(size, 0);
        }
    }

    /// Reset all scratch buffers
    pub fn reset(&mut self) {
        self.aligned_query_float.fill(0.0);
        self.rotated_query.fill(0.0);
        self.aligned_pqtable_dist_scratch.fill(0.0);
        self.aligned_dist_scratch.fill(0.0);
        self.aligned_pq_coord_scratch.fill(0);
    }

    /// Get the data dimension
    pub fn data_dim(&self) -> u64 {
        self.data_dim
    }

    /// Get the number of chunks
    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_scratch() {
        let graph_degree = 512;
        let aligned_dim = 8;

        let mut pq_scratch: PQScratch = PQScratch::new(graph_degree, aligned_dim).unwrap();

        // Check alignment
        assert_eq!(
            (pq_scratch.aligned_pqtable_dist_scratch.as_ptr() as usize) % 256,
            0
        );
        assert_eq!((pq_scratch.aligned_dist_scratch.as_ptr() as usize) % 256, 0);
        assert_eq!(
            (pq_scratch.aligned_pq_coord_scratch.as_ptr() as usize) % 256,
            0
        );
        assert_eq!((pq_scratch.rotated_query.as_ptr() as usize) % 32, 0);
        assert_eq!((pq_scratch.aligned_query_float.as_ptr() as usize) % 32, 0);

        // Test set() method
        let query = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let norm = 2.0f32;
        pq_scratch.set::<u8>(query.len(), &query, norm);

        (0..query.len()).for_each(|i| {
            assert_eq!(pq_scratch.rotated_query[i], query[i] as f32 / norm);
            assert_eq!(pq_scratch.aligned_query_float[i], query[i] as f32 / norm);
        });
    }

    #[test]
    fn test_pq_query_scratch_creation() {
        let scratch = PQQueryScratch::new(128, 8);
        
        assert_eq!(scratch.data_dim(), 128);
        assert_eq!(scratch.n_chunks(), 8);
        assert_eq!(scratch.aligned_query_float.len(), 128);
        assert_eq!(scratch.rotated_query.len(), 128);
        assert_eq!(scratch.aligned_pqtable_dist_scratch.len(), 256 * 8);
    }

    #[test]
    fn test_pq_query_scratch_initialize() {
        let mut scratch = PQQueryScratch::new(64, 4);
        let query_data = vec![1.0f32; 64];
        
        scratch.initialize(64, &query_data);
        
        assert_eq!(scratch.data_dim(), 64);
        assert_eq!(scratch.aligned_query_float[0], 1.0);
        assert_eq!(scratch.rotated_query[0], 1.0);
    }

    #[test]
    fn test_pq_query_scratch_resize() {
        let mut scratch = PQQueryScratch::new(64, 4);
        
        scratch.ensure_dist_scratch_size(2000);
        assert_eq!(scratch.aligned_dist_scratch.len(), 2000);
        
        scratch.ensure_pq_coord_scratch_size(3000);
        assert_eq!(scratch.aligned_pq_coord_scratch.len(), 3000);
    }

    #[test]
    fn test_pq_query_scratch_reset() {
        let mut scratch = PQQueryScratch::new(64, 4);
        
        // Fill with some data
        scratch.aligned_query_float.fill(1.0);
        scratch.rotated_query.fill(2.0);
        scratch.aligned_pqtable_dist_scratch.fill(3.0);
        
        scratch.reset();
        
        assert_eq!(scratch.aligned_query_float[0], 0.0);
        assert_eq!(scratch.rotated_query[0], 0.0);
        assert_eq!(scratch.aligned_pqtable_dist_scratch[0], 0.0);
    }
}
