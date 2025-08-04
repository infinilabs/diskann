/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![allow(dead_code)] // Todo: Remove this when the disk index query code is complete.
use std::sync::Arc;

use crate::model::scratch::ssd_query_scratch::SSDQueryScratch;
use crate::model::scratch::pq_scratch::PQQueryScratch;
use crate::model::scratch::ssd_io_context::IOContext;

/// Thread-specific data for SSD-based search operations
pub struct SSDThreadData<T> {
    /// I/O context for this thread
    pub ctx: IOContext,
    /// Query scratch space
    pub scratch: SSDQueryScratch<T>,
}

impl<T> SSDThreadData<T>
where
    T: Default + Copy,
{
    /// Create a new SSD thread data instance
    pub fn new(
        file_path: &str,
        aligned_dim: usize,
        sector_size: usize,
        max_queue_size: usize,
        data_dim: u64,
        n_chunks: u64,
    ) -> crate::common::ANNResult<Self> {
        let ctx = IOContext::new(file_path)?;
        let pq_scratch = PQQueryScratch::new(data_dim, n_chunks);
        let scratch = SSDQueryScratch::new(aligned_dim, sector_size, max_queue_size, pq_scratch);

        Ok(Self { ctx, scratch })
    }

    /// Get mutable reference to the I/O context
    pub fn ctx_mut(&mut self) -> &mut IOContext {
        &mut self.ctx
    }

    /// Get reference to the I/O context
    pub fn ctx(&self) -> &IOContext {
        &self.ctx
    }

    /// Get mutable reference to the scratch space
    pub fn scratch_mut(&mut self) -> &mut SSDQueryScratch<T> {
        &mut self.scratch
    }

    /// Get reference to the scratch space
    pub fn scratch(&self) -> &SSDQueryScratch<T> {
        &self.scratch
    }

    /// Reset the scratch space for a new query
    pub fn reset_scratch(&mut self) {
        self.scratch.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_ssd_thread_data_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");
        
        // Create a test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        let thread_data = SSDThreadData::<f32>::new(
            file_path.to_str().unwrap(),
            128,
            4096,
            100,
            128,
            8,
        );

        assert!(thread_data.is_ok());
        let thread_data = thread_data.unwrap();
        
        assert_eq!(thread_data.scratch.aligned_query_T.len(), 128);
        assert_eq!(thread_data.scratch.coord_scratch.len(), 128);
        assert_eq!(thread_data.scratch.sector_scratch.len(), 4096);
    }

    #[test]
    fn test_ssd_thread_data_reset() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");
        
        // Create a test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        let mut thread_data = SSDThreadData::<f32>::new(
            file_path.to_str().unwrap(),
            128,
            4096,
            100,
            128,
            8,
        ).unwrap();

        // Add some data to scratch
        thread_data.scratch.mark_visited(1);
        thread_data.scratch.mark_visited(2);
        thread_data.scratch.retset.insert(crate::model::neighbor::Neighbor::new(1, 0.5));

        // Reset
        thread_data.reset_scratch();

        assert_eq!(thread_data.scratch.visited_count(), 0);
        assert_eq!(thread_data.scratch.retset.size(), 0);
        assert_eq!(thread_data.scratch.sector_idx, 0);
    }
}
