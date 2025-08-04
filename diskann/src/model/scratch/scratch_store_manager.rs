/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::sync::Arc;
use crossbeam::queue::ArrayQueue;

use crate::common::ANNResult;
use crate::model::scratch::ssd_thread_data::SSDThreadData;

/// Manager for thread-safe scratch space allocation
pub struct ScratchStoreManager<T> {
    /// Queue of available scratch spaces
    queue: Arc<ArrayQueue<T>>,
    /// Maximum number of scratch spaces
    max_count: usize,
}

impl<T> ScratchStoreManager<T>
where
    T: Send + Sync,
{
    /// Create a new scratch store manager
    pub fn new(max_count: usize) -> Self {
        let queue = Arc::new(ArrayQueue::new(max_count));
        
        // Pre-populate with scratch spaces
        for _ in 0..max_count {
            // This will be implemented by the specific type
            // For now, we'll create empty spaces
        }
        
        Self { queue, max_count }
    }

    /// Get a scratch space from the pool
    pub fn scratch_space(&self) -> T {
        // For now, return a default instance
        // This will be properly implemented with the specific type
        unimplemented!("Scratch space allocation needs to be implemented for specific types")
    }

    /// Return a scratch space to the pool
    pub fn return_scratch_space(&self, _scratch: T) {
        // This will be implemented by the specific type
        // For now, just drop the scratch space
    }

    /// Get the maximum number of scratch spaces
    pub fn max_count(&self) -> usize {
        self.max_count
    }

    /// Get the current number of available scratch spaces
    pub fn available_count(&self) -> usize {
        self.queue.len()
    }

    /// Check if scratch spaces are available
    pub fn has_available(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Destroy the scratch store manager
    pub fn destroy(&self) {
        // Clear the queue
        while self.queue.pop().is_some() {
            // Drain all items
        }
    }
}

/// Specialized scratch store manager for SSD thread data
pub struct SSDScratchStoreManager {
    /// Queue of available SSD thread data instances
    queue: Arc<ArrayQueue<SSDThreadData<f32>>>,
    /// Maximum number of thread data instances
    max_count: usize,
    /// File path for I/O operations
    file_path: String,
    /// Aligned dimension for vectors
    aligned_dim: usize,
    /// Sector size for I/O operations
    sector_size: usize,
    /// Maximum queue size for priority queue
    max_queue_size: usize,
    /// Data dimension
    data_dim: u64,
    /// Number of PQ chunks
    n_chunks: u64,
}

impl SSDScratchStoreManager {
    /// Create a new SSD scratch store manager
    pub fn new(
        max_count: usize,
        file_path: String,
        aligned_dim: usize,
        sector_size: usize,
        max_queue_size: usize,
        data_dim: u64,
        n_chunks: u64,
    ) -> ANNResult<Self> {
        let queue = Arc::new(ArrayQueue::new(max_count));
        
        // Pre-populate with SSD thread data instances
        for _ in 0..max_count {
            let thread_data = SSDThreadData::new(
                &file_path,
                aligned_dim,
                sector_size,
                max_queue_size,
                data_dim,
                n_chunks,
            )?;
            
            if queue.push(thread_data).is_err() {
                return Err(crate::common::ANNError::log_index_error(
                    "Failed to initialize scratch store manager".to_string(),
                ));
            }
        }
        
        Ok(Self {
            queue,
            max_count,
            file_path,
            aligned_dim,
            sector_size,
            max_queue_size,
            data_dim,
            n_chunks,
        })
    }

    /// Get a scratch space from the pool
    pub fn scratch_space(&self) -> SSDThreadData<f32> {
        if let Some(thread_data) = self.queue.pop() {
            thread_data
        } else {
            // Create a new instance if none are available
            // This should not happen in normal operation
            SSDThreadData::new(
                &self.file_path,
                self.aligned_dim,
                self.sector_size,
                self.max_queue_size,
                self.data_dim,
                self.n_chunks,
            ).unwrap_or_else(|_| {
                panic!("Failed to create SSD thread data")
            })
        }
    }

    /// Return a scratch space to the pool
    pub fn return_scratch_space(&self, mut thread_data: SSDThreadData<f32>) {
        // Reset the scratch space before returning
        thread_data.reset_scratch();
        
        // Try to return to the queue
        if self.queue.push(thread_data).is_err() {
            // Queue is full, just drop the thread data
            // This should not happen in normal operation
        }
    }

    /// Get the maximum number of scratch spaces
    pub fn max_count(&self) -> usize {
        self.max_count
    }

    /// Get the current number of available scratch spaces
    pub fn available_count(&self) -> usize {
        self.queue.len()
    }

    /// Check if scratch spaces are available
    pub fn has_available(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Destroy the scratch store manager
    pub fn destroy(&self) {
        // Clear the queue
        while self.queue.pop().is_some() {
            // Drain all items
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_ssd_scratch_store_manager_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");
        
        // Create a test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        let manager = SSDScratchStoreManager::new(
            5,
            file_path.to_str().unwrap().to_string(),
            128,
            4096,
            100,
            128,
            8,
        );

        assert!(manager.is_ok());
        let manager = manager.unwrap();
        
        assert_eq!(manager.max_count(), 5);
        assert_eq!(manager.available_count(), 5);
        assert!(manager.has_available());
    }

    #[test]
    fn test_ssd_scratch_store_manager_scratch_space_allocation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");
        
        // Create a test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        let manager = SSDScratchStoreManager::new(
            3,
            file_path.to_str().unwrap().to_string(),
            128,
            4096,
            100,
            128,
            8,
        ).unwrap();

        // Allocate scratch spaces
        let scratch1 = manager.scratch_space();
        let scratch2 = manager.scratch_space();
        let scratch3 = manager.scratch_space();

        assert_eq!(manager.available_count(), 0);
        assert!(!manager.has_available());

        // Return scratch spaces
        manager.return_scratch_space(scratch1);
        manager.return_scratch_space(scratch2);
        manager.return_scratch_space(scratch3);

        assert_eq!(manager.available_count(), 3);
        assert!(manager.has_available());
    }
}
