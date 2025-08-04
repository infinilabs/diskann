/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

use crate::common::{ANNError, ANNResult};

/// Ultra-high-performance vectorized data structure with aligned memory
pub struct VectorizedStorage {
    /// Aligned data buffer (64-byte aligned for AVX-512)
    data: *mut f32,
    /// Number of elements
    len: usize,
    /// Capacity in elements
    capacity: usize,
    /// Memory layout
    layout: Layout,
}

impl VectorizedStorage {
    /// Create new vectorized storage with 64-byte alignment
    pub fn new(capacity: usize) -> ANNResult<Self> {
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<f32>(),
            64, // 64-byte alignment for AVX-512
        ).map_err(|e| ANNError::log_index_error(format!("Layout error: {}", e)))?;

        let data = unsafe { alloc_zeroed(layout) } as *mut f32;
        if data.is_null() {
            return Err(ANNError::log_index_error("Failed to allocate aligned memory".to_string()));
        }

        Ok(Self {
            data,
            len: 0,
            capacity,
            layout,
        })
    }

    /// Get mutable slice to the data
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }

    /// Get immutable slice to the data
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Resize the storage
    pub fn resize(&mut self, new_len: usize) -> ANNResult<()> {
        if new_len > self.capacity {
            return Err(ANNError::log_index_error("Cannot resize beyond capacity".to_string()));
        }
        self.len = new_len;
        Ok(())
    }

    /// Ultra-fast copy data from slice with SIMD optimization
    pub fn copy_from_slice_simd(&mut self, src: &[f32]) -> ANNResult<()> {
        if src.len() > self.capacity {
            return Err(ANNError::log_index_error("Source too large for storage".to_string()));
        }

        self.len = src.len();
        
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                if is_x86_feature_detected!("avx512f") {
                    // Ultra-fast AVX-512 copying with larger chunks
                    let aligned_len = src.len() - (src.len() % 32); // Process 32 elements at once
                    for i in (0..aligned_len).step_by(32) {
                        let src_vec1 = _mm512_loadu_ps(&src[i]);
                        let src_vec2 = _mm512_loadu_ps(&src[i + 16]);
                        _mm512_store_ps(&mut self.data.add(i), src_vec1);
                        _mm512_store_ps(&mut self.data.add(i + 16), src_vec2);
                    }
                    // Copy remaining elements
                    for i in aligned_len..src.len() {
                        self.data.add(i).write(src[i]);
                    }
                } else if is_x86_feature_detected!("avx2") {
                    // Fast AVX2 copying with larger chunks
                    let aligned_len = src.len() - (src.len() % 16); // Process 16 elements at once
                    for i in (0..aligned_len).step_by(16) {
                        let src_vec1 = _mm256_loadu_ps(&src[i]);
                        let src_vec2 = _mm256_loadu_ps(&src[i + 8]);
                        _mm256_store_ps(&mut self.data.add(i), src_vec1);
                        _mm256_store_ps(&mut self.data.add(i + 8), src_vec2);
                    }
                    // Copy remaining elements
                    for i in aligned_len..src.len() {
                        self.data.add(i).write(src[i]);
                    }
                } else {
                    // Fallback to standard copy
                    ptr::copy_nonoverlapping(src.as_ptr(), self.data, src.len());
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), self.data, src.len());
            }
        }

        Ok(())
    }

    /// Ultra-fast batch copy with SIMD optimization
    pub fn copy_batch_simd(&mut self, src: &[f32], batch_size: usize) -> ANNResult<()> {
        if src.len() > self.capacity {
            return Err(ANNError::log_index_error("Source too large for storage".to_string()));
        }

        self.len = src.len();
        
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                if is_x86_feature_detected!("avx512f") {
                    // Process in larger batches for better cache performance
                    let batch_aligned = batch_size - (batch_size % 32);
                    for batch_start in (0..src.len()).step_by(batch_size) {
                        let batch_end = std::cmp::min(batch_start + batch_size, src.len());
                        let batch_len = batch_end - batch_start;
                        
                        // Process aligned portion of batch
                        let aligned_batch_len = batch_len - (batch_len % 32);
                        for i in (0..aligned_batch_len).step_by(32) {
                            let src_vec1 = _mm512_loadu_ps(&src[batch_start + i]);
                            let src_vec2 = _mm512_loadu_ps(&src[batch_start + i + 16]);
                            _mm512_store_ps(&mut self.data.add(batch_start + i), src_vec1);
                            _mm512_store_ps(&mut self.data.add(batch_start + i + 16), src_vec2);
                        }
                        
                        // Process remaining elements in batch
                        for i in aligned_batch_len..batch_len {
                            self.data.add(batch_start + i).write(src[batch_start + i]);
                        }
                    }
                } else if is_x86_feature_detected!("avx2") {
                    // Process in larger batches for better cache performance
                    let batch_aligned = batch_size - (batch_size % 16);
                    for batch_start in (0..src.len()).step_by(batch_size) {
                        let batch_end = std::cmp::min(batch_start + batch_size, src.len());
                        let batch_len = batch_end - batch_start;
                        
                        // Process aligned portion of batch
                        let aligned_batch_len = batch_len - (batch_len % 16);
                        for i in (0..aligned_batch_len).step_by(16) {
                            let src_vec1 = _mm256_loadu_ps(&src[batch_start + i]);
                            let src_vec2 = _mm256_loadu_ps(&src[batch_start + i + 8]);
                            _mm256_store_ps(&mut self.data.add(batch_start + i), src_vec1);
                            _mm256_store_ps(&mut self.data.add(batch_start + i + 8), src_vec2);
                        }
                        
                        // Process remaining elements in batch
                        for i in aligned_batch_len..batch_len {
                            self.data.add(batch_start + i).write(src[batch_start + i]);
                        }
                    }
                } else {
                    // Fallback to standard copy
                    ptr::copy_nonoverlapping(src.as_ptr(), self.data, src.len());
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), self.data, src.len());
            }
        }

        Ok(())
    }
}

impl Drop for VectorizedStorage {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data as *mut u8, self.layout);
        }
    }
}

/// Ultra-high-performance batch processor with memory pooling and aggressive batching
pub struct BatchProcessor {
    /// Thread-local storage pools for memory reuse
    storage_pools: Arc<Mutex<Vec<VectorizedStorage>>>,
    /// Ultra-large batch size for maximum throughput
    batch_size: usize,
    /// Memory pool for temporary buffers
    temp_buffers: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl BatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        Self {
            storage_pools: Arc::new(Mutex::new(Vec::new())),
            batch_size,
            temp_buffers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get a temporary buffer from the pool
    pub fn get_temp_buffer(&self, size: usize) -> Vec<f32> {
        let mut buffers = self.temp_buffers.lock().unwrap();
        if let Some(mut buffer) = buffers.pop() {
            if buffer.len() >= size {
                buffer.truncate(size);
                buffer
            } else {
                vec![0.0; size]
            }
        } else {
            vec![0.0; size]
        }
    }

    /// Return a temporary buffer to the pool
    pub fn return_temp_buffer(&self, mut buffer: Vec<f32>) {
        buffer.clear();
        let mut buffers = self.temp_buffers.lock().unwrap();
        buffers.push(buffer);
    }

    /// Ultra-fast k-means batch processing with vectorized operations
    pub fn process_kmeans_batch(
        &self,
        data: &[f32],
        centers: &mut [f32],
        num_points: usize,
        num_centers: usize,
        dim: usize,
    ) -> ANNResult<Vec<f32>> {
        let mut distances = self.get_temp_buffer(num_points * num_centers);
        
        // Use ultra-aggressive vectorized batch processing
        calc_distances_batch_ultra_vectorized(data, centers, num_points, num_centers, dim, &mut distances);
        
        Ok(distances)
    }

    /// Ultra-fast batch distance calculation with memory pooling
    pub fn process_distance_batch(
        &self,
        queries: &[f32],
        targets: &[f32],
        num_queries: usize,
        num_targets: usize,
        dim: usize,
    ) -> ANNResult<Vec<f32>> {
        let mut distances = self.get_temp_buffer(num_queries * num_targets);
        
        // Use ultra-aggressive vectorized batch processing
        calc_distances_batch_ultra_vectorized(queries, targets, num_queries, num_targets, dim, &mut distances);
        
        Ok(distances)
    }

    /// Ultra-fast batch vector operations
    pub fn process_vector_batch(
        &self,
        vectors: &[f32],
        num_vectors: usize,
        dim: usize,
        operation: &dyn Fn(&[f32], &mut [f32]),
    ) -> ANNResult<Vec<f32>> {
        let mut result = self.get_temp_buffer(num_vectors * dim);
        
        // Process in ultra-large batches for maximum throughput
        let ultra_batch_size = 8192; // 8KB batches for optimal cache performance
        for batch_start in (0..num_vectors).step_by(ultra_batch_size) {
            let batch_end = std::cmp::min(batch_start + ultra_batch_size, num_vectors);
            let batch_size = batch_end - batch_start;
            
            let batch_input = &vectors[batch_start * dim..batch_end * dim];
            let batch_output = &mut result[batch_start * dim..batch_end * dim];
            
            operation(batch_input, batch_output);
        }
        
        Ok(result)
    }
}

/// Memory-mapped file reader for high-performance I/O with batching
pub struct MemoryMappedReader {
    /// Memory-mapped file data
    data: *const u8,
    /// File size
    size: usize,
    /// Batch size for I/O operations
    batch_size: usize,
}

impl MemoryMappedReader {
    #[cfg(target_os = "linux")]
    pub fn new(file_path: &str) -> ANNResult<Self> {
        use std::fs::File;
        use std::os::unix::io::AsRawFd;
        
        let file = File::open(file_path)?;
        let size = file.metadata()?.len() as usize;
        
        // Memory map the file
        let data = unsafe {
            let fd = file.as_raw_fd();
            let ptr = mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            );
            
            if ptr == libc::MAP_FAILED {
                return Err(ANNError::log_index_error("Failed to memory map file".to_string()));
            }
            
            ptr as *const u8
        };
        
        Ok(Self { 
            data, 
            size,
            batch_size: 1024 * 1024, // 1MB batches for optimal I/O performance
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn new(_file_path: &str) -> ANNResult<Self> {
        Err(ANNError::log_index_error("Memory mapping not supported on this platform".to_string()))
    }
    
    /// Read f32 data from memory-mapped file with batching
    pub fn read_f32_slice(&self, offset: usize, len: usize) -> &[f32] {
        let start = unsafe { self.data.add(offset) };
        unsafe {
            std::slice::from_raw_parts(start as *const f32, len)
        }
    }

    /// Read f32 data in batches for better performance
    pub fn read_f32_batch(&self, offset: usize, len: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(len);
        
        // Read in batches for better cache performance
        for batch_start in (0..len).step_by(self.batch_size) {
            let batch_end = std::cmp::min(batch_start + self.batch_size, len);
            let batch_len = batch_end - batch_start;
            
            let batch_data = self.read_f32_slice(offset + batch_start * 4, batch_len);
            result.extend_from_slice(batch_data);
        }
        
        result
    }
}

impl Drop for MemoryMappedReader {
    #[cfg(target_os = "linux")]
    fn drop(&mut self) {
        unsafe {
            munmap(self.data as *mut libc::c_void, self.size);
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn drop(&mut self) {
        // No-op on non-Linux platforms
    }
}

// Import SIMD intrinsics
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Import libc for memory mapping
#[cfg(target_os = "linux")]
extern "C" {
    fn mmap(
        addr: *mut libc::c_void,
        length: libc::size_t,
        prot: libc::c_int,
        flags: libc::c_int,
        fd: libc::c_int,
        offset: libc::off_t,
    ) -> *mut libc::c_void;
    
    fn munmap(addr: *mut libc::c_void, length: libc::size_t) -> libc::c_int;
}

// Import the ultra-vectorized distance calculation function
use crate::utils::math_util::calc_distances_batch_ultra_vectorized;

/// Ultra-aggressive batch processor for maximum throughput
pub struct UltraBatchProcessor {
    /// Thread-local storage pools for memory reuse
    storage_pools: Arc<Mutex<Vec<VectorizedStorage>>>,
    /// Ultra-large batch size for maximum throughput
    batch_size: usize,
    /// Memory pool for temporary buffers
    temp_buffers: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl UltraBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        Self {
            storage_pools: Arc::new(Mutex::new(Vec::new())),
            batch_size,
            temp_buffers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Ultra-fast batch distance calculation with maximum SIMD utilization
    pub fn process_distance_batch_ultra(
        &self,
        queries: &[f32],
        targets: &[f32],
        num_queries: usize,
        num_targets: usize,
        dim: usize,
    ) -> ANNResult<Vec<f32>> {
        let mut distances = vec![0.0; num_queries * num_targets];
        
        // Use ultra-aggressive vectorized batch processing
        calc_distances_batch_ultra_vectorized(queries, targets, num_queries, num_targets, dim, &mut distances);
        
        Ok(distances)
    }

    /// Ultra-fast batch vector operations with memory pooling
    pub fn process_vector_batch_ultra(
        &self,
        vectors: &[f32],
        num_vectors: usize,
        dim: usize,
        operation: &dyn Fn(&[f32], &mut [f32]),
    ) -> ANNResult<Vec<f32>> {
        let mut result = vec![0.0; num_vectors * dim];
        
        // Process in ultra-large batches for maximum throughput
        let ultra_batch_size = 16384; // 16KB batches for optimal cache performance
        for batch_start in (0..num_vectors).step_by(ultra_batch_size) {
            let batch_end = std::cmp::min(batch_start + ultra_batch_size, num_vectors);
            let batch_size = batch_end - batch_start;
            
            let batch_input = &vectors[batch_start * dim..batch_end * dim];
            let batch_output = &mut result[batch_start * dim..batch_end * dim];
            
            operation(batch_input, batch_output);
        }
        
        Ok(result)
    }
} 