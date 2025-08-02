/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]
#![cfg_attr(test, allow(clippy::unused_io_amount))]
#![doc = include_str!("../../README.md")]

//! # DiskANN - Approximate Nearest Neighbor Search in Rust
//!
//! DiskANN is a high-performance, scalable approximate nearest neighbor (ANN) search library
//! implemented in Rust. It provides both in-memory and disk-based indexing for large-scale
//! vector search with high recall and low latency.
//!
//! ## Key Features
//!
//! - **Pure Rust implementation** - No C/C++ dependencies, leveraging Rust's memory safety
//! - **Disk-based indexing** - Support for datasets that don't fit in memory
//! - **High performance** - Optimized for fast approximate nearest neighbor search
//! - **Parallel processing** - Leverages Rust's concurrency features
//! - **Multiple distance metrics** - Support for L2, cosine, and other distance functions
//! - **Flexible data types** - Support for f32, f16, and other numeric types
//!
//! ## Quick Start
//!
//! ```rust
//! use diskann::{IndexBuilder, Metric, SearchParams};
//!
//! // Create an in-memory index
//! let mut index = IndexBuilder::new()
//!     .with_dimension(128)
//!     .with_metric(Metric::L2)
//!     .with_max_degree(64)
//!     .with_search_list_size(100)
//!     .build_in_memory()?;
//!
//! // Insert vectors
//! let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
//! index.insert_batch(&vectors)?;
//!
//! // Search for nearest neighbors
//! let query = vec![1.0, 2.0, 3.0];
//! let results = index.search(&query, 5, 50)?;
//!
//! println!("Found {} nearest neighbors", results.len());
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for complete working examples:
//! - `basic_usage.rs` - Basic in-memory index usage
//! - `disk_index.rs` - Disk-based index for large datasets
//! - `batch_operations.rs` - Batch insert and search operations
//! - `custom_metrics.rs` - Using custom distance metrics

pub mod algorithm;
pub mod common;
pub mod disk_search;
pub mod index;
pub mod instrumentation;
pub mod model;
pub mod utils;

#[cfg(feature = "disk_store")]
pub mod storage;

#[cfg(test)]
pub mod test_utils;

// Re-export commonly used types for convenience
pub use common::{ANNError, ANNResult};
pub use model::configuration::index_configuration::IndexConfiguration;
pub use model::configuration::index_write_parameters::IndexWriteParametersBuilder;
pub use vector::Metric;

/// High-level index builder for easy configuration
pub struct IndexBuilder {
    dimension: usize,
    metric: Metric,
    max_degree: u32,
    search_list_size: u32,
    alpha: f32,
    num_threads: u32,
    use_opq: bool,
}

impl IndexBuilder {
    /// Create a new index builder with default settings
    pub fn new() -> Self {
        Self {
            dimension: 128,
            metric: Metric::L2,
            max_degree: 64,
            search_list_size: 100,
            alpha: 1.2,
            num_threads: 1,
            use_opq: false,
        }
    }

    /// Set the dimension of vectors
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Set the distance metric
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the maximum degree of graph nodes
    pub fn with_max_degree(mut self, max_degree: u32) -> Self {
        self.max_degree = max_degree;
        self
    }

    /// Set the search list size
    pub fn with_search_list_size(mut self, search_list_size: u32) -> Self {
        self.search_list_size = search_list_size;
        self
    }

    /// Set the alpha parameter (controls graph density)
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the number of threads
    pub fn with_num_threads(mut self, num_threads: u32) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Enable/disable OPQ compression
    pub fn with_opq(mut self, use_opq: bool) -> Self {
        self.use_opq = use_opq;
        self
    }

    /// Build an in-memory index
    pub fn build_in_memory<T>(self) -> ANNResult<InMemoryIndex<T>>
    where
        T: Default + Copy + Sync + Send + Into<f32> + 'static,
        [T; 104]: vector::FullPrecisionDistance<T, 104>,
        [T; 128]: vector::FullPrecisionDistance<T, 128>,
        [T; 256]: vector::FullPrecisionDistance<T, 256>,
        [T; 512]: vector::FullPrecisionDistance<T, 512>,
    {
        InMemoryIndex::new(
            self.dimension,
            self.metric,
            self.max_degree,
            self.search_list_size,
            self.alpha,
            self.num_threads,
            self.use_opq,
        )
    }

    /// Build a disk-based index
    pub fn build_disk_index<T>(self, index_path: &str) -> ANNResult<DiskIndex<T>>
    where
        T: Default + Copy + Sync + Send + Into<f32> + 'static,
    {
        DiskIndex::new(
            index_path,
            self.dimension,
            self.metric,
            self.max_degree,
            self.search_list_size,
            self.alpha,
            self.num_threads,
            self.use_opq,
        )
    }
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level in-memory index for vector search
pub struct InMemoryIndex<T> {
    inner: Box<dyn index::ANNInmemIndex<T>>,
}

impl<T> InMemoryIndex<T>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
    [T; 104]: vector::FullPrecisionDistance<T, 104>,
    [T; 128]: vector::FullPrecisionDistance<T, 128>,
    [T; 256]: vector::FullPrecisionDistance<T, 256>,
    [T; 512]: vector::FullPrecisionDistance<T, 512>,
{
    /// Create a new in-memory index
    pub fn new(
        dimension: usize,
        metric: Metric,
        max_degree: u32,
        search_list_size: u32,
        alpha: f32,
        num_threads: u32,
        use_opq: bool,
    ) -> ANNResult<Self> {
        let write_params = IndexWriteParametersBuilder::new(search_list_size, max_degree)
            .with_alpha(alpha)
            .with_saturate_graph(false)
            .with_num_threads(num_threads)
            .build();

        let config = IndexConfiguration::new(
            metric,
            dimension,
            utils::round_up(dimension as u64, 8) as usize,
            0, // Will be set during build
            false,
            0,
            use_opq,
            0,
            2.0,
            write_params,
        );

        let inner = index::create_inmem_index::<T>(config)?;
        Ok(Self { inner })
    }

    /// Insert a single vector
    pub fn insert(&mut self, vector: &[T]) -> ANNResult<()> {
        let vectors = vec![vector.to_vec()];
        self.inner.insert_vector(&vectors)?;
        Ok(())
    }

    /// Insert multiple vectors in batch
    pub fn insert_batch(&mut self, vectors: &[Vec<T>]) -> ANNResult<()> {
        let vectors = vectors.to_vec();
        self.inner.insert_vector(&vectors)?;
        Ok(())
    }

    /// Build the index from vectors
    pub fn build(&mut self, vectors: &[Vec<T>]) -> ANNResult<()> {
        let vectors = vectors.to_vec();
        self.inner.build_vector(&vectors)
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[T], k: usize, l: u32) -> ANNResult<Vec<SearchResult>> {
        let mut indices = vec![0; k];
        let mut distances = vec![0.0; k];

        self.inner
            .search_with_distance(query, k, l, &mut indices, &mut distances)?;

        let results = indices
            .into_iter()
            .zip(distances)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect();

        Ok(results)
    }

    /// Save the index to disk
    pub fn save(&mut self, path: &str) -> ANNResult<()> {
        self.inner.save(path)
    }

    /// Load the index from disk
    pub fn load(&mut self, _path: &str) -> ANNResult<()> {
        // Note: This would need to be implemented in the underlying index
        todo!("Load functionality not yet implemented")
    }
}

/// High-level disk-based index for large datasets
pub struct DiskIndex<T> {
    inner: Box<dyn index::ann_disk_index::ANNDiskIndex<T>>,
}

impl<T> DiskIndex<T>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
{
    /// Create a new disk-based index
    pub fn new(
        _index_path: &str,
        _dimension: usize,
        _metric: Metric,
        _max_degree: u32,
        _search_list_size: u32,
        _alpha: f32,
        _num_threads: u32,
        _use_opq: bool,
    ) -> ANNResult<Self> {
        // Implementation for disk index creation
        todo!("Implement disk index creation")
    }

    /// Search for nearest neighbors
    pub fn search(&self, _query: &[T], _k: usize, _l: u32) -> ANNResult<Vec<SearchResult>> {
        // Implementation for disk search
        todo!("Implement disk search")
    }
}

/// Result of a search operation
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The ID of the found vector
    pub id: u32,
    /// The distance to the query vector
    pub distance: f32,
}

/// Search parameters for fine-tuning search behavior
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Number of results to return
    pub k: usize,
    /// Search list size (beam width)
    pub l: u32,
    /// Whether to return distances
    pub return_distances: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            l: 50,
            return_distances: true,
        }
    }
}
