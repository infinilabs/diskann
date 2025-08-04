// DiskANN Benchmarks Library
// This crate provides benchmarking utilities for the DiskANN project

pub mod utils;

/// Common utilities for benchmarks
pub mod common {
    use rand::Rng;
    
    /// Generate random vectors for benchmarking
    pub fn generate_random_vectors(dimension: usize, count: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut vectors = Vec::with_capacity(count);
        
        for _ in 0..count {
            let mut vector = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                vector.push(rng.gen_range(-1.0..1.0));
            }
            vectors.push(vector);
        }
        
        vectors
    }
    
    /// Generate random query vectors
    pub fn generate_query_vectors(dimension: usize, count: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut queries = Vec::with_capacity(count);
        
        for _ in 0..count {
            let mut query = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                query.push(rng.gen_range(-1.0..1.0));
            }
            queries.push(query);
        }
        
        queries
    }
} 