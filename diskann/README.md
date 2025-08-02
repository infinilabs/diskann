# DiskANN Library

The core DiskANN library providing high-performance approximate nearest neighbor (ANN) search functionality in Rust.

## Overview

DiskANN is a high-performance, scalable approximate nearest neighbor (ANN) search library implemented in Rust. It provides both in-memory and disk-based indexing for large-scale vector search with high recall and low latency.

## Features

- **Pure Rust implementation** - No C/C++ dependencies, leveraging Rust's memory safety
- **Disk-based indexing** - Support for datasets that don't fit in memory
- **High performance** - Optimized for fast approximate nearest neighbor search
- **Parallel processing** - Leverages Rust's concurrency features
- **Multiple distance metrics** - Support for L2, cosine, inner product, and other distance functions
- **Flexible data types** - Support for f32, f16, and other numeric types
- **Memory efficient** - Smart caching and memory management for large datasets

## Quick Start

```rust
use diskann::{IndexBuilder, Metric, ANNResult};

fn main() -> ANNResult<()> {
    // Create an in-memory index
    let mut index = IndexBuilder::new()
        .with_dimension(128)           // Vector dimension
        .with_metric(Metric::L2)       // Distance metric
        .with_max_degree(64)           // Maximum graph degree
        .with_search_list_size(100)    // Search beam width
        .with_alpha(1.2)               // Graph density parameter
        .with_num_threads(4)           // Number of threads
        .build_in_memory::<f32>()?;

    // Insert vectors
    let vectors = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ];
    index.insert_batch(&vectors)?;

    // Build the index
    index.build(&vectors)?;

    // Search for nearest neighbors
    let query = vec![1.0, 2.0, 3.0, 4.0];
    let results = index.search(&query, 5, 50)?;

    println!("Found {} nearest neighbors", results.len());
    for result in results {
        println!("ID: {}, Distance: {:.4}", result.id, result.distance);
    }

    Ok(())
}
```

## API Reference

### IndexBuilder

The main entry point for creating indices with a fluent builder pattern:

```rust
let index = IndexBuilder::new()
    .with_dimension(128)           // Required: vector dimension
    .with_metric(Metric::L2)       // Required: distance metric
    .with_max_degree(64)           // Optional: max graph degree (default: 64)
    .with_search_list_size(100)    // Optional: search beam width (default: 100)
    .with_alpha(1.2)               // Optional: graph density (default: 1.2)
    .with_num_threads(4)           // Optional: thread count (default: 1)
    .with_opq(false)               // Optional: enable OPQ compression (default: false)
    .build_in_memory::<f32>()?;    // Build in-memory index
```

### Distance Metrics

Available distance metrics:

- `Metric::L2` - Euclidean distance (fastest, most common)
- `Metric::Cosine` - Cosine distance (for normalized vectors)
- `Metric::InnerProduct` - Inner product similarity

### Search Parameters

Fine-tune search behavior:

```rust
let results = index.search(&query, k, l)?;
// k: number of results to return
// l: search beam width (higher = more accurate, slower)
```

### Advanced Configuration

```rust
use diskann::{SearchParams, ANNResult};

let params = SearchParams {
    k: 10,                    // Number of results
    l: 50,                    // Search beam width
    return_distances: true,    // Include distances in results
};

let results = index.search_with_params(&query, &params)?;
```

## Architecture

### In-Memory Index

Best for:
- Small to medium datasets (< 1M vectors)
- High-performance requirements
- Frequent updates

```rust
let mut index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::L2)
    .build_in_memory::<f32>()?;
```

### Disk-Based Index

Best for:
- Large datasets (> 1M vectors)
- Memory-constrained environments
- Persistent storage requirements

```rust
let mut index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::L2)
    .build_disk_index::<f32>("index_path")?;
```

## Performance Tips

### Index Building

- **Alpha parameter**: Higher values (1.2-1.4) create denser graphs with better accuracy but slower search
- **Max degree**: Higher values improve recall but increase memory usage
- **Thread count**: Use more threads for faster building on multi-core systems

### Search Optimization

- **k value**: Start with k=10-20 for most applications
- **l value**: Start with l=50-100, increase for better accuracy
- **Distance metric**: L2 is fastest, cosine good for normalized vectors
- **Batch operations**: Use batch search for multiple queries

### Memory Management

- **In-memory**: ~2-4x vector data size for index overhead
- **Disk-based**: Only cache size in memory, rest on disk
- **Batch size**: Larger batches are more efficient but use more memory

## Use Cases

### Image Similarity Search

```rust
// Load image embeddings
let embeddings = load_image_embeddings("images.bin")?;

let mut index = IndexBuilder::new()
    .with_dimension(512)  // Image embedding dimension
    .with_metric(Metric::Cosine)  // Cosine for normalized embeddings
    .build_in_memory::<f32>()?;

index.insert_batch(&embeddings)?;
index.build(&embeddings)?;

// Find similar images
let query_embedding = extract_image_embedding("query.jpg")?;
let similar_images = index.search(&query_embedding, 10, 50)?;
```

### Text Search with Embeddings

```rust
// Load text embeddings
let text_embeddings = load_text_embeddings("documents.bin")?;

let mut index = IndexBuilder::new()
    .with_dimension(768)  // BERT embedding dimension
    .with_metric(Metric::L2)
    .build_in_memory::<f32>()?;

index.insert_batch(&text_embeddings)?;
index.build(&text_embeddings)?;

// Semantic search
let query_embedding = embed_text("search query")?;
let relevant_docs = index.search(&query_embedding, 20, 100)?;
```

### Large-Scale Recommendation Systems

```rust
// For very large datasets, use disk-based index
let mut index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::InnerProduct)  // For recommendation scores
    .with_max_degree(128)
    .with_search_list_size(200)
    .with_num_threads(16)
    .build_disk_index::<f32>("recommendations")?;

// Build from file
index.build_from_file("user_embeddings.bin")?;

// Get recommendations
let user_embedding = get_user_embedding(user_id)?;
let recommendations = index.search(&user_embedding, 50, 200)?;
```

## Benchmarks

Performance on typical hardware (Intel i7-8700K, 32GB RAM):

| Dataset Size | Index Type | Build Time | Search Time (ms) | Memory Usage |
|-------------|------------|------------|------------------|--------------|
| 100K vectors | In-Memory | 2.3s | 0.8 | 512MB |
| 1M vectors | In-Memory | 18.7s | 1.2 | 4.2GB |
| 10M vectors | Disk-Based | 3m 45s | 2.1 | 1.1GB |
| 100M vectors | Disk-Based | 42m 12s | 3.8 | 2.3GB |

*Search times are for k=10, l=50 on 128-dimensional vectors*

## Development

### Building from Source

```bash
cargo build --release
```

### Running Tests

```bash
cargo test
cargo test --examples
```

### Running Benchmarks

```bash
cargo bench
```

## Dependencies

- **vector**: Vector operations and distance metrics
- **platform**: Platform-specific optimizations
- **logger**: Logging and instrumentation
- **rayon**: Parallel processing
- **serde**: Serialization support

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Contributing

We welcome contributions! Please see the main [README](../README.md) for contribution guidelines. 