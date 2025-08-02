# Vector Search Demo

A comprehensive demonstration of DiskANN vector search capabilities with interactive HTML output.

## 🎯 Overview

This demo showcases the complete vector search workflow:
- Loading vectors from JSON files
- Building in-memory and disk-based indexes
- Performing similarity searches
- Generating interactive HTML visualizations

## 🚀 Quick Start

```bash
# Run the demo with default settings
cargo run --bin vector-search-demo

# Run with custom data
cargo run --bin vector-search-demo -- --input data.json --output results.html
```

## 📁 Structure

```
vector-search-demo/
├── src/
│   ├── main.rs              # Main demo application
│   ├── mem_ann_store.rs     # Memory-based vector store
│   └── disk_ann_store.rs    # Disk-based vector store
├── output/
│   └── relations.html       # Generated HTML report
└── README.md               # This file
```

## 🔧 Features

### Core Functionality
- **Vector Loading**: Load vectors from JSON format
- **Index Building**: Create memory and disk-based indexes
- **Similarity Search**: Find nearest neighbors
- **HTML Generation**: Create interactive visualizations
- **Performance Monitoring**: Built-in timing and metrics

### Search Capabilities
- **K-NN Search**: Find k nearest neighbors
- **Distance Metrics**: L2, Cosine, Inner Product
- **Parameter Tuning**: Adjustable search parameters
- **Batch Processing**: Handle multiple queries

### Visualization
- **Interactive HTML**: Visual search results
- **Distance Metrics**: Display similarity scores
- **Query Highlighting**: Highlight query vectors
- **Responsive Design**: Works on different screen sizes

## 📊 Usage Examples

### Basic Usage
```rust
use vector_search_demo::mem_ann_store::MemANNStore;
use vector::Metric;

// Create vector store
let mut store = MemANNStore::new(
    Metric::L2,           // Distance metric
    512,                  // Vector dimension
    64,                   // Max degree
    100,                  // Search list size
    1.2,                  // Alpha parameter
    4,                    // Number of threads
    1_000_000,           // Max points
)?;

// Insert vectors
store.insert_data(&vectors)?;

// Search
let mut indices = vec![0; 10];
let mut distances = vec![0.0; 10];
store.query(&query_vector, 10, 50, &mut indices, &mut distances)?;
```

### HTML Output
The demo generates an interactive HTML report (`relations.html`) that includes:
- Query vectors highlighted in yellow
- Similar vectors with distance scores
- Visual layout with image thumbnails
- Performance statistics

## 📈 Performance

### Typical Performance (1M vectors, 512 dimensions)
- **Build Time**: 2-5 minutes
- **Query Latency**: 1-10ms
- **Throughput**: 100-1000 QPS
- **Memory Usage**: 2-4 GB
- **Storage**: 3.81 GB (disk index)

### Configuration Options
```rust
// High-performance configuration
let store = MemANNStore::new(
    Metric::L2,           // L2 distance
    512,                  // 512 dimensions
    128,                  // Higher max degree
    200,                  // Larger search list
    1.4,                  // Higher alpha
    8,                    // More threads
    1_000_000,           // 1M vectors
)?;
```

## 🔍 Search Parameters

### Key Parameters
- **k**: Number of results to return
- **l**: Search list size (higher = more accurate, slower)
- **alpha**: Graph construction parameter
- **max_degree**: Maximum neighbors per node
- **num_threads**: Parallel processing threads

### Parameter Tuning
```rust
// Fast search (lower accuracy)
store.query(&query, 10, 25, &mut indices, &mut distances)?;

// Accurate search (slower)
store.query(&query, 10, 100, &mut indices, &mut distances)?;

// Balanced search
store.query(&query, 10, 50, &mut indices, &mut distances)?;
```

## 📁 Data Formats

### Input JSON Format
```json
[
  {
    "filename": "vector_0000001",
    "embedding": [0.1, 0.2, 0.3, ...]
  },
  {
    "filename": "vector_0000002", 
    "embedding": [0.4, 0.5, 0.6, ...]
  }
]
```

### Output HTML Features
- **Query Visualization**: Query vectors highlighted
- **Result Display**: Similar vectors with distances
- **Performance Stats**: Build and search timing
- **Interactive Layout**: Responsive design

## 🛠️ Development

### Building from Source
```bash
# Build in release mode
cargo build --release

# Run with custom data
cargo run --release --bin vector-search-demo -- --input large_embeddings.json
```

### Customization
```rust
// Custom distance metric
let store = MemANNStore::new(Metric::Cosine, ...)?;

// Custom dimensions
let store = MemANNStore::new(Metric::L2, 256, ...)?;

// Custom search parameters
store.query(&query, k, l, &mut indices, &mut distances)?;
```

## 🔧 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce max_points or use smaller vectors
2. **Slow performance**: Increase threads or reduce search parameters
3. **File not found**: Check input file path
4. **Dimension mismatch**: Ensure all vectors have same dimension

### Debug Mode
```bash
# Run with debug output
RUST_LOG=debug cargo run --bin vector-search-demo
```

## 📚 Related Examples

- [Dataset Generators](../dataset-generators/) - Generate test data
- [Performance Benchmarks](../performance-benchmarks/) - Measure performance
- [Storage Analysis](../storage-analysis/) - Analyze storage requirements

## 📄 License

This demo is part of the DiskANN project and is licensed under the MIT License. 