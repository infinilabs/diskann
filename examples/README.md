# DiskANN Examples

This directory contains comprehensive examples demonstrating various aspects of the DiskANN vector search library.

## 📁 Directory Structure

```
examples/
├── README.md                           # This file
├── vector-search-demo/                 # Main vector search demonstration
│   ├── src/                           # Demo source code
│   ├── output/                        # Generated HTML reports
│   └── README.md                      # Demo documentation
├── dataset-generators/                 # Tools for generating test datasets
│   ├── src/                           # Generator source code
│   ├── data/                          # Generated datasets
│   └── README.md                      # Generator documentation
├── performance-benchmarks/             # Performance testing examples
│   ├── src/                           # Benchmark source code
│   └── README.md                      # Benchmark documentation
├── storage-analysis/                   # Storage requirement analysis
│   ├── vector_storage_comparison.md   # Engine comparison
│   └── README.md                      # Analysis documentation
└── cmd_drivers/                       # Command-line tools
    ├── build_disk_index/              # Disk index builder
    ├── build_memory_index/            # Memory index builder
    ├── search_memory_index/           # Memory index searcher
    └── README.md                      # CLI tools documentation
```

## 🚀 Quick Start

### 1. Vector Search Demo
```bash
# Run the main vector search demonstration
cargo run --bin vector-search-demo

# Generate large dataset first
cargo run --bin generate-large-dataset
```

### 2. Generate Test Datasets
```bash
# Generate 1M vectors for testing
cd examples/dataset-generators
cargo run --bin generate-large-dataset
```

### 3. Performance Benchmarks
```bash
# Run performance benchmarks
cd examples/performance-benchmarks
cargo run --bin benchmark-vector-search
```

## 📊 Examples Overview

### Vector Search Demo (`vector-search-demo/`)
- **Purpose**: Main demonstration of vector search capabilities
- **Features**:
  - Load vectors from JSON files
  - Perform similarity search
  - Generate HTML visualization reports
  - Memory and disk-based indexing
- **Output**: Interactive HTML reports showing search results

### Dataset Generators (`dataset-generators/`)
- **Purpose**: Generate synthetic datasets for testing
- **Features**:
  - Generate random vectors with specified dimensions
  - Create datasets of various sizes (1K to 1M vectors)
  - Export in multiple formats (JSON, binary)
- **Use Cases**: Performance testing, benchmarking, development

### Performance Benchmarks (`performance-benchmarks/`)
- **Purpose**: Measure and compare performance metrics
- **Features**:
  - Query latency measurements
  - Throughput testing (QPS)
  - Memory usage analysis
  - Build time comparisons
- **Metrics**: Latency, throughput, memory usage, build time

### Storage Analysis (`storage-analysis/`)
- **Purpose**: Analyze storage requirements and efficiency
- **Features**:
  - Storage requirement calculations
  - Comparison with other vector engines
  - Compression analysis
  - Cost-benefit analysis
- **Output**: Detailed storage comparison reports

### Command Line Tools (`cmd_drivers/`)
- **Purpose**: Production-ready command-line utilities
- **Features**:
  - Build disk and memory indexes
  - Search and query operations
  - Data conversion utilities
  - Batch processing tools

## 🔧 Usage Examples

### Basic Vector Search
```rust
use diskann::index::inmem_index::ANNInmemIndex;
use diskann::common::Metric;

// Create index
let mut index = build_memory_index(
    Metric::L2,
    512,
    64,    // max degree
    100,    // search list size
    1.2,    // alpha
    4,      // threads
)?;

// Insert vectors
index.insert_data(&vectors)?;

// Search
let mut indices = vec![0; 10];
let mut distances = vec![0.0; 10];
index.query(&query_vector, 10, 50, &mut indices, &mut distances)?;
```

### Generate Large Dataset
```rust
use serde_json::{json, Value};
use rand::Rng;

let num_vectors = 1_000_000;
let dimension = 512;
let mut rng = rand::thread_rng();

for i in 0..num_vectors {
    let mut embedding = Vec::new();
    for _ in 0..dimension {
        embedding.push(rng.gen_range(-1.0..1.0));
    }
    
    embeddings.push(json!({
        "filename": format!("vector_{:07}", i),
        "embedding": embedding
    }));
}
```

## 📈 Performance Metrics

### Typical Performance (1M vectors, 512 dimensions)
- **Build Time**: 2-5 minutes
- **Query Latency**: 1-10ms
- **Throughput**: 100-1000 QPS
- **Storage**: 3.81 GB (disk index)
- **Memory Usage**: 2-4 GB during queries

### Storage Requirements
- **Raw Data**: 1.90 GB (1M × 512 × 4 bytes)
- **Disk Index**: 3.81 GB (sector-aligned)
- **PQ Compressed**: 0.12 GB (with pivots: 0.18 GB)

## 🛠️ Development

### Adding New Examples
1. Create a new directory in `examples/`
2. Add a `Cargo.toml` with dependencies
3. Create `src/main.rs` with your example
4. Add documentation in `README.md`
5. Update this main README

### Running Examples
```bash
# Run specific example
cargo run --bin example-name

# Run with custom parameters
cargo run --bin example-name -- --input data.json --output results.html

# Build all examples
cargo build --release --all-targets
```

## 📚 Documentation

- [Vector Search Demo](vector-search-demo/README.md)
- [Dataset Generators](dataset-generators/README.md)
- [Performance Benchmarks](performance-benchmarks/README.md)
- [Storage Analysis](storage-analysis/README.md)
- [Command Line Tools](cmd_drivers/README.md)

## 🤝 Contributing

When adding new examples:
1. Follow the existing directory structure
2. Include comprehensive documentation
3. Add performance metrics where applicable
4. Provide clear usage instructions
5. Include sample data and outputs

## 📄 License

This examples directory is part of the DiskANN project and is licensed under the MIT License. 