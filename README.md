# DiskANN - Approximate Nearest Neighbor Search in Rust

[![Crates.io](https://img.shields.io/crates/v/diskann)](https://crates.io/crates/diskann)
[![Documentation](https://docs.rs/diskann/badge.svg)](https://docs.rs/diskann)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DiskANN is a high-performance, scalable approximate nearest neighbor (ANN) search library implemented in Rust. It provides both in-memory and disk-based indexing for large-scale vector search with high recall and low latency.

## 🚀 Key Features

- **Pure Rust implementation** - No C/C++ dependencies, leveraging Rust's memory safety
- **Disk-based indexing** - Support for datasets that don't fit in memory
- **High performance** - Optimized for fast approximate nearest neighbor search
- **Parallel processing** - Leverages Rust's concurrency features
- **Multiple distance metrics** - Support for L2, cosine, and other distance functions
- **Flexible data types** - Support for f32, f16, and other numeric types
- **Comprehensive examples** - Complete working examples for all functionality

## 📦 Installation

Add DiskANN to your `Cargo.toml`:

```toml
[dependencies]
diskann = "0.1.0"
```

## 🎯 Quick Start

### Basic In-Memory Search

```rust
use diskann::{IndexBuilder, Metric, SearchParams};

// Create an in-memory index
let mut index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::L2)
    .with_max_degree(64)
    .with_search_list_size(100)
    .build_in_memory()?;

// Insert vectors
let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
index.insert_batch(&vectors)?;

// Search for nearest neighbors
let query = vec![1.0, 2.0, 3.0];
let results = index.search(&query, 5, 50)?;

println!("Found {} nearest neighbors", results.len());
```

### Disk-Based Search

```rust
use diskann::disk_search::{BeamSearch, SearchParameters, SimpleFileReader};

// Initialize disk-based search
let file_reader = std::sync::Arc::new(SimpleFileReader::new(4096));
let beam_search = BeamSearch::<f32>::new(
    "index_path",
    "pq_file.bin",
    10,  // num_medoids
    4,   // num_centroids
    128, // data_dim
    8,   // n_chunks
    Metric::L2,
    file_reader,
)?;

// Set search parameters
let search_params = SearchParameters {
    k_search: 10,
    l_search: 50,
    beam_width: 100,
    io_limit: 1000,
    use_reorder_data: false,
    use_filter: false,
    filter_label: 0,
};

// Perform search
let query: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
let results = beam_search.search(&query, search_params)?;

println!("Found {} results", results.len());
```

## 📚 Examples

The project includes comprehensive examples demonstrating all functionality:

### 🎯 **Disk-Based Search Examples**

#### 1. **`disk_search_demo`** - Complete Disk Search Workflow ⭐
**Status**: ✅ **FULLY WORKING**

This is the most comprehensive example showing complete disk-based search functionality:

```bash
# Build an index
cargo run --package disk_search_demo build --data_path data.bin --index_prefix index

# Search using the built index
cargo run --package disk_search_demo search --index_prefix index --k 10

# Run complete demo workflow
cargo run --package disk_search_demo demo
```

**Features**:
- Complete end-to-end disk-based search workflow
- Build, search, and demo commands
- Proper error handling and file management
- Self-contained with all necessary functionality

#### 2. **`search_disk_index`** - Simple Command-Line Interface
**Status**: ✅ **WORKING**

Simple command-line interface for disk-based search:

```bash
cargo run --package search_disk_index <index_path> <query_file> <result_file> [options]
```

**Options**:
- `--k <num>` - Number of results (default: 10)
- `--l <num>` - Search list size (default: 50)
- `--beam <num>` - Beam width (default: 100)
- `--io-limit <num>` - I/O limit (default: 1000)
- `--metric <metric>` - Distance metric: L2, Cosine (default: L2)

### 🏗️ **Index Building Examples**

#### 3. **`build_disk_index`** - Disk Index Construction
**Status**: ✅ **WORKING**

Build disk-based indexes for large datasets:

```bash
cargo run --package build_disk_index -- --help
```

**Features**:
- Build disk indexes from vector data
- Configurable parameters for index optimization
- Support for various data formats

#### 4. **`build_memory_index`** - In-Memory Index Construction
**Status**: ✅ **WORKING**

Build in-memory indexes for smaller datasets:

```bash
cargo run --package build_memory_index -- --help
```

### 🔍 **Search Examples**

#### 5. **`search_memory_index`** - In-Memory Search
**Status**: ✅ **WORKING**

Search in-memory indexes:

```bash
cargo run --package search_memory_index -- --help
```

### 🔄 **Data Management Examples**

#### 6. **`load_and_insert_memory_index`** - Load and Insert
**Status**: ✅ **WORKING**

Load existing indexes and insert new data:

```bash
cargo run --package load_and_insert_memory_index -- --help
```

#### 7. **`build_and_insert_memory_index`** - Build and Insert
**Status**: ✅ **WORKING**

Build indexes and insert data in one operation:

```bash
cargo run --package build_and_insert_memory_index -- --help
```

#### 8. **`build_and_insert_delete_memory_index`** - Build, Insert, and Delete
**Status**: ✅ **WORKING**

Complete CRUD operations on indexes:

```bash
cargo run --package build_and_insert_delete_memory_index -- --help
```

### 🔧 **Utility Examples**

#### 9. **`convert_f32_to_bf16`** - Data Type Conversion
**Status**: ✅ **WORKING**

Convert between different numeric formats:

```bash
cargo run --package convert_f32_to_bf16 -- --help
```

## 💾 **Loading Saved Indexes and Performing Searches**

### **Disk-Based Index Loading and Search**

The `disk_search_demo` example provides the most comprehensive demonstration of loading saved indexes and performing searches:

#### **Step 1: Build an Index**
```bash
# Create test data and build index
cargo run --package disk_search_demo build --data_path test_data.bin --index_prefix my_index
```

#### **Step 2: Load and Search the Index**
```bash
# Search using the saved index
cargo run --package disk_search_demo search --index_prefix my_index --k 10
```

#### **Step 3: Complete Demo Workflow**
```bash
# Run the complete workflow (build + search)
cargo run --package disk_search_demo demo
```

### **In-Memory Index Loading and Search**

For in-memory indexes, use the `load_and_insert_memory_index` example:

```bash
# Load existing index and perform searches
cargo run --package load_and_insert_memory_index -- --help
```

### **Programmatic Usage**

```rust
use diskann::disk_search::{BeamSearch, SearchParameters, SimpleFileReader};

// Load a saved disk index
let file_reader = std::sync::Arc::new(SimpleFileReader::new(4096));
let beam_search = BeamSearch::<f32>::new(
    "saved_index_path",  // Path to saved index
    "saved_pq_file.bin", // Path to PQ file
    10,  // num_medoids
    4,   // num_centroids
    128, // data_dim
    8,   // n_chunks
    Metric::L2,
    file_reader,
)?;

// Set search parameters
let search_params = SearchParameters {
    k_search: 10,        // Number of results to return
    l_search: 50,        // Search list size
    beam_width: 100,     // Beam width
    io_limit: 1000,      // I/O limit
    use_reorder_data: false,
    use_filter: false,
    filter_label: 0,
};

// Perform search
let query: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
let results = beam_search.search(&query, search_params)?;

// Process results
for (id, distance) in results {
    println!("ID: {}, Distance: {:.6}", id, distance);
}
```

## 🧪 **Testing All Examples**

To verify all examples are working:

```bash
# Build all examples
cargo build --workspace

# Test specific examples
cargo run --package disk_search_demo -- --help
cargo run --package search_disk_index -- --help
cargo run --package build_disk_index -- --help
```

## 📊 **Performance**

DiskANN is optimized for high-performance vector search:

- **Memory efficiency** - Disk-based indexing for datasets that don't fit in memory
- **Parallel processing** - Leverages Rust's concurrency features
- **Optimized algorithms** - Beam search and other advanced search algorithms
- **Flexible data types** - Support for various numeric formats

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Original DiskANN paper and implementation
- Rust community for excellent tooling and ecosystem
- Contributors and maintainers

---

**Ready to get started?** Check out the examples directory for complete working code!