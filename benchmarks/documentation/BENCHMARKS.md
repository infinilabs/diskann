# DiskANN Performance Benchmarks

This document describes the comprehensive benchmarking system for DiskANN, designed to measure indexing and search performance across various parameters and dataset sizes.

## 🚀 Quick Start

### Run All Benchmarks
```bash
./run_benchmarks.sh
```

### Run Specific Benchmark Categories
```bash
# Indexing performance
./run_benchmarks.sh indexing

# Search performance
./run_benchmarks.sh search

# Memory usage analysis
./run_benchmarks.sh memory

# Quick benchmarks (smaller datasets)
./run_benchmarks.sh quick
```

## 📊 Benchmark Categories

### 1. Index Building Performance (`indexing_benchmarks.rs`)

Measures the time and resources required to build indices with different parameters:

#### **Dataset Size Scaling**
- **Sizes**: 1K, 10K, 50K, 100K, 500K vectors
- **Dimensions**: 64, 128, 256 dimensions
- **Metrics**: Build time, memory usage, throughput

#### **Thread Scaling**
- **Threads**: 1, 2, 4, 8, 16 threads
- **Metrics**: Build time vs thread count, scalability

#### **Parameter Optimization**
- **Alpha Values**: 0.8, 1.0, 1.2, 1.4, 1.6
- **Distance Metrics**: L2, Cosine, InnerProduct
- **Max Degrees**: 32, 64, 128, 256

#### **Disk Index Building**
- **Large Datasets**: 50K, 100K, 500K vectors
- **Memory Efficiency**: Disk vs memory comparison

### 2. Search Performance (`search_benchmarks.rs`)

Measures query performance and accuracy:

#### **K-Value Scaling**
- **K Values**: 1, 5, 10, 20, 50, 100 results
- **Metrics**: Query latency, throughput

#### **L-Value Optimization**
- **L Values**: 10, 25, 50, 100, 200, 500 beam width
- **Metrics**: Accuracy vs speed trade-off

#### **Dataset Size Impact**
- **Sizes**: 10K, 50K, 100K, 500K vectors
- **Metrics**: Query time scaling

#### **Dimension Scaling**
- **Dimensions**: 64, 128, 256, 512 dimensions
- **Metrics**: Performance vs dimensionality

#### **Batch Processing**
- **Batch Sizes**: 1, 10, 50, 100, 500 queries
- **Metrics**: Throughput optimization

#### **Distance Metrics Comparison**
- **Metrics**: L2, Cosine, InnerProduct
- **Metrics**: Performance characteristics

### 3. Memory Usage Analysis (`memory_usage_benchmarks.rs`)

Measures memory consumption patterns:

#### **Building Memory Usage**
- **Dataset Sizes**: 1K, 10K, 50K, 100K, 500K vectors
- **Metrics**: Peak memory, memory per vector

#### **Search Memory Overhead**
- **Query Counts**: 10, 100, 1000 queries
- **Metrics**: Memory overhead per query

#### **Dimension Impact**
- **Dimensions**: 64, 128, 256, 512, 1024
- **Metrics**: Memory scaling with dimensions

#### **Parameter Impact**
- **Alpha Values**: 0.8, 1.0, 1.2, 1.4, 1.6
- **Max Degrees**: 32, 64, 128, 256
- **Metrics**: Memory vs parameter trade-offs

#### **Disk Index Memory**
- **Large Datasets**: 100K, 500K, 1M vectors
- **Metrics**: Memory efficiency vs accuracy

## 📈 Performance Metrics

### Index Building Metrics
- **Build Time**: Total time to construct index
- **Memory Usage**: Peak memory consumption
- **Throughput**: Vectors processed per second
- **Memory per Vector**: MB per vector in index

### Search Metrics
- **Query Latency**: Time per search query
- **Throughput**: Queries per second
- **Recall@K**: Accuracy measurement
- **Memory Overhead**: Additional memory per query

### Memory Efficiency Metrics
- **Index Overhead**: Memory beyond raw vectors
- **Search Overhead**: Memory used during search
- **Memory Scaling**: How memory grows with dataset size

## 🔧 Configuration

### Benchmark Parameters

The benchmarks can be configured by modifying the benchmark files:

```rust
// In indexing_benchmarks.rs
let dataset_sizes = [1000, 10000, 50000, 100000, 500000];
let dimensions = [64, 128, 256];
let thread_counts = [1, 2, 4, 8, 16];
let alpha_values = [0.8, 1.0, 1.2, 1.4, 1.6];
```

### System Requirements

#### **Minimum Requirements**
- **CPU**: 4+ cores for parallel processing
- **RAM**: 8GB for medium datasets
- **Storage**: 10GB free space for disk indices

#### **Recommended Requirements**
- **CPU**: 8+ cores for optimal performance
- **RAM**: 16GB+ for large in-memory indices
- **Storage**: NVMe SSD for best disk performance

### Environment Variables

```bash
# Run quick benchmarks (smaller datasets)
export QUICK_BENCH=1
./run_benchmarks.sh

# Set number of benchmark iterations
export CRITERION_ITERATIONS=100
./run_benchmarks.sh
```

## 📊 Results Interpretation

### Performance Guidelines

#### **Index Building Performance**
- **Small Datasets (< 100K)**: Should build in < 30 seconds
- **Medium Datasets (100K-1M)**: Should build in < 5 minutes
- **Large Datasets (> 1M)**: Should build in < 30 minutes

#### **Search Performance**
- **Query Latency**: < 10ms for small datasets, < 50ms for large
- **Throughput**: > 100 QPS for small datasets, > 10 QPS for large
- **Memory Overhead**: < 1MB per query

#### **Memory Efficiency**
- **Index Overhead**: < 4x raw vector size
- **Memory per Vector**: < 1KB per vector for 128-dim
- **Search Overhead**: < 100KB per query

### Optimization Recommendations

#### **For Small Datasets (< 100K vectors)**
```rust
let index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::L2)
    .with_max_degree(64)
    .with_search_list_size(100)
    .with_alpha(1.2)
    .with_num_threads(4)
    .build_in_memory::<f32>()?;
```

#### **For Medium Datasets (100K-1M vectors)**
```rust
let index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::L2)
    .with_max_degree(128)
    .with_search_list_size(200)
    .with_alpha(1.4)
    .with_num_threads(8)
    .build_in_memory::<f32>()?;
```

#### **For Large Datasets (> 1M vectors)**
```rust
let index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::L2)
    .with_max_degree(64)
    .with_search_list_size(100)
    .with_alpha(1.2)
    .with_num_threads(16)
    .build_disk_index::<f32>("index_path")?;
```

## 🛠️ Custom Benchmarks

### Adding New Benchmarks

1. **Create a new benchmark file**:
```rust
// benches/custom_benchmark.rs
use criterion::{criterion_group, criterion_main, Criterion};
use diskann::{IndexBuilder, Metric};

fn my_custom_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_benchmark");
    
    group.bench_function("my_test", |b| {
        b.iter(|| {
            // Your benchmark code here
        });
    });
    
    group.finish();
}

criterion_group!(benches, my_custom_benchmark);
criterion_main!(benches);
```

2. **Add to Cargo.toml**:
```toml
[[bench]]
name = "custom_benchmark"
path = "custom_benchmark.rs"
harness = false
```

3. **Run the benchmark**:
```bash
cargo bench --bench custom_benchmark
```

### Benchmark Utilities

The benchmark crate provides utilities in `src/lib.rs`:

```rust
use diskann_benchmarks::common::{generate_random_vectors, generate_query_vectors};
use diskann_benchmarks::utils::{get_memory_usage, format_bytes, format_duration};

// Generate test data
let vectors = generate_random_vectors(128, 10000);
let queries = generate_query_vectors(128, 100);

// Measure memory usage
let memory = get_memory_usage();
println!("Memory usage: {} MB", memory);
```

## 📋 Output Files

### Generated Reports

After running benchmarks, the following files are generated:

- `benchmark_results/indexing_output.txt` - Index building results
- `benchmark_results/search_output.txt` - Search performance results
- `benchmark_results/memory_output.txt` - Memory usage analysis
- `benchmark_results/benchmark_summary.md` - Summary report

### Criterion HTML Reports

Criterion generates detailed HTML reports in:
- `target/criterion/report/index.html` - Main report
- `target/criterion/*/report/index.html` - Individual benchmark reports

## 🔍 Troubleshooting

### Common Issues

#### **Out of Memory**
```bash
# Reduce dataset sizes for quick testing
export QUICK_BENCH=1
./run_benchmarks.sh quick
```

#### **Long Build Times**
```bash
# Run only specific benchmarks
./run_benchmarks.sh indexing
```

#### **Network Issues**
```bash
# Use offline mode if available
cargo bench --offline
```

### Performance Tips

1. **Use Release Mode**: Always run benchmarks in release mode
2. **Close Other Applications**: Free up memory and CPU
3. **Use SSD Storage**: For disk index benchmarks
4. **Monitor System Resources**: Use `htop` or `top` during benchmarks

## 📚 Further Reading

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [DiskANN Paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
- [Performance Optimization Guide](PERFORMANCE.md)

---

**Note**: These benchmarks are designed to help optimize DiskANN for your specific use case. Results may vary based on hardware, dataset characteristics, and system configuration. 