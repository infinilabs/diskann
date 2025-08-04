# DiskANN Benchmarks

This directory contains all benchmark-related code, results, and documentation for the DiskANN project.

## Directory Structure

```
benchmarks/
├── performance/          # Criterion.rs performance benchmarks
│   ├── Cargo.toml       # Benchmark dependencies
│   ├── src/             # Benchmark utilities
│   ├── simple_benchmark.rs
│   ├── search_benchmarks.rs
│   ├── memory_usage_benchmarks.rs
│   ├── indexing_benchmarks.rs
│   └── build_benchmark.rs
├── comparison/          # Implementation comparisons
│   ├── Cargo.toml
│   └── src/main.rs      # DiskANN vs DiskANN-RS comparison
├── debug/              # Debug and progress tracking tools
│   ├── Cargo.toml
│   ├── src/main.rs      # Progress tracking debug tool
│   ├── debug_progress.rs
│   └── debug_benchmark.rs
├── results/            # Benchmark results and reports
│   ├── BENCHMARK_RESULTS.md
│   ├── ACTUAL_BENCHMARK_COMPARISON.md
│   └── benchmark_comparison_results.json
├── documentation/      # Benchmark documentation
│   ├── BENCHMARKS.md
│   └── COMPARISON_WITH_DISKANN_RS.md
├── README.md          # This file
├── QUICK_REFERENCE.md # Quick reference guide
├── run_all_benchmarks.sh # Automated benchmark runner
└── run_benchmarks.sh  # Legacy benchmark runner
```

## Quick Start

### Running Performance Benchmarks

```bash
# Run all performance benchmarks
cargo bench -p diskann_benchmarks

# Run specific benchmark
cargo bench --bench simple_benchmark
cargo bench --bench search_benchmarks
cargo bench --bench memory_usage_benchmarks
cargo bench --bench indexing_benchmarks
cargo bench --bench build_benchmark
```

### Running Comparison Benchmarks

```bash
# Compare our DiskANN with DiskANN-RS
cargo run -p compare_implementations
```

### Running Debug Tools

```bash
# Debug progress tracking
cargo run -p debug_benchmark
```

## Benchmark Categories

### 1. Performance Benchmarks (`performance/`)

**Purpose**: Measure the performance characteristics of our DiskANN implementation.

**Benchmarks**:
- **Simple Benchmark**: Basic index building and search performance
- **Search Benchmarks**: Search performance across different parameters
- **Memory Usage Benchmarks**: Memory consumption analysis
- **Indexing Benchmarks**: Index building performance with various configurations
- **Build Benchmark**: Detailed build process analysis

**Usage**:
```bash
cd benchmarks/performance
cargo bench
```

### 2. Comparison Benchmarks (`comparison/`)

**Purpose**: Compare our implementation with other DiskANN implementations.

**Features**:
- DiskANN vs DiskANN-RS comparison
- Build time, search time, and memory usage analysis
- Comprehensive benchmark reports

**Usage**:
```bash
cd benchmarks/comparison
cargo run
```

### 3. Debug Tools (`debug/`)

**Purpose**: Debug and track the progress of index building and search operations.

**Tools**:
- Progress tracking for index building
- Performance analysis tools
- Debug utilities

**Usage**:
```bash
cd benchmarks/debug
cargo run
```

## Results

### Performance Summary

| Metric | Value |
|--------|-------|
| Index Creation | ~3.6ms |
| Vector Insertion | ~65ms (1000 vectors) |
| Graph Building | ~141ms (1000 vectors) |
| Search Time | ~6-7μs per query |
| Search Throughput | ~150,000 queries/second |

### Key Findings

1. **Build Performance**: 
   - 1000 vectors: ~210ms total build time
   - Scales with dataset size and complexity
   - Parallel processing with 4 threads

2. **Search Performance**:
   - Very fast: 6-7 microseconds per search
   - High throughput: 150,000+ searches/second
   - Excellent for real-time applications

3. **Memory Usage**:
   - Efficient memory usage
   - Scales linearly with dataset size
   - Optimized for both small and large datasets

## Configuration

### Benchmark Parameters

```rust
// Common benchmark configuration
let index = IndexBuilder::new()
    .with_dimension(128)
    .with_metric(Metric::L2)
    .with_max_degree(64)
    .with_search_list_size(100)
    .with_alpha(1.2)
    .with_num_threads(4)
    .build_in_memory::<f32>()
    .unwrap();
```

### Dataset Sizes

- **Small**: 100-1,000 vectors
- **Medium**: 1,000-10,000 vectors  
- **Large**: 10,000+ vectors

## Running Custom Benchmarks

### Creating New Benchmarks

1. Add your benchmark file to `performance/`
2. Update `performance/Cargo.toml` with new benchmark
3. Run with `cargo bench --bench your_benchmark`

### Custom Dataset Generation

```rust
use rand::Rng;

fn generate_test_vectors(dimension: usize, count: usize) -> Vec<Vec<f32>> {
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
```

## Analysis Tools

### Progress Tracking

The debug tools help track the progress of long-running operations:

```bash
# Track index building progress
cargo run -p debug_benchmark
```

### Performance Analysis

```bash
# Generate detailed performance reports
cargo bench --bench build_benchmark
```

## Contributing

When adding new benchmarks:

1. **Follow the naming convention**: `category_benchmark.rs`
2. **Add documentation**: Include purpose and expected results
3. **Update this README**: Add new benchmarks to the appropriate section
4. **Test thoroughly**: Ensure benchmarks are reliable and reproducible

## Troubleshooting

### Common Issues

1. **Build timeouts**: Increase timeout in Criterion configuration
2. **Memory issues**: Reduce dataset size or optimize memory usage
3. **Threading issues**: Adjust thread count in benchmark configuration

### Debug Commands

```bash
# Run with debug output
RUST_LOG=debug cargo bench

# Run specific benchmark with verbose output
cargo bench --bench simple_benchmark -- --verbose
```

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/)
- [DiskANN Paper](https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8a6-Abstract.html)
- [Performance Analysis Guide](benchmarks/documentation/BENCHMARKS.md) 