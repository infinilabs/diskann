# DiskANN Benchmarks - Quick Reference

## 🚀 Quick Start

```bash
# Run all benchmarks
./benchmarks/run_all_benchmarks.sh

# Run specific benchmark categories
cargo bench -p diskann_performance_benchmarks
cargo run -p diskann_comparison_benchmarks
cargo run -p diskann_debug_tools
```

## 📁 Directory Structure

```
benchmarks/
├── performance/          # Performance benchmarks (Criterion.rs)
├── comparison/          # Implementation comparisons
├── debug/              # Debug and progress tracking tools
├── results/            # Benchmark results and reports
├── documentation/      # Benchmark documentation
├── README.md          # Comprehensive documentation
├── QUICK_REFERENCE.md # This file
└── run_all_benchmarks.sh # Automated benchmark runner
```

## 🎯 Common Commands

### Performance Benchmarks
```bash
# Run all performance benchmarks
cargo bench -p diskann_performance_benchmarks

# Run specific benchmark
cargo bench --bench simple_benchmark
cargo bench --bench search_benchmarks
cargo bench --bench memory_usage_benchmarks
cargo bench --bench indexing_benchmarks
cargo bench --bench build_benchmark
```

### Comparison Benchmarks
```bash
# Compare DiskANN implementations
cargo run -p diskann_comparison_benchmarks
```

### Debug Tools
```bash
# Progress tracking and debugging
cargo run -p diskann_debug_tools
```

## 📊 Performance Summary

| Metric | Value |
|--------|-------|
| Index Creation | ~3.6ms |
| Vector Insertion | ~65ms (1000 vectors) |
| Graph Building | ~141ms (1000 vectors) |
| Search Time | ~6-7μs per query |
| Search Throughput | ~150,000 queries/second |

## 🔧 Configuration

### Common Benchmark Parameters
```rust
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

## 📈 Results Location

- **Performance Results**: `benchmarks/results/YYYYMMDD_HHMMSS/`
- **Comparison Results**: `benchmarks/results/YYYYMMDD_HHMMSS/comparison_results.txt`
- **Debug Results**: `benchmarks/results/YYYYMMDD_HHMMSS/debug_results.txt`

## 🐛 Troubleshooting

### Common Issues
1. **Build timeouts**: Increase timeout in Criterion configuration
2. **Memory issues**: Reduce dataset size
3. **Threading issues**: Adjust thread count

### Debug Commands
```bash
# Run with debug output
RUST_LOG=debug cargo bench

# Run specific benchmark with verbose output
cargo bench --bench simple_benchmark -- --verbose
```

## 📚 Documentation

- **Full Documentation**: `benchmarks/README.md`
- **Performance Guide**: `benchmarks/documentation/BENCHMARKS.md`
- **Comparison Guide**: `benchmarks/documentation/COMPARISON_WITH_DISKANN_RS.md`
- **Results**: `benchmarks/results/BENCHMARK_RESULTS.md`

## 🎯 What Each Benchmark Tests

### Performance Benchmarks
- **Simple**: Basic index building and search
- **Search**: Search performance with different parameters
- **Memory**: Memory consumption analysis
- **Indexing**: Index building with various configurations
- **Build**: Detailed build process analysis

### Comparison Benchmarks
- **DiskANN vs DiskANN-RS**: Implementation comparison
- **Build time analysis**: Performance comparison
- **Search time analysis**: Query performance comparison
- **Memory usage analysis**: Resource consumption comparison

### Debug Tools
- **Progress tracking**: Monitor long-running operations
- **Performance analysis**: Detailed timing breakdown
- **Memory profiling**: Resource usage analysis 