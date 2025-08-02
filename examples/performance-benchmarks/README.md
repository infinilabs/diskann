# Performance Benchmarks

Comprehensive performance testing and benchmarking tools for DiskANN vector search operations.

## 🎯 Overview

This directory contains tools to measure and analyze DiskANN performance:
- Query latency measurements
- Throughput testing (QPS)
- Memory usage analysis
- Build time comparisons
- Performance regression testing

## 🚀 Quick Start

```bash
# Run basic benchmarks
cargo run --bin benchmark-vector-search

# Run comprehensive benchmarks
cargo run --bin benchmark-comprehensive -- --dataset large_embeddings.json
```

## 📁 Structure

```
performance-benchmarks/
├── src/
│   ├── main.rs                    # Main benchmark runner
│   ├── benchmarks/
│   │   ├── latency.rs            # Latency measurements
│   │   ├── throughput.rs         # Throughput testing
│   │   ├── memory.rs             # Memory usage analysis
│   │   └── build_time.rs         # Build time measurements
│   └── utils/
│       ├── metrics.rs            # Performance metrics
│       └── reporting.rs          # Report generation
├── results/
│   ├── latency_report.json       # Latency results
│   ├── throughput_report.json    # Throughput results
│   └── memory_report.json        # Memory usage results
└── README.md                     # This file
```

## 🔧 Available Benchmarks

### 1. Latency Benchmark (`benchmark-latency`)
- **Purpose**: Measure query response times
- **Metrics**:
  - Average latency
  - P50, P95, P99 percentiles
  - Latency distribution
  - Cold vs warm query performance
- **Output**: Detailed latency statistics

### 2. Throughput Benchmark (`benchmark-throughput`)
- **Purpose**: Measure queries per second (QPS)
- **Metrics**:
  - Maximum QPS
  - Sustained QPS
  - QPS under load
  - Concurrent query performance
- **Output**: Throughput analysis

### 3. Memory Benchmark (`benchmark-memory`)
- **Purpose**: Analyze memory usage patterns
- **Metrics**:
  - Peak memory usage
  - Memory during queries
  - Memory efficiency
  - Garbage collection impact
- **Output**: Memory usage reports

### 4. Build Time Benchmark (`benchmark-build-time`)
- **Purpose**: Measure index construction time
- **Metrics**:
  - Build time vs dataset size
  - Build time vs parameters
  - Memory usage during build
  - Build optimization analysis
- **Output**: Build performance data

## 📊 Usage Examples

### Basic Latency Benchmark
```rust
use std::time::Instant;
use diskann::index::inmem_index::ANNInmemIndex;

fn benchmark_latency(index: &mut ANNInmemIndex<f32>, queries: &[Vec<f32>]) -> Vec<f64> {
    let mut latencies = Vec::new();
    
    for query in queries {
        let start = Instant::now();
        
        let mut indices = vec![0; 10];
        let mut distances = vec![0.0; 10];
        index.query(query, 10, 50, &mut indices, &mut distances)?;
        
        let latency = start.elapsed().as_micros() as f64;
        latencies.push(latency);
    }
    
    latencies
}
```

### Throughput Benchmark
```rust
fn benchmark_throughput(index: &mut ANNInmemIndex<f32>, queries: &[Vec<f32>], duration: Duration) -> f64 {
    let start = Instant::now();
    let mut query_count = 0;
    
    while start.elapsed() < duration {
        for query in queries {
            let mut indices = vec![0; 10];
            let mut distances = vec![0.0; 10];
            index.query(query, 10, 50, &mut indices, &mut distances)?;
            query_count += 1;
        }
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    query_count as f64 / elapsed
}
```

### Memory Usage Benchmark
```rust
use std::alloc::{alloc, dealloc, Layout};

fn benchmark_memory_usage() -> MemoryMetrics {
    let before = get_memory_usage();
    
    // Perform operations
    let mut index = build_index(&vectors)?;
    let during = get_memory_usage();
    
    // Cleanup
    drop(index);
    let after = get_memory_usage();
    
    MemoryMetrics {
        before,
        during,
        after,
        peak: during.max(before),
    }
}
```

## 📈 Benchmark Results

### Typical Performance (1M vectors, 512 dimensions)

#### Latency Results
```
Query Latency Statistics:
├── Average: 2.1ms
├── P50: 1.8ms
├── P95: 4.2ms
├── P99: 7.1ms
└── P99.9: 12.3ms
```

#### Throughput Results
```
Throughput Analysis:
├── Maximum QPS: 1,247
├── Sustained QPS: 892
├── QPS under load: 654
└── Concurrent QPS: 1,156
```

#### Memory Results
```
Memory Usage:
├── Peak usage: 3.2 GB
├── Query memory: 2.1 GB
├── Index memory: 1.8 GB
└── Memory efficiency: 85%
```

#### Build Time Results
```
Build Performance:
├── Build time: 3.2 minutes
├── Memory during build: 4.1 GB
├── Build speed: 5,208 vectors/second
└── Optimization level: High
```

## 🔧 Configuration Options

### Benchmark Parameters
```bash
# Basic benchmark
cargo run --bin benchmark-latency

# Custom parameters
cargo run --bin benchmark-latency -- \
    --dataset large_embeddings.json \
    --queries 1000 \
    --k 10 \
    --l 50 \
    --threads 4
```

### Configuration File
```json
{
  "benchmark": {
    "name": "comprehensive_benchmark",
    "dataset": "large_embeddings.json",
    "queries": 1000,
    "k": 10,
    "l": 50,
    "threads": 4
  },
  "metrics": {
    "latency": true,
    "throughput": true,
    "memory": true,
    "build_time": true
  },
  "output": {
    "format": "json",
    "file": "benchmark_results.json",
    "detailed": true
  }
}
```

## 📊 Performance Metrics

### Latency Metrics
- **Average Latency**: Mean query response time
- **Percentiles**: P50, P95, P99, P99.9
- **Latency Distribution**: Histogram of response times
- **Cold vs Warm**: First query vs subsequent queries

### Throughput Metrics
- **QPS**: Queries per second
- **Maximum QPS**: Peak throughput
- **Sustained QPS**: Long-term throughput
- **Concurrent QPS**: Multi-threaded performance

### Memory Metrics
- **Peak Memory**: Maximum memory usage
- **Query Memory**: Memory during search
- **Index Memory**: Memory for index storage
- **Memory Efficiency**: Memory usage ratio

### Build Metrics
- **Build Time**: Total construction time
- **Build Speed**: Vectors per second
- **Memory During Build**: Peak memory during construction
- **Optimization Level**: Build quality metrics

## 🛠️ Development

### Adding New Benchmarks
1. Create new file in `src/benchmarks/`
2. Implement benchmark trait
3. Add to main.rs
4. Update documentation

### Benchmark Template
```rust
pub trait Benchmark {
    fn name(&self) -> &str;
    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult;
    fn validate(&self, result: &BenchmarkResult) -> bool;
}

pub struct BenchmarkConfig {
    pub dataset: String,
    pub queries: usize,
    pub k: usize,
    pub l: usize,
    pub threads: usize,
}
```

## 📈 Performance Analysis

### Performance Factors
1. **Dataset Size**: Larger datasets = slower queries
2. **Vector Dimensions**: Higher dimensions = more computation
3. **Search Parameters**: Higher k/l = slower but more accurate
4. **Hardware**: CPU cores, memory, disk speed
5. **Index Type**: Memory vs disk-based performance

### Optimization Tips
1. **Use appropriate k/l values**: Balance speed vs accuracy
2. **Optimize thread count**: Match CPU cores
3. **Use disk index for large datasets**: Memory efficiency
4. **Tune alpha parameter**: Graph construction optimization
5. **Monitor memory usage**: Prevent OOM errors

## 🔧 Troubleshooting

### Common Issues
1. **Out of memory**: Reduce dataset size or use disk index
2. **Slow benchmarks**: Increase thread count or reduce parameters
3. **Inconsistent results**: Use fixed random seeds
4. **High latency**: Check hardware and configuration

### Debug Mode
```bash
# Run with debug output
RUST_LOG=debug cargo run --bin benchmark-latency
```

## 📚 Related Examples

- [Vector Search Demo](../vector-search-demo/) - Test with real data
- [Dataset Generators](../dataset-generators/) - Generate test data
- [Storage Analysis](../storage-analysis/) - Analyze storage impact

## 📄 License

This benchmark suite is part of the DiskANN project and is licensed under the MIT License. 