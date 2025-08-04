#!/bin/bash

# DiskANN Benchmark Runner
# Runs comprehensive performance benchmarks and generates reports

set -e

echo "🚀 Starting DiskANN Performance Benchmarks"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create benchmark output directory
BENCHMARK_DIR="benchmark_results"
mkdir -p "$BENCHMARK_DIR"

# Function to run benchmarks with progress
run_benchmark() {
    local name=$1
    local command=$2
    
    echo -e "${BLUE}Running $name benchmarks...${NC}"
    
    # Run the benchmark
    if eval "$command" > "$BENCHMARK_DIR/${name}_output.txt" 2>&1; then
        echo -e "${GREEN}✅ $name benchmarks completed${NC}"
    else
        echo -e "${RED}❌ $name benchmarks failed${NC}"
        echo "Check $BENCHMARK_DIR/${name}_output.txt for details"
        return 1
    fi
}

# Function to generate summary report
generate_report() {
    echo -e "${YELLOW}📊 Generating benchmark summary...${NC}"
    
    cat > "$BENCHMARK_DIR/benchmark_summary.md" << 'EOF'
# DiskANN Benchmark Results

## Overview
This report contains performance benchmarks for the DiskANN vector search library.

## Benchmark Categories

### 1. Index Building Performance
- **Dataset Sizes**: 1K, 10K, 50K, 100K, 500K vectors
- **Dimensions**: 64, 128, 256 dimensions
- **Thread Scaling**: 1, 2, 4, 8, 16 threads
- **Alpha Values**: 0.8, 1.0, 1.2, 1.4, 1.6
- **Distance Metrics**: L2, Cosine, InnerProduct

### 2. Search Performance
- **K Values**: 1, 5, 10, 20, 50, 100 results
- **L Values**: 10, 25, 50, 100, 200, 500 beam width
- **Dataset Sizes**: 10K, 50K, 100K, 500K vectors
- **Dimensions**: 64, 128, 256, 512 dimensions
- **Batch Sizes**: 1, 10, 50, 100, 500 queries

### 3. Memory Usage
- **Building Memory**: Memory consumption during index construction
- **Search Memory**: Memory overhead during search operations
- **Dimension Scaling**: Memory usage vs vector dimensions
- **Alpha Impact**: Memory usage vs alpha parameter
- **Max Degree Impact**: Memory usage vs graph degree

### 4. Disk Index Performance
- **Large Datasets**: 50K, 100K, 500K vectors on disk
- **Memory Efficiency**: Disk vs memory index comparison

## Performance Metrics

### Index Building
- **Build Time**: Time to construct the index
- **Memory Usage**: Peak memory consumption during building
- **Throughput**: Vectors processed per second

### Search Performance
- **Query Latency**: Time per search query
- **Throughput**: Queries per second
- **Accuracy**: Recall@K for different parameters
- **Memory Overhead**: Additional memory per query

### Memory Efficiency
- **Memory per Vector**: MB per vector in index
- **Index Overhead**: Additional memory beyond raw vectors
- **Search Overhead**: Memory used during search

## Recommendations

### For Small Datasets (< 100K vectors)
- Use in-memory index
- Alpha = 1.2, Max Degree = 64
- L = 50-100 for good accuracy/speed balance

### For Medium Datasets (100K - 1M vectors)
- Use in-memory index if memory available
- Consider disk index for memory-constrained environments
- Alpha = 1.2-1.4 for better accuracy

### For Large Datasets (> 1M vectors)
- Use disk index
- Optimize for memory usage
- Consider batch processing for queries

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores for parallel processing
- **RAM**: 8GB for medium datasets
- **Storage**: SSD recommended for disk indices

### Recommended Requirements
- **CPU**: 8+ cores for optimal performance
- **RAM**: 16GB+ for large in-memory indices
- **Storage**: NVMe SSD for best disk performance

EOF

    echo -e "${GREEN}✅ Benchmark summary generated: $BENCHMARK_DIR/benchmark_summary.md${NC}"
}

# Function to check system resources
check_system() {
    echo -e "${YELLOW}🔍 Checking system resources...${NC}"
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        echo "Available memory: ${MEMORY_GB}GB"
        
        if [ "$MEMORY_GB" -lt 8 ]; then
            echo -e "${YELLOW}⚠️  Warning: Less than 8GB RAM available. Large benchmarks may fail.${NC}"
        fi
    fi
    
    # Check CPU cores
    CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    echo "CPU cores: $CPU_CORES"
    
    # Check available disk space
    DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    echo "Available disk space: ${DISK_GB}GB"
    
    if [ "$DISK_GB" -lt 10 ]; then
        echo -e "${YELLOW}⚠️  Warning: Less than 10GB disk space available.${NC}"
    fi
}

# Function to run quick smoke test
smoke_test() {
    echo -e "${BLUE}🧪 Running smoke test...${NC}"
    
    if cargo test --quiet --lib; then
        echo -e "${GREEN}✅ Smoke test passed${NC}"
    else
        echo -e "${RED}❌ Smoke test failed${NC}"
        exit 1
    fi
}

# Main benchmark execution
main() {
    echo "Starting benchmarks at $(date)"
    echo "=========================================="
    
    # Check system resources
    check_system
    echo ""
    
    # Run smoke test
    smoke_test
    echo ""
    
    # Run indexing benchmarks
    run_benchmark "indexing" "cargo bench --bench indexing_benchmarks"
    
    # Run search benchmarks
    run_benchmark "search" "cargo bench --bench search_benchmarks"
    
    # Run memory usage benchmarks
    run_benchmark "memory" "cargo bench --bench memory_usage_benchmarks"
    
    # Generate summary report
    generate_report
    
    echo ""
    echo -e "${GREEN}🎉 All benchmarks completed successfully!${NC}"
    echo "Results available in: $BENCHMARK_DIR/"
    echo "Summary report: $BENCHMARK_DIR/benchmark_summary.md"
    echo ""
    echo "To view detailed results:"
    echo "  open $BENCHMARK_DIR/indexing_output.txt"
    echo "  open $BENCHMARK_DIR/search_output.txt"
    echo "  open $BENCHMARK_DIR/memory_output.txt"
}

# Handle command line arguments
case "${1:-}" in
    "indexing")
        run_benchmark "indexing" "cargo bench --bench indexing_benchmarks"
        ;;
    "search")
        run_benchmark "search" "cargo bench --bench search_benchmarks"
        ;;
    "memory")
        run_benchmark "memory" "cargo bench --bench memory_usage_benchmarks"
        ;;
    "quick")
        echo -e "${YELLOW}Running quick benchmarks (smaller datasets)...${NC}"
        # Modify benchmark parameters for quick run
        export QUICK_BENCH=1
        main
        ;;
    "help"|"-h"|"--help")
        echo "DiskANN Benchmark Runner"
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  (no args)  Run all benchmarks"
        echo "  indexing    Run only indexing benchmarks"
        echo "  search      Run only search benchmarks"
        echo "  memory      Run only memory usage benchmarks"
        echo "  quick       Run quick benchmarks with smaller datasets"
        echo "  help        Show this help message"
        ;;
    *)
        main
        ;;
esac 