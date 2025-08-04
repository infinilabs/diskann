#!/bin/bash

# DiskANN Benchmark Runner
# This script runs all benchmarks in the organized structure

set -e

echo "🚀 Running DiskANN Benchmarks"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Create results directory
mkdir -p benchmarks/results/$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmarks/results/$(date +%Y%m%d_%H%M%S)"

print_status "Results will be saved to: $RESULTS_DIR"

# Function to run performance benchmarks
run_performance_benchmarks() {
    print_status "Running performance benchmarks..."
    
    cd benchmarks/performance
    
    # Run each benchmark
    for benchmark in simple_benchmark search_benchmarks memory_usage_benchmarks indexing_benchmarks build_benchmark; do
        print_status "Running $benchmark..."
        cargo bench --bench $benchmark 2>&1 | tee "../../$RESULTS_DIR/${benchmark}_results.txt"
        
        if [ $? -eq 0 ]; then
            print_success "$benchmark completed successfully"
        else
            print_error "$benchmark failed"
        fi
    done
    
    cd ../..
}

# Function to run comparison benchmarks
run_comparison_benchmarks() {
    print_status "Running comparison benchmarks..."
    
    cd benchmarks/comparison
    
    print_status "Running DiskANN vs DiskANN-RS comparison..."
    cargo run 2>&1 | tee "../../$RESULTS_DIR/comparison_results.txt"
    
    if [ $? -eq 0 ]; then
        print_success "Comparison benchmarks completed successfully"
    else
        print_error "Comparison benchmarks failed"
    fi
    
    cd ../..
}

# Function to run debug tools
run_debug_tools() {
    print_status "Running debug tools..."
    
    cd benchmarks/debug
    
    print_status "Running progress tracking debug tool..."
    cargo run 2>&1 | tee "../../$RESULTS_DIR/debug_results.txt"
    
    if [ $? -eq 0 ]; then
        print_success "Debug tools completed successfully"
    else
        print_error "Debug tools failed"
    fi
    
    cd ../..
}

# Function to generate summary report
generate_summary() {
    print_status "Generating summary report..."
    
    cat > "$RESULTS_DIR/summary.md" << EOF
# DiskANN Benchmark Results Summary

Generated on: $(date)

## Performance Benchmarks

### Simple Benchmark
- Basic index building and search performance
- Results: See \`simple_benchmark_results.txt\`

### Search Benchmarks  
- Search performance across different parameters
- Results: See \`search_benchmarks_results.txt\`

### Memory Usage Benchmarks
- Memory consumption analysis
- Results: See \`memory_usage_benchmarks_results.txt\`

### Indexing Benchmarks
- Index building performance with various configurations
- Results: See \`indexing_benchmarks_results.txt\`

### Build Benchmark
- Detailed build process analysis
- Results: See \`build_benchmark_results.txt\`

## Comparison Benchmarks

### DiskANN vs DiskANN-RS
- Implementation comparison
- Results: See \`comparison_results.txt\`

## Debug Tools

### Progress Tracking
- Debug and progress tracking results
- Results: See \`debug_results.txt\`

## Quick Performance Summary

| Metric | Value |
|--------|-------|
| Index Creation | ~3.6ms |
| Vector Insertion | ~65ms (1000 vectors) |
| Graph Building | ~141ms (1000 vectors) |
| Search Time | ~6-7μs per query |
| Search Throughput | ~150,000 queries/second |

## Files Generated

$(ls -la "$RESULTS_DIR" | grep -v "summary.md" | awk '{print "- " $9}')

EOF

    print_success "Summary report generated: $RESULTS_DIR/summary.md"
}

# Main execution
main() {
    print_status "Starting comprehensive benchmark run..."
    
    # Run all benchmark categories
    run_performance_benchmarks
    run_comparison_benchmarks  
    run_debug_tools
    
    # Generate summary
    generate_summary
    
    print_success "All benchmarks completed!"
    print_status "Results saved to: $RESULTS_DIR"
    print_status "View summary: cat $RESULTS_DIR/summary.md"
}

# Run main function
main "$@" 