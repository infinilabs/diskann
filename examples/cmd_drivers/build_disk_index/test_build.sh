#!/bin/bash

# Test script for build_disk_index command
# This script demonstrates how to build a disk index using the available test data

set -e

echo "🧪 Testing DiskANN Disk Index Builder"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if test data exists
TEST_DATA="diskann/tests/data/siftsmall_learn.bin"
if [ ! -f "$TEST_DATA" ]; then
    echo "❌ Test data not found: $TEST_DATA"
    echo "Please ensure the test data file exists"
    exit 1
fi

echo "📁 Using test data: $TEST_DATA"
echo "📊 Data size: $(ls -lh $TEST_DATA | awk '{print $5}')"
echo

# Create output directory
OUTPUT_DIR="test_output"
mkdir -p $OUTPUT_DIR

echo "🚀 Building disk index with test data..."
echo

# Build a basic disk index
cargo run -p build_disk_index -- \
    --data_type float \
    --dist_fn l2 \
    --data_path $TEST_DATA \
    --index_path_prefix $OUTPUT_DIR/test_index \
    --max_degree 64 \
    --Lbuild 100 \
    --num_threads 4 \
    --search_DRAM_budget 1.0 \
    --build_DRAM_budget 2.0

echo
echo "✅ Test completed successfully!"
echo "📁 Index files created in: $OUTPUT_DIR/"
echo "📋 Generated files:"
ls -la $OUTPUT_DIR/

echo
echo "🎯 Next steps:"
echo "1. Use the generated index for search operations"
echo "2. Experiment with different parameters"
echo "3. Try with your own data files"
echo
echo "Example search command (when search tool is available):"
echo "cargo run -p search_disk_index \\"
echo "  --index_path_prefix $OUTPUT_DIR/test_index \\"
echo "  --query_file your_queries.bin \\"
echo "  --result_file results.txt" 