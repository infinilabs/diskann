#!/bin/bash

# DiskANN Cleanup Script
# Removes generated files, test outputs, and build artifacts

echo "🧹 Cleaning DiskANN project..."

# Remove Rust build artifacts
echo "Removing Rust build artifacts..."
cargo clean

# Remove generated data files
echo "Removing generated data files..."
find . -name "*.json" -not -path "./target/*" -not -name "Cargo.lock" -not -name "package.json" -delete
find . -name "*.bin" -not -path "./target/*" -not -path "./diskann/tests/data/*" -delete
find . -name "*.index" -not -path "./target/*" -not -path "./diskann/tests/data/*" -delete
find . -name "*.data" -not -path "./target/*" -not -path "./diskann/tests/data/*" -delete

# Remove test outputs
echo "Removing test outputs..."
find . -name "output" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "test_output" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "temp" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "tmp" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove log files
echo "Removing log files..."
find . -name "*.log" -not -path "./target/*" -delete

# Remove HTML reports
echo "Removing HTML reports..."
find . -name "*.html" -not -path "./target/*" -delete

# Remove temporary files
echo "Removing temporary files..."
find . -name "*.tmp" -not -path "./target/*" -delete
find . -name "*.bak" -not -path "./target/*" -delete
find . -name "*.backup" -not -path "./target/*" -delete

# Remove OS generated files
echo "Removing OS generated files..."
find . -name ".DS_Store" -delete
find . -name "Thumbs.db" -delete

echo "✅ Cleanup complete!"
echo ""
echo "Note: Important test data in diskann/tests/data/ has been preserved."
echo "To regenerate test data, run the appropriate example commands."
