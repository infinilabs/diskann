# DiskANN Disk Index Search

A command-line tool for searching disk-based DiskANN indices that have been previously built.

## Overview

The `search_disk_index` command loads an existing disk-based index and performs nearest neighbor search on it. This tool is designed to work with indices created by the `build_disk_index` command.

## Quick Start

### Basic Search

```bash
# Search an existing disk index
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix data/index \
  --query_file queries.bin \
  --result_file results.txt
```

### Advanced Search

```bash
# Search with custom parameters
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn cosine \
  --index_path_prefix data/index \
  --query_file queries.bin \
  --result_file results.txt \
  --k 20 \
  --l 100 \
  --search_list_size 200 \
  --num_threads 8 \
  --search_DRAM_budget 4.0
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data_type` | Data type of vectors | `float`, `int8`, `uint8`, `f16` |
| `--dist_fn` | Distance function | `l2`, `cosine` |
| `--index_path_prefix` | Path prefix of the built index | `data/index` |
| `--query_file` | Input query file path | `queries.bin` |
| `--result_file` | Output result file path | `results.txt` |

### Optional Arguments

| Argument | Alias | Default | Description |
|----------|-------|---------|-------------|
| `--k` | | `10` | Number of results to return |
| `--l` | | `50` | Search complexity (higher = better recall) |
| `--search_list_size` | `-L` | `100` | Search list size |
| `--num_threads` | `-T` | CPU cores | Number of search threads |
| `--search_DRAM_budget` | `-B` | `1.0` | Search RAM limit in GB |

## Data Format

### Query File Format

The query file should be in binary format with the following structure:

```
[number_of_queries: u32][dimension: u32][data: T * number_of_queries * dimension]
```

Where `T` is the data type (f32, i8, u8, or f16).

### Result File Format

The result file contains search results in a human-readable format, including:
- Search timing information
- Number of results found
- Top results with IDs and distances

## Examples

### Example 1: Basic Search

```bash
# Search for 10 nearest neighbors
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix data/index \
  --query_file queries.bin \
  --result_file results.txt
```

### Example 2: High-Recall Search

```bash
# Search with higher recall settings
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn cosine \
  --index_path_prefix data/index \
  --query_file queries.bin \
  --result_file results.txt \
  --k 50 \
  --l 200 \
  --search_list_size 500
```

### Example 3: High-Performance Search

```bash
# Search optimized for speed
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix data/index \
  --query_file queries.bin \
  --result_file results.txt \
  --k 10 \
  --l 30 \
  --search_list_size 50 \
  --num_threads 16 \
  --search_DRAM_budget 8.0
```

## Parameter Guidelines

### Search Quality Parameters

- **K (number of results)**: Controls how many nearest neighbors to return
  - `10-50`: Good for most applications
  - `100+`: For applications requiring many results

- **L (search complexity)**: Controls search quality vs. speed trade-off
  - `30-50`: Fast search, moderate recall
  - `100-200`: High recall, slower search
  - `500+`: Maximum recall, slowest search

- **Search List Size**: Internal search parameter
  - `50-100`: Fast search
  - `200-500`: Better recall
  - `1000+`: Maximum recall

### Performance Parameters

- **Number of Threads**: Parallel search execution
  - `1-4`: For small datasets or single-threaded applications
  - `8-16`: For large datasets and multi-core systems
  - `32+`: For very large datasets on high-core systems

- **Search RAM Budget**: Memory limit for search operations
  - `1-2 GB`: For small to medium datasets
  - `4-8 GB`: For large datasets
  - `16+ GB`: For very large datasets

## Integration with Build Process

### Complete Workflow

1. **Build the index**:
   ```bash
   cargo run -p build_disk_index -- \
     --data_type float \
     --dist_fn l2 \
     --data_path vectors.bin \
     --index_path_prefix data/index
   ```

2. **Search the index**:
   ```bash
   cargo run -p search_disk_index -- \
     --data_type float \
     --dist_fn l2 \
     --index_path_prefix data/index \
     --query_file queries.bin \
     --result_file results.txt
   ```

### Index Files

After building, the following files are created:
- `{prefix}_disk.index`: Main disk index file
- `{prefix}_mem.index.data`: Memory-mapped data file
- `{prefix}_sample_data.bin`: Sample data for queries
- `{prefix}_sample_ids.bin`: Sample IDs for queries

## Performance Tips

1. **For high-recall applications**: Use higher L values and search list sizes
2. **For high-throughput applications**: Use lower L values and more threads
3. **For memory-constrained systems**: Reduce search RAM budget
4. **For large datasets**: Increase search RAM budget and use more threads

## Troubleshooting

### Common Issues

1. **"Index not found"**
   - Ensure the index was built successfully
   - Check that the index path prefix is correct
   - Verify all index files exist

2. **"Query file not found"**
   - Check that the query file exists and is readable
   - Ensure the query file format matches the index data type

3. **"Out of memory during search"**
   - Increase `--search_DRAM_budget`
   - Reduce `--search_list_size`
   - Use fewer threads

4. **"Slow search performance"**
   - Increase `--search_DRAM_budget`
   - Use more threads
   - Reduce L value for faster search

### Error Messages

- **"Missing required arguments"**: Ensure all required arguments are provided
- **"Invalid data type"**: Use one of: `int8`, `uint8`, `float`, `f16`
- **"Invalid distance function"**: Use one of: `l2`, `cosine`
- **"Index load failed"**: Check index file integrity and parameters

## Advanced Usage

### Batch Processing

For processing multiple query files:

```bash
#!/bin/bash
for query_file in queries/*.bin; do
    result_file="results/$(basename $query_file .bin)_results.txt"
    cargo run -p search_disk_index -- \
      --data_type float \
      --dist_fn l2 \
      --index_path_prefix data/index \
      --query_file $query_file \
      --result_file $result_file
done
```

### Performance Benchmarking

```bash
# Benchmark search performance
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix data/index \
  --query_file benchmark_queries.bin \
  --result_file benchmark_results.txt \
  --k 10 \
  --l 50 \
  --num_threads 8
```

## See Also

- [Disk Index Builder](../build_disk_index/README.md)
- [Memory Index Search](../search_memory_index/README.md)
- [Main DiskANN Documentation](../../../README.md) 