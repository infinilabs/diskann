# DiskANN Disk Index Builder

A command-line tool for building disk-based DiskANN indices for large-scale vector search.

## Overview

The `build_disk_index` command creates disk-based indices that can handle datasets larger than available memory. These indices are optimized for:

- **Large-scale datasets**: Can handle millions of vectors
- **Memory efficiency**: Most data resides on disk, with smart caching
- **High performance**: Optimized for both build and search operations
- **Flexible configuration**: Supports various data types and distance metrics

## Quick Start

### Basic Usage

```bash
# Build a basic disk index
cargo run -p build_disk_index \
  --data_type float \
  --dist_fn l2 \
  --data_path data/vectors.bin \
  --index_path_prefix data/index
```

### Advanced Usage

```bash
# Build with custom parameters for large datasets
cargo run -p build_disk_index \
  --data_type float \
  --dist_fn cosine \
  --data_path data/vectors.bin \
  --index_path_prefix data/index \
  --max_degree 128 \
  --Lbuild 200 \
  --num_threads 8 \
  --search_DRAM_budget 4.0 \
  --build_DRAM_budget 8.0 \
  --build_PQ_bytes 64 \
  --use_opq true
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data_type` | Data type of vectors | `float`, `int8`, `uint8`, `f16` |
| `--dist_fn` | Distance function | `l2`, `cosine` |
| `--data_path` | Input data file path | `data/vectors.bin` |
| `--index_path_prefix` | Output index file prefix | `data/index` |

### Optional Arguments

| Argument | Alias | Default | Description |
|----------|-------|---------|-------------|
| `--max_degree` | `-R` | `64` | Maximum graph degree |
| `--Lbuild` | `-L` | `100` | Build complexity (higher = better quality) |
| `--num_threads` | `-T` | CPU cores | Number of build threads |
| `--search_DRAM_budget` | `-B` | `0.0` | Search RAM limit in GB |
| `--build_DRAM_budget` | `-M` | `0.0` | Build RAM limit in GB |
| `--build_PQ_bytes` | | `0` | PQ compression bytes (0 = no compression) |
| `--use_opq` | | `false` | Use OPQ compression |

## Data Format

The input data should be in binary format with the following structure:

```
[number_of_points: u32][dimension: u32][data: T * number_of_points * dimension]
```

Where `T` is the data type (f32, i8, u8, or f16).

## Examples

### Example 1: Basic Index for Small Dataset

```bash
# Build a simple index for 100K vectors
cargo run -p build_disk_index \
  --data_type float \
  --dist_fn l2 \
  --data_path data/small_vectors.bin \
  --index_path_prefix data/small_index
```

### Example 2: High-Quality Index for Large Dataset

```bash
# Build a high-quality index for 1M+ vectors
cargo run -p build_disk_index \
  --data_type float \
  --dist_fn cosine \
  --data_path data/large_vectors.bin \
  --index_path_prefix data/large_index \
  --max_degree 128 \
  --Lbuild 200 \
  --num_threads 16 \
  --search_DRAM_budget 8.0 \
  --build_DRAM_budget 16.0
```

### Example 3: Compressed Index for Memory-Constrained Systems

```bash
# Build a compressed index to save memory
cargo run -p build_disk_index \
  --data_type float \
  --dist_fn l2 \
  --data_path data/vectors.bin \
  --index_path_prefix data/compressed_index \
  --build_PQ_bytes 64 \
  --use_opq true \
  --search_DRAM_budget 2.0
```

## Parameter Guidelines

### Graph Parameters

- **Max Degree (R)**: Controls graph connectivity
  - `32-64`: Good for most use cases
  - `128-256`: For high-recall applications
  - Higher values = better recall but slower search

- **Build Complexity (L)**: Controls build quality
  - `50-100`: Fast builds, good quality
  - `200-400`: High quality, slower builds
  - Higher values = better graph quality

### Memory Parameters

- **Search RAM Budget**: Memory used during search
  - `1-4 GB`: For small to medium datasets
  - `8-16 GB`: For large datasets
  - Higher values = faster search

- **Build RAM Budget**: Memory used during build
  - `4-8 GB`: For small datasets
  - `16-32 GB`: For large datasets
  - Higher values = faster builds

### Compression Parameters

- **PQ Bytes**: Product quantization compression
  - `0`: No compression (full precision)
  - `32-128`: Moderate compression
  - Higher values = smaller index but lower accuracy

- **Use OPQ**: Optimized product quantization
  - `false`: Standard PQ compression
  - `true`: Better compression with minimal accuracy loss

## Output Files

The index builder creates several files with the specified prefix:

- `{prefix}.index`: Main index file
- `{prefix}.data`: Vector data file
- `{prefix}.graph`: Graph structure file
- `{prefix}.pq`: PQ compression data (if enabled)

## Performance Tips

1. **For large datasets**: Use higher Lbuild values and more threads
2. **For memory-constrained systems**: Use PQ compression
3. **For high-recall applications**: Use higher max_degree values
4. **For fast builds**: Use more threads and higher build RAM budget

## Troubleshooting

### Common Issues

1. **Out of memory during build**
   - Increase `--build_DRAM_budget`
   - Use PQ compression to reduce memory usage

2. **Slow build times**
   - Increase `--num_threads`
   - Increase `--build_DRAM_budget`
   - Reduce `--Lbuild` for faster builds

3. **Poor search quality**
   - Increase `--max_degree`
   - Increase `--Lbuild`
   - Reduce PQ compression

4. **Slow search times**
   - Increase `--search_DRAM_budget`
   - Reduce `--max_degree`
   - Use PQ compression

### Error Messages

- **"Missing required arguments"**: Ensure all required arguments are provided
- **"Invalid data type"**: Use one of: `int8`, `uint8`, `float`, `f16`
- **"Invalid distance function"**: Use one of: `l2`, `cosine`
- **"File not found"**: Check that the data file exists and is readable

## Integration with Search

After building the index, you can use it with the search tools:

```bash
# Search the built index
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix data/index \
  --query_file queries.bin \
  --result_file results.txt
```

### Complete Workflow Example

1. **Build the disk index**:
   ```bash
   cargo run -p build_disk_index -- \
     --data_type float \
     --dist_fn l2 \
     --data_path vectors.bin \
     --index_path_prefix data/index \
     --max_degree 64 \
     --Lbuild 100 \
     --num_threads 8
   ```

2. **Search the built index**:
   ```bash
   cargo run -p search_disk_index -- \
     --data_type float \
     --dist_fn l2 \
     --index_path_prefix data/index \
     --query_file queries.bin \
     --result_file results.txt \
     --k 10 \
     --l 50 \
     --num_threads 8
   ```

### Loading and Searching Existing Indices

Once you've built a disk index, you can load and search it multiple times without rebuilding:

```bash
# Load and search an existing index
cargo run -p search_disk_index -- \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix data/index \
  --query_file new_queries.bin \
  --result_file new_results.txt
```

The index files are persistent and can be reused for multiple search operations.

## See Also

- [Memory Index Builder](../build_memory_index/README.md)
- [Search Tools](../search_memory_index/README.md)
- [Main DiskANN Documentation](../../../README.md) 