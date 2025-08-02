# Dataset Generators

Tools for generating synthetic vector datasets for testing and benchmarking DiskANN performance.

## 🎯 Overview

This directory contains utilities to generate various types of vector datasets:
- Random vectors for performance testing
- Structured datasets for algorithm validation
- Large-scale datasets for benchmarking
- Multiple output formats (JSON, binary)

## 🚀 Quick Start

```bash
# Generate 1M random vectors
cargo run --bin generate-large-dataset

# Generate custom dataset
cargo run --bin generate-custom-dataset -- --size 100000 --dim 256
```

## 📁 Structure

```
dataset-generators/
├── src/
│   ├── main.rs                    # Main generator
│   └── generators/
│       ├── random.rs              # Random vector generator
│       ├── structured.rs          # Structured dataset generator
│       └── benchmark.rs           # Benchmark dataset generator
├── data/
│   ├── large_embeddings.json     # 1M vectors (13GB)
│   ├── small_dataset.json         # 10K vectors
│   └── benchmark_data.json        # Benchmark dataset
└── README.md                     # This file
```

## 🔧 Available Generators

### 1. Large Dataset Generator (`generate-large-dataset`)
- **Purpose**: Generate massive datasets for performance testing
- **Features**:
  - Configurable vector count (1K to 10M)
  - Configurable dimensions (64 to 2048)
  - Progress tracking
  - Memory-efficient generation
- **Output**: JSON format with metadata

### 2. Custom Dataset Generator (`generate-custom-dataset`)
- **Purpose**: Generate datasets with specific characteristics
- **Features**:
  - Custom vector distributions
  - Structured patterns
  - Multiple output formats
  - Validation tools

### 3. Benchmark Dataset Generator (`generate-benchmark-dataset`)
- **Purpose**: Generate datasets for algorithm benchmarking
- **Features**:
  - Known similarity patterns
  - Ground truth data
  - Multiple difficulty levels
  - Performance metrics

## 📊 Usage Examples

### Generate Large Dataset
```rust
use serde_json::{json, Value};
use rand::Rng;

fn generate_large_dataset(num_vectors: usize, dimension: usize) -> std::io::Result<()> {
    let mut rng = rand::thread_rng();
    let mut embeddings = Vec::new();
    
    for i in 0..num_vectors {
        let mut embedding = Vec::new();
        for _ in 0..dimension {
            embedding.push(rng.gen_range(-1.0..1.0));
        }
        
        embeddings.push(json!({
            "filename": format!("vector_{:07}", i),
            "embedding": embedding
        }));
        
        if i % 100_000 == 0 {
            println!("Generated {} vectors...", i);
        }
    }
    
    // Write to file
    let mut file = File::create("large_embeddings.json")?;
    let json_string = serde_json::to_string_pretty(&embeddings)?;
    file.write_all(json_string.as_bytes())?;
    
    println!("Generated {} vectors", num_vectors);
    Ok(())
}
```

### Generate Custom Dataset
```rust
fn generate_structured_dataset() -> std::io::Result<()> {
    let mut embeddings = Vec::new();
    
    // Generate clusters
    for cluster in 0..10 {
        let center = generate_random_vector(512);
        
        for i in 0..1000 {
            let vector = add_noise(&center, 0.1);
            embeddings.push(json!({
                "filename": format!("cluster_{}_vector_{}", cluster, i),
                "embedding": vector,
                "cluster": cluster
            }));
        }
    }
    
    // Write to file
    let mut file = File::create("structured_dataset.json")?;
    let json_string = serde_json::to_string_pretty(&embeddings)?;
    file.write_all(json_string.as_bytes())?;
    
    Ok(())
}
```

## 📈 Dataset Specifications

### Large Dataset (1M vectors, 512 dimensions)
- **Size**: 13GB JSON file
- **Format**: JSON array with metadata
- **Structure**: Random vectors with uniform distribution
- **Use Case**: Performance testing, benchmarking

### Small Dataset (10K vectors, 512 dimensions)
- **Size**: 130MB JSON file
- **Format**: JSON array
- **Structure**: Random vectors
- **Use Case**: Development, testing

### Benchmark Dataset (100K vectors, 512 dimensions)
- **Size**: 1.3GB JSON file
- **Format**: JSON with ground truth
- **Structure**: Clustered data with known similarities
- **Use Case**: Algorithm validation, accuracy testing

## 🔧 Configuration Options

### Command Line Parameters
```bash
# Basic usage
cargo run --bin generate-large-dataset

# Custom parameters
cargo run --bin generate-large-dataset -- \
    --vectors 1000000 \
    --dimensions 512 \
    --output large_dataset.json \
    --format json
```

### Configuration File
```json
{
  "dataset": {
    "name": "large_benchmark",
    "vectors": 1000000,
    "dimensions": 512,
    "distribution": "uniform",
    "output_format": "json",
    "output_file": "large_embeddings.json"
  },
  "generation": {
    "batch_size": 10000,
    "progress_interval": 100000,
    "validate": true,
    "compress": false
  }
}
```

## 📊 Output Formats

### JSON Format
```json
[
  {
    "filename": "vector_0000001",
    "embedding": [0.1, 0.2, 0.3, ...],
    "metadata": {
      "cluster": 0,
      "generated_at": "2024-01-01T00:00:00Z"
    }
  }
]
```

### Binary Format
```
[4 bytes] number of vectors (u32)
[4 bytes] vector dimension (u32)
[4 * dimension * num_vectors bytes] vector data (f32)
```

### CSV Format
```csv
filename,embedding_0,embedding_1,...,embedding_511
vector_0000001,0.1,0.2,0.3,...,0.512
vector_0000002,0.4,0.5,0.6,...,0.513
```

## 🛠️ Development

### Adding New Generators
1. Create new file in `src/generators/`
2. Implement generator trait
3. Add to main.rs
4. Update documentation

### Generator Template
```rust
pub trait DatasetGenerator {
    fn generate(&self, config: &GeneratorConfig) -> std::io::Result<()>;
    fn validate(&self, data: &[Vec<f32>]) -> bool;
    fn get_metadata(&self) -> DatasetMetadata;
}

pub struct GeneratorConfig {
    pub vectors: usize,
    pub dimensions: usize,
    pub output_file: String,
    pub format: OutputFormat,
}
```

## 📈 Performance

### Generation Speed
- **1M vectors (512 dims)**: ~2-5 minutes
- **10M vectors (512 dims)**: ~20-50 minutes
- **Memory usage**: 2-4 GB during generation

### Storage Requirements
- **JSON format**: ~6.8x larger than raw data
- **Binary format**: Raw data size
- **Compressed**: ~2-4x smaller than JSON

## 🔧 Troubleshooting

### Common Issues
1. **Out of memory**: Reduce batch size or use streaming
2. **Slow generation**: Increase batch size or use parallel processing
3. **File too large**: Use binary format or compression
4. **Invalid data**: Check dimension consistency

### Debug Mode
```bash
# Run with debug output
RUST_LOG=debug cargo run --bin generate-large-dataset
```

## 📚 Related Examples

- [Vector Search Demo](../vector-search-demo/) - Use generated datasets
- [Performance Benchmarks](../performance-benchmarks/) - Test with datasets
- [Storage Analysis](../storage-analysis/) - Analyze dataset storage

## 📄 License

This generator is part of the DiskANN project and is licensed under the MIT License. 