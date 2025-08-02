# Vector Search Demo Application

A demonstration application showcasing DiskANN's memory-based vector search capabilities with real-world examples including image similarity search and HTML visualization.

## Overview

The Vector Search Demo Application demonstrates how to use DiskANN for practical vector search applications. It includes examples of building indices, inserting vectors, performing searches, and visualizing results. The application is designed to be educational and showcase real-world usage patterns.

## Features

- **Memory-based vector storage** - Efficient in-memory vector indexing
- **Vector insertion and deletion** - Dynamic index management
- **Similarity search** - Find nearest neighbors in vector space
- **HTML visualization** - Generate interactive HTML reports of search results
- **JSON data support** - Load vectors from JSON files
- **Performance monitoring** - Built-in timing and performance metrics
- **Real-world examples** - Practical use cases with sample data

## Quick Start

```bash
# Run the demo application
cargo run --bin vector-search-demo

# Or run with custom data
cargo run --bin vector-search-demo -- --input embeddings.json --output results.html

# The demo will generate a relations.html file with visual search results
```

## Usage Examples

### Basic Vector Search

```rust
use vector_search_demo::mem_ann_store::MemANNStore;
use vector::Metric;

// Create a memory-based vector store
let mut store = MemANNStore::new(
    Metric::L2,           // Distance metric
    512,                  // Vector dimension
    64,                   // Max degree
    100,                  // Search list size
    1.2,                  // Alpha parameter
    4,                    // Number of threads
    16,                   // Max points
)?;

// Insert vectors
let vectors = load_vectors_from_file("data.json")?;
store.insert_data(&vectors)?;

// Search for similar vectors
let query = vec![1.0; 512];
let mut indices = vec![0; 10];
let mut distances = vec![0.0; 10];

store.query(&query, 10, 50, &mut indices, &mut distances)?;

println!("Found {} similar vectors", indices.len());
```

### Image Similarity Search

```rust
// Load image embeddings
let embeddings = load_image_embeddings("images.json")?;

let mut store = MemANNStore::new(
    Metric::Cosine,       // Cosine distance for normalized embeddings
    512,                  // Image embedding dimension
    64,                   // Max degree
    100,                  // Search list size
    1.2,                  // Alpha parameter
    4,                    // Number of threads
    16,                   // Max points
)?;

store.insert_data(&embeddings)?;

// Find similar images
let query_embedding = extract_image_embedding("query.jpg")?;
let mut indices = vec![0; 5];
let mut distances = vec![0.0; 5];

store.query(&query_embedding, 5, 50, &mut indices, &mut distances)?;

// Generate HTML visualization
generate_html_report(&embeddings, &indices, &distances, "results.html")?;
```

## Data Formats

### JSON Input Format

```json
[
  {
    "filename": "image1.jpg",
    "embedding": [0.1, 0.2, 0.3, ...]
  },
  {
    "filename": "image2.jpg", 
    "embedding": [0.4, 0.5, 0.6, ...]
  }
]
```

### Binary Input Format

For large datasets, binary format is supported:

```
[4 bytes] number of vectors (u32)
[4 bytes] vector dimension (u32)
[4 * dimension * num_vectors bytes] vector data (f32)
```

## HTML Output

The demo generates a comprehensive HTML report (`relations.html`) that includes:

- **Visual Search Results**: Each query vector with its similar vectors displayed
- **Distance Metrics**: L2 distance scores for each result (lower = more similar)
- **Query Highlighting**: Query images highlighted in yellow
- **Interactive Layout**: Responsive design with image thumbnails
- **Performance Data**: Search timing and result statistics

The HTML file provides an intuitive way to visualize the vector search results and evaluate the quality of the similarity search algorithm.

## Configuration Options

### Store Parameters

- **Metric** - Distance function (L2, Cosine, InnerProduct)
- **Dimension** - Vector dimension (must match your data)
- **Max Degree** - Maximum graph degree (64-128 recommended)
- **Search List Size** - Beam width for search (50-200)
- **Alpha** - Graph density parameter (1.0-1.4)
- **Threads** - Number of parallel threads
- **Max Points** - Maximum vectors in store

### Search Parameters

- **k** - Number of results to return
- **l** - Search beam width (higher = more accurate, slower)
- **Distance threshold** - Filter results by distance

## HTML Visualization

The application generates interactive HTML reports showing:

- **Query visualization** - The input vector/image
- **Result grid** - Similar vectors with distances
- **Distance distribution** - Histogram of result distances
- **Performance metrics** - Search time and throughput

### Example HTML Output

```html
<!DOCTYPE html>
<html>
<head>
    <title>Vector Search Results</title>
    <style>
        .query-row { margin-bottom: 30px; }
        .image-box { display: inline-block; text-align: center; margin: 5px; }
        img { width: 100px; height: 100px; object-fit: cover; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Vector Search Results</h1>
    
    <div class="query-row">
        <h2>Query: image1.jpg</h2>
        <div class="image-box">
            <img src="images/image1.jpg" alt="Query Image">
        </div>
        
        <div class="results">
            <div class="image-box">
                <img src="images/image2.jpg" alt="Result 1">
                <div>Distance: 0.123</div>
            </div>
            <!-- More results... -->
        </div>
    </div>
</body>
</html>
```

## Performance Optimization

### Memory Usage

```rust
// For large datasets, consider data type conversion
let store = MemANNStore::new(
    Metric::L2,
    512,
    64,
    100,
    1.2,
    4,
    1000000,  // 1M vectors max
)?;
```

### Search Performance

```rust
// Optimize search parameters
let mut indices = vec![0; k];
let mut distances = vec![0.0; k];

// Use appropriate beam width
let l_value = k + 50;  // l = k + 50 is a good starting point

store.query(&query, k, l_value, &mut indices, &mut distances)?;
```

### Batch Operations

```rust
// Process multiple queries efficiently
let queries = load_queries("queries.json")?;

for (i, query) in queries.iter().enumerate() {
    let mut indices = vec![0; 10];
    let mut distances = vec![0.0; 10];
    
    store.query(query, 10, 50, &mut indices, &mut distances)?;
    
    // Process results
    process_results(i, &indices, &distances)?;
}
```

## Real-World Use Cases

### Image Similarity Search

```rust
// Load image embeddings from a deep learning model
let embeddings = load_image_embeddings("image_embeddings.json")?;

let mut store = MemANNStore::new(
    Metric::Cosine,  // Cosine for normalized embeddings
    512,             // ResNet/Inception embedding dimension
    64,
    100,
    1.2,
    4,
    100000,         // 100K images
)?;

store.insert_data(&embeddings)?;

// Find similar images
let query_embedding = extract_embedding("query_image.jpg")?;
let mut indices = vec![0; 10];
let mut distances = vec![0.0; 10];

store.query(&query_embedding, 10, 50, &mut indices, &mut distances)?;
```

### Document Similarity Search

```rust
// Load document embeddings (e.g., from BERT)
let embeddings = load_document_embeddings("documents.json")?;

let mut store = MemANNStore::new(
    Metric::L2,      // L2 for document embeddings
    768,             // BERT embedding dimension
    64,
    100,
    1.2,
    4,
    50000,          // 50K documents
)?;

store.insert_data(&embeddings)?;

// Find similar documents
let query_embedding = embed_document("query document")?;
let mut indices = vec![0; 20];
let mut distances = vec![0.0; 20];

store.query(&query_embedding, 20, 100, &mut indices, &mut distances)?;
```

### Recommendation Systems

```rust
// Load user/item embeddings
let embeddings = load_recommendation_embeddings("users.json")?;

let mut store = MemANNStore::new(
    Metric::InnerProduct,  // Inner product for recommendations
    128,                   // User embedding dimension
    64,
    100,
    1.2,
    4,
    1000000,              // 1M users
)?;

store.insert_data(&embeddings)?;

// Get recommendations
let user_embedding = get_user_embedding(user_id)?;
let mut indices = vec![0; 50];
let mut distances = vec![0.0; 50];

store.query(&user_embedding, 50, 200, &mut indices, &mut distances)?;
```

## Error Handling

### Common Issues

1. **Memory errors:**
   ```rust
   // Reduce max points or use smaller vectors
   let store = MemANNStore::new(
       Metric::L2,
       128,        // Smaller dimension
       32,         // Smaller max degree
       50,         // Smaller search list
       1.0,        // Lower alpha
       2,          // Fewer threads
       10000,      // Fewer max points
   )?;
   ```

2. **Dimension mismatch:**
   ```rust
   // Ensure vector dimensions match
   assert_eq!(vectors[0].len(), 512, "Vector dimension must be 512");
   ```

3. **Performance issues:**
   ```rust
   // Optimize search parameters
   let l_value = k * 2;  // l = 2*k for better accuracy
   store.query(&query, k, l_value, &mut indices, &mut distances)?;
   ```

## Development

### Building

```bash
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Running Examples

```bash
# Basic demo
cargo run --bin vector-search-demo

# With custom data
cargo run --bin vector-search-demo -- --input data.json --output results.html

# Performance test
cargo run --bin vector-search-demo -- --benchmark
```

## API Reference

### Core Types

- `MemANNStore<T>` - Memory-based vector store
- `Embeddings` - JSON data structure for vectors
- `SearchResult` - Search result with ID and distance

### Main Functions

- `MemANNStore::new()` - Create new vector store
- `MemANNStore::insert_data()` - Insert vectors
- `MemANNStore::query()` - Search for similar vectors
- `MemANNStore::save_to_file()` - Save store to disk
- `MemANNStore::load_from_file()` - Load store from disk

### Utility Functions

- `load_embeddings_from_json()` - Load vectors from JSON
- `generate_html_report()` - Generate HTML visualization
- `calculate_recall()` - Calculate search recall
- `benchmark_performance()` - Performance benchmarking

## Dependencies

- **diskann** - Core DiskANN library
- **vector** - Vector operations and metrics
- **serde** - JSON serialization
- **serde_json** - JSON parsing
- **chrono** - Timestamp handling

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Contributing

We welcome contributions! Please see the main [README](../README.md) for contribution guidelines. 