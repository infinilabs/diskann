use diskann::{IndexBuilder, Metric};
use std::time::Instant;

fn main() {
    println!("Starting progress tracking...");

    // Generate test data
    let mut vectors = Vec::new();
    for i in 0..1000 {
        let mut vector = Vec::new();
        for j in 0..128 {
            vector.push((i + j) as f32 / 1000.0);
        }
        vectors.push(vector);
    }

    println!("Generated {} vectors", vectors.len());

    // Create index
    let start = Instant::now();
    println!("Creating index...");

    let mut index = IndexBuilder::new()
        .with_dimension(128)
        .with_metric(Metric::L2)
        .with_max_degree(64)
        .with_search_list_size(100)
        .with_alpha(1.2)
        .with_num_threads(4)
        .build_in_memory::<f32>()
        .unwrap();

    println!("Index created in {:?}", start.elapsed());

    // Insert vectors
    let insert_start = Instant::now();
    println!("Inserting vectors...");
    index.insert_batch(&vectors).unwrap();
    println!("Vectors inserted in {:?}", insert_start.elapsed());

    // Build index with progress tracking
    let build_start = Instant::now();
    println!("Building index...");
    println!("This will take time because it's:");
    println!(
        "1. Finding nearest neighbors for each of {} points",
        vectors.len()
    );
    println!("2. Building a graph with max degree {}", 64);
    println!("3. Optimizing the graph structure");
    println!("4. Each point needs to find ~{} nearest neighbors", 64);
    println!(
        "5. Total work: {} points × {} searches = {} operations",
        vectors.len(),
        64,
        vectors.len() * 64
    );
    println!(
        "6. With 4 threads, each thread handles ~{} points",
        vectors.len() / 4
    );

    index.build(&vectors).unwrap();
    println!("Index built in {:?}", build_start.elapsed());

    println!("Total time: {:?}", start.elapsed());

    // Test search performance
    let query = vectors[0].clone();
    let search_start = Instant::now();
    let results = index.search(&query, 10, 50).unwrap();
    let search_time = search_start.elapsed();

    println!("Search performance:");
    println!("- Found {} results", results.len());
    println!("- Search time: {:?}", search_time);
    println!(
        "- Throughput: {:.2} searches/second",
        1.0 / search_time.as_secs_f64()
    );

    // Show some results
    println!("Top 5 results:");
    for (i, result) in results.iter().take(5).enumerate() {
        println!(
            "  {}. ID: {}, Distance: {:.4}",
            i + 1,
            result.id,
            result.distance
        );
    }
}
