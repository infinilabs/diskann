use diskann::{IndexBuilder, Metric};
use std::time::Instant;

fn main() {
    println!("Starting debug benchmark...");

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

    // Build index
    let build_start = Instant::now();
    println!("Building index...");
    index.build(&vectors).unwrap();
    println!("Index built in {:?}", build_start.elapsed());

    println!("Total time: {:?}", start.elapsed());
}
