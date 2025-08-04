use criterion::{criterion_group, criterion_main, Criterion};
use diskann::{IndexBuilder, Metric};
use rand::Rng;
use std::time::Instant;

/// Generate random vectors for benchmarking
fn generate_random_vectors(dimension: usize, count: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(count);

    for _ in 0..count {
        let mut vector = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            vector.push(rng.gen_range(-1.0..1.0));
        }
        vectors.push(vector);
    }

    vectors
}

fn benchmark_build_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_only");

    let vectors = generate_random_vectors(128, 1000);

    group.bench_function("build_index_only", |b| {
        b.iter(|| {
            let mut index = IndexBuilder::new()
                .with_dimension(128)
                .with_metric(Metric::L2)
                .with_max_degree(64)
                .with_search_list_size(100)
                .with_alpha(1.2)
                .with_num_threads(4)
                .build_in_memory::<f32>()
                .unwrap();

            // Insert vectors first
            index.insert_batch(&vectors).unwrap();
            
            // Then build the index
            let start = Instant::now();
            index.build(&vectors).unwrap();
            let elapsed = start.elapsed();
            
            // Print progress for debugging
            if elapsed.as_millis() > 100 {
                println!("Build took: {:?}", elapsed);
            }
        });
    });

    group.finish();
}

fn benchmark_smaller_dataset(c: &mut Criterion) {
    let mut group = c.benchmark_group("smaller_dataset");

    let vectors = generate_random_vectors(128, 100); // Smaller dataset

    group.bench_function("build_small_index", |b| {
        b.iter(|| {
            let mut index = IndexBuilder::new()
                .with_dimension(128)
                .with_metric(Metric::L2)
                .with_max_degree(32) // Smaller max degree
                .with_search_list_size(50) // Smaller search list
                .with_alpha(1.2)
                .with_num_threads(2) // Fewer threads
                .build_in_memory::<f32>()
                .unwrap();

            index.insert_batch(&vectors).unwrap();
            index.build(&vectors).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_build_only, benchmark_smaller_dataset);
criterion_main!(benches); 