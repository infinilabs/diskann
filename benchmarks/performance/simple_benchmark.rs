use criterion::{criterion_group, criterion_main, Criterion};
use diskann::{IndexBuilder, Metric};
use rand::Rng;

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

fn benchmark_simple_index_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_index_building");

    let vectors = generate_random_vectors(128, 1000);

    group.bench_function("build_in_memory_index", |b| {
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

            index.insert_batch(&vectors).unwrap();
            // Skip the build step for faster benchmarking
            // index.build(&vectors).unwrap();
        });
    });

    group.finish();
}

fn benchmark_simple_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_search");

    let vectors = generate_random_vectors(128, 1000);
    let query = generate_random_vectors(128, 1)[0].clone();

    let mut index = IndexBuilder::new()
        .with_dimension(128)
        .with_metric(Metric::L2)
        .with_max_degree(64)
        .with_search_list_size(100)
        .with_alpha(1.2)
        .with_num_threads(4)

        .build_in_memory::<f32>()
        .unwrap();

    index.insert_batch(&vectors).unwrap();
    // Skip the build step for faster benchmarking
    // index.build(&vectors).unwrap();

    group.bench_function("search_k10_l50", |b| {
        b.iter(|| {
            index.search(&query, 10, 50).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_simple_index_building,
    benchmark_simple_search
);
criterion_main!(benches);
