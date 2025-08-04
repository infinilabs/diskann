use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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

/// Generate random query vectors
fn generate_query_vectors(dimension: usize, count: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut queries = Vec::with_capacity(count);

    for _ in 0..count {
        let mut query = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            query.push(rng.gen_range(-1.0..1.0));
        }
        queries.push(query);
    }

    queries
}

/// Benchmark search performance with different k values
fn benchmark_search_k_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_k_values");

    let vectors = generate_random_vectors(128, 100000);
    let queries = generate_query_vectors(128, 100);

    // Build index once
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
    index.build(&vectors).unwrap();

    let k_values = [1, 5, 10, 20, 50, 100];

    for &k in &k_values {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| {
                for query in &queries {
                    index.search(query, k, 100).unwrap();
                }
            });
        });
    }

    group.finish();
}

/// Benchmark search performance with different L values
fn benchmark_search_l_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_l_values");

    let vectors = generate_random_vectors(128, 100000);
    let queries = generate_query_vectors(128, 100);

    // Build index once
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
    index.build(&vectors).unwrap();

    let l_values = [10, 25, 50, 100, 200, 500];

    for &l in &l_values {
        group.bench_with_input(BenchmarkId::new("l", l), &l, |b, &l| {
            b.iter(|| {
                for query in &queries {
                    index.search(query, 10, l).unwrap();
                }
            });
        });
    }

    group.finish();
}

/// Benchmark search performance with different dataset sizes
fn benchmark_search_dataset_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_dataset_sizes");

    let dataset_sizes = [10000, 50000, 100000, 500000];
    let queries = generate_query_vectors(128, 50);

    for &size in &dataset_sizes {
        group.bench_with_input(BenchmarkId::new("dataset_size", size), &size, |b, &size| {
            b.iter(|| {
                let vectors = generate_random_vectors(128, size);

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
                index.build(&vectors).unwrap();

                for query in &queries {
                    index.search(query, 10, 100).unwrap();
                }
            });
        });
    }

    group.finish();
}

/// Benchmark search performance with different dimensions
fn benchmark_search_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_dimensions");

    let dimensions = [64, 128, 256, 512];
    let _queries = generate_query_vectors(512, 50);

    for &dim in &dimensions {
        group.bench_with_input(BenchmarkId::new("dimension", dim), &dim, |b, &dim| {
            b.iter(|| {
                let vectors = generate_random_vectors(dim, 50000);
                let queries = generate_query_vectors(dim, 50);

                let mut index = IndexBuilder::new()
                    .with_dimension(dim)
                    .with_metric(Metric::L2)
                    .with_max_degree(64)
                    .with_search_list_size(100)
                    .with_alpha(1.2)
                    .with_num_threads(4)
                    .build_in_memory::<f32>()
                    .unwrap();

                index.insert_batch(&vectors).unwrap();
                index.build(&vectors).unwrap();

                for query in &queries {
                    index.search(query, 10, 100).unwrap();
                }
            });
        });
    }

    group.finish();
}

/// Benchmark search performance with different distance metrics
fn benchmark_search_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_metrics");

    let vectors = generate_random_vectors(128, 100000);
    let queries = generate_query_vectors(128, 100);
    let metrics = [
        (Metric::L2, "L2"),
        (Metric::Cosine, "Cosine"),
        // InnerProduct not available in current implementation
    ];

    for (metric, name) in &metrics {
        group.bench_with_input(BenchmarkId::new("metric", name), metric, |b, &metric| {
            b.iter(|| {
                let mut index = IndexBuilder::new()
                    .with_dimension(128)
                    .with_metric(metric)
                    .with_max_degree(64)
                    .with_search_list_size(100)
                    .with_alpha(1.2)
                    .with_num_threads(4)
                    .build_in_memory::<f32>()
                    .unwrap();

                index.insert_batch(&vectors).unwrap();
                index.build(&vectors).unwrap();

                for query in &queries {
                    index.search(query, 10, 100).unwrap();
                }
            });
        });
    }

    group.finish();
}

/// Benchmark batch search performance
fn benchmark_batch_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_search");

    let vectors = generate_random_vectors(128, 100000);
    let queries = generate_query_vectors(128, 1000);

    // Build index once
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
    index.build(&vectors).unwrap();

    let batch_sizes = [1, 10, 50, 100, 500];

    for &batch_size in &batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for query in queries.iter().take(batch_size) {
                        results.push(index.search(query, 10, 100).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark disk index search performance
fn benchmark_disk_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("disk_search");

    let dataset_sizes = [50000, 100000, 500000];
    let queries = generate_query_vectors(128, 50);

    for &size in &dataset_sizes {
        group.bench_with_input(BenchmarkId::new("disk_search", size), &size, |b, &size| {
            b.iter(|| {
                let _vectors = generate_random_vectors(128, size);
                let temp_dir = tempfile::tempdir().unwrap();
                let index_path = temp_dir.path().join("benchmark_index");

                let index = IndexBuilder::new()
                    .with_dimension(128)
                    .with_metric(Metric::L2)
                    .with_max_degree(64)
                    .with_search_list_size(100)
                    .with_alpha(1.2)
                    .with_num_threads(4)
                    .build_disk_index::<f32>(index_path.to_str().unwrap())
                    .unwrap();

                // DiskIndex doesn't support insert_batch and build methods
                // These operations would need to be done differently for disk indices

                for query in &queries {
                    index.search(query, 10, 100).unwrap();
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_search_k_values,
    benchmark_search_l_values,
    benchmark_search_dataset_sizes,
    benchmark_search_dimensions,
    benchmark_search_metrics,
    benchmark_batch_search,
    benchmark_disk_search
);
criterion_main!(benches);
