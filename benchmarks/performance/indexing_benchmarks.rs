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

fn benchmark_index_building_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_building_sizes");

    let sizes = [1000, 5000, 10000, 25000];

    for size in sizes {
        let vectors = generate_random_vectors(128, size);

        group.bench_with_input(
            BenchmarkId::new("build_in_memory", size),
            &size,
            |b, &_size| {
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
                    index.build(&vectors).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_index_building_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_building_dimensions");

    let dimensions = [64, 128, 256, 512];

    for dim in dimensions {
        let vectors = generate_random_vectors(dim, 5000);

        group.bench_with_input(BenchmarkId::new("build_in_memory", dim), &dim, |b, &dim| {
            b.iter(|| {
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
            });
        });
    }

    group.finish();
}

fn benchmark_index_building_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_building_metrics");

    let metrics = [(Metric::L2, "L2"), (Metric::Cosine, "Cosine")];

    for (metric, name) in metrics {
        let vectors = generate_random_vectors(128, 5000);

        group.bench_with_input(
            BenchmarkId::new("build_in_memory", name),
            &name,
            |b, &_name| {
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
                });
            },
        );
    }

    group.finish();
}

fn benchmark_index_building_parameters(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_building_parameters");

    let max_degrees = [32, 64, 128];
    let alphas = [1.0, 1.2, 1.4];

    for max_degree in max_degrees {
        let vectors = generate_random_vectors(128, 5000);

        group.bench_with_input(
            BenchmarkId::new("max_degree", max_degree),
            &max_degree,
            |b, &max_degree| {
                b.iter(|| {
                    let mut index = IndexBuilder::new()
                        .with_dimension(128)
                        .with_metric(Metric::L2)
                        .with_max_degree(max_degree)
                        .with_search_list_size(100)
                        .with_alpha(1.2)
                        .with_num_threads(4)
                        .build_in_memory::<f32>()
                        .unwrap();

                    index.insert_batch(&vectors).unwrap();
                    index.build(&vectors).unwrap();
                });
            },
        );
    }

    for alpha in alphas {
        let vectors = generate_random_vectors(128, 5000);

        group.bench_with_input(BenchmarkId::new("alpha", alpha), &alpha, |b, &alpha| {
            b.iter(|| {
                let mut index = IndexBuilder::new()
                    .with_dimension(128)
                    .with_metric(Metric::L2)
                    .with_max_degree(64)
                    .with_search_list_size(100)
                    .with_alpha(alpha)
                    .with_num_threads(4)
                    .build_in_memory::<f32>()
                    .unwrap();

                index.insert_batch(&vectors).unwrap();
                index.build(&vectors).unwrap();
            });
        });
    }

    group.finish();
}

fn benchmark_index_building_threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_building_threads");

    let thread_counts = [1, 2, 4, 8];

    for threads in thread_counts {
        let vectors = generate_random_vectors(128, 10000);

        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    let mut index = IndexBuilder::new()
                        .with_dimension(128)
                        .with_metric(Metric::L2)
                        .with_max_degree(64)
                        .with_search_list_size(100)
                        .with_alpha(1.2)
                        .with_num_threads(threads)
                        .build_in_memory::<f32>()
                        .unwrap();

                    index.insert_batch(&vectors).unwrap();
                    index.build(&vectors).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_index_building_sizes,
    benchmark_index_building_dimensions,
    benchmark_index_building_metrics,
    benchmark_index_building_parameters,
    benchmark_index_building_threads
);
criterion_main!(benches);
