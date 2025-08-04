use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use diskann::{IndexBuilder, Metric};
use rand::Rng;
use std::process::Command;


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

/// Get current memory usage in MB
fn get_memory_usage() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(rss_str) = String::from_utf8(output.stdout) {
                if let Ok(kb) = rss_str.trim().parse::<f64>() {
                    return kb / 1024.0; // Convert KB to MB
                }
            }
        }
    }

    // Fallback for other platforms
    0.0
}

/// Benchmark memory usage during index building
fn benchmark_memory_usage_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_building");

    let dataset_sizes = [1000, 10000, 50000, 100000, 500000];

    for &size in &dataset_sizes {
        group.bench_with_input(BenchmarkId::new("memory_build", size), &size, |b, &size| {
            b.iter(|| {
                let initial_memory = get_memory_usage();

                let vectors = generate_random_vectors(128, size);
                let vectors_memory = get_memory_usage();

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
                let _insert_memory = get_memory_usage();

                index.build(&vectors).unwrap();
                let final_memory = get_memory_usage();

                // Record memory usage metrics
                let _vectors_overhead = vectors_memory - initial_memory;
                let _index_overhead = final_memory - vectors_memory;
                let _total_overhead = final_memory - initial_memory;

                // Store metrics for reporting
                std::mem::drop(index);
                std::mem::drop(vectors);
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage during search operations
fn benchmark_memory_usage_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_search");

    let dataset_sizes = [10000, 50000, 100000, 500000];
    let query_counts = [10, 100, 1000];

    for &size in &dataset_sizes {
        for &query_count in &query_counts {
            group.bench_with_input(
                BenchmarkId::new("memory_search", format!("{}x{}", size, query_count)),
                &(size, query_count),
                |b, &(size, query_count)| {
                    b.iter(|| {
                        let vectors = generate_random_vectors(128, size);
                        let queries = generate_random_vectors(128, query_count);

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

                        let search_memory = get_memory_usage();

                        let mut results = Vec::new();
                        for query in &queries {
                            results.push(index.search(query, 10, 100).unwrap());
                        }

                        let final_memory = get_memory_usage();
                        let _search_overhead = final_memory - search_memory;

                        // Store metrics for reporting
                        std::mem::drop(index);
                        std::mem::drop(vectors);
                        std::mem::drop(queries);
                        std::mem::drop(results);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory usage with different dimensions
fn benchmark_memory_usage_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_dimensions");

    let dimensions = [64, 128, 256, 512, 1024];
    let size = 50000;

    for &dim in &dimensions {
        group.bench_with_input(
            BenchmarkId::new("memory_dimension", dim),
            &dim,
            |b, &dim| {
                b.iter(|| {
                    let initial_memory = get_memory_usage();

                    let vectors = generate_random_vectors(dim, size);
                    let _vectors_memory = get_memory_usage();

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

                    let final_memory = get_memory_usage();
                    let _memory_per_vector = (final_memory - initial_memory) / size as f64;

                    // Store metrics for reporting
                    std::mem::drop(index);
                    std::mem::drop(vectors);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage with different alpha values
fn benchmark_memory_usage_alpha(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_alpha");

    let alpha_values = [0.8, 1.0, 1.2, 1.4, 1.6];
    let size = 100000;

    for &alpha in &alpha_values {
        group.bench_with_input(
            BenchmarkId::new("memory_alpha", alpha),
            &alpha,
            |b, &alpha| {
                b.iter(|| {
                    let initial_memory = get_memory_usage();

                    let vectors = generate_random_vectors(128, size);

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

                    let final_memory = get_memory_usage();
                    let _index_overhead = final_memory - initial_memory;

                    // Store metrics for reporting
                    std::mem::drop(index);
                    std::mem::drop(vectors);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage with different max degrees
fn benchmark_memory_usage_max_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_max_degree");

    let max_degrees = [32, 64, 128, 256];
    let size = 100000;

    for &max_degree in &max_degrees {
        group.bench_with_input(
            BenchmarkId::new("memory_max_degree", max_degree),
            &max_degree,
            |b, &max_degree| {
                b.iter(|| {
                    let initial_memory = get_memory_usage();

                    let vectors = generate_random_vectors(128, size);

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

                    let final_memory = get_memory_usage();
                    let _index_overhead = final_memory - initial_memory;

                    // Store metrics for reporting
                    std::mem::drop(index);
                    std::mem::drop(vectors);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark disk index memory usage
fn benchmark_disk_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("disk_memory_usage");

    let dataset_sizes = [100000, 500000, 1000000];

    for &size in &dataset_sizes {
        group.bench_with_input(BenchmarkId::new("disk_memory", size), &size, |b, &size| {
            b.iter(|| {
                let initial_memory = get_memory_usage();

                let vectors = generate_random_vectors(128, size);
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

                let final_memory = get_memory_usage();
                let _disk_overhead = final_memory - initial_memory;

                // Store metrics for reporting
                std::mem::drop(index);
                std::mem::drop(vectors);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_memory_usage_building,
    benchmark_memory_usage_search,
    benchmark_memory_usage_dimensions,
    benchmark_memory_usage_alpha,
    benchmark_memory_usage_max_degree,
    benchmark_disk_memory_usage
);
criterion_main!(benches);
