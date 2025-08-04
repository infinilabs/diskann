/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::env;

use diskann::{
    common::{ANNError, ANNResult},
    index::ann_disk_index::create_disk_index,
    model::{
        default_param_vals::ALPHA,
        vertex::{DIM_104, DIM_128, DIM_256, DIM_512},
        DiskIndexBuildParameters, IndexConfiguration, IndexWriteParametersBuilder,
    },
    storage::DiskIndexStorage,
    utils::round_up,
    utils::{load_metadata_from_file, Timer},
};

use diskann_vector::{FullPrecisionDistance, Half, Metric};

/// Configuration for building a disk index
#[derive(Debug)]
struct BuildConfig {
    data_type: String,
    dist_fn: String,
    data_path: String,
    index_path_prefix: String,
    max_degree: u32,
    l_build: u32,
    num_threads: u32,
    search_ram_limit_gb: f64,
    build_ram_limit_gb: f64,
    build_pq_bytes: u32,
    use_opq: bool,
}

impl BuildConfig {
    fn new() -> Self {
        Self {
            data_type: String::new(),
            dist_fn: String::new(),
            data_path: String::new(),
            index_path_prefix: String::new(),
            max_degree: 64,
            l_build: 100,
            num_threads: 0, // Will be set to CPU count if 0
            search_ram_limit_gb: 0.0,
            build_ram_limit_gb: 0.0,
            build_pq_bytes: 0,
            use_opq: false,
        }
    }

    fn validate(&self) -> ANNResult<()> {
        if self.data_type.is_empty() {
            return Err(ANNError::log_index_config_error(
                "data_type".to_string(),
                "Missing required argument: --data_type".to_string(),
            ));
        }
        if self.dist_fn.is_empty() {
            return Err(ANNError::log_index_config_error(
                "dist_fn".to_string(),
                "Missing required argument: --dist_fn".to_string(),
            ));
        }
        if self.data_path.is_empty() {
            return Err(ANNError::log_index_config_error(
                "data_path".to_string(),
                "Missing required argument: --data_path".to_string(),
            ));
        }
        if self.index_path_prefix.is_empty() {
            return Err(ANNError::log_index_config_error(
                "index_path_prefix".to_string(),
                "Missing required argument: --index_path_prefix".to_string(),
            ));
        }
        Ok(())
    }
}

/// The main function to build a disk index
#[allow(clippy::too_many_arguments)]
fn build_disk_index<T>(
    metric: Metric,
    data_path: &str,
    r: u32,
    l: u32,
    index_path_prefix: &str,
    num_threads: u32,
    search_ram_limit_gb: f64,
    index_build_ram_limit_gb: f64,
    num_pq_chunks: usize,
    use_opq: bool,
) -> ANNResult<()>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_512]: FullPrecisionDistance<T, DIM_512>,
{
    println!("📊 Loading data metadata...");
    let (data_num, data_dim) = load_metadata_from_file(data_path)?;
    println!("  - Data points: {}", data_num);
    println!("  - Dimensions: {}", data_dim);
    println!();

    println!("⚙️  Configuring index parameters...");
    let disk_index_build_parameters =
        DiskIndexBuildParameters::new(search_ram_limit_gb, index_build_ram_limit_gb)?;

    let index_write_parameters = IndexWriteParametersBuilder::new(l, r)
        .with_saturate_graph(true)
        .with_num_threads(num_threads)
        .build();

    let config = IndexConfiguration::new(
        metric,
        data_dim,
        round_up(data_dim as u64, 8_u64) as usize,
        data_num,
        num_pq_chunks > 0,
        num_pq_chunks,
        use_opq,
        0,
        1f32,
        index_write_parameters,
    );
    println!("  - Index configuration created");
    println!();

    println!("💾 Initializing disk storage...");
    let storage = DiskIndexStorage::new(data_path.to_string(), index_path_prefix.to_string())?;
    println!("  - Storage initialized");
    println!();

    println!("🔨 Creating disk index...");
    let mut index = create_disk_index::<T>(Some(disk_index_build_parameters), config, storage)?;
    println!("  - Index created");
    println!();

    println!("🚀 Starting index build process...");
    println!("  - This may take a while depending on dataset size and parameters");
    println!(
        "  - Building graph with {} points, {} dimensions",
        data_num, data_dim
    );
    println!("  - Max degree: {}, Build complexity: {}", r, l);
    println!();

    let timer = Timer::new();
    index.build("")?;
    let build_time = timer.elapsed().unwrap_or_default();

    println!("✅ Index build completed!");
    println!(
        "  - Total build time: {:.2} seconds",
        build_time.as_secs_f64()
    );
    println!(
        "  - Build rate: {:.2} points/second",
        data_num as f64 / build_time.as_secs_f64()
    );
    println!("  - Index files saved with prefix: {}", index_path_prefix);

    Ok(())
}

fn main() -> ANNResult<()> {
    // Initialize tracing
    diskann::instrumentation::init_tracing();

    let mut config = BuildConfig::new();
    let args: Vec<String> = env::args().collect();
    let mut iter = args.iter().skip(1).peekable();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--data_type" => {
                config.data_type = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "data_type".to_string(),
                            "Missing data type".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--dist_fn" => {
                config.dist_fn = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "dist_fn".to_string(),
                            "Missing distance function".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--data_path" => {
                config.data_path = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "data_path".to_string(),
                            "Missing data path".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--index_path_prefix" => {
                config.index_path_prefix = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "index_path_prefix".to_string(),
                            "Missing index path prefix".to_string(),
                        )
                    })?
                    .to_owned();
            }
            "--max_degree" | "-R" => {
                config.max_degree = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "max_degree".to_string(),
                            "Missing max degree".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "max_degree".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--Lbuild" | "-L" => {
                config.l_build = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "Lbuild".to_string(),
                            "Missing build complexity".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "Lbuild".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--num_threads" | "-T" => {
                config.num_threads = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "num_threads".to_string(),
                            "Missing number of threads".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "num_threads".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--build_PQ_bytes" => {
                config.build_pq_bytes = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "build_PQ_bytes".to_string(),
                            "Missing PQ bytes".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "build_PQ_bytes".to_string(),
                            format!("ParseIntError: {}", err),
                        )
                    })?;
            }
            "--use_opq" => {
                config.use_opq = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "use_opq".to_string(),
                            "Missing use_opq flag".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "use_opq".to_string(),
                            format!("ParseBoolError: {}", err),
                        )
                    })?;
            }
            "--search_DRAM_budget" | "-B" => {
                config.search_ram_limit_gb = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "search_DRAM_budget".to_string(),
                            "Missing search_DRAM_budget value".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "search_DRAM_budget".to_string(),
                            format!("ParseFloatError: {}", err),
                        )
                    })?;
            }
            "--build_DRAM_budget" | "-M" => {
                config.build_ram_limit_gb = iter
                    .next()
                    .ok_or_else(|| {
                        ANNError::log_index_config_error(
                            "build_DRAM_budget".to_string(),
                            "Missing build_DRAM_budget value".to_string(),
                        )
                    })?
                    .parse()
                    .map_err(|err| {
                        ANNError::log_index_config_error(
                            "build_DRAM_budget".to_string(),
                            format!("ParseFloatError: {}", err),
                        )
                    })?;
            }
            _ => {
                return Err(ANNError::log_index_config_error(
                    String::from(""),
                    format!("Unknown argument: {}", arg),
                ));
            }
        }
    }

    // Validate configuration
    config.validate()?;

    // Set default number of threads if not specified
    if config.num_threads == 0 {
        config.num_threads = num_cpus::get() as u32;
    }

    let metric = config
        .dist_fn
        .parse::<Metric>()
        .map_err(|err| ANNError::log_index_config_error("dist_fn".to_string(), err.to_string()))?;

    println!("🚀 Starting DiskANN disk index build");
    println!("=====================================");
    println!("Configuration:");
    println!("  Data type: {}", config.data_type);
    println!("  Distance function: {}", config.dist_fn);
    println!("  Data path: {}", config.data_path);
    println!("  Index path prefix: {}", config.index_path_prefix);
    println!("  Max degree (R): {}", config.max_degree);
    println!("  Build complexity (L): {}", config.l_build);
    println!("  Alpha: {}", ALPHA);
    println!("  Threads: {}", config.num_threads);
    println!("  Search RAM budget: {:.2} GB", config.search_ram_limit_gb);
    println!("  Build RAM budget: {:.2} GB", config.build_ram_limit_gb);
    println!("  PQ bytes: {}", config.build_pq_bytes);
    println!("  Use OPQ: {}", config.use_opq);
    println!();

    let result = match config.data_type.as_str() {
        "int8" => build_disk_index::<i8>(
            metric,
            &config.data_path,
            config.max_degree,
            config.l_build,
            &config.index_path_prefix,
            config.num_threads,
            config.search_ram_limit_gb,
            config.build_ram_limit_gb,
            config.build_pq_bytes as usize,
            config.use_opq,
        ),
        "uint8" => build_disk_index::<u8>(
            metric,
            &config.data_path,
            config.max_degree,
            config.l_build,
            &config.index_path_prefix,
            config.num_threads,
            config.search_ram_limit_gb,
            config.build_ram_limit_gb,
            config.build_pq_bytes as usize,
            config.use_opq,
        ),
        "float" => build_disk_index::<f32>(
            metric,
            &config.data_path,
            config.max_degree,
            config.l_build,
            &config.index_path_prefix,
            config.num_threads,
            config.search_ram_limit_gb,
            config.build_ram_limit_gb,
            config.build_pq_bytes as usize,
            config.use_opq,
        ),
        "f16" => build_disk_index::<Half>(
            metric,
            &config.data_path,
            config.max_degree,
            config.l_build,
            &config.index_path_prefix,
            config.num_threads,
            config.search_ram_limit_gb,
            config.build_ram_limit_gb,
            config.build_pq_bytes as usize,
            config.use_opq,
        ),
        _ => {
            println!("❌ Unsupported data type: {}", config.data_type);
            println!("Supported types: int8, uint8, float, f16");
            return Err(ANNError::log_index_config_error(
                "data_type".to_string(),
                format!("Invalid data type: {}", config.data_type),
            ));
        }
    };

    match result {
        Ok(_) => {
            println!("✅ Index build completed successfully!");
            println!(
                "📁 Index files saved with prefix: {}",
                config.index_path_prefix
            );
            Ok(())
        }
        Err(err) => {
            eprintln!("❌ Error building index: {:?}", err);
            Err(err)
        }
    }
}

fn print_help() {
    println!("🚀 DiskANN Disk Index Builder");
    println!("==============================");
    println!();
    println!("Builds a disk-based DiskANN index for large-scale vector search.");
    println!();
    println!("USAGE:");
    println!("  cargo run --bin build_disk_index [OPTIONS]");
    println!();
    println!("REQUIRED ARGUMENTS:");
    println!("  --data_type <TYPE>           Data type: int8, uint8, float, f16");
    println!("  --dist_fn <FUNCTION>         Distance function: l2, cosine");
    println!("  --data_path <PATH>           Input data file in binary format");
    println!("  --index_path_prefix <PREFIX> Path prefix for saving index files");
    println!();
    println!("OPTIONAL ARGUMENTS:");
    println!("  --help, -h                   Show this help message");
    println!("  --max_degree, -R <N>         Maximum graph degree (default: 64)");
    println!("  --Lbuild, -L <N>             Build complexity (default: 100)");
    println!("  --num_threads, -T <N>        Number of threads (default: CPU cores)");
    println!("  --search_DRAM_budget <GB>    Search RAM limit in GB");
    println!("  --build_DRAM_budget <GB>     Build RAM limit in GB");
    println!("  --build_PQ_bytes <N>         PQ compression bytes (default: 0)");
    println!("  --use_opq <BOOL>             Use OPQ compression (default: false)");
    println!();
    println!("EXAMPLES:");
    println!("  # Build a basic index");
    println!("  cargo run --bin build_disk_index \\");
    println!("    --data_type float \\");
    println!("    --dist_fn l2 \\");
    println!("    --data_path data/vectors.bin \\");
    println!("    --index_path_prefix data/index");
    println!();
    println!("  # Build with custom parameters");
    println!("  cargo run --bin build_disk_index \\");
    println!("    --data_type float \\");
    println!("    --dist_fn cosine \\");
    println!("    --data_path data/vectors.bin \\");
    println!("    --index_path_prefix data/index \\");
    println!("    --max_degree 128 \\");
    println!("    --Lbuild 200 \\");
    println!("    --num_threads 8 \\");
    println!("    --search_DRAM_budget 4.0 \\");
    println!("    --build_DRAM_budget 8.0");
    println!();
    println!("NOTES:");
    println!("  - Higher Lbuild values result in better graph quality but slower builds");
    println!("  - RAM budgets control memory usage during build and search");
    println!("  - PQ compression reduces index size but may affect accuracy");
    println!("  - Use OPQ for better compression with minimal accuracy loss");
}
