/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
mod tracing_index_logger;
pub use tracing_index_logger::{IndexLogger, TracingIndexLogger};

mod tracing_disk_index_build_logger;
pub use tracing_disk_index_build_logger::{DiskIndexBuildLogger, TracingDiskIndexBuildLogger, DiskIndexConstructionCheckpoint};

mod tracing_init;
pub use tracing_init::performance;
pub use tracing_init::structured;
pub use tracing_init::{
    init_tracing, init_tracing_json, init_tracing_with_file, init_tracing_with_level,
};

mod tracing_adapter;
pub use tracing_adapter::{
    init_logging, init_logging_with_config, PerfMetrics, TracingLogger, TracingPerfLogger,
    TracingTraceLogger,
};
