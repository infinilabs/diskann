/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use std::sync::Once;
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    fmt::{self, time::ChronoUtc},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};

static INIT: Once = Once::new();

/// Initialize tracing with default configuration
pub fn init_tracing() {
    init_tracing_with_level(Level::INFO);
}

/// Initialize tracing with custom log level
pub fn init_tracing_with_level(level: Level) {
    INIT.call_once(|| {
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(format!("diskann={}", level.as_str())));

        Registry::default()
            .with(env_filter)
            .with(fmt::layer().with_timer(ChronoUtc::rfc_3339()))
            .init();
    });
}

/// Initialize tracing with file output
pub fn init_tracing_with_file(log_dir: &str, log_file: &str) {
    INIT.call_once(|| {
        let file_appender = tracing_appender::rolling::RollingFileAppender::new(
            tracing_appender::rolling::Rotation::DAILY,
            log_dir,
            log_file,
        );

        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("diskann=info"));

        Registry::default()
            .with(env_filter)
            .with(fmt::layer().with_timer(ChronoUtc::rfc_3339()))
            .with(fmt::layer().with_writer(file_appender))
            .init();
    });
}

/// Initialize tracing with JSON output
pub fn init_tracing_json() {
    INIT.call_once(|| {
        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("diskann=info"));

        Registry::default()
            .with(env_filter)
            .with(fmt::layer().json().with_timer(ChronoUtc::rfc_3339()))
            .init();
    });
}

/// Create a custom subscriber for specific use cases
pub fn create_subscriber() -> impl Subscriber {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("diskann=info"));

    Registry::default()
        .with(env_filter)
        .with(fmt::layer().with_timer(ChronoUtc::rfc_3339()))
}

/// Structured logging helpers for DiskANN operations
pub mod structured {
    use tracing::{error, info, instrument, warn};

    /// Log index construction progress
    #[instrument(skip_all, fields(percentage_complete, time_spent, g_cycles))]
    pub fn log_index_construction_progress(
        percentage_complete: f32,
        time_spent: f32,
        g_cycles: f32,
    ) {
        info!(
            percentage_complete = percentage_complete,
            time_spent = time_spent,
            g_cycles = g_cycles,
            "Index construction progress"
        );
    }

    /// Log disk index construction checkpoint
    #[instrument(skip_all, fields(checkpoint, time_spent, g_cycles))]
    pub fn log_disk_index_checkpoint(checkpoint: &str, time_spent: f32, g_cycles: f32) {
        info!(
            checkpoint = checkpoint,
            time_spent = time_spent,
            g_cycles = g_cycles,
            "Disk index construction checkpoint"
        );
    }

    /// Log error with context
    #[instrument(skip_all, fields(error_message))]
    pub fn log_error_with_context(error_message: &str) {
        error!(error_message = error_message, "Operation failed");
    }

    /// Log performance metrics
    #[instrument(skip_all, fields(operation, duration_ms, memory_mb))]
    pub fn log_performance_metrics(operation: &str, duration_ms: f64, memory_mb: f64) {
        info!(
            operation = operation,
            duration_ms = duration_ms,
            memory_mb = memory_mb,
            "Performance metrics"
        );
    }

    /// Log vector operations
    #[instrument(skip_all, fields(operation, vector_count, dimension))]
    pub fn log_vector_operation(operation: &str, vector_count: usize, dimension: usize) {
        info!(
            operation = operation,
            vector_count = vector_count,
            dimension = dimension,
            "Vector operation"
        );
    }

    /// Log search operations
    #[instrument(skip_all, fields(query_count, k, l, avg_latency_ms))]
    pub fn log_search_operation(query_count: usize, k: usize, l: usize, avg_latency_ms: f64) {
        info!(
            query_count = query_count,
            k = k,
            l = l,
            avg_latency_ms = avg_latency_ms,
            "Search operation completed"
        );
    }
}

/// Performance tracing helpers
pub mod performance {
    use tracing::{info_span, instrument};

    /// Create a span for index construction
    pub fn index_construction_span() -> tracing::Span {
        info_span!("index_construction")
    }

    /// Create a span for disk index construction
    pub fn disk_index_construction_span() -> tracing::Span {
        info_span!("disk_index_construction")
    }

    /// Create a span for vector operations
    pub fn vector_operation_span(operation: &str) -> tracing::Span {
        info_span!("vector_operation", operation = operation)
    }

    /// Create a span for search operations
    pub fn search_operation_span() -> tracing::Span {
        info_span!("search_operation")
    }

    /// Instrument a function for performance tracing
    #[instrument(skip_all, fields(operation = %operation))]
    pub fn trace_operation<T, F>(operation: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tracing() {
        init_tracing();
        // Should not panic
    }

    #[test]
    fn test_structured_logging() {
        init_tracing();
        structured::log_index_construction_progress(50.0, 10.5, 1000.0);
        structured::log_error_with_context("Test error");
        structured::log_performance_metrics("test_op", 100.0, 512.0);
    }
}
