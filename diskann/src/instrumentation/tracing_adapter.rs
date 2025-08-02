/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use tracing::{info, warn, error, debug, trace, info_span, Span};
use serde_json;

/// Tracing-based logger that provides compatibility with the existing logger interface
pub struct TracingLogger {
    span: Option<Span>,
}

impl TracingLogger {
    /// Create a new tracing logger
    pub fn new() -> Self {
        Self { span: None }
    }

    /// Create a new tracing logger with a span
    pub fn with_span(span: Span) -> Self {
        Self { span: Some(span) }
    }

    /// Log an info message
    pub fn info(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(span) = &self.span {
            span.in_scope(|| info!("{}", message));
        } else {
            info!("{}", message);
        }
        Ok(())
    }

    /// Log a warning message
    pub fn warn(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(span) = &self.span {
            span.in_scope(|| warn!("{}", message));
        } else {
            warn!("{}", message);
        }
        Ok(())
    }

    /// Log an error message
    pub fn error(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(span) = &self.span {
            span.in_scope(|| error!("{}", message));
        } else {
            error!("{}", message);
        }
        Ok(())
    }

    /// Log a debug message
    pub fn debug(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(span) = &self.span {
            span.in_scope(|| debug!("{}", message));
        } else {
            debug!("{}", message);
        }
        Ok(())
    }

    /// Log a trace message
    pub fn trace(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(span) = &self.span {
            span.in_scope(|| trace!("{}", message));
        } else {
            trace!("{}", message);
        }
        Ok(())
    }

    /// Log structured data
    pub fn info_with_fields(
        &self,
        message: &str,
        fields: serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(span) = &self.span {
            span.in_scope(|| {
                if let serde_json::Value::Object(map) = fields {
                    for (key, value) in map {
                        tracing::info!(%key, value = %value.as_str().unwrap_or(""), "{}", message);
                    }
                } else {
                    info!("{}", message);
                }
            });
        } else {
            if let serde_json::Value::Object(map) = fields {
                for (key, value) in map {
                    tracing::info!(%key, value = %value.as_str().unwrap_or(""), "{}", message);
                }
            } else {
                info!("{}", message);
            }
        }
        Ok(())
    }
}

/// Tracing-based trace logger for performance monitoring
pub struct TracingTraceLogger {
    span: Span,
}

impl TracingTraceLogger {
    /// Create a new trace logger
    pub fn new(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let span = info_span!("trace", name = name);
        Ok(Self { span })
    }

    /// Start a trace span
    pub fn start(&mut self, operation: &str) -> Result<(), Box<dyn std::error::Error>> {
        let _guard = self.span.enter();
        info!("Starting operation: {}", operation);
        Ok(())
    }

    /// End a trace span
    pub fn end(&mut self, operation: &str) -> Result<(), Box<dyn std::error::Error>> {
        info!("Completed operation: {}", operation);
        Ok(())
    }

    /// Finish the trace
    pub fn finish(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Trace completed");
        Ok(())
    }

    /// Record a metric
    pub fn record_metric(&mut self, name: &str, value: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.span.record(name, &value);
        Ok(())
    }
}

/// Tracing-based performance logger
pub struct TracingPerfLogger {
    span: Span,
}

impl TracingPerfLogger {
    /// Create a new performance logger
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let span = info_span!("performance");
        Ok(Self { span })
    }

    /// Start monitoring
    pub fn start_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let _guard = self.span.enter();
        info!("Performance monitoring started");
        Ok(())
    }

    /// Record a metric
    pub fn record_metric(&mut self, name: &str, value: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.span.record(name, &value);
        Ok(())
    }

    /// Get metrics (placeholder for compatibility)
    pub fn get_metrics(&self) -> Result<PerfMetrics, Box<dyn std::error::Error>> {
        Ok(PerfMetrics {
            cpu_percent: 0.0,
            memory_mb: 0.0,
            io_operations: 0,
            custom_metrics: std::collections::HashMap::new(),
        })
    }
}

/// Performance metrics structure for compatibility
pub struct PerfMetrics {
    pub cpu_percent: f64,
    pub memory_mb: f64,
    pub io_operations: u64,
    pub custom_metrics: std::collections::HashMap<String, f64>,
}

/// Initialize tracing with default configuration
pub fn init_logging() -> Result<(), Box<dyn std::error::Error>> {
    super::init_tracing();
    Ok(())
}

/// Initialize tracing with custom configuration
pub fn init_logging_with_config(
    level: &str,
    output_file: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    match level {
        "debug" => super::init_tracing_with_level(tracing::Level::DEBUG),
        "info" => super::init_tracing_with_level(tracing::Level::INFO),
        "warn" => super::init_tracing_with_level(tracing::Level::WARN),
        "error" => super::init_tracing_with_level(tracing::Level::ERROR),
        _ => super::init_tracing_with_level(tracing::Level::INFO),
    }

    if let Some(file) = output_file {
        if let Some(dir) = std::path::Path::new(file).parent() {
            super::init_tracing_with_file(
                dir.to_str().unwrap_or("logs"),
                std::path::Path::new(file)
                    .file_stem()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap_or("diskann"),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_logger() {
        init_logging().unwrap();
        
        let logger = TracingLogger::new();
        logger.info("Test info message").unwrap();
        logger.warn("Test warning message").unwrap();
        logger.error("Test error message").unwrap();
    }

    #[test]
    fn test_trace_logger() {
        init_logging().unwrap();
        
        let mut trace = TracingTraceLogger::new("test_trace").unwrap();
        trace.start("test_operation").unwrap();
        trace.record_metric("test_metric", 42.0).unwrap();
        trace.end("test_operation").unwrap();
        trace.finish().unwrap();
    }

    #[test]
    fn test_perf_logger() {
        init_logging().unwrap();
        
        let mut perf = TracingPerfLogger::new().unwrap();
        perf.start_monitoring().unwrap();
        perf.record_metric("cpu_usage", 75.5).unwrap();
        perf.record_metric("memory_usage", 1024.0).unwrap();
        let metrics = perf.get_metrics().unwrap();
        assert_eq!(metrics.cpu_percent, 0.0); // Placeholder value
    }
} 