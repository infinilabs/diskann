/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use crate::log_error::LogError;
use crate::logger::indexlog::DiskIndexConstructionCheckpoint;
use crate::logger::indexlog::Log;
use crate::logger::indexlog::LogLevel;

use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;
use std::thread;

#[cfg(target_os = "windows")]
use win_etw_macros::trace_logging_provider;

trait MessagePublisher {
    fn publish(&self, log_level: LogLevel, message: &str);
}

#[cfg(target_os = "windows")]
// ETW provider - the GUID specified here is that of the default provider for Geneva Metric Extensions
// We are just using it as a placeholder until we have a version of OpenTelemetry exporter for Rust
#[trace_logging_provider(guid = "edc24920-e004-40f6-a8e1-0e6e48f39d84")]
trait EtwTraceProvider {
    fn write(msg: &str);
}

#[cfg(target_os = "windows")]
struct EtwPublisher {
    provider: EtwTraceProvider,
    publish_to_stdout: bool,
}

#[cfg(target_os = "windows")]
impl EtwPublisher {
    pub fn new() -> Result<Self, win_etw_provider::Error> {
        let provider = EtwTraceProvider::new();
        Ok(EtwPublisher {
            provider,
            publish_to_stdout: true,
        })
    }
}

#[cfg(target_os = "windows")]
fn log_level_to_etw(level: LogLevel) -> win_etw_provider::Level {
    match level {
        LogLevel::Error => win_etw_provider::Level::ERROR,
        LogLevel::Warn => win_etw_provider::Level::WARN,
        LogLevel::Info => win_etw_provider::Level::INFO,
        LogLevel::Debug => win_etw_provider::Level::VERBOSE,
        LogLevel::Trace => win_etw_provider::Level(6),
        LogLevel::Unspecified => win_etw_provider::Level(6),
    }
}

#[cfg(not(target_os = "windows"))]
// Cross-platform fallback publisher
struct CrossPlatformPublisher {
    publish_to_stdout: bool,
}

#[cfg(not(target_os = "windows"))]
impl CrossPlatformPublisher {
    pub fn new() -> Result<Self, LogError> {
        Ok(CrossPlatformPublisher {
            publish_to_stdout: true,
        })
    }
}

fn i32_to_log_level(value: i32) -> LogLevel {
    match value {
        0 => LogLevel::Unspecified,
        1 => LogLevel::Error,
        2 => LogLevel::Warn,
        3 => LogLevel::Info,
        4 => LogLevel::Debug,
        5 => LogLevel::Trace,
        _ => LogLevel::Unspecified,
    }
}

#[cfg(target_os = "windows")]
impl MessagePublisher for EtwPublisher {
    fn publish(&self, log_level: LogLevel, message: &str) {
        let options = win_etw_provider::EventOptions {
            level: Some(log_level_to_etw(log_level)),
            ..Default::default()
        };
        self.provider.write(Some(&options), message);

        if self.publish_to_stdout {
            println!("{}", message);
        }
    }
}

#[cfg(not(target_os = "windows"))]
impl MessagePublisher for CrossPlatformPublisher {
    fn publish(&self, _log_level: LogLevel, message: &str) {
        if self.publish_to_stdout {
            println!("{}", message);
        }
    }
}

struct MessageProcessor {
    sender: Mutex<Sender<Log>>,
}

impl MessageProcessor {
    pub fn start_processing() -> Self {
        let (sender, receiver) = mpsc::channel::<Log>();
        thread::spawn(move || -> Result<(), LogError> {
            for message in receiver {
                // Process the received message
                if let Some(indexlog) = message.index_construction_log {
                    let str = format!(
                        "Time for {}% of index build completed: {:.3} seconds, {:.3}B cycles",
                        indexlog.percentage_complete,
                        indexlog.time_spent_in_seconds,
                        indexlog.g_cycles_spent
                    );
                    publish(i32_to_log_level(indexlog.log_level), &str)?;
                }

                if let Some(disk_index_log) = message.disk_index_construction_log {
                    let checkpoint_name =
                        match DiskIndexConstructionCheckpoint::from_i32(disk_index_log.checkpoint)
                            .unwrap_or(DiskIndexConstructionCheckpoint::None)
                        {
                            DiskIndexConstructionCheckpoint::None => "None",
                            DiskIndexConstructionCheckpoint::PqConstruction => "PQ Construction",
                            DiskIndexConstructionCheckpoint::InmemIndexBuild => {
                                "In-memory Index Build"
                            }
                            DiskIndexConstructionCheckpoint::DiskLayout => "Disk Layout",
                        };
                    let str = format!(
                        "Disk index construction checkpoint [{}]: {:.3} seconds, {:.3}B cycles",
                        checkpoint_name,
                        disk_index_log.time_spent_in_seconds,
                        disk_index_log.g_cycles_spent
                    );
                    publish(i32_to_log_level(disk_index_log.log_level), &str)?;
                }

                if let Some(trace_log) = message.trace_log {
                    let str = format!("[{}] {}", trace_log.log_level, trace_log.log_line);
                    publish(i32_to_log_level(trace_log.log_level), &str)?;
                }

                if let Some(error_log) = message.error_log {
                    let str = format!("[ERROR] {}", error_log.error_message);
                    publish(i32_to_log_level(error_log.log_level), &str)?;
                }
            }
            Ok(())
        });
        MessageProcessor {
            sender: Mutex::new(sender),
        }
    }

    pub fn log(&self, message: Log) -> Result<(), LogError> {
        let sender = self.sender.lock().unwrap();
        sender
            .send(message)
            .map_err(|e| LogError::SendError { err: e })?;
        Ok(())
    }
}

lazy_static::lazy_static! {
    static ref MESSAGE_PROCESSOR: MessageProcessor = MessageProcessor::start_processing();
}

pub fn send_log(message: Log) -> Result<(), LogError> {
    MESSAGE_PROCESSOR.log(message)
}

#[cfg(target_os = "windows")]
fn publish(log_level: LogLevel, message: &str) -> Result<(), LogError> {
    let publisher = EtwPublisher::new()?;
    publisher.publish(log_level, message);
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn publish(log_level: LogLevel, message: &str) -> Result<(), LogError> {
    let publisher = CrossPlatformPublisher::new()?;
    publisher.publish(log_level, message);
    Ok(())
}
