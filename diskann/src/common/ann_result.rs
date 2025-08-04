/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::fmt;
use std::io;
use std::sync::PoisonError;

use thiserror::Error;

use crate::model::configuration::index_configuration::IndexConfiguration;

use tracing as trace;

/// ANN Error types
#[derive(Error, Debug)]
pub enum ANNError {
    #[error("IndexError: {err}")]
    IndexError { err: String },

    #[error("IndexConfigError: {parameter} is invalid, err={err}")]
    IndexConfigError { parameter: String, err: String },

    #[error("TryFromIntError: {err}")]
    TryFromIntError {
        #[from]
        err: std::num::TryFromIntError,
    },

    #[error("IOError: {err}")]
    IOError {
        #[from]
        err: io::Error,
    },

    #[error("MemoryAllocLayoutError: {err}")]
    MemoryAllocLayoutError {
        #[from]
        err: std::alloc::LayoutError,
    },

    #[error("LockPoisonError: {err}")]
    LockPoisonError { err: String },

    #[error("DiskIOAlignmentError: {err}")]
    DiskIOAlignmentError { err: String },

    #[error("LogError: {err}")]
    LogError { err: String },

    #[error("PQError: {err}")]
    PQError { err: String },

    #[error("Error try creating array from slice: {err}")]
    TryFromSliceError {
        #[from]
        err: std::array::TryFromSliceError,
    },

    #[error("Error file size not match: {message}, {actual_size} != {expected_actual_file_size}")]
    FileSizeNotMatchError {
        message: String,
        actual_size: u64,
        expected_actual_file_size: u64,
    },
}

impl ANNError {
    pub fn log_index_error(err: String) -> Self {
        Self::IndexError { err }
    }

    pub fn log_index_config_error(parameter: String, err: String) -> Self {
        Self::IndexConfigError { parameter, err }
    }

    pub fn log_io_error(err: io::Error) -> Self {
        Self::IOError { err }
    }

    pub fn log_memory_alloc_layout_error(err: std::alloc::LayoutError) -> Self {
        Self::MemoryAllocLayoutError { err }
    }

    pub fn log_lock_poison_error(err: String) -> Self {
        Self::LockPoisonError { err }
    }

    pub fn log_disk_io_alignment_error(err: String) -> Self {
        Self::DiskIOAlignmentError { err }
    }

    pub fn log_error(err: String) -> Self {
        Self::LogError { err }
    }

    pub fn log_pq_error(err: String) -> Self {
        Self::PQError { err }
    }

    pub fn log_file_size_not_match_error(
        message: String,
        actual_size: u64,
        expected_actual_file_size: u64,
    ) -> Self {
        Self::FileSizeNotMatchError {
            message,
            actual_size,
            expected_actual_file_size,
        }
    }
}

/// ANN Result type
pub type ANNResult<T> = Result<T, ANNError>;

impl<T> From<PoisonError<T>> for ANNError {
    fn from(err: PoisonError<T>) -> Self {
        Self::LockPoisonError {
            err: err.to_string(),
        }
    }
}

impl From<ANNError> for io::Error {
    fn from(err: ANNError) -> Self {
        io::Error::new(io::ErrorKind::Other, err.to_string())
    }
}

#[cfg(test)]
mod ann_result_test {
    use super::*;

    #[test]
    fn ann_err_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ANNError>();
    }
}
