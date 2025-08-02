/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![allow(dead_code)] // Todo: Remove this when the disk index query code is complete.
use crate::common::ANNError;

#[cfg(target_os = "windows")]
use platform::{FileHandle, IOCompletionPort};

#[cfg(not(target_os = "windows"))]
pub struct FileHandle;
#[cfg(not(target_os = "windows"))]
impl Default for FileHandle {
    fn default() -> Self { FileHandle }
}

#[cfg(not(target_os = "windows"))]
pub struct IOCompletionPort;
#[cfg(not(target_os = "windows"))]
impl Default for IOCompletionPort {
    fn default() -> Self { IOCompletionPort }
}

// The IOContext struct for disk I/O. One for each thread.
pub struct IOContext {
    pub status: Status,
    pub file_handle: FileHandle,
    pub io_completion_port: IOCompletionPort,
}

impl Default for IOContext {
    fn default() -> Self {
        IOContext {
            status: Status::ReadWait,
            file_handle: FileHandle::default(),
            io_completion_port: IOCompletionPort::default(),
        }
    }
}

impl IOContext {
    pub fn new() -> Self {
        Self::default()
    }
}

pub enum Status {
    ReadWait,
    ReadSuccess,
    ReadFailed(ANNError),
    ProcessComplete,
}
