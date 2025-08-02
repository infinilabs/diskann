/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! Disk graph storage

use std::sync::Arc;

use crate::{
    common::ANNResult,

    disk_search::aligned_file_reader::AlignedRead,
};

/// Graph storage for disk index
/// One thread has one storage instance
pub struct DiskGraphStorage {
    // Placeholder implementation
    pub dummy: i32,
}

impl DiskGraphStorage {
    pub fn new(_disk_graph_reader: Arc<dyn crate::disk_search::aligned_file_reader::AlignedFileReader>) -> ANNResult<Self> {
        Ok(Self { dummy: 0 })
    }

    pub fn read(&self, _read_requests: &mut [AlignedRead]) -> ANNResult<()> {
        Ok(())
    }
}
