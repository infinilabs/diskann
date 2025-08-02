/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#[allow(clippy::module_inception)]
mod disk_index;
pub use disk_index::DiskIndex;

pub mod ann_disk_index;
pub mod utils;
pub mod percentile_stats;

pub mod aligned_file_reader;

#[cfg(target_os = "linux")]
pub mod linux_aligned_file_reader;
