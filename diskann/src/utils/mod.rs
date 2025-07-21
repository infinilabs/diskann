/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use cfg_if::cfg_if;
pub mod file_util;
pub use file_util::*;

#[allow(clippy::module_inception)]
pub mod utils;
pub use utils::*;

pub mod bit_vec_extension;
pub use bit_vec_extension::*;

pub mod rayon_util;
pub use rayon_util::*;

cfg_if! {
if #[cfg(target_os = "windows")] {
    pub mod timer;
    pub use timer::*;
} else {
    pub mod perf_linux;
    pub use perf_linux::*;

    pub mod timer_linux;
    pub use timer_linux::*;
}
}

pub mod cached_reader;
pub use cached_reader::*;

pub mod cached_writer;
pub use cached_writer::*;

pub mod partition;
pub use partition::*;

pub mod math_util;
pub use math_util::*;

pub mod kmeans;
pub use kmeans::*;
