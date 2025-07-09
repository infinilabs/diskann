/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
mod inmem_index;
pub use inmem_index::ann_inmem_index::*;
pub use inmem_index::{InmemIndex, INIT_WARMUP_DATA_LEN};

#[cfg(feature = "disk_store")]
mod disk_index;

#[cfg(feature = "disk_store")]
pub use disk_index::*;

