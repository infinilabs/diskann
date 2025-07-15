/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
pub mod neighbor;
use cfg_if::cfg_if;
pub use neighbor::Neighbor;
pub use neighbor::NeighborPriorityQueue;

pub mod data_store;
pub use data_store::InmemDataset;

pub mod graph;
pub use graph::InMemoryGraph;
pub use graph::VertexAndNeighbors;

pub mod configuration;
pub use configuration::*;

pub mod scratch;
pub use scratch::*;

pub mod vertex;
pub use vertex::Vertex;

cfg_if! {
    if #[cfg(feature = "disk_store")] {
        pub mod pq;
        pub use pq::*;

        cfg_if! {
            if #[cfg(target_os = "windows")] {
                pub mod windows_aligned_file_reader;
                pub use windows_aligned_file_reader::*;
            } else {
                //pub mod linux_aligned_file_reader;
                //pub use linux_aligned_file_reader::*;
            }
        }
    }
}
