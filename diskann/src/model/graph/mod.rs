/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#[allow(clippy::module_inception)]
mod inmem_graph;
use cfg_if::cfg_if;
pub use inmem_graph::InMemoryGraph;

pub mod vertex_and_neighbors;
pub use vertex_and_neighbors::VertexAndNeighbors;

mod adjacency_list;
pub use adjacency_list::AdjacencyList;

cfg_if! {
if #[cfg(feature = "disk_store")] {
mod sector_graph;
pub use sector_graph::*;

mod disk_graph;
pub use disk_graph::*;
}
}
