/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![allow(dead_code)] // Todo: Remove this when the disk index query code is complete.
use std::collections::HashSet;

use crate::model::neighbor::{Neighbor, NeighborPriorityQueue};
use crate::model::scratch::pq_scratch::PQQueryScratch;

/// Scratch space for SSD-based query operations
pub struct SSDQueryScratch<T> {
    /// Aligned query vector for distance calculations
    pub aligned_query_T: Vec<T>,
    /// Coordinate scratch buffer for temporary data
    pub coord_scratch: Vec<T>,
    /// Sector scratch buffer for disk I/O
    pub sector_scratch: Vec<u8>,
    /// Current sector index
    pub sector_idx: u64,
    /// Set of visited node IDs
    pub visited: HashSet<u64>,
    /// Priority queue for beam search
    pub retset: NeighborPriorityQueue,
    /// Full result set for final sorting
    pub full_retset: Vec<Neighbor>,
    /// PQ query scratch space
    pub pq_scratch: PQQueryScratch,
}

impl<T> SSDQueryScratch<T>
where
    T: Default + Copy,
{
    /// Create a new SSD query scratch space
    pub fn new(
        aligned_dim: usize,
        sector_size: usize,
        max_queue_size: usize,
        pq_scratch: PQQueryScratch,
    ) -> Self {
        Self {
            aligned_query_T: vec![T::default(); aligned_dim],
            coord_scratch: vec![T::default(); aligned_dim],
            sector_scratch: vec![0u8; sector_size],
            sector_idx: 0,
            visited: HashSet::new(),
            retset: NeighborPriorityQueue::with_capacity(max_queue_size),
            full_retset: Vec::new(),
            pq_scratch,
        }
    }

    /// Reset the scratch space for a new query
    pub fn reset(&mut self) {
        self.sector_idx = 0;
        self.visited.clear();
        self.retset.clear();
        self.full_retset.clear();
    }

    /// Get mutable reference to aligned query buffer
    pub fn aligned_query_T(&mut self) -> &mut [T] {
        &mut self.aligned_query_T
    }

    /// Get mutable reference to coordinate scratch buffer
    pub fn coord_scratch(&mut self) -> &mut [T] {
        &mut self.coord_scratch
    }

    /// Get mutable reference to sector scratch buffer
    pub fn sector_scratch(&mut self) -> &mut [u8] {
        &mut self.sector_scratch
    }

    /// Get the current sector index
    pub fn sector_idx(&self) -> u64 {
        self.sector_idx
    }

    /// Set the sector index
    pub fn set_sector_idx(&mut self, idx: u64) {
        self.sector_idx = idx;
    }

    /// Increment the sector index
    pub fn increment_sector_idx(&mut self) {
        self.sector_idx += 1;
    }

    /// Check if a node has been visited
    pub fn is_visited(&self, node_id: u64) -> bool {
        self.visited.contains(&node_id)
    }

    /// Mark a node as visited
    pub fn mark_visited(&mut self, node_id: u64) {
        self.visited.insert(node_id);
    }

    /// Get the number of visited nodes
    pub fn visited_count(&self) -> usize {
        self.visited.len()
    }

    /// Get mutable reference to the priority queue
    pub fn retset_mut(&mut self) -> &mut NeighborPriorityQueue {
        &mut self.retset
    }

    /// Get reference to the priority queue
    pub fn retset(&self) -> &NeighborPriorityQueue {
        &self.retset
    }

    /// Get mutable reference to the full result set
    pub fn full_retset_mut(&mut self) -> &mut Vec<Neighbor> {
        &mut self.full_retset
    }

    /// Get reference to the full result set
    pub fn full_retset(&self) -> &[Neighbor] {
        &self.full_retset
    }

    /// Get mutable reference to PQ scratch space
    pub fn pq_scratch_mut(&mut self) -> &mut PQQueryScratch {
        &mut self.pq_scratch
    }

    /// Get reference to PQ scratch space
    pub fn pq_scratch(&self) -> &PQQueryScratch {
        &self.pq_scratch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::scratch::pq_scratch::PQQueryScratch;

    #[test]
    fn test_ssd_query_scratch_creation() {
        let pq_scratch = PQQueryScratch::new(128, 256);
        let scratch = SSDQueryScratch::<f32>::new(128, 4096, 100, pq_scratch);

        assert_eq!(scratch.aligned_query_T.len(), 128);
        assert_eq!(scratch.coord_scratch.len(), 128);
        assert_eq!(scratch.sector_scratch.len(), 4096);
        assert_eq!(scratch.sector_idx, 0);
        assert!(scratch.visited.is_empty());
    }

    #[test]
    fn test_ssd_query_scratch_reset() {
        let pq_scratch = PQQueryScratch::new(128, 256);
        let mut scratch = SSDQueryScratch::<f32>::new(128, 4096, 100, pq_scratch);

        // Add some data
        scratch.mark_visited(1);
        scratch.mark_visited(2);
        scratch.retset.insert(Neighbor::new(1, 0.5));
        scratch.full_retset.push(Neighbor::new(1, 0.5));
        scratch.set_sector_idx(5);

        // Reset
        scratch.reset();

        assert_eq!(scratch.visited_count(), 0);
        assert_eq!(scratch.retset.size(), 0);
        assert_eq!(scratch.full_retset.len(), 0);
        assert_eq!(scratch.sector_idx, 0);
    }

    #[test]
    fn test_ssd_query_scratch_visited_tracking() {
        let pq_scratch = PQQueryScratch::new(128, 256);
        let mut scratch = SSDQueryScratch::<f32>::new(128, 4096, 100, pq_scratch);

        assert!(!scratch.is_visited(1));
        scratch.mark_visited(1);
        assert!(scratch.is_visited(1));
        assert_eq!(scratch.visited_count(), 1);
    }

    #[test]
    fn test_ssd_query_scratch_sector_management() {
        let pq_scratch = PQQueryScratch::new(128, 256);
        let mut scratch = SSDQueryScratch::<f32>::new(128, 4096, 100, pq_scratch);

        assert_eq!(scratch.sector_idx(), 0);
        scratch.set_sector_idx(5);
        assert_eq!(scratch.sector_idx(), 5);
        scratch.increment_sector_idx();
        assert_eq!(scratch.sector_idx(), 6);
    }
}
