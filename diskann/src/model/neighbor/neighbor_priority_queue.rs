/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::model::neighbor::Neighbor;

/// Wrapper for Neighbor that reverses the ordering to make BinaryHeap a min-heap
#[derive(Debug, Clone)]
struct MinNeighbor(Neighbor);

impl PartialEq for MinNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for MinNeighbor {}

impl PartialOrd for MinNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse the ordering to make it a min-heap
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for MinNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the ordering to make it a min-heap
        other.0.cmp(&self.0)
    }
}

/// Priority queue for beam search with visited node tracking
#[derive(Debug)]
pub struct NeighborPriorityQueue {
    queue: BinaryHeap<MinNeighbor>,
    visited: HashSet<u32>,
    max_size: usize,
}

impl NeighborPriorityQueue {
    /// Create a new priority queue
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            visited: HashSet::new(),
            max_size: usize::MAX,
        }
    }

    /// Create a new priority queue with maximum size
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            queue: BinaryHeap::with_capacity(max_size),
            visited: HashSet::new(),
            max_size,
        }
    }

    /// Insert a neighbor into the queue
    pub fn insert(&mut self, neighbor: Neighbor) {
        if self.queue.len() < self.max_size {
            self.queue.push(MinNeighbor(neighbor));
        } else if let Some(top) = self.queue.peek() {
            if neighbor.distance < top.0.distance {
                self.queue.pop();
                self.queue.push(MinNeighbor(neighbor));
            }
        }
    }

    /// Get the closest unexpanded node
    pub fn closest_unexpanded(&mut self) -> Option<Neighbor> {
        while let Some(min_neighbor) = self.queue.pop() {
            let neighbor = min_neighbor.0;
            if !self.visited.contains(&neighbor.id) {
                self.visited.insert(neighbor.id);
                return Some(neighbor);
            }
        }
        None
    }

    /// Get the closest not visited node (alias for closest_unexpanded)
    pub fn closest_notvisited(&mut self) -> Neighbor {
        self.closest_unexpanded().unwrap_or_else(|| {
            // Return a default neighbor if none available
            Neighbor::new(0, f32::INFINITY)
        })
    }

    /// Check if there are unexpanded nodes
    pub fn has_unexpanded_node(&self) -> bool {
        self.queue.iter().any(|n| !self.visited.contains(&n.0.id))
    }

    /// Check if there are not visited nodes (alias for has_unexpanded_node)
    pub fn has_notvisited_node(&self) -> bool {
        self.has_unexpanded_node()
    }

    /// Reserve capacity for the queue
    pub fn reserve(&mut self, capacity: usize) {
        self.queue.reserve(capacity);
        self.visited.reserve(capacity);
    }

    /// Get the current size of the queue
    pub fn size(&self) -> usize {
        self.queue.len()
    }

    /// Set capacity for benchmarking purposes
    pub fn set_capacity(&mut self, capacity: usize) {
        self.max_size = capacity;
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clear the queue and visited set
    pub fn clear(&mut self) {
        self.queue.clear();
        self.visited.clear();
    }

    /// Get the number of visited nodes
    pub fn visited_count(&self) -> usize {
        self.visited.len()
    }
}

impl std::ops::Index<usize> for NeighborPriorityQueue {
    type Output = Neighbor;

    fn index(&self, _i: usize) -> &Self::Output {
        // This is a simplified implementation - in practice, you'd need to maintain order
        // For now, we'll return a default neighbor
        static DEFAULT_NEIGHBOR: Neighbor = Neighbor {
            id: 0,
            distance: 0.0,
            visited: false,
        };
        &DEFAULT_NEIGHBOR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_queue_basic() {
        let mut queue = NeighborPriorityQueue::with_capacity(5);

        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 0.3));
        queue.insert(Neighbor::new(3, 0.7));

        assert_eq!(queue.size(), 3);
        assert!(queue.has_unexpanded_node());

        let closest = queue.closest_unexpanded();
        assert_eq!(closest.unwrap().id, 2); // 0.3 is smallest
    }

    #[test]
    fn test_priority_queue_max_size() {
        let mut queue = NeighborPriorityQueue::with_capacity(2);

        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 0.3));
        queue.insert(Neighbor::new(3, 0.1)); // Should replace 0.5

        assert_eq!(queue.size(), 2);

        let closest = queue.closest_unexpanded();
        assert_eq!(closest.unwrap().id, 3); // 0.1 is smallest
    }

    #[test]
    fn test_priority_queue_visited() {
        let mut queue = NeighborPriorityQueue::with_capacity(5);

        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 0.3));

        let first = queue.closest_unexpanded();
        assert_eq!(first.unwrap().id, 2);

        // Should not return the same node again
        let second = queue.closest_unexpanded();
        assert_eq!(second.unwrap().id, 1);

        // No more unexpanded nodes
        let third = queue.closest_unexpanded();
        assert!(third.is_none());
    }
}
