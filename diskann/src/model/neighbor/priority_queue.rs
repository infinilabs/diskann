use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::model::neighbor::Neighbor;

/// Priority queue for beam search with visited node tracking
pub struct NeighborPriorityQueue {
    queue: BinaryHeap<Neighbor>,
    visited: HashSet<u32>,
    max_size: usize,
}

impl NeighborPriorityQueue {
    /// Create a new priority queue with maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: BinaryHeap::with_capacity(max_size),
            visited: HashSet::new(),
            max_size,
        }
    }

    /// Insert a neighbor into the queue
    pub fn insert(&mut self, neighbor: Neighbor) {
        if self.queue.len() < self.max_size {
            self.queue.push(neighbor);
        } else if let Some(top) = self.queue.peek() {
            if neighbor.distance < top.distance {
                self.queue.pop();
                self.queue.push(neighbor);
            }
        }
    }

    /// Get the closest unexpanded node
    pub fn closest_unexpanded(&mut self) -> Option<Neighbor> {
        while let Some(neighbor) = self.queue.pop() {
            if !self.visited.contains(&neighbor.id) {
                self.visited.insert(neighbor.id);
                return Some(neighbor);
            }
        }
        None
    }

    /// Check if there are unexpanded nodes
    pub fn has_unexpanded_node(&self) -> bool {
        self.queue.iter().any(|n| !self.visited.contains(&n.id))
    }

    /// Reserve capacity for the queue
    pub fn reserve(&mut self, capacity: usize) {
        self.queue.reserve(capacity);
        self.visited.reserve(capacity);
    }

    /// Get the current size of the queue
    pub fn len(&self) -> usize {
        self.queue.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_queue_basic() {
        let mut queue = NeighborPriorityQueue::new(5);

        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 0.3));
        queue.insert(Neighbor::new(3, 0.7));

        assert_eq!(queue.len(), 3);
        assert!(queue.has_unexpanded_node());

        let closest = queue.closest_unexpanded();
        assert_eq!(closest.unwrap().id, 2); // 0.3 is smallest
    }

    #[test]
    fn test_priority_queue_max_size() {
        let mut queue = NeighborPriorityQueue::new(2);

        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 0.3));
        queue.insert(Neighbor::new(3, 0.1)); // Should replace 0.5

        assert_eq!(queue.len(), 2);

        let closest = queue.closest_unexpanded();
        assert_eq!(closest.unwrap().id, 3); // 0.1 is smallest
    }

    #[test]
    fn test_priority_queue_visited() {
        let mut queue = NeighborPriorityQueue::new(5);

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
