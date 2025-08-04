/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::cmp::Ordering;

/// Neighbor node
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    /// The id of the node
    pub id: u32,

    /// The distance from the query node to current node
    pub distance: f32,

    /// Whether the current is visited or not
    pub visited: bool,
}

impl Neighbor {
    /// Create the neighbor node and it has not been visited
    pub fn new(id: u32, distance: f32) -> Self {
        Self {
            id,
            distance,
            visited: false,
        }
    }
}

impl Default for Neighbor {
    fn default() -> Self {
        Self {
            id: 0,
            distance: 0.0_f32,
            visited: false,
        }
    }
}

impl PartialEq for Neighbor {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        let ord = self
            .distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal);

        if ord == Ordering::Equal {
            return self.id.cmp(&other.id);
        }

        ord
    }
}

impl PartialOrd for Neighbor {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod neighbor_test {
    use super::*;

    #[test]
    fn eq_lt_works() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);
        let n3 = Neighbor::new(1, 1.1);

        assert!(n1 != n2);
        assert!(n1 < n2);
        assert!(n1 == n3);
    }

    #[test]
    fn full_ordering_works() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);
        let n3 = Neighbor::new(3, 1.1);

        assert!(n1 < n2);
        assert!(n2 > n1);
        assert!(n1 <= n2);
        assert!(n2 >= n1);
        assert!(n1 <= n3); // Same distance, different ID
        assert!(n3 >= n1);
    }

    #[test]
    fn ordering_with_same_distance() {
        let n1 = Neighbor::new(1, 1.0);
        let n2 = Neighbor::new(2, 1.0);

        assert!(n1 < n2); // Lower ID comes first
        assert!(n2 > n1);
    }
}
