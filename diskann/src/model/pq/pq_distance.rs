use crate::common::{ANNError, ANNResult};

/// Aggregate coordinates from multiple vectors
pub fn aggregate_coords(
    ids: &[u32],
    all_coords: &[u8],
    ndims: u64,
    out: &mut [u8],
) -> ANNResult<()> {
    let chunk_size = (ndims / ids.len() as u64) as usize;
    let out_size = ids.len() * chunk_size;
    
    if out.len() < out_size {
        return Err(ANNError::log_index_error(
            "Output buffer too small for aggregated coordinates".to_string(),
        ));
    }
    
    for (i, &id) in ids.iter().enumerate() {
        let src_start = (id as usize) * chunk_size;
        let src_end = src_start + chunk_size;
        let dst_start = i * chunk_size;
        let dst_end = dst_start + chunk_size;
        
        if src_end > all_coords.len() {
            return Err(ANNError::log_index_error(
                format!("Source coordinates out of bounds for id {}", id),
            ));
        }
        
        out[dst_start..dst_end].copy_from_slice(&all_coords[src_start..src_end]);
    }
    
    Ok(())
}

/// Lookup PQ distances using precomputed tables
pub fn pq_dist_lookup(
    pq_ids: &[u8],
    n_pts: usize,
    pq_nchunks: usize,
    pq_dists: &[f32],
    dists_out: &mut [f32],
) -> ANNResult<()> {
    if dists_out.len() < n_pts {
        return Err(ANNError::log_index_error(
            "Output buffer too small for distance results".to_string(),
        ));
    }
    
    for pt_idx in 0..n_pts {
        let mut total_dist = 0.0;
        
        for chunk in 0..pq_nchunks {
            let centroid_id = pq_ids[pt_idx * pq_nchunks + chunk] as usize;
            let table_offset = chunk * 256 + centroid_id;
            
            if table_offset < pq_dists.len() {
                total_dist += pq_dists[table_offset];
            }
        }
        
        dists_out[pt_idx] = total_dist;
    }
    
    Ok(())
}

/// Calculate L2 distance between two vectors
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    let mut dist = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        dist += diff * diff;
    }
    
    dist.sqrt()
}

/// Calculate inner product between two vectors
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let mut product = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    
    product
}

/// Calculate cosine distance between two vectors
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 1.0; // Maximum distance for different lengths
    }
    
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }
    
    1.0 - (dot_product / (norm_a * norm_b))
}

/// Normalize a vector to unit length
pub fn normalize_vector(vec: &mut [f32]) -> ANNResult<()> {
    let mut norm = 0.0;
    for &val in vec.iter() {
        norm += val * val;
    }
    norm = norm.sqrt();
    
    if norm == 0.0 {
        return Err(ANNError::log_index_error(
            "Cannot normalize zero vector".to_string(),
        ));
    }
    
    for val in vec.iter_mut() {
        *val /= norm;
    }
    
    Ok(())
}

/// Convert distance to similarity score
pub fn distance_to_similarity(distance: f32, metric: crate::vector::Metric) -> f32 {
    match metric {
        crate::vector::Metric::L2 => (-distance).exp(),
        crate::vector::Metric::Cosine => 1.0 - distance,
        crate::vector::Metric::InnerProduct => distance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_coords() {
        let ids = vec![0u32, 2u32];
        let all_coords = vec![1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8];
        let mut out = vec![0u8; 4];
        
        let result = aggregate_coords(&ids, &all_coords, 4, &mut out);
        assert!(result.is_ok());
        
        // Check that coordinates were aggregated correctly
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 2);
        assert_eq!(out[2], 5);
        assert_eq!(out[3], 6);
    }

    #[test]
    fn test_pq_dist_lookup() {
        let pq_ids = vec![0u8, 1u8, 2u8, 3u8]; // 2 points, 2 chunks each
        let pq_dists = vec![0.1f32, 0.2f32, 0.3f32, 0.4f32, 0.5f32, 0.6f32, 0.7f32, 0.8f32];
        let mut dists_out = vec![0.0f32; 2];
        
        let result = pq_dist_lookup(&pq_ids, 2, 2, &pq_dists, &mut dists_out);
        assert!(result.is_ok());
        
        // Check that distances were calculated correctly
        assert!((dists_out[0] - 0.4).abs() < 1e-6); // 0.1 + 0.3
        assert!((dists_out[1] - 1.3).abs() < 1e-6); // 0.6 + 0.7
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0f32, 2.0f32, 3.0f32];
        let b = vec![4.0f32, 5.0f32, 6.0f32];
        
        let distance = l2_distance(&a, &b);
        let expected = ((3.0 * 3.0) + (3.0 * 3.0) + (3.0 * 3.0)).sqrt();
        
        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0f32, 2.0f32, 3.0f32];
        let b = vec![4.0f32, 5.0f32, 6.0f32];
        
        let product = inner_product(&a, &b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
        
        assert!((product - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0f32, 0.0f32];
        let b = vec![0.0f32, 1.0f32];
        
        let distance = cosine_distance(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6); // Perpendicular vectors
        
        let c = vec![1.0f32, 0.0f32];
        let d = vec![1.0f32, 0.0f32];
        
        let distance = cosine_distance(&c, &d);
        assert!((distance - 0.0).abs() < 1e-6); // Same direction
    }

    #[test]
    fn test_normalize_vector() {
        let mut vec = vec![3.0f32, 4.0f32];
        
        let result = normalize_vector(&mut vec);
        assert!(result.is_ok());
        
        let expected_norm = (3.0 * 3.0 + 4.0 * 4.0).sqrt();
        assert!((vec[0] - 3.0 / expected_norm).abs() < 1e-6);
        assert!((vec[1] - 4.0 / expected_norm).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut vec = vec![0.0f32, 0.0f32];
        
        let result = normalize_vector(&mut vec);
        assert!(result.is_err());
    }

    #[test]
    fn test_distance_to_similarity() {
        let l2_sim = distance_to_similarity(1.0, crate::vector::Metric::L2);
        assert!(l2_sim > 0.0 && l2_sim < 1.0);
        
        let cosine_sim = distance_to_similarity(0.5, crate::vector::Metric::Cosine);
        assert_eq!(cosine_sim, 0.5);
        
        let ip_sim = distance_to_similarity(2.0, crate::vector::Metric::InnerProduct);
        assert_eq!(ip_sim, 2.0);
    }
} 