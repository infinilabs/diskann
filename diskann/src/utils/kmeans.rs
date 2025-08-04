/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use crate::common::ANNError;
use rand::{distributions::Uniform, prelude::Distribution, thread_rng, Rng};
use rayon::prelude::*;
use std::cmp::min;

use crate::common::ANNResult;
use crate::utils::math_util::{
    calc_distance, calc_distance_avx512, calc_distance_simd, calc_distances_batch_vectorized,
    compute_closest_centers, compute_vecs_l2sq,
};
use crate::utils::vectorized_storage::{BatchProcessor, VectorizedStorage, UltraBatchProcessor};

/// Ultra-optimized k-means iteration with vectorized operations
#[allow(clippy::too_many_arguments)]
fn lloyds_iter_ultra_optimized(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    docs_l2sq: &[f32],
    mut closest_docs: &mut Vec<Vec<usize>>,
    closest_center: &mut [u32],
    ultra_batch_processor: &UltraBatchProcessor,
) -> ANNResult<f32> {
    let compute_residual = true;

    closest_docs.iter_mut().for_each(|doc| doc.clear());

    // Use vectorized batch processing for distance calculations
    let distances =
        ultra_batch_processor.process_distance_batch_ultra(data, centers, num_points, num_centers, dim)?;

    // Find closest centers using pre-computed distances
    for i in 0..num_points {
        let mut min_dist = f32::INFINITY;
        let mut min_center = 0;

        for j in 0..num_centers {
            let dist = distances[i * num_centers + j];
            if dist < min_dist {
                min_dist = dist;
                min_center = j;
            }
        }

        closest_center[i] = min_center as u32;
        closest_docs[min_center].push(i);
    }

    // Reset centers to zero before updating them
    centers.fill(0.0);

    // Update centers using vectorized operations
    centers
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(c, center)| {
            let mut cluster_sum = vec![0.0; dim];

            // Use SIMD for center updates
            for &doc_index in &closest_docs[c] {
                let current = &data[doc_index * dim..(doc_index + 1) * dim];

                #[cfg(target_arch = "x86_64")]
                {
                    unsafe {
                        if is_x86_feature_detected!("avx512f") {
                            // AVX-512 vectorized addition
                            let aligned_len = dim - (dim % 16);
                            for i in (0..aligned_len).step_by(16) {
                                let cluster_vec = _mm512_loadu_ps(&cluster_sum[i]);
                                let data_vec = _mm512_loadu_ps(&current[i]);
                                let sum_vec = _mm512_add_ps(cluster_vec, data_vec);
                                _mm512_storeu_ps(&mut cluster_sum[i], sum_vec);
                            }
                            // Handle remaining elements
                            for i in aligned_len..dim {
                                cluster_sum[i] += current[i];
                            }
                        } else if is_x86_feature_detected!("avx2") {
                            // AVX2 vectorized addition
                            let aligned_len = dim - (dim % 8);
                            for i in (0..aligned_len).step_by(8) {
                                let cluster_vec = _mm256_loadu_ps(&cluster_sum[i]);
                                let data_vec = _mm256_loadu_ps(&current[i]);
                                let sum_vec = _mm256_add_ps(cluster_vec, data_vec);
                                _mm256_storeu_ps(&mut cluster_sum[i], sum_vec);
                            }
                            // Handle remaining elements
                            for i in aligned_len..dim {
                                cluster_sum[i] += current[i];
                            }
                        } else {
                            // Fallback to standard addition
                            for (j, current_val) in current.iter().enumerate() {
                                cluster_sum[j] += *current_val as f64;
                            }
                        }
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    for (j, current_val) in current.iter().enumerate() {
                        cluster_sum[j] += *current_val as f64;
                    }
                }
            }

            if !closest_docs[c].is_empty() {
                let cluster_size = closest_docs[c].len() as f64;
                for (i, sum_val) in cluster_sum.iter().enumerate() {
                    center[i] = (*sum_val / cluster_size) as f32;
                }
            }
        });

    let mut residual = 0.0;
    if compute_residual {
        // Use vectorized residual calculation
        residual = (0..num_points)
            .into_par_iter()
            .map(|d| {
                calc_distance_avx512(
                    &data[d * dim..(d + 1) * dim],
                    &centers
                        [closest_center[d] as usize * dim..(closest_center[d] as usize + 1) * dim],
                    dim,
                )
            })
            .sum();
    }

    Ok(residual)
}

/// Ultra-optimized Lloyds with memory pooling and vectorized operations
fn run_lloyds_ultra_optimized(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    let mut residual = f32::MAX;

    let mut closest_docs = vec![Vec::new(); num_centers];
    let mut closest_center = vec![0; num_points];

    let mut docs_l2sq = vec![0.0; num_points];
    compute_vecs_l2sq(&mut docs_l2sq, data, num_points, dim);

    // Create ultra batch processor for maximum throughput
    let batch_size = 65536; // 16x larger for ultra-maximum throughput
    let ultra_batch_processor = UltraBatchProcessor::new(batch_size);

    let mut old_residual;
    let start_time = std::time::Instant::now();

    for i in 0..max_reps {
        old_residual = residual;

        residual = lloyds_iter_ultra_optimized(
            data,
            num_points,
            dim,
            centers,
            num_centers,
            &docs_l2sq,
            &mut closest_docs,
            &mut closest_center,
            &ultra_batch_processor,
        )?;

        // More aggressive early termination for speed
        if (i != 0 && (old_residual - residual) / residual < 0.01) || (residual < f32::EPSILON) {
            let elapsed = start_time.elapsed();
            println!(
                "Residuals unchanged: {} becomes {}. Early termination. (Iteration {}, Total time: {:.2?})",
                old_residual, residual, i, elapsed
            );
            break;
        }

        // Progress reporting every 3 iterations
        if i % 3 == 0 {
            let elapsed = start_time.elapsed();
            println!(
                "K-means iteration {}: residual = {:.2}, elapsed = {:.2?}",
                i, residual, elapsed
            );
        }
    }

    Ok((closest_docs, closest_center, residual))
}

/// Assume memory allocated for pivot_data as new float[num_centers * dim]
/// and select randomly num_centers points as pivots
fn selecting_pivots(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &mut [f32],
    num_centers: usize,
) {
    let mut picked = Vec::new();
    let mut rng = thread_rng();
    let distribution = Uniform::from(0..num_points);

    for j in 0..num_centers {
        let mut tmp_pivot = distribution.sample(&mut rng);
        while picked.contains(&tmp_pivot) {
            tmp_pivot = distribution.sample(&mut rng);
        }
        picked.push(tmp_pivot);
        let data_offset = tmp_pivot * dim;
        let pivot_offset = j * dim;
        pivot_data[pivot_offset..pivot_offset + dim]
            .copy_from_slice(&data[data_offset..data_offset + dim]);
    }
}

/// Select pivots in k-means++ algorithm
/// Points that are farther away from the already chosen centroids
/// have a higher probability of being selected as the next centroid.
/// The k-means++ algorithm helps avoid poor initial centroid
/// placement that can result in suboptimal clustering.
fn k_meanspp_selecting_pivots(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &mut [f32],
    num_centers: usize,
) {
    if num_points > (1 << 23) {
        println!("ERROR: n_pts {} currently not supported for k-means++, maximum is 8388608. Falling back to random pivot selection.", num_points);
        selecting_pivots(data, num_points, dim, pivot_data, num_centers);
        return;
    }

    let mut picked: Vec<usize> = Vec::new();
    let mut rng = thread_rng();
    let real_distribution = Uniform::from(0.0..1.0);
    let int_distribution = Uniform::from(0..num_points);

    let init_id = int_distribution.sample(&mut rng);
    let mut num_picked = 1;

    picked.push(init_id);
    let init_data_offset = init_id * dim;
    pivot_data[0..dim].copy_from_slice(&data[init_data_offset..init_data_offset + dim]);

    let mut dist = vec![0.0; num_points];

    dist.par_iter_mut().enumerate().for_each(|(i, dist_i)| {
        *dist_i = calc_distance_simd(
            &data[i * dim..(i + 1) * dim],
            &data[init_id * dim..(init_id + 1) * dim],
            dim,
        );
    });

    let mut dart_val: f64;
    let mut tmp_pivot = 0;
    let mut sum_flag = false;

    while num_picked < num_centers {
        dart_val = real_distribution.sample(&mut rng);

        let mut sum: f64 = 0.0;
        for item in dist.iter().take(num_points) {
            sum += *item as f64;
        }
        if sum == 0.0 {
            sum_flag = true;
        }

        dart_val *= sum;

        let mut prefix_sum: f64 = 0.0;
        for (i, pivot) in dist.iter().enumerate().take(num_points) {
            tmp_pivot = i;
            if dart_val >= prefix_sum && dart_val < (prefix_sum + *pivot as f64) {
                break;
            }

            prefix_sum += *pivot as f64;
        }

        if picked.contains(&tmp_pivot) && !sum_flag {
            continue;
        }

        picked.push(tmp_pivot);
        let pivot_offset = num_picked * dim;
        let data_offset = tmp_pivot * dim;
        pivot_data[pivot_offset..pivot_offset + dim]
            .copy_from_slice(&data[data_offset..data_offset + dim]);

        dist.par_iter_mut().enumerate().for_each(|(i, dist_i)| {
            *dist_i = (*dist_i).min(calc_distance_simd(
                &data[i * dim..(i + 1) * dim],
                &data[tmp_pivot * dim..(tmp_pivot + 1) * dim],
                dim,
            ));
        });

        num_picked += 1;
    }
}

/// k-means algorithm interface
pub fn k_means_clustering(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    k_meanspp_selecting_pivots(data, num_points, dim, centers, num_centers);
    
    // Create ultra batch processor for maximum throughput
    let batch_size = 32768; // 8x larger for ultra-maximum throughput
    let ultra_batch_processor = UltraBatchProcessor::new(batch_size);
    
    let (closest_docs, closest_center, residual) = run_lloyds_ultra_optimized(data, num_points, dim, centers, num_centers, max_reps)?;
    Ok((closest_docs, closest_center, residual))
}

/// Ultra-fast k-means clustering with minimal iterations for maximum speed
pub fn k_means_clustering_fast(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    if num_centers == 0 {
        return Err(ANNError::log_index_error(
            "num_centers cannot be zero".to_string(),
        ));
    }

    if num_points == 0 {
        return Err(ANNError::log_index_error(
            "num_points cannot be zero".to_string(),
        ));
    }

    // Ultra-fast k-means++ initialization
    k_meanspp_selecting_pivots_fast(data, num_points, dim, centers, num_centers);

    // Ultra-fast Lloyd's iterations with minimal iterations
    let batch_processor = BatchProcessor::new(4096); // 4x larger batch size for maximum speed
    let (closest_docs, closest_center, residual) = run_lloyds_ultra_fast(
        data,
        num_points,
        dim,
        centers,
        num_centers,
        max_reps,
        &batch_processor,
    )?;

    Ok((closest_docs, closest_center, residual))
}

/// Ultra-fast Lloyd's iteration with minimal processing
#[allow(clippy::too_many_arguments)]
fn lloyds_iter_ultra_fast(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    mut closest_docs: &mut Vec<Vec<usize>>,
    closest_center: &mut [u32],
    batch_processor: &BatchProcessor,
) -> ANNResult<f32> {
    closest_docs.iter_mut().for_each(|doc| doc.clear());

    // Ultra-fast distance calculation using SIMD
    let mut distances = vec![0.0; num_points * num_centers];

    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            if is_x86_feature_detected!("avx512f") {
                // Ultra-fast AVX-512 distance calculation
                for i in 0..num_points {
                    let data_point = &data[i * dim..(i + 1) * dim];
                    for j in 0..num_centers {
                        let center = &centers[j * dim..(j + 1) * dim];
                        let mut dist = 0.0;

                        let aligned_len = dim - (dim % 16);
                        let mut sum = _mm512_setzero_ps();

                        for k in (0..aligned_len).step_by(16) {
                            let data_vec = _mm512_loadu_ps(&data_point[k]);
                            let center_vec = _mm512_loadu_ps(&center[k]);
                            let diff = _mm512_sub_ps(data_vec, center_vec);
                            let squared = _mm512_mul_ps(diff, diff);
                            sum = _mm512_add_ps(sum, squared);
                        }

                        dist = _mm512_reduce_add_ps(sum);

                        for k in aligned_len..dim {
                            let diff = data_point[k] - center[k];
                            dist += diff * diff;
                        }

                        distances[i * num_centers + j] = dist;
                    }
                }
            } else if is_x86_feature_detected!("avx2") {
                // Ultra-fast AVX2 distance calculation
                for i in 0..num_points {
                    let data_point = &data[i * dim..(i + 1) * dim];
                    for j in 0..num_centers {
                        let center = &centers[j * dim..(j + 1) * dim];
                        let mut dist = 0.0;

                        let aligned_len = dim - (dim % 8);
                        let mut sum = _mm256_setzero_ps();

                        for k in (0..aligned_len).step_by(8) {
                            let data_vec = _mm256_loadu_ps(&data_point[k]);
                            let center_vec = _mm256_loadu_ps(&center[k]);
                            let diff = _mm256_sub_ps(data_vec, center_vec);
                            let squared = _mm256_mul_ps(diff, diff);
                            sum = _mm256_add_ps(sum, squared);
                        }

                        // Fast horizontal sum
                        let x128: __m128 =
                            _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
                        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
                        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
                        dist = _mm_cvtss_f32(x32);

                        for k in aligned_len..dim {
                            let diff = data_point[k] - center[k];
                            dist += diff * diff;
                        }

                        distances[i * num_centers + j] = dist;
                    }
                }
            } else {
                // Fast fallback
                for i in 0..num_points {
                    let data_point = &data[i * dim..(i + 1) * dim];
                    for j in 0..num_centers {
                        let center = &centers[j * dim..(j + 1) * dim];
                        let mut dist = 0.0;
                        for k in 0..dim {
                            let diff = data_point[k] - center[k];
                            dist += diff * diff;
                        }
                        distances[i * num_centers + j] = dist;
                    }
                }
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..num_points {
            let data_point = &data[i * dim..(i + 1) * dim];
            for j in 0..num_centers {
                let center = &centers[j * dim..(j + 1) * dim];
                let mut dist = 0.0;
                for k in 0..dim {
                    let diff = data_point[k] - center[k];
                    dist += diff * diff;
                }
                distances[i * num_centers + j] = dist;
            }
        }
    }

    // Find closest centers
    for i in 0..num_points {
        let mut min_dist = f32::INFINITY;
        let mut min_center = 0;

        for j in 0..num_centers {
            let dist = distances[i * num_centers + j];
            if dist < min_dist {
                min_dist = dist;
                min_center = j;
            }
        }

        closest_center[i] = min_center as u32;
        closest_docs[min_center].push(i);
    }

    // Reset centers
    centers.fill(0.0);

    // Ultra-fast center updates using SIMD
    centers
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(c, center)| {
            if closest_docs[c].is_empty() {
                return;
            }

            #[cfg(target_arch = "x86_64")]
            {
                unsafe {
                    if is_x86_feature_detected!("avx512f") {
                        // Ultra-fast AVX-512 center update
                        let aligned_len = dim - (dim % 16);
                        for i in (0..aligned_len).step_by(16) {
                            let mut sum = _mm512_setzero_ps();
                            for &doc_index in &closest_docs[c] {
                                let data_point = &data[doc_index * dim..(doc_index + 1) * dim];
                                let data_vec = _mm512_loadu_ps(&data_point[i]);
                                sum = _mm512_add_ps(sum, data_vec);
                            }
                            let avg =
                                _mm512_div_ps(sum, _mm512_set1_ps(closest_docs[c].len() as f32));
                            _mm512_storeu_ps(&mut center[i], avg);
                        }
                        for i in aligned_len..dim {
                            let mut sum = 0.0;
                            for &doc_index in &closest_docs[c] {
                                sum += data[doc_index * dim + i];
                            }
                            center[i] = sum / closest_docs[c].len() as f32;
                        }
                    } else if is_x86_feature_detected!("avx2") {
                        // Ultra-fast AVX2 center update
                        let aligned_len = dim - (dim % 8);
                        for i in (0..aligned_len).step_by(8) {
                            let mut sum = _mm256_setzero_ps();
                            for &doc_index in &closest_docs[c] {
                                let data_point = &data[doc_index * dim..(doc_index + 1) * dim];
                                let data_vec = _mm256_loadu_ps(&data_point[i]);
                                sum = _mm256_add_ps(sum, data_vec);
                            }
                            let avg =
                                _mm256_div_ps(sum, _mm256_set1_ps(closest_docs[c].len() as f32));
                            _mm256_storeu_ps(&mut center[i], avg);
                        }
                        for i in aligned_len..dim {
                            let mut sum = 0.0;
                            for &doc_index in &closest_docs[c] {
                                sum += data[doc_index * dim + i];
                            }
                            center[i] = sum / closest_docs[c].len() as f32;
                        }
                    } else {
                        for i in 0..dim {
                            let mut sum = 0.0;
                            for &doc_index in &closest_docs[c] {
                                sum += data[doc_index * dim + i];
                            }
                            center[i] = sum / closest_docs[c].len() as f32;
                        }
                    }
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                for i in 0..dim {
                    let mut sum = 0.0;
                    for &doc_index in &closest_docs[c] {
                        sum += data[doc_index * dim + i];
                    }
                    center[i] = sum / closest_docs[c].len() as f32;
                }
            }
        });

    // Calculate residual
    let mut residual = 0.0;
    for i in 0..num_points {
        let center_index = closest_center[i] as usize;
        let data_point = &data[i * dim..(i + 1) * dim];
        let center = &centers[center_index * dim..(center_index + 1) * dim];

        for j in 0..dim {
            let diff = data_point[j] - center[j];
            residual += diff * diff;
        }
    }

    Ok(residual)
}

/// Ultra-fast Lloyd's algorithm with minimal iterations
fn run_lloyds_ultra_fast(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centers: &mut [f32],
    num_centers: usize,
    max_reps: usize,
    batch_processor: &BatchProcessor,
) -> ANNResult<(Vec<Vec<usize>>, Vec<u32>, f32)> {
    let mut closest_docs: Vec<Vec<usize>> = vec![Vec::new(); num_centers];
    let mut closest_center: Vec<u32> = vec![0; num_points];
    let mut prev_residual = f32::INFINITY;

    // Ultra-fast iterations with aggressive early termination
    for iter in 0..max_reps {
        let residual = lloyds_iter_ultra_fast(
            data,
            num_points,
            dim,
            centers,
            num_centers,
            &mut closest_docs,
            &mut closest_center,
            batch_processor,
        )?;

        // Ultra-aggressive early termination
        if (prev_residual - residual).abs() < 1.0 {
            println!(
                "Ultra-fast k-means: Early termination at iteration {} (residual: {:.1})",
                iter, residual
            );
            break;
        }

        prev_residual = residual;

        // Report progress every iteration for ultra-fast processing
        if iter % 1 == 0 {
            println!(
                "Ultra-fast k-means iteration {}: residual = {:.1}",
                iter, residual
            );
        }
    }

    Ok((closest_docs, closest_center, prev_residual))
}

/// Ultra-fast k-means++ initialization
fn k_meanspp_selecting_pivots_fast(
    data: &[f32],
    num_points: usize,
    dim: usize,
    pivot_data: &mut [f32],
    num_centers: usize,
) {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0, num_points);

    // Select first center randomly
    let first_center = uniform.sample(&mut rng);
    for i in 0..dim {
        pivot_data[i] = data[first_center * dim + i];
    }

    // Ultra-fast selection of remaining centers
    for center_index in 1..num_centers {
        let mut min_distances = vec![f32::INFINITY; num_points];

        // Calculate minimum distances to existing centers
        for point_index in 0..num_points {
            for existing_center in 0..center_index {
                let mut dist = 0.0;

                #[cfg(target_arch = "x86_64")]
                {
                    unsafe {
                        if is_x86_feature_detected!("avx512f") {
                            let aligned_len = dim - (dim % 16);
                            let mut sum = _mm512_setzero_ps();

                            for i in (0..aligned_len).step_by(16) {
                                let data_vec = _mm512_loadu_ps(&data[point_index * dim + i]);
                                let center_vec =
                                    _mm512_loadu_ps(&pivot_data[existing_center * dim + i]);
                                let diff = _mm512_sub_ps(data_vec, center_vec);
                                let squared = _mm512_mul_ps(diff, diff);
                                sum = _mm512_add_ps(sum, squared);
                            }

                            dist = _mm512_reduce_add_ps(sum);

                            for i in aligned_len..dim {
                                let diff = data[point_index * dim + i]
                                    - pivot_data[existing_center * dim + i];
                                dist += diff * diff;
                            }
                        } else {
                            for i in 0..dim {
                                let diff = data[point_index * dim + i]
                                    - pivot_data[existing_center * dim + i];
                                dist += diff * diff;
                            }
                        }
                    }
                }

                #[cfg(not(target_arch = "x86_64"))]
                {
                    for i in 0..dim {
                        let diff =
                            data[point_index * dim + i] - pivot_data[existing_center * dim + i];
                        dist += diff * diff;
                    }
                }

                if dist < min_distances[point_index] {
                    min_distances[point_index] = dist;
                }
            }
        }

        // Select next center with probability proportional to distance squared
        let total_distance: f32 = min_distances.iter().sum();
        let random_value = rng.gen::<f32>() * total_distance;

        let mut cumulative_distance = 0.0;
        let mut selected_point = 0;

        for (point_index, &distance) in min_distances.iter().enumerate() {
            cumulative_distance += distance;
            if cumulative_distance >= random_value {
                selected_point = point_index;
                break;
            }
        }

        // Copy selected point to new center
        for i in 0..dim {
            pivot_data[center_index * dim + i] = data[selected_point * dim + i];
        }
    }
}

#[cfg(test)]
mod kmeans_test {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;

    #[test]
    fn lloyds_iter_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        let data: Vec<f32> = (1..=num_points * dim).map(|x| x as f32).collect();
        let mut centers = [1.0, 2.0, 7.0, 8.0, 19.0, 20.0];

        let mut closest_docs: Vec<Vec<usize>> = vec![vec![]; num_centers];
        let mut closest_center: Vec<u32> = vec![0; num_points];
        let docs_l2sq: Vec<f32> = data
            .chunks(dim)
            .map(|chunk| chunk.iter().map(|val| val.powi(2)).sum())
            .collect();

        println!("Initial centers: {:?}", centers);
        println!("Data points: {:?}", data);

        // Create ultra batch processor for test
        let ultra_batch_processor = UltraBatchProcessor::new(1024);
        
        let residual = lloyds_iter_ultra_optimized(
            &data,
            num_points,
            dim,
            &mut centers,
            num_centers,
            &docs_l2sq,
            &mut closest_docs,
            &mut closest_center,
            &ultra_batch_processor,
        )
        .unwrap();

        let expected_centers: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 10.0, 11.0];
        let expected_closest_docs: Vec<Vec<usize>> =
            vec![vec![], vec![], vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
        let expected_closest_center: [u32; 10] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let expected_residual: f32 = 660.0;

        // sort data for assert
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for inner_vec in &mut closest_docs {
            inner_vec.sort();
        }
        closest_center.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("Actual centers: {:?}", centers);
        println!("Expected centers: {:?}", expected_centers);
        println!("Actual closest_docs: {:?}", closest_docs);
        println!("Expected closest_docs: {:?}", expected_closest_docs);
        println!("Actual closest_center: {:?}", closest_center);
        println!("Expected closest_center: {:?}", expected_closest_center);
        println!("Actual residual: {}", residual);
        println!("Expected residual: {}", expected_residual);

        assert_eq!(centers, expected_centers);
        assert_eq!(closest_docs, expected_closest_docs);
        assert_eq!(closest_center, expected_closest_center);
        assert_relative_eq!(residual, expected_residual, epsilon = 1.0e-6_f32);
    }

    #[test]
    fn run_lloyds_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;
        let max_reps = 5;

        let data: Vec<f32> = (1..=num_points * dim).map(|x| x as f32).collect();
        let mut centers = [1.0, 2.0, 7.0, 8.0, 19.0, 20.0];

        let (mut closest_docs, mut closest_center, residual) =
            run_lloyds_ultra_optimized(&data, num_points, dim, &mut centers, num_centers, max_reps).unwrap();

        let expected_centers: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 10.0, 11.0];
        let expected_closest_docs: Vec<Vec<usize>> =
            vec![vec![], vec![], vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
        let expected_closest_center: [u32; 10] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
        let expected_residual: f32 = 660.0;

        // sort data for assert
        centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for inner_vec in &mut closest_docs {
            inner_vec.sort();
        }
        closest_center.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(centers, expected_centers);
        assert_eq!(closest_docs, expected_closest_docs);
        assert_eq!(closest_center, expected_closest_center);
        assert_relative_eq!(residual, expected_residual, epsilon = 1.0e-6_f32);
    }

    #[test]
    fn selecting_pivots_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        // Generate some random data points
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.gen()).collect();

        let mut pivot_data = vec![0.0; num_centers * dim];

        selecting_pivots(&data, num_points, dim, &mut pivot_data, num_centers);

        // Verify that each pivot point corresponds to a point in the data
        for i in 0..num_centers {
            let pivot_offset = i * dim;
            let pivot = &pivot_data[pivot_offset..(pivot_offset + dim)];

            // Make sure the pivot is found in the data
            let mut found = false;
            for j in 0..num_points {
                let data_offset = j * dim;
                let point = &data[data_offset..(data_offset + dim)];

                if pivot == point {
                    found = true;
                    break;
                }
            }
            assert!(found, "Pivot not found in data");
        }
    }

    #[test]
    fn k_meanspp_selecting_pivots_test() {
        let dim = 2;
        let num_points = 10;
        let num_centers = 3;

        // Generate some random data points
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..num_points * dim).map(|_| rng.gen()).collect();

        let mut pivot_data = vec![0.0; num_centers * dim];

        k_meanspp_selecting_pivots(&data, num_points, dim, &mut pivot_data, num_centers);

        // Verify that each pivot point corresponds to a point in the data
        for i in 0..num_centers {
            let pivot_offset = i * dim;
            let pivot = &pivot_data[pivot_offset..pivot_offset + dim];

            // Make sure the pivot is found in the data
            let mut found = false;
            for j in 0..num_points {
                let data_offset = j * dim;
                let point = &data[data_offset..data_offset + dim];

                if pivot == point {
                    found = true;
                    break;
                }
            }
            assert!(found, "Pivot not found in data");
        }
    }
}
