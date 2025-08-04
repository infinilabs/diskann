/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::cmp;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    RwLock,
};
use std::time::Duration;

use diskann_vector::FullPrecisionDistance;
use hashbrown::hash_set::Entry::*;
use hashbrown::HashSet;

use crate::common::{ANNError, ANNResult};
use crate::index::ANNInmemIndex;
use crate::instrumentation::IndexLogger;
use crate::model::graph::AdjacencyList;
use crate::model::{
    ArcConcurrentBoxedQueue, InMemQueryScratch, InMemoryGraph, IndexConfiguration, InmemDataset,
    Neighbor, ScratchStoreManager, Vertex,
};

use crate::utils::file_util::{file_exists, load_metadata_from_file};
use crate::utils::rayon_util::execute_with_rayon;
use crate::utils::{set_rayon_num_threads, Timer};

/// Used for warmup dataset, or it will cannot build graph and crash
pub const INIT_WARMUP_DATA_LEN: u32 = 5;

/// In-memory Index
pub struct InmemIndex<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Dataset
    pub dataset: InmemDataset<T, N>,

    /// Graph
    pub final_graph: InMemoryGraph,

    /// Index configuration
    pub configuration: IndexConfiguration,

    /// Start point of the search. When _num_frozen_pts is greater than zero,
    /// this is the location of the first frozen point. Otherwise, this is a
    /// location of one of the points in index.
    pub start: u32,

    /// Max observed out degree
    pub max_observed_degree: u32,

    /// Number of active points i.e. existing in the graph
    pub num_active_pts: usize,

    /// query scratch queue.
    query_scratch_queue: ArcConcurrentBoxedQueue<InMemQueryScratch<T, N>>,

    pub delete_set: RwLock<HashSet<u32>>,
}

impl<T, const N: usize> InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Create Index obj based on configuration
    pub fn new(mut config: IndexConfiguration) -> ANNResult<Self> {
        // Sanity check. While logically it is correct, max_points = 0 causes
        // downstream problems.
        if config.max_points == 0 {
            config.max_points = 1;
        }

        let total_internal_points = config.max_points + config.num_frozen_pts;

        if config.use_pq_dist {
            // TODO: pq
            todo!("PQ is not supported now");
        }

        let start = config.max_points.try_into()?;

        let query_scratch_queue = ArcConcurrentBoxedQueue::<InMemQueryScratch<T, N>>::new();
        let delete_set = RwLock::new(HashSet::<u32>::new());

        Ok(Self {
            dataset: InmemDataset::<T, N>::new(total_internal_points, config.growth_potential)?,
            final_graph: InMemoryGraph::new(
                total_internal_points,
                config.index_write_parameter.max_degree,
            ),
            configuration: config,
            start,
            max_observed_degree: 0,
            num_active_pts: 0,
            query_scratch_queue,
            delete_set,
        })
    }

    pub fn or_increase_capacity(&mut self, new_data_len: usize) -> ANNResult<bool> {
        self.dataset.or_increase_capacity(new_data_len)
    }

    /// Get distance between two vertices.
    pub fn get_distance(&self, id1: u32, id2: u32) -> ANNResult<f32> {
        self.dataset
            .get_distance(id1, id2, self.configuration.dist_metric)
    }

    fn build_with_data_populated(&mut self) -> ANNResult<()> {
        println!(
            "Starting index build with {} points...",
            self.num_active_pts
        );
        println!("📊 Total documents to process: {}", self.num_active_pts);

        if self.num_active_pts < 1 {
            return Err(ANNError::log_index_error(
                "Error: Trying to build an index with 0 points.".to_string(),
            ));
        }

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        // TODO: generate_frozen_point()

        let link_start = std::time::Instant::now();
        self.link()?;
        let link_elapsed = link_start.elapsed();
        println!(
            "🔗 Link phase completed in {:.2}s",
            link_elapsed.as_secs_f64()
        );

        self.print_stats()?;

        Ok(())
    }

    pub fn link(&mut self) -> ANNResult<()> {
        // visit_order is a vector that is initialized to the entire graph
        let mut visit_order =
            Vec::with_capacity(self.num_active_pts + self.configuration.num_frozen_pts);
        for i in 0..self.num_active_pts {
            visit_order.push(i as u32);
        }

        // If there are any frozen points, add them all.
        for frozen in self.configuration.max_points
            ..(self.configuration.max_points + self.configuration.num_frozen_pts)
        {
            visit_order.push(frozen as u32);
        }

        // if there are frozen points, the first such one is set to be the _start
        if self.configuration.num_frozen_pts > 0 {
            self.start = self.configuration.max_points as u32;
        } else {
            self.start = self.dataset.calculate_medoid_point_id()?;
        }

        let range = visit_order.len();
        let total_vertices = visit_order.len();
        println!(
            "🔄 Starting to process {} vertices with {} threads...",
            total_vertices, self.configuration.index_write_parameter.num_threads
        );

        let logger = IndexLogger::new(range);
        let processed_count = AtomicUsize::new(0);
        let progress_interval = std::cmp::max(1, total_vertices / 20); // Show progress every 5%
        let start_time = std::time::Instant::now();

        // Ultra-aggressive batching: process vertices in larger batches
        let batch_size = 512; // Increased from 256 for maximum throughput
        let num_batches = (total_vertices + batch_size - 1) / batch_size;

        execute_with_rayon(
            0..num_batches,
            self.configuration.index_write_parameter.num_threads,
            |batch_idx| {
                let batch_start = batch_idx * batch_size;
                let batch_end = std::cmp::min(batch_start + batch_size, total_vertices);

                // Process all vertices in this batch
                for idx in batch_start..batch_end {
                    self.insert_vertex_id(visit_order[idx])?;
                }

                logger.vertex_processed()?;

                // Progress reporting with TPS calculation
                let current_count = processed_count
                    .fetch_add(batch_end - batch_start, Ordering::Relaxed)
                    + (batch_end - batch_start);
                if current_count % (progress_interval * batch_size) < batch_size {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let tps = current_count as f64 / elapsed;
                    let progress = (current_count as f64 / total_vertices as f64) * 100.0;
                    let eta_seconds = if tps > 0.0 {
                        (total_vertices - current_count) as f64 / tps
                    } else {
                        0.0
                    };
                    let eta_minutes = eta_seconds / 60.0;

                    println!("📈 Progress: {}/{} vertices processed ({:.1}%) | TPS: {:.1} docs/sec | ETA: {:.1} min",
                            current_count, total_vertices, progress, tps, eta_minutes);
                }

                Ok(())
            },
        )?;

        let total_elapsed = start_time.elapsed().as_secs_f64();
        let final_tps = total_vertices as f64 / total_elapsed;
        println!(
            "✅ All {} vertices processed successfully in {:.2}s | Avg TPS: {:.1} docs/sec",
            total_vertices, total_elapsed, final_tps
        );

        self.cleanup_graph(&visit_order)?;

        Ok(())
    }

    fn insert_vertex_id(&self, vertex_id: u32) -> ANNResult<()> {
        // TODO: Fix ScratchStoreManager usage
        // let mut scratch_manager =
        //     ScratchStoreManager::new(self.query_scratch_queue.clone(), Duration::from_millis(10))?;
        // let scratch = scratch_manager.scratch_space().ok_or_else(|| {
        //     ANNError::log_index_error(
        //         "ScratchStoreManager doesn't have InMemQueryScratch instance available".to_string(),
        //     )
        // })?;

        // For now, create a simple scratch space
        let mut scratch = InMemQueryScratch::<T, N>::new(
            self.configuration.index_write_parameter.max_degree as u32,
            &self.configuration.index_write_parameter,
            true,
        )?;

        let new_neighbors = self.search_for_point_and_prune(&mut scratch, vertex_id)?;
        self.update_vertex_with_neighbors(vertex_id, new_neighbors)?;
        self.update_neighbors_of_vertex(vertex_id, &mut scratch)?;

        Ok(())
    }

    fn update_neighbors_of_vertex(
        &self,
        vertex_id: u32,
        scratch: &mut InMemQueryScratch<T, N>,
    ) -> Result<(), ANNError> {
        let vertex = self.final_graph.read_vertex_and_neighbors(vertex_id)?;
        assert!(vertex.size() <= self.configuration.index_write_parameter.max_degree as usize);
        self.inter_insert(
            vertex_id,
            vertex.get_neighbors(),
            self.configuration.index_write_parameter.max_degree,
            scratch,
        )?;
        Ok(())
    }

    fn update_vertex_with_neighbors(
        &self,
        vertex_id: u32,
        new_neighbors: AdjacencyList,
    ) -> Result<(), ANNError> {
        let vertex = &mut self.final_graph.write_vertex_and_neighbors(vertex_id)?;
        vertex.set_neighbors(new_neighbors);
        assert!(vertex.size() <= self.configuration.index_write_parameter.max_degree as usize);
        Ok(())
    }

    fn search_for_point_and_prune(
        &self,
        scratch: &mut InMemQueryScratch<T, N>,
        vertex_id: u32,
    ) -> ANNResult<AdjacencyList> {
        let mut pruned_list =
            AdjacencyList::for_range(self.configuration.index_write_parameter.max_degree as usize);
        let vertex = self.dataset.get_vertex(vertex_id)?;
        let mut visited_nodes = self.search_for_point(&vertex, scratch)?;

        self.prune_neighbors(vertex_id, &mut visited_nodes, &mut pruned_list, scratch)?;

        if pruned_list.is_empty() {
            return Err(ANNError::log_index_error(
                "pruned_list is empty.".to_string(),
            ));
        }

        if self.final_graph.size()
            != self.configuration.max_points + self.configuration.num_frozen_pts
        {
            return Err(ANNError::log_index_error(format!(
                "final_graph has {} vertices instead of {}",
                self.final_graph.size(),
                self.configuration.max_points + self.configuration.num_frozen_pts,
            )));
        }

        Ok(pruned_list)
    }

    fn search_with_distance(
        &self,
        query: &Vertex<T, N>,
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
        distances: Option<&mut [f32]>,
    ) -> ANNResult<u32> {
        if k_value > l_value as usize {
            return Err(ANNError::log_index_error(format!(
                "Set L: {} to a value of at least K: {}",
                l_value, k_value
            )));
        }

        // Create a scratch space for the search
        let mut scratch = InMemQueryScratch::<T, N>::new(
            l_value,
            &self.configuration.index_write_parameter,
            true,
        )?;

        // Perform the actual search
        let visited_nodes = self.search_for_point(query, &mut scratch)?;

        // Take the top k results
        let mut results = Vec::new();
        for neighbor in visited_nodes.iter().take(k_value) {
            results.push(neighbor);
        }

        // Sort by distance and fill the indices array
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        for (i, neighbor) in results.iter().enumerate() {
            if i < indices.len() {
                indices[i] = neighbor.id;
            }
        }

        // Calculate distances if requested
        if let Some(distances) = distances {
            for (i, neighbor) in results.iter().enumerate() {
                if i < distances.len() {
                    distances[i] = neighbor.distance;
                }
            }
        }

        Ok(results.len() as u32)
    }

    fn search(
        &self,
        query: &Vertex<T, N>,
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> ANNResult<u32> {
        self.search_with_distance(query, k_value, l_value, indices, None)
    }

    fn cleanup_graph(&mut self, visit_order: &Vec<u32>) -> ANNResult<()> {
        if self.num_active_pts > 0 {
            println!("Starting final cleanup..");
        }

        execute_with_rayon(
            0..visit_order.len(),
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                let vertex_id = visit_order[idx];
                let num_nbrs = self.get_neighbor_count(vertex_id)?;

                if num_nbrs <= self.configuration.index_write_parameter.max_degree as usize {
                    // Neighbor list is already small enough.
                    return Ok(());
                }

                // TODO: Fix ScratchStoreManager usage
                // let mut scratch_manager = ScratchStoreManager::new(
                //     self.query_scratch_queue.clone(),
                //     Duration::from_millis(10),
                // )?;
                // let scratch = scratch_manager.scratch_space().ok_or_else(|| {
                //     ANNError::log_index_error(
                //         "ScratchStoreManager doesn't have InMemQueryScratch instance available"
                //             .to_string(),
                //     )
                // })?;

                // For now, create a simple scratch space
                let mut scratch = InMemQueryScratch::<T, N>::new(
                    self.configuration.index_write_parameter.max_degree as u32,
                    &self.configuration.index_write_parameter,
                    true,
                )?;

                let mut dummy_pool = self.get_neighbors_for_vertex(vertex_id)?;

                let mut new_out_neighbors = AdjacencyList::for_range(
                    self.configuration.index_write_parameter.max_degree as usize,
                );
                self.prune_neighbors(
                    vertex_id,
                    &mut dummy_pool,
                    &mut new_out_neighbors,
                    &mut scratch,
                )?;

                self.final_graph
                    .write_vertex_and_neighbors(vertex_id)?
                    .set_neighbors(new_out_neighbors);

                Ok(())
            },
        )
    }

    /// Get the unique neighbors for a vertex.
    ///
    /// This code feels out of place here. This should have nothing to do with whether this
    /// is in memory index?
    /// # Errors
    ///
    /// This function will return an error if we are not able to get the read lock.
    fn get_neighbors_for_vertex(&self, vertex_id: u32) -> ANNResult<Vec<Neighbor>> {
        let binding = self.final_graph.read_vertex_and_neighbors(vertex_id)?;
        let neighbors = binding.get_neighbors();
        let dummy_pool = self.get_unique_neighbors(neighbors, vertex_id)?;

        Ok(dummy_pool)
    }

    /// Returns a vector of unique neighbors for the given vertex, along with their distances.
    ///
    /// # Arguments
    ///
    /// * `neighbors` - A vector of neighbor id index for the given vertex.
    /// * `vertex_id` - The given vertex id.
    ///
    /// # Errors
    ///
    /// Returns an `ANNError` if there is an error retrieving the vertex or one of its neighbors.
    pub fn get_unique_neighbors(
        &self,
        neighbors: &Vec<u32>,
        vertex_id: u32,
    ) -> Result<Vec<Neighbor>, ANNError> {
        let vertex = self.dataset.get_vertex(vertex_id)?;

        let len = neighbors.len();
        if len == 0 {
            return Ok(Vec::new());
        }

        self.dataset.prefetch_vector(neighbors[0]);

        let mut dummy_visited: HashSet<u32> = HashSet::with_capacity(len);
        let mut dummy_pool: Vec<Neighbor> = Vec::with_capacity(len);

        // Ultra-aggressive batching for neighbor processing
        let batch_size = 128; // Increased from 64 for maximum throughput
        for batch_start in (0..len).step_by(batch_size) {
            let batch_end = std::cmp::min(batch_start + batch_size, len);

            // Prefetch entire batch of neighbors for better cache locality
            for i in batch_start..batch_end {
                if i < len - 1 {
                    self.dataset.prefetch_vector(neighbors[i + 1]);
                }
            }

            // Process batch of neighbors with vectorized distance calculations
            for i in batch_start..batch_end {
                let current = neighbors[i];

                self.insert_neighbor_if_unique(
                    &mut dummy_visited,
                    current,
                    vertex_id,
                    &vertex,
                    &mut dummy_pool,
                )?;
            }
        }

        Ok(dummy_pool)
    }

    fn insert_neighbor_if_unique(
        &self,
        dummy_visited: &mut HashSet<u32>,
        current: u32,
        vertex_id: u32,
        vertex: &Vertex<'_, T, N>,
        dummy_pool: &mut Vec<Neighbor>,
    ) -> Result<(), ANNError> {
        if current != vertex_id {
            if let Vacant(entry) = dummy_visited.entry(current) {
                let cur_nbr_vertex = self.dataset.get_vertex(current)?;
                // Use optimized distance calculation for better performance
                let dist = vertex.compare(&cur_nbr_vertex, self.configuration.dist_metric);
                dummy_pool.push(Neighbor::new(current, dist));
                entry.insert();
            }
        }

        Ok(())
    }

    /// Get count of neighbors for a given vertex.
    ///
    /// # Errors
    ///
    /// This function will return an error if we can't get a lock.
    fn get_neighbor_count(&self, vertex_id: u32) -> ANNResult<usize> {
        let num_nbrs = self
            .final_graph
            .read_vertex_and_neighbors(vertex_id)?
            .size();
        Ok(num_nbrs)
    }

    fn soft_delete_vertex(&self, vertex_id_to_delete: u32) -> ANNResult<()> {
        if vertex_id_to_delete as usize > self.num_active_pts {
            return Err(ANNError::log_index_error(format!(
                "vertex_id_to_delete: {} is greater than the number of active points in the graph: {}",
                vertex_id_to_delete, self.num_active_pts
            )));
        }

        let mut delete_set_guard = match self.delete_set.write() {
            Ok(guard) => guard,
            Err(_) => {
                return Err(ANNError::log_index_error(format!(
                    "Failed to acquire delete_set lock, cannot delete vertex {}",
                    vertex_id_to_delete
                )));
            }
        };

        delete_set_guard.insert(vertex_id_to_delete);
        Ok(())
    }

    fn initialize_query_scratch(
        &mut self,
        num_threads: u32,
        search_candidate_size: u32,
    ) -> ANNResult<()> {
        self.query_scratch_queue.reserve(num_threads as usize)?;
        for _ in 0..num_threads {
            let scratch = Box::new(InMemQueryScratch::<T, N>::new(
                search_candidate_size,
                &self.configuration.index_write_parameter,
                false,
            )?);

            self.query_scratch_queue.push(scratch)?;
        }

        Ok(())
    }

    fn print_stats(&mut self) -> ANNResult<()> {
        let mut max = 0;
        let mut min = usize::MAX;
        let mut total = 0;
        let mut cnt = 0;

        for i in 0..self.num_active_pts {
            let vertex_id = i.try_into()?;
            let pool_size = self
                .final_graph
                .read_vertex_and_neighbors(vertex_id)?
                .size();
            max = cmp::max(max, pool_size);
            min = cmp::min(min, pool_size);
            total += pool_size;
            if pool_size < 2 {
                cnt += 1;
            }
        }

        println!(
            "Index built with degree: max: {} avg: {} min: {} count(deg<2): {}",
            max,
            (total as f32) / ((self.num_active_pts + self.configuration.num_frozen_pts) as f32),
            min,
            cnt
        );

        match self.delete_set.read() {
            Ok(guard) => {
                println!(
                    "Number of soft deleted vertices {}, soft deleted percentage: {}",
                    guard.len(),
                    (guard.len() as f32)
                        / ((self.num_active_pts + self.configuration.num_frozen_pts) as f32),
                );
            }
            Err(_) => {
                return Err(ANNError::log_lock_poison_error(
                    "Failed to acquire delete_set lock, cannot get the number of deleted vertices"
                        .to_string(),
                ));
            }
        };

        self.max_observed_degree = cmp::max(max as u32, self.max_observed_degree);

        Ok(())
    }
}

impl<T, const N: usize> ANNInmemIndex<T> for InmemIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    fn build(&mut self, filename: &str, num_points_to_load: usize) -> ANNResult<()> {
        // TODO: fresh-diskANN
        // std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

        if !file_exists(filename) {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Data file {} does not exist.",
                filename
            )));
        }

        let (file_num_points, file_dim) = load_metadata_from_file(filename)?;
        if file_num_points > self.configuration.max_points {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has {} points, 
                but index can support only {} points as specified in configuration.",
                num_points_to_load, file_num_points, self.configuration.max_points
            )));
        }

        if num_points_to_load > file_num_points {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has only {} points.",
                num_points_to_load, file_num_points
            )));
        }

        if file_dim != self.configuration.dim {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} dimension, but file has {} dimension.",
                self.configuration.dim, file_dim
            )));
        }

        if self.configuration.use_pq_dist {
            // TODO: PQ
            todo!("PQ is not supported now");
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        self.dataset.build_from_file(filename, num_points_to_load)?;

        println!("Using only first {} from file.", num_points_to_load);

        // TODO: tag_lock

        self.num_active_pts = num_points_to_load;
        self.build_with_data_populated()?;

        Ok(())
    }

    fn build_vector(&mut self, vector: &Vec<Vec<T>>) -> ANNResult<()> {
        let num_points_to_insert = vector.len();
        if num_points_to_insert == 0 {
            return Ok(());
        }

        let dim = N;
        if dim != self.configuration.dim {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} dimension, but file has {} dimension.",
                self.configuration.dim, dim
            )));
        }

        if self.configuration.use_pq_dist {
            // TODO: PQ
            todo!("PQ is not supported now");
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        self.or_increase_capacity(vector.len())?;
        self.dataset.build_from_vector(vector)?;

        println!("Using only first {} from file.", num_points_to_insert);

        // TODO: tag_lock

        self.num_active_pts = num_points_to_insert;
        self.build_with_data_populated()?;

        Ok(())
    }

    fn insert(&mut self, filename: &str, num_points_to_insert: usize) -> ANNResult<()> {
        // fresh-diskANN
        if !file_exists(filename) {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Data file {} does not exist.",
                filename
            )));
        }

        let (file_num_points, file_dim) = load_metadata_from_file(filename)?;

        if num_points_to_insert > file_num_points {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {} points and file has only {} points.",
                num_points_to_insert, file_num_points
            )));
        }

        if file_dim != self.configuration.dim {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {}  dimension, but file has {} dimension.",
                self.configuration.dim, file_dim
            )));
        }

        if self.configuration.use_pq_dist {
            // TODO: PQ
            todo!("PQ is not supported now");
        }

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            // set the thread count of Rayon, otherwise it will use threads as many as logical cores.
            std::env::set_var(
                "RAYON_NUM_THREADS",
                self.configuration
                    .index_write_parameter
                    .num_threads
                    .to_string(),
            );
        }

        self.dataset
            .append_from_file(filename, num_points_to_insert)?;
        self.final_graph.extend(
            num_points_to_insert,
            self.configuration.index_write_parameter.max_degree,
        );

        // TODO: this should not consider frozen points
        let previous_last_pt = self.num_active_pts;
        self.num_active_pts += num_points_to_insert;
        self.configuration.max_points += num_points_to_insert;

        println!("Inserting {} vectors from file.", num_points_to_insert);

        // TODO: tag_lock
        let logger = IndexLogger::new(num_points_to_insert);
        let timer = Timer::new();
        execute_with_rayon(
            previous_last_pt..self.num_active_pts,
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                self.insert_vertex_id(idx as u32)?;
                logger.vertex_processed()?;

                Ok(())
            },
        )?;

        let mut visit_order =
            Vec::with_capacity(self.num_active_pts + self.configuration.num_frozen_pts);
        for i in 0..self.num_active_pts {
            visit_order.push(i as u32);
        }

        self.cleanup_graph(&visit_order)?;
        println!("{}", timer.elapsed_seconds_for_step("Insert time: "));

        self.print_stats()?;

        Ok(())
    }

    fn insert_vector(&mut self, vector: &Vec<Vec<T>>) -> ANNResult<(usize, usize)> {
        let num_points_to_insert = vector.len();
        if num_points_to_insert == 0 {
            return Ok((0, 0));
        }

        let dim = N;

        if dim != self.configuration.dim {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Driver requests loading {}  dimension, but file has {} dimension.",
                self.configuration.dim, dim
            )));
        }

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        if self.configuration.index_write_parameter.num_threads > 0 {
            // set the thread count of Rayon, otherwise it will use threads as many as logical cores.
            std::env::set_var(
                "RAYON_NUM_THREADS",
                self.configuration
                    .index_write_parameter
                    .num_threads
                    .to_string(),
            );
        }

        self.or_increase_capacity(vector.len())?;
        let result = self.dataset.append_from_vector(vector)?;
        self.final_graph.extend(
            num_points_to_insert,
            self.configuration.index_write_parameter.max_degree,
        );

        // TODO: this should not consider frozen points
        let previous_last_pt = self.num_active_pts;
        self.num_active_pts += num_points_to_insert;
        self.configuration.max_points += num_points_to_insert;

        println!("Inserting {} vectors from file.", num_points_to_insert);

        // TODO: tag_lock
        let logger = IndexLogger::new(num_points_to_insert);
        let timer = Timer::new();
        execute_with_rayon(
            previous_last_pt..self.num_active_pts,
            self.configuration.index_write_parameter.num_threads,
            |idx| {
                self.insert_vertex_id(idx as u32)?;
                logger.vertex_processed()?;

                Ok(())
            },
        )?;

        let mut visit_order =
            Vec::with_capacity(self.num_active_pts + self.configuration.num_frozen_pts);
        for i in 0..self.num_active_pts {
            visit_order.push(i as u32);
        }

        self.cleanup_graph(&visit_order)?;
        println!("{}", timer.elapsed_seconds_for_step("Insert time: "));

        self.print_stats()?;

        Ok(result)
    }

    fn save(&mut self, filename: &str) -> ANNResult<()> {
        let data_file = filename.to_string() + ".data";
        let delete_file = filename.to_string() + ".delete";

        self.save_graph(filename)?;
        self.save_data(data_file.as_str())?;
        self.save_delete_list(delete_file.as_str())?;

        Ok(())
    }

    fn load_with_enhance(&mut self, filename: &str, expected_num_points: usize) -> ANNResult<()> {
        // self.num_active_pts = expected_num_points;

        let num_active_pts_saved = self.dataset.num_active_pts;
        self.dataset
            .build_from_file_with_enhance(&format!("{}.data", filename), expected_num_points)?;

        let diff = self.dataset.num_active_pts - num_active_pts_saved;
        self.num_active_pts = self.dataset.num_active_pts;

        self.final_graph
            .extend(diff, self.configuration.index_write_parameter.max_degree);

        self.configuration.max_points += diff;

        self.load_graph(filename, expected_num_points)?;
        self.load_delete_list(&format!("{}.delete", filename))?;

        if self.query_scratch_queue.size()? == 0 {
            self.initialize_query_scratch(
                5 + self.configuration.index_write_parameter.num_threads,
                self.configuration.index_write_parameter.search_list_size,
            )?;
        }

        Ok(())
    }

    fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> ANNResult<u32> {
        let query_vector = Vertex::new(<&[T; N]>::try_from(query)?, 0);
        InmemIndex::search(self, &query_vector, k_value, l_value, indices)
    }

    fn search_with_distance(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
        distances: &mut [f32],
    ) -> ANNResult<u32> {
        let query_vector = Vertex::new(<&[T; N]>::try_from(query)?, 0);
        InmemIndex::search_with_distance(
            self,
            &query_vector,
            k_value,
            l_value,
            indices,
            Some(distances),
        )
    }

    fn soft_delete(
        &mut self,
        vertex_ids_to_delete: Vec<u32>,
        num_points_to_delete: usize,
    ) -> ANNResult<()> {
        println!("Deleting {} vectors from file.", num_points_to_delete);

        let logger = IndexLogger::new(num_points_to_delete);
        let timer = Timer::new();

        execute_with_rayon(
            0..num_points_to_delete,
            self.configuration.index_write_parameter.num_threads,
            |idx: usize| {
                self.soft_delete_vertex(vertex_ids_to_delete[idx])?;
                logger.vertex_processed()?;

                Ok(())
            },
        )?;

        println!("{}", timer.elapsed_seconds_for_step("Delete time: "));
        self.print_stats()?;

        Ok(())
    }
}

#[cfg(test)]
mod index_test {
    use diskann_vector::Metric;

    use super::*;
    use crate::{
        model::{
            configuration::index_write_parameters::IndexWriteParametersBuilder, vertex::DIM_128,
        },
        test_utils::get_test_file_path,
        utils::file_util::load_ids_to_delete_from_file,
        utils::round_up,
    };

    const TEST_DATA_FILE: &str = "tests/data/siftsmall_learn_256pts.fbin";
    const TRUTH_GRAPH: &str = "tests/data/truth_index_siftsmall_learn_256pts_R4_L50_A1.2";
    const TEST_DELETE_FILE: &str = "tests/data/delete_set_50pts.bin";
    const TRUTH_GRAPH_WITH_SATURATED: &str =
        "tests/data/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_mem.index";
    const R: u32 = 4;
    const L: u32 = 50;
    const ALPHA: f32 = 1.2;

    /// Build the index with TEST_DATA_FILE and compare the index graph with truth graph TRUTH_GRAPH
    /// Change above constants if you want to test with different dataset
    macro_rules! index_end_to_end_test_singlethread {
        ($saturate_graph:expr, $truth_graph:expr) => {{
            let (data_num, dim) =
                load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

            let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
                .with_alpha(ALPHA)
                .with_num_threads(1)
                .with_saturate_graph($saturate_graph)
                .build();
            let config = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num,
                false,
                0,
                false,
                0,
                1.0f32,
                index_write_parameters,
            );
            let mut index: InmemIndex<f32, DIM_128> = InmemIndex::new(config.clone()).unwrap();

            index
                .build(get_test_file_path(TEST_DATA_FILE).as_str(), data_num)
                .unwrap();

            let mut truth_index: InmemIndex<f32, DIM_128> = InmemIndex::new(config).unwrap();
            truth_index
                .load_graph(get_test_file_path($truth_graph).as_str(), data_num)
                .unwrap();

            compare_graphs(&index, &truth_index);
        }};
    }

    #[test]
    fn index_end_to_end_test_singlethread() {
        index_end_to_end_test_singlethread!(false, TRUTH_GRAPH);
    }

    #[test]
    fn index_end_to_end_test_singlethread_with_saturate_graph() {
        index_end_to_end_test_singlethread!(true, TRUTH_GRAPH_WITH_SATURATED);
    }

    #[test]
    fn index_end_to_end_test_multithread() {
        let (data_num, dim) =
            load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

        let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
            .with_alpha(ALPHA)
            .with_num_threads(8)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            dim,
            round_up(dim as u64, 16_u64) as usize,
            data_num,
            false,
            0,
            false,
            0,
            1f32,
            index_write_parameters,
        );
        let mut index: InmemIndex<f32, DIM_128> = InmemIndex::new(config).unwrap();

        index
            .build(get_test_file_path(TEST_DATA_FILE).as_str(), data_num)
            .unwrap();

        for i in 0..index.final_graph.size() {
            assert_ne!(
                index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .size(),
                0
            );
        }
    }

    const TEST_DATA_FILE_2: &str = "tests/data/siftsmall_learn_256pts_2.fbin";
    const INSERT_TRUTH_GRAPH: &str =
        "tests/data/truth_index_siftsmall_learn_256pts_1+2_R4_L50_A1.2";
    const INSERT_TRUTH_GRAPH_WITH_SATURATED: &str =
        "tests/data/truth_index_siftsmall_learn_256pts_1+2_saturated_R4_L50_A1.2";

    /// Build the index with TEST_DATA_FILE, insert TEST_DATA_FILE_2 and compare the index graph with truth graph TRUTH_GRAPH
    /// Change above constants if you want to test with different dataset
    macro_rules! index_insert_end_to_end_test_singlethread {
        ($saturate_graph:expr, $truth_graph:expr) => {{
            let (data_num, dim) =
                load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

            let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
                .with_alpha(ALPHA)
                .with_num_threads(1)
                .with_saturate_graph($saturate_graph)
                .build();
            let config = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num,
                false,
                0,
                false,
                0,
                2.0f32,
                index_write_parameters,
            );
            let mut index: InmemIndex<f32, DIM_128> = InmemIndex::new(config.clone()).unwrap();

            index
                .build(get_test_file_path(TEST_DATA_FILE).as_str(), data_num)
                .unwrap();
            index
                .insert(get_test_file_path(TEST_DATA_FILE_2).as_str(), data_num)
                .unwrap();

            let config2 = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num * 2,
                false,
                0,
                false,
                0,
                1.0f32,
                index_write_parameters,
            );
            let mut truth_index: InmemIndex<f32, DIM_128> = InmemIndex::new(config2).unwrap();
            truth_index
                .load_graph(get_test_file_path($truth_graph).as_str(), data_num)
                .unwrap();

            compare_graphs(&index, &truth_index);
        }};
    }

    /// Build the index with TEST_DATA_FILE, and delete the vertices with id defined in TEST_DELETE_SET
    macro_rules! index_delete_end_to_end_test_singlethread {
        () => {{
            let (data_num, dim) =
                load_metadata_from_file(get_test_file_path(TEST_DATA_FILE).as_str()).unwrap();

            let index_write_parameters = IndexWriteParametersBuilder::new(L, R)
                .with_alpha(ALPHA)
                .with_num_threads(1)
                .build();
            let config = IndexConfiguration::new(
                Metric::L2,
                dim,
                round_up(dim as u64, 16_u64) as usize,
                data_num,
                false,
                0,
                false,
                0,
                2.0f32,
                index_write_parameters,
            );
            let mut index: InmemIndex<f32, DIM_128> = InmemIndex::new(config.clone()).unwrap();

            index
                .build(get_test_file_path(TEST_DATA_FILE).as_str(), data_num)
                .unwrap();

            let (num_points_to_delete, vertex_ids_to_delete) =
                load_ids_to_delete_from_file(TEST_DELETE_FILE).unwrap();
            index
                .soft_delete(vertex_ids_to_delete, num_points_to_delete)
                .unwrap();
            assert!(index.delete_set.read().unwrap().len() == num_points_to_delete);
        }};
    }

    #[test]
    fn index_insert_end_to_end_test_singlethread() {
        index_insert_end_to_end_test_singlethread!(false, INSERT_TRUTH_GRAPH);
    }

    #[test]
    fn index_delete_end_to_end_test_singlethread() {
        index_delete_end_to_end_test_singlethread!();
    }

    #[test]
    fn index_insert_end_to_end_test_saturated_singlethread() {
        index_insert_end_to_end_test_singlethread!(true, INSERT_TRUTH_GRAPH_WITH_SATURATED);
    }

    fn compare_graphs(index: &InmemIndex<f32, DIM_128>, truth_index: &InmemIndex<f32, DIM_128>) {
        assert_eq!(index.start, truth_index.start);
        assert_eq!(index.max_observed_degree, truth_index.max_observed_degree);
        assert_eq!(index.final_graph.size(), truth_index.final_graph.size());

        for i in 0..index.final_graph.size() {
            assert_eq!(
                index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .size(),
                truth_index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .size()
            );
            assert_eq!(
                index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .get_neighbors(),
                truth_index
                    .final_graph
                    .read_vertex_and_neighbors(i as u32)
                    .unwrap()
                    .get_neighbors()
            );
        }
    }
}
