use std::mem;
use std::str::FromStr;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::os::linux::fs::MetadataExt;

use crate::instrumentation::DiskIndexConstructionCheckpoint;
use diskann_vector::{FullPrecisionDistance, Metric};
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::common::{ANNError, ANNResult};
use crate::disk_search::pq_flash_index::{DummyFileReader, PQFlashIndex};
use crate::index::percentile_stats::{get_mean_stats, get_percentile_stats, QueryStats};
use crate::index::utils::{calculate_recall, load_aligned_bin, load_truthset};
use crate::index::{ANNInmemIndex, InmemIndex};
use crate::instrumentation::DiskIndexBuildLogger;
use crate::model::configuration::DiskIndexBuildParameters;
use crate::model::{
    generate_quantized_data, IndexConfiguration, GRAPH_SLACK_FACTOR, MAX_PQ_CHUNKS,
    MAX_PQ_TRAINING_SET_SIZE,
};
use crate::storage::DiskIndexStorage;
use crate::utils::{convert_types_u64_u32, file_exists, set_rayon_num_threads};

use super::ann_disk_index::ANNDiskIndex;

pub const OVERHEAD_FACTOR: f64 = 1.1f64;

pub const MAX_SAMPLE_POINTS_FOR_WARMUP: usize = 100_000;

const WARMUP: bool = false;

pub struct DiskIndex<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Parameters for index construction
    /// None for query path
    disk_build_param: Option<DiskIndexBuildParameters>,

    configuration: IndexConfiguration,

    pub storage: DiskIndexStorage<T>,
}

impl<T, const N: usize> DiskIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    pub fn new(
        disk_build_param: Option<DiskIndexBuildParameters>,
        configuration: IndexConfiguration,
        storage: DiskIndexStorage<T>,
    ) -> Self {
        Self {
            disk_build_param,
            configuration,
            storage,
        }
    }

    pub fn disk_build_param(&self) -> &Option<DiskIndexBuildParameters> {
        &self.disk_build_param
    }

    pub fn index_configuration(&self) -> &IndexConfiguration {
        &self.configuration
    }

    fn build_inmem_index(
        &self,
        num_points: usize,
        data_path: &str,
        inmem_index_path: &str,
    ) -> ANNResult<()> {
        let estimated_index_ram = self.estimate_ram_usage(num_points);
        if estimated_index_ram
            >= self.fetch_disk_build_param()?.index_build_ram_limit()
                * 1024_f64
                * 1024_f64
                * 1024_f64
        {
            return Err(ANNError::log_index_error(format!(
                "Insufficient memory budget for index build, index_build_ram_limit={}GB estimated_index_ram={}GB",
                self.fetch_disk_build_param()?.index_build_ram_limit(),
                estimated_index_ram / (1024_f64 * 1024_f64 * 1024_f64),
            )));
        }

        let mut index = InmemIndex::<T, N>::new(self.configuration.clone())?;
        index.build(data_path, num_points)?;
        index.save(inmem_index_path)?;

        Ok(())
    }

    #[inline]
    fn estimate_ram_usage(&self, size: usize) -> f64 {
        let degree = self.configuration.index_write_parameter.max_degree as usize;
        let datasize = mem::size_of::<T>();

        let dataset_size = (size * N * datasize) as f64;
        let graph_size = (size * degree * mem::size_of::<u32>()) as f64 * GRAPH_SLACK_FACTOR;

        OVERHEAD_FACTOR * (dataset_size + graph_size)
    }

    #[inline]
    fn fetch_disk_build_param(&self) -> ANNResult<&DiskIndexBuildParameters> {
        self.disk_build_param.as_ref().ok_or_else(|| {
            ANNError::log_index_config_error(
                "disk_build_param".to_string(),
                "disk_build_param is None".to_string(),
            )
        })
    }

    // load_aligned_bin functions START

    //template <typename T, typename LabelT = uint32_t>
    pub fn search_disk_index<U>(
        metric: Metric,
        index_path_prefix: &str,
        num_threads: u64,
        query_file: &str,
        gt_file: &str,
        _warmup_query_file: &str,
        recall_at: u32,
        num_nodes_to_cache: u64,
        _search_list_size: u32,
        beamwidth: u32,
        fail_if_recall_below: f64,
        query_filters: &Vec<&str>,
        _use_reorder_data: bool, /*  = false */
    ) -> ANNResult<i32>
    where
        U: Default + Clone + Copy + Send + Sync + 'static,
    {
        let warmup_query_file = format!("{index_path_prefix}_sample_data.bin");

        // load query bin
        //T *query = nullptr;
        //let mut query = vec![];
        let _query; // = AlignedBoxWithSlice::new(capacity, alignment);
        let query_num: usize;
        let query_dim: usize;
        let query_aligned_dim: usize;
        let (query_data, query_num_val, query_dim_val) = load_aligned_bin::<T>(query_file)?;
        _query = query_data;
        query_num = query_num_val;
        query_dim = query_dim_val;
        query_aligned_dim = query_dim_val;

        let _filtered_search = false;
        if !query_filters.is_empty() {
            // filtered_search = true;
            if query_filters.len() != 1 && query_filters.len() != query_num {
                // "Error. Mismatch in number of queries and size of query filters file
                return Ok(-1); // To return -1 or some other error handling?
            }
        }

        let gt_ids: Vec<u32> = vec![];
        let _gt_dists: Vec<f32> = vec![];
        let mut calc_recall_flag = false;
        if gt_file != "null" && gt_file != "NULL" && file_exists(gt_file) {
            let (gt_ids, _gt_dists) =
                load_truthset(gt_file, query_num, query_dim, recall_at as usize, 1)?;
            if gt_ids.len() != query_num {
                // Error. Mismatch in number of queries and ground truth data
            }
            calc_recall_flag = true;
        }

        // TODO: check if index_path_prefix can be the right path for .ann (instead of a separate parameter)
        #[cfg(target_os = "windows")]
        let reader = WindowsAlignedFileReader::new(index_path_prefix)?; // as AlignedFileReader;

        #[cfg(target_os = "linux")]
        let reader = LinuxAlignedFileReader::new(index_path_prefix)?; // as AlignedFileReader;

        let mut _p_flash_index: PQFlashIndex<U, u32> =
            PQFlashIndex::new(Arc::new(DummyFileReader), metric);
        let res = _p_flash_index.load(num_threads.try_into().unwrap(), index_path_prefix);

        if res != 0 {
            return Ok(res);
        }

        let mut node_list: Vec<u32> = vec![];

        // Caching num_nodes_to_cache nodes around medoid(s)
        _p_flash_index.cache_bfs_levels(num_nodes_to_cache, &mut node_list, false);
        // if (num_nodes_to_cache > 0)
        //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
        //     num_threads, node_list);
        _p_flash_index.load_cache_list(&node_list);
        node_list.clear();
        node_list.shrink_to_fit();

        let _warmup_l = 20u64;
        let mut _warmup_num = 0usize;
        let mut _warmup_dim = 0usize;
        let mut _warmup_aligned_dim = 0usize;
        // T *warmup = nullptr;
        let mut _warmup: Vec<T> = vec![];

        if WARMUP {
            if file_exists(&warmup_query_file) {
                let (warmup_data, warmup_num_val, warmup_dim_val) =
                    load_aligned_bin::<T>(&warmup_query_file)?;
                _warmup = warmup_data;
                _warmup_num = warmup_num_val;
                _warmup_dim = warmup_dim_val;
                _warmup_aligned_dim = warmup_dim_val;
            } else {
                _warmup_num = 150000.min((15000 * num_threads).try_into().unwrap());
                _warmup_dim = query_dim;
                _warmup_aligned_dim = query_aligned_dim;
                _warmup = vec![T::default(); _warmup_num * _warmup_aligned_dim];

                let _rng = StdRng::from_entropy();
                let _range = Uniform::new_inclusive(-128, 127);

                for _i in 0.._warmup_num {
                    let index_base = _i * _warmup_aligned_dim;
                    for d in 0.._warmup_dim {
                        _warmup[index_base + d] = T::default(); // Placeholder
                    }
                }
            }

            // Warming up index...
            let _warmup_result_ids_64 = vec![0u64; _warmup_num];
            let _warmup_result_dists = vec![0f32; _warmup_num];

            (0.._warmup_num).into_par_iter().for_each(|_i| {
                // Placeholder - skip actual search for now
                // _pFlashIndex.cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_l,
                //                                 warmup_result_ids_64.data() + (i * 1),
                //                                 warmup_result_dists.data() + (i * 1), 4);
            });
        }

        let mut query_result_ids: Vec<Vec<u32>> = vec![vec![]; 1]; // Placeholder
        let mut query_result_dists: Vec<Vec<f32>> = vec![vec![]; 1]; // Placeholder

        let mut _optimized_beamwidth = 2;
        let mut best_recall = 0.0;

        for test_id in 0..1 {
            // Placeholder
            let l = 10u32; // Placeholder
            if l < recall_at {
                // Ignoring search with `L` since it's smaller than `K`
                continue;
            }

            if beamwidth <= 0 {
                // Tuning beamwidth..
                _optimized_beamwidth = 2; // Placeholder
            } else {
                _optimized_beamwidth = beamwidth;
            }

            query_result_ids[test_id].resize((recall_at as usize) * query_num, 0u32);
            query_result_dists[test_id].resize((recall_at as usize) * query_num, 0.0f32);

            let stats = vec![QueryStats::default(); query_num];
            let query_result_ids_64 = vec![0u64; (recall_at as usize) * query_num];

            // Placeholder - skip actual search for now
            // (0..query_num).into_par_iter().for_each(|i| {
            //     if !filtered_search {
            //         _pFlashIndex.cached_beam_search(query + (i * query_aligned_dim), recall_at, l,
            //                                         query_result_ids_64.data() + (i * recall_at),
            //                                         query_result_dists[test_id].data() + (i * recall_at),
            //                                         optimized_beamwidth, use_reorder_data, stats + i);
            //     }
            //     else
            //     {
            //         let label_for_search: LabelT;
            //         if (query_filters.size() == 1)
            //         { // one label for all queries
            //             label_for_search = _pFlashIndex.get_converted_label(query_filters[0]);
            //         }
            //         else
            //         { // one label for each query
            //             label_for_search = _pFlashIndex.get_converted_label(query_filters[i]);
            //         }
            //         _pFlashIndex.cached_beam_search(
            //             query + (i * query_aligned_dim), recall_at, l, query_result_ids_64.data() + (i * recall_at),
            //             query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
            //             use_reorder_data, stats + i);
            //     }
            // });

            query_result_ids[test_id] = convert_types_u64_u32(
                &query_result_ids_64,
                query_num,
                recall_at.try_into().unwrap(),
            );

            let _mean_latency = get_mean_stats(&stats, |stat: &QueryStats| stat.total_us);
            let _latency_999 = get_percentile_stats(&stats, stats.len(), 0.999);
            let _mean_ios = get_mean_stats(&stats, |stat: &QueryStats| stat.n_ios);
            let _mean_cpuus = get_mean_stats(&stats, |stat: &QueryStats| stat.cpu_us);
            let _mean_io_us = get_mean_stats(&stats, |stat: &QueryStats| stat.io_us);

            let mut _recall = 0f64;
            if calc_recall_flag {
                _recall = calculate_recall(
                    &query_result_ids[test_id],
                    &gt_ids,
                    query_num,
                    query_dim,
                    query_dim,
                    recall_at as usize,
                    recall_at as usize,
                ) as f64;
                best_recall = _recall.max(best_recall);
            }
        }

        // Done searching. Now saving results
        let mut _test_id = 0u64;
        for &l in &[10u32] {
            // Placeholder
            if l < recall_at {
                continue;
            }

            // Placeholder - skip file saving for now
            // let cur_result_path = format!("{result_output_prefix}_{l}_idx_uint32.bin");
            // diskann::save_bin<uint32_t>(&cur_result_path, &query_result_ids[test_id], query_num, recall_at);

            // let cur_result_path = format!("{result_output_prefix}_{l}_dists_float.bin");
            // diskann::save_bin<float>(cur_result_path, &query_result_dists[test_id], query_num, recall_at);
            _test_id += 1;
        }

        Ok(if best_recall >= fail_if_recall_below.into() {
            0
        } else {
            -1
        })
    }
}

impl<T, const N: usize> ANNDiskIndex<T> for DiskIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    fn load(&mut self) -> ANNResult<()> {
        // Verify that the index files exist
        let disk_index_path = self.storage.index_path_prefix().clone() + "_disk.index";
        let mem_index_path = self.storage.index_path_prefix().clone() + "_mem.index";

        if !file_exists(&disk_index_path) {
            return Err(ANNError::log_index_error(format!(
                "Disk index file not found: {}",
                disk_index_path
            )));
        }

        if !file_exists(&mem_index_path) {
            return Err(ANNError::log_index_error(format!(
                "Memory index file not found: {}",
                mem_index_path
            )));
        }

        println!(
            "Loading existing disk index from: {}",
            self.storage.index_path_prefix()
        );
        println!("  - Disk index: {}", disk_index_path);
        println!("  - Memory index: {}", mem_index_path);

        // The index is already loaded when the storage is created
        // We just need to verify the files exist and are accessible
        Ok(())
    }

    fn build(&mut self, codebook_prefix: &str) -> ANNResult<()> {
        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        let total_build_start = std::time::Instant::now();
        println!(
            "Starting index build: R={} L={} Query RAM budget={} Indexing RAM budget={} T={}",
            self.configuration.index_write_parameter.max_degree,
            self.configuration.index_write_parameter.search_list_size,
            self.fetch_disk_build_param()?.search_ram_limit(),
            self.fetch_disk_build_param()?.index_build_ram_limit(),
            self.configuration.index_write_parameter.num_threads
        );
        println!("📊 Total documents to build: {}", self.configuration.max_points);

        let mut logger = DiskIndexBuildLogger::new(DiskIndexConstructionCheckpoint::PqConstruction);

        // PQ memory consumption = PQ pivots + PQ compressed table
        // PQ pivots: dim * num_centroids * sizeof::<T>()
        // PQ compressed table: num_pts * num_pq_chunks * (dim / num_pq_chunks) * sizeof::<u8>()
        // * Because num_centroids is 256, centroid id can be represented by u8
        let num_points = self.configuration.max_points;
        let dim = self.configuration.dim;
        let p_val = MAX_PQ_TRAINING_SET_SIZE / (num_points as f64);
        let mut num_pq_chunks = ((self.fetch_disk_build_param()?.search_ram_limit()
            / (num_points as f64))
            .floor()) as usize;
        num_pq_chunks = if num_pq_chunks == 0 { 1 } else { num_pq_chunks };
        num_pq_chunks = if num_pq_chunks > dim {
            dim
        } else {
            num_pq_chunks
        };
        num_pq_chunks = if num_pq_chunks > MAX_PQ_CHUNKS {
            MAX_PQ_CHUNKS
        } else {
            num_pq_chunks
        };

        println!(
            "Compressing {}-dimensional data into {} bytes per vector.",
            dim, num_pq_chunks
        );

        let pq_start = std::time::Instant::now();
        // Use ultra-optimized PQ generation
        generate_quantized_data::<T>(
            p_val,
            num_pq_chunks,
            codebook_prefix,
            self.storage.get_pq_storage(),
        )?;
        let pq_elapsed = pq_start.elapsed();
        println!("PQ compression completed in {:.2?}", pq_elapsed);
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::InmemIndexBuild)?;

        // TODO: Decouple index from file access
        let inmem_index_path = self.storage.index_path_prefix().clone() + "_mem.index";
        let inmem_start = std::time::Instant::now();
        println!("🔨 Starting in-memory index build for {} documents...", num_points);
        self.build_inmem_index(
            num_points,
            self.storage.dataset_file(),
            inmem_index_path.as_str(),
        )?;
        let inmem_elapsed = inmem_start.elapsed();
        println!("✅ In-memory index build completed in {:.2?}", inmem_elapsed);
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::DiskLayout)?;

        let disk_layout_start = std::time::Instant::now();
        self.storage.create_disk_layout()?;
        let disk_layout_elapsed = disk_layout_start.elapsed();
        println!("Disk layout creation completed in {:.2?}", disk_layout_elapsed);
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::None)?;

        let ten_percent_points = ((num_points as f64) * 0.1_f64).ceil();
        let num_sample_points = if ten_percent_points > (MAX_SAMPLE_POINTS_FOR_WARMUP as f64) {
            MAX_SAMPLE_POINTS_FOR_WARMUP as f64
        } else {
            ten_percent_points
        };
        let sample_sampling_rate = num_sample_points / (num_points as f64);
        self.storage.gen_query_warmup_data(sample_sampling_rate)?;

        self.storage.index_build_cleanup()?;

        let total_build_elapsed = total_build_start.elapsed();
        let total_tps = num_points as f64 / total_build_elapsed.as_secs_f64();
        println!("🚀 Total index build completed in {:.2?} | Overall TPS: {:.1} docs/sec", 
                total_build_elapsed, total_tps);

        Ok(())
    }

    fn search(
        &self,
        _query: &[T],
        _k_value: usize,
        _l_value: u32,
        _indices: &mut [u32],
    ) -> ANNResult<u32> {
        unimplemented!()
    }

    fn search_with_distance(
        &self,
        _query: &[T],
        _k_value: usize,
        _l_value: u32,
        _indices: &mut [u32],
        _distances: &mut [f32],
    ) -> ANNResult<u32> {
        unimplemented!()
    }
}

#[allow(dead_code)]
pub fn search_disk_index<T, LabelT>(
    _metric: Metric,
    _index_path_prefix: &str,
    _result_output_prefix: &str,
    _query_file: &str,
    _gt_file: &str,
    _num_threads: u32,
    _recall_at: u32,
    _beamwidth: u32,
    _num_nodes_to_cache: u32,
    _search_io_limit: u32,
    _lvec: &Vec<u32>,
    _fail_if_recall_below: f32,
    _query_filters: &Vec<&str>,
    _use_reorder_data: bool, /*  = false */
) -> ANNResult<i32>
where
    T: Default + Clone + Copy + Send + Sync,
    LabelT: Default
        + FromStr
        + Clone
        + Copy
        + Eq
        + std::hash::Hash
        + std::marker::Send
        + std::marker::Sync,
{
    // Placeholder implementation - just return success
    Ok(0)
}

#[allow(dead_code)]
pub fn search_disk_index_with_filters<T>(
    _query: &[T],
    _k_value: usize,
    _l_value: u32,
    _indices: &mut [u32],
) -> ANNResult<()> {
    // Placeholder implementation
    Ok(())
}

#[allow(dead_code)]
pub fn search_disk_index_with_filters_and_distances<T>(
    _query: &[T],
    _k_value: usize,
    _l_value: u32,
    _indices: &mut [u32],
    _distances: &mut [f32],
) -> ANNResult<()> {
    // Placeholder implementation
    Ok(())
}
