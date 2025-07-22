use std::io::Read;
/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::mem;

use std::fs::File;

cfg_if! {
    if #[cfg(target_os = "windows")] {
        use std::os::windows::fs::MetadataExt;
    } else {
        use std::os::linux::fs::MetadataExt;
    }
}

use byteorder::{NativeEndian, ReadBytesExt};
use cfg_if::cfg_if;
use logger::logger::indexlog::DiskIndexConstructionCheckpoint;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use vector::{FullPrecisionDistance, Metric};

use crate::common::{ANNError, ANNResult, AlignedBoxWithSlice};
//use crate::index::percentile_stats::{get_mean_stats, get_percentile_stats, QueryStats};
//use crate::index::utils::{calculate_recall, load_aligned_bin, load_truthset};
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

macro_rules! round_up {
    ($x:expr, $y:expr) => {
        (($x / $y) + (if $x % $y != 0 { 1 } else { 0 })) * $y
    };
}

macro_rules! div_round_up {
    ($x:expr, $y:expr) => {
        ($x / $y) + (if $x % $y != 0 { 1 } else { 0 })
    };
}

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
    /* disk part
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // load_aligned_bin functions START

    //template <typename T, typename LabelT = uint32_t>
    fn search_disk_index(metric: Metric, index_path_prefix: &str,
                        result_output_prefix: &str, query_file: &str, gt_file: &str,
                        num_threads: u32, recall_at: u32, beamwidth: u32,
                        num_nodes_to_cache: u32, search_io_limit: u32,
                        Lvec: &Vec<u32>, fail_if_recall_below: f32,
                        query_filters: &Vec<&str>, use_reorder_data: bool/*  = false */) -> ANNResult<i32>
    {
        let warmup_query_file = format!("{index_path_prefix}_sample_data.bin");

        // load query bin
        //T *query = nullptr;
        //let mut query = vec![];
        let mut query; // = AlignedBoxWithSlice::new(capacity, alignment);
        let mut query_num: usize;
        let mut query_dim: usize;
        let mut query_aligned_dim: usize;
        load_aligned_bin<T>(query_file, &mut query, &mut query_num, &mut query_dim, &mut query_aligned_dim)?;

        let filtered_search = false;
        if (!query_filters.is_empty()) {
            filtered_search = true;
            if (query_filters.len() != 1 && query_filters.len() != query_num) {
                // "Error. Mismatch in number of queries and size of query filters file
                return -1; // To return -1 or some other error handling?
            }
        }

        let mut gt_ids;
        let mut gt_dists;
        let mut calc_recall_flag = false;
        if (gt_file != "null" && gt_file != "NULL" && file_exists(gt_file)) {
            let mut gt_num;
            let mut gt_dim;
            load_truthset(gt_file, &mut gt_ids, &mut gt_dists, &mut gt_num, &mut gt_dim)?;
            if (gt_num != query_num) {
                // Error. Mismatch in number of queries and ground truth data
            }
            calc_recall_flag = true;
        }

        //std::shared_ptr<AlignedFileReader> reader = nullptr;
        cfg_if! {
            let reader: AlignedFileReader = if #[cfg(target_os = "windows")] {
                reader.reset(new WindowsAlignedFileReader())
            } else {
                reader.reset(new LinuxAlignedFileReader())
            };
        };

        //std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(new diskann::PQFlashIndex<T, LabelT>(reader, metric));
        let _pFlashIndex: PQFlashIndex<T, LabelT> = PQFlashIndex::new();

        let res = _pFlashIndex.load(num_threads, index_path_prefix.c_str());

        if (res != 0) {
            return res;
        }

        std::vector<uint32_t> node_list;
        diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
        _pFlashIndex.cache_bfs_levels(num_nodes_to_cache, node_list);
        // if (num_nodes_to_cache > 0)
        //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
        //     num_threads, node_list);
        _pFlashIndex.load_cache_list(node_list);
        node_list.clear();
        node_list.shrink_to_fit();

        let warmup_l = 20u64;
        let warmup_num = 0usize;
        let warmup_dim = 0usize;
        let warmup_aligned_dim = 0usize;
        // T *warmup = nullptr;
        let mut warmup; // = AlignedBoxWithSlice::new(capacity, alignment);

        if (WARMUP)
        {
            if (file_exists(&warmup_query_file)) {
                load_aligned_bin<T>(&warmup_query_file, &mut warmup, &mut warmup_num, &mut warmup_dim, &mut warmup_aligned_dim);
            }
            else
            {
                warmup_num = 150000.min(15000 * num_threads);
                warmup_dim = query_dim;
                warmup_aligned_dim = query_aligned_dim;
                warmup = AlignedBoxWithSlice::new(warmup_num * warmup_aligned_dim, 8 * mem::sizeof::<T>()).unwrap();


                let mut rng = StdRng::from_entropy();
                let range = Uniform::new_inclusive(-128, 127);

                for i in 0..warmup_num {
                    let index_base = i * warmup_aligned_dim;
                    for d in 0..warmup_dim {
                        warmup[index_base + d] = rng.sample(&range).into();
                    }
                }
            }

            // Warming up index...
            let mut warmup_result_ids_64 = vec![0u64; warmup_num];
            let mut warmup_result_dists = vec![0f32; warmup_num];

            (0..warmup_num).into_par_iter().for_each(|i| {
                _pFlashIndex.cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_l,
                                                warmup_result_ids_64.data() + (i * 1),
                                                warmup_result_dists.data() + (i * 1), 4);
            });
        }

        let mut query_result_ids: Vec<Vec<u32>> = vec![vec![]; Lvec.len()];
        let mut query_result_dists: Vec<Vec<f32>> = vec![vec![]; Lvec.len()];

        let mut optimized_beamwidth = 2;
        let best_recall = 0.0;

        for test_id in 0..Lvec.len() {
            let l = Lvec[test_id];
            if l < recall_at {
                // Ignoring search with `L` since it's smaller than `K`
                continue;
            }

            if beamwidth <= 0 {
                // Tuning beamwidth..
                optimized_beamwidth =
                    optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, l, optimized_beamwidth);
            }
            else {
                optimized_beamwidth = beamwidth;
            }

            query_result_ids[test_id].resize(recall_at * query_num);
            query_result_dists[test_id].resize(recall_at * query_num);

            let mut stats = vec![QueryStats::default(); query_num];
            let mut query_result_ids_64 = vec![0u64; recall_at * query_num];

            (0..query_num).into_par_iter().for_each(|i| {
                if !filtered_search {
                    _pFlashIndex.cached_beam_search(query + (i * query_aligned_dim), recall_at, l,
                                                    query_result_ids_64.data() + (i * recall_at),
                                                    query_result_dists[test_id].data() + (i * recall_at),
                                                    optimized_beamwidth, use_reorder_data, stats + i);
                }
                else
                {
                    let label_for_search: LabelT;
                    if (query_filters.size() == 1)
                    { // one label for all queries
                        label_for_search = _pFlashIndex.get_converted_label(query_filters[0]);
                    }
                    else
                    { // one label for each query
                        label_for_search = _pFlashIndex.get_converted_label(query_filters[i]);
                    }
                    _pFlashIndex.cached_beam_search(
                        query + (i * query_aligned_dim), recall_at, l, query_result_ids_64.data() + (i * recall_at),
                        query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                        use_reorder_data, stats + i);
                }

            });

            query_result_ids[test_id] = convert_types_u64_u32(&query_result_ids_64, query_num, recall_at);

            let mean_latency = get_mean_stats(&stats, |stat: &QueryStats| { stat.total_us });
            let latency_999 = get_percentile_stats(&stats, 0.999, |stat: &QueryStats| { stat.total_us });
            let mean_ios = get_mean_stats(&stats, |stat: &QueryStats| { stat.n_ios });
            let mean_cpuus = get_mean_stats(&stats, |stat: &QueryStats| { stat.cpu_us });
            let mean_io_us = get_mean_stats(&stats, |stat: &QueryStats| { stat.io_us });

            let mut recall = 0f64;
            if (calc_recall_flag)
            {
                recall = calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                                &query_result_ids[test_id], recall_at, recall_at);
                best_recall = recall.max(best_recall);
            }
        }

        // Done searching. Now saving results
        let mut test_id = 0u64;
        for &l in Lvec {
            if (l < recall_at) {
                continue;
            }

            let cur_result_path = format!("{result_output_prefix}_{l}_idx_uint32.bin");
            diskann::save_bin<uint32_t>(&cur_result_path, &query_result_ids[test_id], query_num, recall_at);

            let cur_result_path = format!("{result_output_prefix}_{l}_dists_float.bin");
            diskann::save_bin<float>(cur_result_path, &query_result_dists[test_id], query_num, recall_at);
            test_id += 1;
        }

        Ok(if best_recall >= fail_if_recall_below {0} else{-1})
    }
    */
}

impl<T, const N: usize> ANNDiskIndex<T> for DiskIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    fn build(&mut self, codebook_prefix: &str) -> ANNResult<()> {
        if self.configuration.index_write_parameter.num_threads > 0 {
            set_rayon_num_threads(self.configuration.index_write_parameter.num_threads);
        }

        println!(
            "Starting index build: R={} L={} Query RAM budget={} Indexing RAM budget={} T={}",
            self.configuration.index_write_parameter.max_degree,
            self.configuration.index_write_parameter.search_list_size,
            self.fetch_disk_build_param()?.search_ram_limit(),
            self.fetch_disk_build_param()?.index_build_ram_limit(),
            self.configuration.index_write_parameter.num_threads
        );

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

        // TODO: Decouple PQ from file access
        generate_quantized_data::<T>(
            p_val,
            num_pq_chunks,
            codebook_prefix,
            self.storage.get_pq_storage(),
        )?;
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::InmemIndexBuild)?;

        // TODO: Decouple index from file access
        let inmem_index_path = self.storage.index_path_prefix().clone() + "_mem.index";
        self.build_inmem_index(
            num_points,
            self.storage.dataset_file(),
            inmem_index_path.as_str(),
        )?;
        logger.log_checkpoint(DiskIndexConstructionCheckpoint::DiskLayout)?;

        self.storage.create_disk_layout()?;
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

        Ok(())
    }

    fn search(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
    ) -> ANNResult<u32> {
        unimplemented!()
    }

    fn search_with_distance(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
        distances: &mut [f32],
    ) -> ANNResult<u32> {
        unimplemented!()
    }
}
