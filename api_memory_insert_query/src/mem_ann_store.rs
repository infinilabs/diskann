#![allow(dead_code)]
use std::marker::PhantomData;

use diskann::{
    common::ANNResult,
    index::{ANNInmemIndex, create_inmem_index, INIT_WARMUP_DATA_LEN},
    model::{
        IndexConfiguration,
        configuration::index_write_parameters::IndexWriteParametersBuilder,
        vertex::{DIM_104, DIM_128, DIM_256, DIM_512},
    },
    utils::round_up,
};

use vector::{FullPrecisionDistance, Metric};

pub struct MemANNStore<T>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_512]: FullPrecisionDistance<T, DIM_512>,
{
    metric: Metric,

    max_degree: u32,

    search_list_size: u32,

    alpha: f32,

    num_threads: u32,

    config: IndexConfiguration,

    index: Box<dyn ANNInmemIndex<T>>,

    _phantom_data: PhantomData<T>,
}

impl<T> MemANNStore<T>
where
    T: Default + Copy + Sync + Send + Into<f32> + From<f32> + 'static,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_512]: FullPrecisionDistance<T, DIM_512>,
{
    pub fn new(
        metric: Metric,
        dimension: usize,
        max_degree: u32,
        search_list_size: u32,
        alpha: f32,
        num_threads: u32,
        max_point: usize
    ) -> ANNResult<Self> {
        let index_write_parameters = IndexWriteParametersBuilder::new(search_list_size, max_degree)
            .with_alpha(alpha)
            .with_saturate_graph(false)
            .with_num_threads(num_threads)
            .build();

        let config = IndexConfiguration::new(
            metric,
            dimension,
            round_up(dimension as u64, 8_u64) as usize,
            //8388608,
            max_point,
            false,
            0,
            false,
            0,
            2.0f32,
            index_write_parameters,
        );
        let index = create_inmem_index::<T>(config.clone())?;

        let mut create_points = vec![vec![T::default(); dimension]; 5];

        for i in 0..INIT_WARMUP_DATA_LEN as usize{
            for j in 0..dimension {
                create_points[i][j] = ((i+j) as f32).into();
            }
        }

        let mut slf = Self {
            metric,
            max_degree,
            search_list_size,
            alpha,
            num_threads,
            config,
            index,
            _phantom_data: PhantomData,
        };

        Self::init_data(&mut slf, &create_points)?;

        Ok(slf)
    }

    pub fn init_data(&mut self, data: &Vec<Vec<T>>) -> ANNResult<()> {
        self.index.build_vector(data)
    }

    /// Return (id_start, id_len)
    pub fn insert_data(&mut self, data: &Vec<Vec<T>>) -> ANNResult<(usize, usize)> {
        self.index.insert_vector(data)
    }

    pub fn soft_delete(&mut self, vertex_ids_to_delete: Vec<u32>) -> ANNResult<()> {
        let len = vertex_ids_to_delete.len();
        self.index.soft_delete(vertex_ids_to_delete, len)
    }

    pub fn save_to_file(&mut self, save_path: &str) -> ANNResult<()> {
        self.index.save(save_path)
    }

    pub fn load_from_file(&mut self, save_path: &str) -> ANNResult<()> {
        self.index.load_with_enhance(save_path, 0)
    }

    pub fn query(
        &self,
        query: &[T],
        k_value: usize,
        l_value: u32,
        indices: &mut [u32],
        distances: &mut [f32],
    ) -> ANNResult<u32> {
        self.index
            .search_with_distance(query, k_value, l_value, indices, distances)
    }
}
