/*
use diskann::{
    common::ANNResult,
    index::ann_disk_index::{create_disk_index, ANNDiskIndex},
    model::{
        vertex::{DIM_104, DIM_128, DIM_256, DIM_512}, DiskIndexBuildParameters, IndexConfiguration, IndexWriteParametersBuilder
    },
    storage::DiskIndexStorage,
    utils::{load_metadata_from_file, round_up, Timer},
};

use vector::{FullPrecisionDistance, Metric};

/// The main function to build a disk index
#[allow(clippy::too_many_arguments)]
fn build_disk_index<T>(
    metric: Metric,
    data_path: &str,
    r: u32,
    l: u32,
    index_path_prefix: &str,
    num_threads: u32,
    search_ram_limit_gb: f64,
    index_build_ram_limit_gb: f64,
    num_pq_chunks: usize,
    use_opq: bool,
) -> ANNResult<()>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_512]: FullPrecisionDistance<T, DIM_512>,
{
    let disk_index_build_parameters =
        DiskIndexBuildParameters::new(search_ram_limit_gb, index_build_ram_limit_gb)?;

    let index_write_parameters = IndexWriteParametersBuilder::new(l, r)
        .with_saturate_graph(true)
        .with_num_threads(num_threads)
        .build();

    let (data_num, data_dim) = load_metadata_from_file(data_path)?;

    let config = IndexConfiguration::new(
        metric,
        data_dim,
        round_up(data_dim as u64, 8_u64) as usize,
        data_num,
        num_pq_chunks > 0,
        num_pq_chunks,
        use_opq,
        0,
        1f32,
        index_write_parameters,
    );
    let storage = DiskIndexStorage::new(data_path.to_string(), index_path_prefix.to_string())?;
    let mut index = create_disk_index::<T>(Some(disk_index_build_parameters), config, storage)?;

    let timer = Timer::new();

    index.build("")?;

    let diff = timer.elapsed();
    println!("Indexing time: {}", diff.as_secs_f64());

    Ok(())
}

#[cfg(test)]
mod tests {
    use diskann::common::ANNResult;
    use vector::Metric;
    use crate::disk_ann_store::build_disk_index;

    const VECTOR_FILE: &str = "test/input/embeddings.json";
    const INPUT_VECTOR_FILE: &str = "test/input/saved_vector_file.data";
    //const INPUT_VECTOR_FILE: &str = "test/input/siftsmall_learn.bin";
    const SAVED_VECTOR_FILE: &str = "test/output/saved_vector_file";

    #[test]
    fn test_disk_output() -> ANNResult<()> {
        let err = build_disk_index::<f32>(
            Metric::L2,
            INPUT_VECTOR_FILE,
            11,
            61,
            SAVED_VECTOR_FILE,
            1,
            1.0,
            1.0,
            0,
            false,
        );

        match err {
            Ok(_) => {
                println!("Index build completed successfully");
                Ok(())
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                Err(err)
            }
        }
    }
}
*/