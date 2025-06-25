#![allow(dead_code)]

mod diskann_store;

/// 512 vertex dimension
pub const DIM_512: usize = 512;

use std::{
    fs::File,
    io::{BufReader, Write},
};

use diskann::{
    common::{ANNError, ANNResult},
    index::{ANNInmemIndex, create_inmem_index},
    model::{
        IndexConfiguration,
        configuration::index_write_parameters::IndexWriteParametersBuilder,
        vertex::{DIM_104, DIM_128, DIM_256},
    },
    utils::{Timer, file_exists, load_ids_to_delete_from_file, load_metadata_from_file, round_up},
};

use serde::{Deserialize, Serialize};
use vector::{FullPrecisionDistance, Metric};

use crate::diskann_store::DiskANNStore;

// The main function to build an in-memory index
#[allow(clippy::too_many_arguments)]
fn build_and_insert_delete_in_memory_index<T>(
    metric: Metric,
    data_dim: usize,
    data: &Vec<Vec<T>>,
    delta_data: &Vec<Vec<T>>,
    r: u32,
    l: u32,
    alpha: f32,
    save_path: &str,
    num_threads: u32,
    _use_pq_build: bool,
    _num_pq_bytes: usize,
    use_opq: bool,
    delete_path: &str,
) -> ANNResult<Box<dyn ANNInmemIndex<T>>>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_512]: FullPrecisionDistance<T, DIM_512>,
{
    let index_write_parameters = IndexWriteParametersBuilder::new(l, r)
        .with_alpha(alpha)
        .with_saturate_graph(false)
        .with_num_threads(num_threads)
        .build();

    let config = IndexConfiguration::new(
        metric,
        data_dim,
        round_up(data_dim as u64, 8_u64) as usize,
        8388608,
        false,
        0,
        use_opq,
        0,
        2.0f32,
        index_write_parameters,
    );
    let mut index = create_inmem_index::<T>(config)?;

    let timer = Timer::new();

    if !data.is_empty() {
        index.build_vector(data)?;
    }

    let diff = timer.elapsed();

    println!("Initial indexing time: {}", diff.as_secs_f64());

    if !delta_data.is_empty() {
        index.insert_vector(delta_data)?;
    }

    if !delete_path.is_empty() {
        if !file_exists(delete_path) {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Data file for delete {} does not exist.",
                delete_path
            )));
        }

        let (num_points_to_delete, vertex_ids_to_delete) =
            load_ids_to_delete_from_file(delete_path)?;
        index.soft_delete(vertex_ids_to_delete, num_points_to_delete)?;
    }

    if !save_path.is_empty() {
        index.save(save_path)?;
        index.save(&(save_path.to_string() + "2"))?;
    }

    Ok(index)
}

// The main function to build an in-memory index
#[allow(clippy::too_many_arguments)]
fn build_and_insert_delete_in_memory_index_file<T>(
    metric: Metric,
    data_path: &str,
    delta_path: &str,
    r: u32,
    l: u32,
    alpha: f32,
    save_path: &str,
    num_threads: u32,
    _use_pq_build: bool,
    _num_pq_bytes: usize,
    use_opq: bool,
    delete_path: &str,
) -> ANNResult<Box<dyn ANNInmemIndex<T>>>
where
    T: Default + Copy + Sync + Send + Into<f32> + 'static,
    [T; DIM_104]: FullPrecisionDistance<T, DIM_104>,
    [T; DIM_128]: FullPrecisionDistance<T, DIM_128>,
    [T; DIM_256]: FullPrecisionDistance<T, DIM_256>,
    [T; DIM_512]: FullPrecisionDistance<T, DIM_512>,
{
    let index_write_parameters = IndexWriteParametersBuilder::new(l, r)
        .with_alpha(alpha)
        .with_saturate_graph(false)
        .with_num_threads(num_threads)
        .build();

    let (data_num, data_dim) = load_metadata_from_file(data_path)?;

    let config = IndexConfiguration::new(
        metric,
        data_dim,
        round_up(data_dim as u64, 8_u64) as usize,
        data_num,
        false,
        0,
        use_opq,
        0,
        2.0f32,
        index_write_parameters,
    );
    let mut index = create_inmem_index::<T>(config)?;

    let timer = Timer::new();

    //index.build(data_path, data_num)?;

    let diff = timer.elapsed();

    println!("Initial indexing time: {}", diff.as_secs_f64());

    if !delta_path.is_empty() {
        let (delta_data_num, _) = load_metadata_from_file(delta_path)?;
        index.insert(delta_path, delta_data_num)?;
    }

    if !delete_path.is_empty() {
        if !file_exists(delete_path) {
            return Err(ANNError::log_index_error(format!(
                "ERROR: Data file for delete {} does not exist.",
                delete_path
            )));
        }

        let (num_points_to_delete, vertex_ids_to_delete) =
            load_ids_to_delete_from_file(delete_path)?;
        index.soft_delete(vertex_ids_to_delete, num_points_to_delete)?;
    }

    index.save(save_path)?;

    Ok(index)
}

const TEST_DATA_FILE: &str = "DiskANN\\rust\\diskann\\tests\\data\\siftsmall_learn_256pts.fbin";
const TEST_DELETE_DATA_FILE: &str = "DiskANN\\rust\\diskann\\tests\\data\\delete_set_50pts.bin";
const TEST_SAVE_PATH: &str = "DiskANN\\rust\\api\\memory_insert_query\\output\\output";
const VECTOR_FILE: &str = "input\\embeddings.json";

#[derive(Debug, Serialize, Deserialize)]
struct Embeddings {
    filename: String,
    embedding: Vec<f32>,
}

fn main() -> ANNResult<()> {
    println!("{:?}", std::env::current_dir().unwrap());
    let json_file = File::open(VECTOR_FILE)?;
    let json_reader = BufReader::new(json_file);
    let items: Vec<Embeddings> = serde_json::from_reader(json_reader).unwrap();

    let filenames: Vec<String> = items
        .iter()
        .map(|item| {
            let mut filename = item.filename.clone();
            let mut with_ext = if let Some(pos) = filename.rfind('/') {
                filename.split_off(pos + 1)
            } else {
                filename
            };

            if let Some(pos) = with_ext.rfind('.') {
                //with_ext.split_off(pos);
                with_ext.truncate(pos);
                with_ext
            } else {
                with_ext
            }
        })
        .collect();

    /*     for _ in 0..5 {
        filenames.insert(0, "".to_string());
    } */

    let mut _create_points: Vec<Vec<f32>> = items
        .iter()
        .take(5)
        .map(|item| item.embedding.clone())
        .collect();
    let insert_points: Vec<Vec<f32>> = items
        .into_iter()
        //.skip(5)
        .map(|item| item.embedding)
        .collect();
    for point in &insert_points {
        assert_eq!(point.len(), 512);
    }

    //insert_points[0] = insert_points[100].clone();
    //insert_points.truncate(300);
    //insert_points.clear();

    let mut create_points = vec![vec![0f32; 512]; 5];
    for i in 0..5 {
        for j in 0..512 {
            create_points[i][j] = (i + j) as f32;
        }
    }

    create_points.clear();

    let _data_type = String::from("f32");
    let _data_path = String::from(TEST_DATA_FILE);
    let mut _delta_path = String::from(TEST_DATA_FILE);
    //let delta_path = String::new();
    let _index_path_prefix = String::from(TEST_SAVE_PATH); // save_path
    //let index_path_prefix = String::new();
    //let mut delete_path = String::from(TEST_DELETE_DATA_FILE);
    let _delete_path = String::new();

    let num_threads = 1u32;
    let max_degree = 64u32;
    let search_list_size = 100u32;

    // 控制图构建的"贪婪"程度	1.2，越大越精确，越慢
    let alpha = 1.2f32;

    // Number of PQ bytes to build the index; 0 for full precision build (default: 0)
    let _build_pq_bytes = 0u32;
    let _use_pq_build = false;

    // Set true for OPQ compression while using PQ distance comparisons for building the index, and false for PQ compression (default: false)
    let _use_opq = false;
    let metric = Metric::L2;
    /*
    let index = build_and_insert_delete_in_memory_index_file::<f32>(
        metric,
        &data_path,
        &delta_path,
        r,
        l,
        alpha,
        &index_path_prefix,
        num_threads,
        _use_pq_build,
        build_pq_bytes as usize,
        use_opq,
        &delete_path,
    )?;
    */

    let mut store: DiskANNStore<f32, 512> =
        DiskANNStore::new(metric, max_degree, search_list_size, alpha, num_threads).unwrap();
    //store.init_data(&insert_points).unwrap();
    store.insert_data(&insert_points).unwrap();

    let mut file = File::create("./relations.html").unwrap();
    file.write_fmt(format_args!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Image KNN Evaluation</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .query-row {{ margin-bottom: 30px; }}
        .image-box {{ display: inline-block; text-align: center; margin: 5px; }}
        img {{ width: 100px; height: 100px; object-fit: cover; border: 1px solid #ccc; }}
    </style>
</head>
<body>
<h1>Image KNN Evaluation</h1>
"#
    ))
    .unwrap();

    // k_value: return count 5
    // l_value: search width 50
    for i in 0..insert_points.len() {
        for k_value in 11..12 {
            for l_value in k_value + 50..k_value + 51 {
                let mut indices = vec![0; k_value];
                let mut distances = vec![0f32; k_value];
                let a = store.query(
                    &insert_points[i],
                    k_value,
                    l_value as u32,
                    &mut indices,
                    &mut distances,
                );

                file.write_fmt(format_args!(
                    r#"
<div class="query-row">
<h2>Query: {}</h2>
<div class="image-box" style="background:yellow">
<img src="cleaned_images/{}.jpg" alt="Query Image">
<div>{}</div>
</div>
<div class="image-box"> ⇨ </div>
"#,
                    filenames[i + create_points.len()],
                    filenames[i + create_points.len()],
                    filenames[i + create_points.len()],
                ))
                .unwrap();

                for (i, &j) in indices.iter().enumerate() {
                    if distances[i] > 0.1 {
                        continue;
                    }

                    file.write_fmt(format_args!(
                        r#"
                <div class="image-box">
                <img src="cleaned_images/{}.jpg" alt="{}">
                <div>{} @ {}</div>
                </div>
                "#,
                        filenames[j as usize + create_points.len()],
                        filenames[j as usize + create_points.len()],
                        filenames[j as usize + create_points.len()],
                        distances[i],
                    ))
                    .unwrap();
                }

                file.write_fmt(format_args!("</div><hr>")).unwrap();

                println!("开始: ({i},{k_value},{l_value}) => {a:?}: {indices:?}");
                let rel: Vec<String> = indices
                    .iter()
                    .map(|&i| filenames[i as usize].clone())
                    .collect();
                println!("{} => {:?}", filenames[i + create_points.len()], rel);
            }
        }
    }

    file.write_fmt(format_args!("</body></html>")).unwrap();

    Ok(())
}
