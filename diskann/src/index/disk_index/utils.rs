use std::collections::BTreeSet;
use std::io::BufReader;
use std::os::windows::fs::MetadataExt;
use std::{fs::File, io::Read};

use crate::common::{ANNError, ANNResult};

pub fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

pub fn load_aligned_bin_impl(
    reader: impl Read,
    actual_file_size: u64,
    data: &mut AlignedBoxWithSlice,
    npts: &mut usize,
    dim: &mut usize,
    rounded_dim: &mut usize,
) -> ANNResult<()> {
    let tsize = mem::sizeof::<T>;
    *npts = reader.read_i32::<NativeEndian>()? as usize;
    *dim = reader.read_i32::<NativeEndian>()? as usize;
    *rounded_dim = round_up(*dim, 8);
    let npts = *npts;
    let dim = *dim;
    let rounded_dim = *rounded_dim;

    let expected_actual_file_size = (npts * dim * tsize + 2 * mem::size_of::<u32>()) as u64;
    if actual_file_size != expected_actual_file_size {
        return Err(ANNError::FileSizeMismatch {
            message: "Error. File size mismatch".to_string(),
            actual_size,
            expected_actual_file_size,
        });
    }

    //let allocSize = npts * rounded_dim * mem::sizeof::<T>();
    let alloc_size = npts * rounded_dim;
    // allocating aligned memory of
    *data = AlignedBoxWithSlice::<T>::new(alloc_size, 8 * tsize);
    let data = data.as_mut_slice();

    // done. Copying data to mem_aligned buffer...
    for i in 0..npts {
        let buf = data[i * rounded_dim..i * rounded_dim + dim].as_mut_ptr() as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(buf, dim * tsize) }
        reader.read(buf)?;
    }
}

pub fn load_aligned_bin(
    bin_file: &str,
    data: &mut AlignedBoxWithSlice,
    npts: &mut usize,
    dim: &mut usize,
    rounded_dim: &mut usize,
) -> ANNResult<()> {
    let reader = File::open(bin_file)?;
    let fsize = reader.metadata()?.file_size();
    Self::load_aligned_bin_impl(reader, fsize, data, npts, dim, rounded_dim)
}

pub fn load_truthset(
    bin_file: &str,
    ids: &mut Vec<u32>,
    dists: &mut Vec<f32>,
    npts: &mut usize,
    dim: &mut usize,
) -> ANNResult<()> {
    //let read_blk_size = 64 * 1024 * 1024;
    //cached_ifstream reader(bin_file, read_blk_size);
    let file = File::open(bin_file)?;
    let actual_file_size = file.metadata()?.file_size();

    let reader = BufReader::new(file);
    // Reading truthset file `bin_file` ...

    *npts = reader.read_i32::<NativeEndian>()? as usize;
    *dim = reader.read_i32::<NativeEndian>()? as usize;

    let npts = *npts;
    let dim = *dim;

    // 1 means truthset has ids and distances, 2 means
    let truthset_type = -1;

    // only ids, -1 is error
    let expected_file_size_with_dists = 2 * npts * dim * 4 + 2 * 4;
    if (actual_file_size == expected_file_size_with_dists) {
        truthset_type = 1;
    }

    let expected_file_size_just_ids = npts * dim * 4 + 2 * 4;
    if (actual_file_size == expected_file_size_just_ids) {
        truthset_type = 2;
    }

    if (truthset_type == -1) {
        return Err(ANNError::FileSizeMismatch {
            message: r#"Error. File size mismatch. File should have bin format, with "
                  "npts followed by ngt followed by npts*ngt ids and optionally "
                  "followed by npts*ngt distance values; actual size: "#
                .to_string(),
            actual_size: expected_file_size_with_dists,
            expected_actual_file_size: expected_file_size_just_ids,
        });
    }

    *ids = vec![0; npts * dim];
    let buf = unsafe {
        unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, npts * dim * 4) }
    };

    reader.read(buf);
    if (truthset_type == 1) {
        *dists = vec![0.0; npts * dim];
        let buf = unsafe {
            unsafe {
                std::slice::from_raw_parts_mut(
                    dists.as_mut_ptr() as *mut u8,
                    npts * dim * std::mem::size_of::<f32>(),
                )
            }
        };
        reader.read(buf);
    }
}

pub fn calculate_recall(
    num_queries: u32,
    gold_std: &Vec<u32>,
    gs_dist: &Vec<f32>,
    dim_gs: u32,
    our_results: &Vec<u32>,
    dim_or: u32,
    recall_at: u32,
) -> f64 {
    let dim_gs = dim_gs as usize;
    let dim_or = dim_or as usize;
    let num_queries = num_queries as usize;
    let recall_at = recall_at as usize;
    let mut total_recall = 0;

    for i in 0..num_queries {
        let mut tie_breaker = recall_at;
        if !gs_dist.is_empty() {
            tie_breaker = recall_at - 1;
            let gt_dist_vec = &gs_dist[dim_gs * i..];
            while tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1] {
                tie_breaker += 1;
            }
        }

        let gt_vec = &gold_std[dim_gs * i..dim_gs * i + tie_breaker];
        let res_vec = &our_results[dim_or * i..dim_gs * i + recall_at];

        let mut gt: BTreeSet<u32> = gt_vec.iter().collect();

        // change to recall_at for recall k@k
        let mut res: BTreeSet<u32> = res_vec.iter().collect();

        // or dim_or for k@dim_or
        let mut cur_recall = 0;
        for v in &gt {
            if res.contains(v) {
                cur_recall += 1;
            }
        }

        total_recall += cur_recall;
    }

    (total_recall / num_queries) as f64 * (100.0f64 / recall_at as f64)
}
