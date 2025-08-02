use byteorder::{NativeEndian, ReadBytesExt};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem;

use crate::common::ANNResult;

pub fn load_aligned_bin_impl<T: Default + Clone>(
    reader: &mut impl Read,
    _file_size: u64,
    data: &mut Vec<T>,
    npts: &mut usize,
    dim: &mut usize,
    rounded_dim: usize,
) -> ANNResult<()> {
    let tsize = mem::size_of::<T>();
    *npts = reader.read_i32::<NativeEndian>()? as usize;
    *dim = reader.read_i32::<NativeEndian>()? as usize;

    data.resize(*npts * rounded_dim, T::default());

    for i in 0..*npts {
        let buf = data[i * rounded_dim..i * rounded_dim + *dim].as_mut_ptr() as *mut u8;
        unsafe {
            let _ = std::slice::from_raw_parts_mut(buf, *dim * tsize);
        };
        reader.read_exact(unsafe { std::slice::from_raw_parts_mut(buf, *dim * tsize) })?;
    }
    Ok(())
}

pub fn load_aligned_bin<T: Default + Clone>(fname: &str) -> ANNResult<(Vec<T>, usize, usize)> {
    let mut file = File::open(fname)?;
    let fsize = file.metadata()?.len();
    let mut data = Vec::new();
    let mut npts = 0;
    let mut dim = 0;
    let rounded_dim = 0;

    load_aligned_bin_impl(
        &mut file,
        fsize,
        &mut data,
        &mut npts,
        &mut dim,
        rounded_dim,
    )?;
    Ok((data, npts, dim))
}

pub fn load_truthset(
    truthset_file: &str,
    npts: usize,
    dim: usize,
    recall_at: usize,
    truthset_type: i32,
) -> ANNResult<(Vec<u32>, Vec<f32>)> {
    let file = File::open(truthset_file)?;
    let actual_file_size = file.metadata()?.len();
    let expected_file_size_with_dists = (npts * dim * 4 + 8) as u64;
    let _expected_file_size_just_ids = (npts * recall_at * 4 + 8) as u64;

    if actual_file_size == expected_file_size_with_dists {
        // Truthset has distances
        let mut reader = BufReader::new(file);
        let npts_truth = reader.read_i32::<NativeEndian>()? as usize;
        let dim_truth = reader.read_i32::<NativeEndian>()? as usize;

        let mut gold_std = vec![0u32; npts_truth * dim_truth];
        let _dists = vec![0.0f32; npts_truth * dim_truth];

        let buf = unsafe {
            std::slice::from_raw_parts_mut(
                gold_std.as_mut_ptr() as *mut u8,
                npts_truth * dim_truth * 4,
            )
        };
        reader.read_exact(buf)?;
    }

    if truthset_type == 1 {
        // Truthset has distances - create a new file reader since the previous one was consumed
        let file = File::open(truthset_file)?;
        let mut reader = BufReader::new(file);
        let npts_truth = reader.read_i32::<NativeEndian>()? as usize;
        let dim_truth = reader.read_i32::<NativeEndian>()? as usize;

        let mut gold_std = vec![0u32; npts_truth * dim_truth];
        let _dists = vec![0.0f32; npts_truth * dim_truth];

        let buf = unsafe {
            std::slice::from_raw_parts_mut(
                gold_std.as_mut_ptr() as *mut u8,
                npts_truth * dim_truth * 4,
            )
        };
        reader.read_exact(buf)?;
    }

    Ok((vec![], vec![]))
}

pub fn calculate_recall(
    gold_std: &[u32],
    our_results: &[u32],
    npts: usize,
    dim_gs: usize,
    dim_or: usize,
    recall_at: usize,
    tie_breaker: usize,
) -> f32 {
    let mut total_recall = 0.0;

    for i in 0..npts {
        let gt_vec = &gold_std[dim_gs * i..dim_gs * i + tie_breaker];
        let res_vec = &our_results[dim_or * i..dim_gs * i + recall_at];

        let gt: BTreeSet<u32> = gt_vec.iter().cloned().collect();
        let res: BTreeSet<u32> = res_vec.iter().cloned().collect();

        let intersection = gt.intersection(&res).count();
        total_recall += intersection as f32 / recall_at as f32;
    }

    total_recall / npts as f32
}
