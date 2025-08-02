#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct QueryStats {
    pub n_cache_hits: u32,
    pub n_hops: u32,
    pub n_4k: u32,
    pub n_ios: u32,
    pub io_us: f32,
    pub n_cmps: u32,
    pub cpu_us: f32,
    pub total_us: f32,
}

pub fn get_percentile_stats<T: Default + Clone + PartialOrd + std::marker::Copy>(
    data: &[T],
    len: usize,
    percentile: f32,
) -> T {
    if data.is_empty() {
        return T::default();
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = ((percentile * len as f32) as usize).min(len - 1);
    sorted_data[index]
}

#[inline]
pub fn get_mean_stats<T>(stats: &[QueryStats], member_fn: impl Fn(&QueryStats) -> T) -> f64
where
    T: Into<f64>,
{
    let mut avg = 0.0;
    for e in stats {
        avg += member_fn(e).into();
    }

    return avg / stats.len() as f64;
}
