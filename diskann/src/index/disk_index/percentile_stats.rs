#[derive(Debug, Default)]
pub struct QueryStats {
    pub total_us: f32, // total time to process query in micros
    pub io_us: f32,    // total time spent in IO
    pub cpu_us: f32,   // total time spent in CPU

    n_4k: u32,         // # of 4kB reads
    n_8k: u32,         // # of 8kB reads
    n_12k: u32,        // # of 12kB reads
    pub n_ios: u32,        // total # of IOs issued
    read_size: u32,    // total # of bytes read
    n_cmps_saved: u32, // # cmps saved
    n_cmps: u32,       // # cmps
    n_cache_hits: u32, // # cache_hits
    n_hops: u32,       // # search hops
}

#[inline]
pub fn get_percentile_stats<T: Default>(
    stats: &[QueryStats],
    percentile: f32,
    member_fn: impl Fn(&QueryStats) -> T,
) -> T {
    let len = stats.len();
    let mut vals = vec![T::default(); len];

    for i in 0..len {
        vals[i] = member_fn(&stats[i]);
    }

    vals.sort();
    vals[percentile * len]
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
