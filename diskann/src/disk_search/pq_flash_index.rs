use std::any::Any;

use vector::Metric;

use crate::{
    disk_search::aligned_file_reader::AlignedFileReader,
    model::ScratchStoreManager,
    utils::{div_round_up, is_floating_point},
};

pub struct PQFlashIndex<T, LabelT> {
    _max_node_len: u64,
    _nnodes_per_sector: u64, // 0 for multi-sector nodes, >0 for multi-node sectors
    _max_degree: u64,

    // Data used for searching with re-order vectors
    _ndims_reorder_vecs: u64,
    _reorder_data_start_sector: u64,
    _nvecs_per_sector: u64,

    metric: diskann::Metric,

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    _max_base_norm: f32,

    // data info
    _num_points: u64,
    _num_frozen_points: u64,
    _frozen_location: u64,
    _data_dim: u64,
    _aligned_dim: u64,
    _disk_bytes_per_point: u64, // Number of bytes

    _disk_index_file: String,
    _node_visit_counter: Vec<(u32, u32)>,

    // PQ data
    data: *mut u8,
    _n_chunks: u64,
    _pq_table: FixedChunkPQTable,

    // distance comparator
    _dist_cmp: std::shared_ptr<Distance<T>>,
    _dist_cmp_float: std::shared_ptr<Distance<f32>>,

    // for very large datasets: we use PQ even for the disk resident index
    _use_disk_index_pq: bool,
    _disk_pq_n_chunks: u64,
    _disk_pq_table: FixedChunkPQTable,

    // medoid/start info
    _medoids: *mut u32,
    _num_medoids: usize,
    _centroid_data: *mut f32,

    // nhood_cache
    _nhood_cache_buf: *mut u32,
    _nhood_cache: tsl::robin_map<u32, (u32, *mut u32)>,

    // coord_cache
    _coord_cache_buf: *mut T,
    _coord_cache: tsl::robin_map<u32, *mut T>,

    // thread-specific scratch
    _thread_data: ConcurrentQueue<SSDThreadData<T>>,
    _max_nthreads: u64,
    _load_flag: bool,
    _count_visited_nodes: bool,
    _reorder_data_exists: bool,
    _reoreder_data_offset: u64,

    // filter support
    _pts_to_label_offsets: *mut u32,
    _pts_to_label_counts: *mut u32,
    _pts_to_labels: *mut LabelT,
    _filter_to_medoid_ids: std::unordered_map<LabelT, Vec<u32>>,
    _use_universal_label: bool,
    _universal_filter_label: LabelT,
    _dummy_pts: tsl::robin_set<u32>,
    _has_dummy_pts: tsl::robin_set<u32>,
    _dummy_to_real_map: tsl::robin_map<u32, u32>,
    _real_to_dummy_map: tsl::robin_map<u32, Vec<u32>>,
    _label_map: std::unordered_map<String, LabelT>,
}

impl<T, LabelT> PQFlashIndex<T, LabelT>
where
    T: Any,
{
    pub fn new(file_reader: Arc<dyn AlignedFileReader>, metric: diskann::Metric) -> Self {
        let mut metric_to_invoke = metric;
        if metric == Metric::L2 || m == Metric::Cosine {
            let float_point = is_floating_point::<T>();
            if float_point {

                // Since data is floating point, we assume that it has been appropriately pre-processed
                // (normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we
                // shall invoke an l2 distance function.
                //metric_to_invoke = diskann::Metric::L2;
            } else {
                // WARNING: Cannot normalize integral data types This may result in erroneous results or poor recall
                // Consider using L2 distance with integral data types
            }
        }

        let dist_cmp = diskann::get_distance_function::<T>(metric_to_invoke);
        let dist_cmp_float = diskann::get_distance_function::<f32>(metric_to_invoke);

        Self {
            _max_node_len: 0,
            _nnodes_per_sector: 0,
            _max_degree: 0,
            _ndims_reorder_vecs: 0,
            _reorder_data_start_sector: 0,
            _nvecs_per_sector: 0,
            metric: diskann::Metric::L2,
            _max_base_norm: 0.0,
            _num_points: 0,
            _num_frozen_points: 0,
            _frozen_location: 0,
            _data_dim: 0,
            _aligned_dim: 0,
            _disk_bytes_per_point: 0,
            _disk_index_file: String::new(),
            _node_visit_counter: Vec::new(),
            data: std::ptr::null_mut(),
            _n_chunks: 0,
            _pq_table: FixedChunkPQTable::default(),
            _dist_cmp: std::shared_ptr::default(),
            _dist_cmp_float: std::shared_ptr::default(),
            _use_disk_index_pq: false,
            _disk_pq_n_chunks: 0,
            _disk_pq_table: FixedChunkPQTable::default(),
            _medoids: std::ptr::null_mut(),
            _num_medoids: 0,
            _centroid_data: std::ptr::null_mut(),
            _nhood_cache_buf: std::ptr::null_mut(),
            _nhood_cache: tsl::robin_map::new(),
            _coord_cache_buf: std::ptr::null_mut(),
            _coord_cache: tsl::robin_map::new(),
            _thread_data: ConcurrentQueue::new(),
            _max_nthreads: 0,
            _load_flag: false,
            _count_visited_nodes: false,
            _reorder_data_exists: false,
            _reoreder_data_offset: 0,
            _pts_to_label_offsets: std::ptr::null_mut(),
            _pts_to_label_counts: std::ptr::null_mut(),
            _pts_to_labels: std::ptr::null_mut(),
            _filter_to_medoid_ids: std::unordered_map::new(),
            _use_universal_label: false,
            _universal_filter_label: LabelT::default(),
            _dummy_pts: tsl::robin_set::new(),
            _has_dummy_pts: tsl::robin_set::new(),
            _dummy_to_real_map: tsl::robin_map::new(),
            _real_to_dummy_map: tsl::robin_map::new(),
            _label_map: std::unordered_map::new(),
        }
    }

    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                drop(Box::from_raw(self.data));
            }
        }

        if !self._centroid_data.is_null() {
            unsafe {
                diskann::aligned_free(self._centroid_data);
            }
        }

        // delete backing bufs for nhood and coord cache
        if !self._nhood_cache_buf.is_null() {
            unsafe {
                drop(Box::from_raw(self._nhood_cache_buf));
            }
            diskann::aligned_free(self._coord_cache_buf);
        }

        if self._load_flag {
            println!("Clearing scratch");
            let manager = ScratchStoreManager::<SSDThreadData<T>>::new(self._thread_data);
            manager.destroy();
            self.reader.deregister_all_threads();
            self.reader.close();
        }

        if !self._pts_to_label_offsets.is_null() {
            unsafe {
                drop(Box::from_raw(self._pts_to_label_offsets));
            }
        }

        if !self._pts_to_label_counts.is_null() {
            unsafe {
                drop(Box::from_raw(self._pts_to_label_counts));
            }
        }

        if !self._pts_to_labels.is_null() {
            unsafe {
                drop(Box::from_raw(self._pts_to_labels));
            }
        }

        if !self._medoids.is_null() {
            unsafe {
                drop(Box::from_raw(self._medoids));
            }
        }
    }

    pub fn get_node_sector(&self, node_id: u64) -> u64 {
        1 + if self._nnodes_per_sector > 0 {
            node_id / self._nnodes_per_sector
        } else {
            node_id * div_round_up(_max_node_len, defaults::SECTOR_LEN)
        }
    }

    pub fn offset_to_node(&self, sector_buf: *mut u8, node_id: u64) -> *mut u8 {
        unsafe {
            sector_buf.add(if self._nnodes_per_sector == 0 {
                0
            } else {
                (node_id % self._nnodes_per_sector) * self._max_node_len as usize
            })
        }
    }

    pub fn offset_to_node_nhood(&self, node_buf: *mut u8) -> *mut u32 {
        unsafe { (node_buf.add(self._disk_bytes_per_point as usize) as *mut u32) }
    }

    pub fn offset_to_node_coords(&self, node_buf: *mut u8) -> *mut T {
        node_buf as *mut T
    }

    /// TODO： recheck
    pub fn setup_thread_data(&mut self, nthreads: u64, visited_reserve: u64) {
        println!(
            "Setting up thread-specific contexts for nthreads: {}",
            nthreads
        );

        let thread_data = Arc::new(Mutex::new(Vec::new()));

        (0..nthreads).for_each(|_| {
            let thread_data = Arc::clone(&thread_data);
            let reader = self.reader.clone();

            thread::spawn(move || {
                let mut data = SSDThreadData::new(self._aligned_dim, visited_reserve);
                reader.register_thread();
                data.ctx = reader.get_ctx();

                let mut guard = thread_data.lock().unwrap();
                guard.push(data);
            });
        });

        self._thread_data = Arc::try_unwrap(thread_data).unwrap().into_inner().unwrap();
        self._load_flag = true;
    }

    pub fn read_nodes(
        &self,
        node_ids: &[u32],
        coord_buffers: &mut [*mut T],
        nbr_buffers: &mut [(&mut u32, *mut u32)],
    ) -> Vec<bool> {
        let mut read_reqs = Vec::new();
        let mut retval = vec![true; node_ids.len()];

        let num_sectors = if self._nnodes_per_sector > 0 {
            1
        } else {
            div_round_up(_max_node_len, defaults::SECTOR_LEN)
        };

        let buf_size = node_ids.len() * num_sectors * defaults::SECTOR_LEN;
        let mut buf = unsafe { diskann::aligned_alloc(defaults::SECTOR_LEN, buf_size) };

        // create read requests
        for (i, &node_id) in node_ids.iter().enumerate() {
            let read = AlignedRead {
                len: num_sectors * defaults::SECTOR_LEN,
                buf: unsafe { buf.add(i * num_sectors * defaults::SECTOR_LEN) },
                offset: self.get_node_sector(node_id as u64) * defaults::SECTOR_LEN,
            };
            read_reqs.push(read);
        }

        // borrow thread data and issue reads
        let manager = ScratchStoreManager::new(&self._thread_data)?;
        let this_thread_data = manager.scratch_space();
        let ctx = &this_thread_data.ctx;
        self.reader.read(&read_reqs, ctx);

        // copy reads into buffers
        for i in 0..read_reqs.len() {
            let node_buf = self.offset_to_node(read_reqs[i].buf, node_ids[i] as u64);

            if !coord_buffers[i].is_null() {
                let node_coords = self.offset_to_node_coords(node_buf);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        node_coords,
                        coord_buffers[i],
                        self._disk_bytes_per_point,
                    );
                }
            }

            if !nbr_buffers[i].1.is_null() {
                let node_nhood = self.offset_to_node_nhood(node_buf);
                let num_nbrs = unsafe { *node_nhood };
                *nbr_buffers[i].0 = num_nbrs;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        node_nhood.add(1),
                        nbr_buffers[i].1,
                        num_nbrs as usize,
                    );
                }
            }
        }

        unsafe {
            diskann::aligned_free(buf);
        }

        retval
    }

    pub fn load_cache_list(&mut self, node_list: &[u32]) {
        println!("Loading the cache list into memory..");
        let num_cached_nodes = node_list.len();

        // Allocate space for neighborhood cache
        self._nhood_cache_buf = unsafe {
            let size = num_cached_nodes * (self._max_degree + 1);
            let layout = std::alloc::Layout::array::<u32>(size).unwrap();
            std::alloc::alloc_zeroed(layout) as *mut u32
        };

        // Allocate space for coordinate cache
        let coord_cache_buf_len = num_cached_nodes * self._aligned_dim;
        unsafe {
            diskann::alloc_aligned(
                &mut self._coord_cache_buf as *mut *mut T as *mut *mut std::ffi::c_void,
                coord_cache_buf_len * std::mem::size_of::<T>(),
                8 * std::mem::size_of::<T>(),
            );
            std::ptr::write_bytes(
                self._coord_cache_buf,
                0,
                coord_cache_buf_len * std::mem::size_of::<T>(),
            );
        }

        const BLOCK_SIZE: usize = 8;
        let num_blocks = (num_cached_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for block in 0..num_blocks {
            let start_idx = block * BLOCK_SIZE;
            let end_idx = std::cmp::min(num_cached_nodes, (block + 1) * BLOCK_SIZE);

            // Prepare buffers for reading
            let mut nodes_to_read = Vec::new();
            let mut coord_buffers = Vec::new();
            let mut nbr_buffers = Vec::new();
            for node_idx in start_idx..end_idx {
                nodes_to_read.push(node_list[node_idx]);
                unsafe {
                    coord_buffers.push(self._coord_cache_buf.add(node_idx * self._aligned_dim));
                    nbr_buffers.push((
                        0,
                        self._nhood_cache_buf.add(node_idx * (self._max_degree + 1)),
                    ));
                }
            }

            // Issue the reads
            let read_status = self.read_nodes(&nodes_to_read, &mut coord_buffers, &mut nbr_buffers);

            // Insert into cache
            for i in 0..read_status.len() {
                if read_status[i] {
                    unsafe {
                        self._coord_cache.insert(nodes_to_read[i], coord_buffers[i]);
                        self._nhood_cache
                            .insert(nodes_to_read[i], (nbr_buffers[i].0, nbr_buffers[i].1));
                    }
                }
            }
        }
        println!("..done.");
    }

    pub fn generate_cache_list_from_sample_queries(
        &mut self,
        sample_bin: &str,
        l_search: u64,
        beamwidth: u64,
        num_nodes_to_cache: u64,
        nthreads: u32,
        node_list: &mut Vec<u32>,
    ) {
        if num_nodes_to_cache >= self._num_points {
            node_list.resize(self._num_points as usize, 0);
            for i in 0..self._num_points {
                node_list[i as usize] = i as u32;
            }
            return;
        }

        self._count_visited_nodes = true;
        self._node_visit_counter.clear();
        self._node_visit_counter
            .resize(self._num_points as usize, (0, 0));
        for i in 0..self._node_visit_counter.len() {
            self._node_visit_counter[i] = (i as u32, 0);
        }

        let (samples, sample_num, sample_dim, sample_aligned_dim) = {
            if file_exists(sample_bin) {
                diskann::load_aligned_bin::<T>(sample_bin)
            } else {
                eprintln!("Sample bin file not found. Not generating cache.");
                return;
            }
        };

        let mut tmp_result_ids_64 = vec![0; sample_num as usize];
        let mut tmp_result_dists = vec![0.0; sample_num as usize];
        let mut filtered_search = false;
        let mut random_query_filters = vec![LabelT::default(); sample_num as usize];

        if !self._filter_to_medoid_ids.is_empty() {
            filtered_search = true;
            self.generate_random_labels(&mut random_query_filters, sample_num as u32, nthreads);
        }

        (0..sample_num as i64).into_par_iter().for_each(|i| {
            let label_for_search = random_query_filters[i as usize];
            self.cached_beam_search(
                &samples[(i as usize * sample_aligned_dim as usize)..],
                1,
                l_search,
                &mut tmp_result_ids_64[i as usize],
                &mut tmp_result_dists[i as usize],
                beamwidth,
                filtered_search,
                label_for_search,
                false,
            );
        });

        self._node_visit_counter.sort_by(|a, b| b.1.cmp(&a.1));
        node_list.clear();
        node_list.shrink_to_fit();
        let num_nodes_to_cache =
            std::cmp::min(num_nodes_to_cache, self._node_visit_counter.len() as u64);
        node_list.reserve(num_nodes_to_cache as usize);
        for i in 0..num_nodes_to_cache {
            node_list.push(self._node_visit_counter[i as usize].0);
        }
        self._count_visited_nodes = false;
        unsafe {
            diskann::aligned_free(samples.as_ptr() as *mut std::ffi::c_void);
        }
    }

    pub fn cache_bfs_levels(
        &mut self,
        num_nodes_to_cache: u64,
        node_list: &mut Vec<u32>,
        shuffle: bool,
    ) {
        let mut rng = rand::thread_rng();
        let mut node_set = tsl::robin_set::RobinSet::new();

        let tenp_nodes = (self._num_points as f64 * 0.1).round() as u64;
        let num_nodes_to_cache = if num_nodes_to_cache > tenp_nodes {
            println!(
                "Reducing nodes to cache from: {} to: {}(10 percent of total nodes:{})",
                num_nodes_to_cache, tenp_nodes, self._num_points
            );
            if tenp_nodes == 0 {
                1
            } else {
                tenp_nodes
            }
        } else {
            num_nodes_to_cache
        };
        println!("Caching {}...", num_nodes_to_cache);

        let mut cur_level = tsl::robin_set::RobinSet::new();
        let mut prev_level = tsl::robin_set::RobinSet::new();

        for miter in 0..self._num_medoids {
            if cur_level.len() >= num_nodes_to_cache as usize {
                break;
            }
            cur_level.insert(self._medoids[miter as usize]);
        }

        if !self._filter_to_medoid_ids.is_empty() && cur_level.len() < num_nodes_to_cache as usize {
            for x in self._filter_to_medoid_ids.values() {
                for &y in x {
                    cur_level.insert(y);
                    if cur_level.len() == num_nodes_to_cache as usize {
                        break;
                    }
                }
                if cur_level.len() == num_nodes_to_cache as usize {
                    break;
                }
            }
        }

        let mut lvl = 1;
        let mut prev_node_set_size = 0;
        while node_set.len() + cur_level.len() < num_nodes_to_cache as usize
            && !cur_level.is_empty()
        {
            std::mem::swap(&mut prev_level, &mut cur_level);
            cur_level.clear();

            let mut nodes_to_expand = Vec::new();
            for &id in prev_level.iter() {
                if !node_set.contains(&id) {
                    node_set.insert(id);
                    nodes_to_expand.push(id);
                }
            }

            if shuffle {
                nodes_to_expand.shuffle(&mut rng);
            } else {
                nodes_to_expand.sort();
            }

            print!("Level: {}", lvl);
            std::io::stdout().flush().unwrap();
            let mut finish_flag = false;

            const BLOCK_SIZE: usize = 1024;
            let nblocks = (nodes_to_expand.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for block in 0..nblocks {
                if finish_flag {
                    break;
                }
                print!(".");
                std::io::stdout().flush().unwrap();
                let start = block * BLOCK_SIZE;
                let end = std::cmp::min((block + 1) * BLOCK_SIZE, nodes_to_expand.len());

                let mut nodes_to_read = Vec::new();
                let mut coord_buffers = vec![std::ptr::null_mut(); end - start];
                let mut nbr_buffers = Vec::new();

                for cur_pt in start..end {
                    nodes_to_read.push(nodes_to_expand[cur_pt]);
                    nbr_buffers.push((0, vec![0u32; self._max_degree as usize + 1].as_mut_ptr()));
                }

                let read_status =
                    self.read_nodes(&nodes_to_read, &mut coord_buffers, &mut nbr_buffers);

                for i in 0..read_status.len() {
                    if !read_status[i] {
                        continue;
                    }
                    let nnbrs = nbr_buffers[i].0;
                    let nbrs = nbr_buffers[i].1;

                    for j in 0..nnbrs {
                        if finish_flag {
                            break;
                        }
                        if !node_set.contains(&nbrs[j]) {
                            cur_level.insert(nbrs[j]);
                        }
                        if cur_level.len() + node_set.len() >= num_nodes_to_cache as usize {
                            finish_flag = true;
                        }
                    }
                    unsafe {
                        std::alloc::dealloc(
                            nbr_buffers[i].1 as *mut u8,
                            std::alloc::Layout::array::<u32>(self._max_degree as usize + 1)
                                .unwrap(),
                        );
                    }
                }
            }

            println!(
                ". #nodes: {}, #nodes thus far: {}",
                node_set.len() - prev_node_set_size,
                node_set.len()
            );
            prev_node_set_size = node_set.len();
            lvl += 1;
        }

        assert!(
            node_set.len() + cur_level.len() == num_nodes_to_cache as usize || cur_level.is_empty()
        );

        node_list.clear();
        node_list.reserve(node_set.len() + cur_level.len());
        for &node in node_set.iter() {
            node_list.push(node);
        }
        for &node in cur_level.iter() {
            node_list.push(node);
        }

        println!(
            "Level: {}. #nodes: {}, #nodes thus far: {}",
            lvl,
            node_list.len() - prev_node_set_size,
            node_list.len()
        );
        println!("done");
    }

    pub fn use_medoids_data_as_centroids(&mut self) {
        if !self._centroid_data.is_null() {
            unsafe {
                diskann::aligned_free(self._centroid_data as *mut std::ffi::c_void);
            }
        }
        unsafe {
            diskann::alloc_aligned(
                &mut self._centroid_data as *mut *mut f32 as *mut *mut std::ffi::c_void,
                self._num_medoids * self._aligned_dim * std::mem::size_of::<f32>(),
                32,
            );
            std::ptr::write_bytes(
                self._centroid_data,
                0,
                self._num_medoids * self._aligned_dim * std::mem::size_of::<f32>(),
            );
        }

        println!(
            "Loading centroid data from medoids vector data of {} medoid(s)",
            self._num_medoids
        );

        let mut nodes_to_read = Vec::new();
        let mut medoid_bufs = Vec::new();
        let mut nbr_bufs = Vec::new();

        for cur_m in 0..self._num_medoids {
            nodes_to_read.push(self._medoids[cur_m as usize]);
            medoid_bufs.push(vec![T::default(); self._data_dim as usize].as_mut_ptr());
            nbr_bufs.push((0, std::ptr::null_mut()));
        }

        let read_status = self.read_nodes(&nodes_to_read, &mut medoid_bufs, &mut nbr_bufs);

        for cur_m in 0..self._num_medoids {
            if !read_status[cur_m as usize] {
                panic!("Unable to read a medoid");
            }
            if !self._use_disk_index_pq {
                for i in 0..self._data_dim {
                    unsafe {
                        *self
                            ._centroid_data
                            .add((cur_m * self._aligned_dim + i) as usize) =
                            *medoid_bufs[cur_m as usize].add(i as usize);
                    }
                }
            } else {
                unsafe {
                    self._disk_pq_table.inflate_vector(
                        medoid_bufs[cur_m as usize] as *const u8,
                        self._centroid_data.add(cur_m * self._aligned_dim),
                    );
                }
            }
            unsafe {
                std::alloc::dealloc(
                    medoid_bufs[cur_m as usize] as *mut u8,
                    std::alloc::Layout::array::<T>(self._data_dim as usize).unwrap(),
                );
            }
        }
    }

    pub fn generate_random_labels(&self, labels: &mut Vec<LabelT>, num_labels: u32, nthreads: u32) {
        let mut rng = rand::thread_rng();
        labels.clear();
        labels.resize(num_labels as usize, LabelT::default());

        let num_total_labels = self._pts_to_label_offsets[self._num_points as usize - 1]
            + self._pts_to_label_counts[self._num_points as usize - 1];
        if num_total_labels == 0 {
            eprintln!("No labels found in data. Not sampling random labels");
            panic!("No labels found in data");
        }
        let dist = rand::distributions::Uniform::new(0, num_total_labels - 1);

        (0..num_labels as i64).into_par_iter().for_each(|i| {
            let rnd_loc = dist.sample(&mut rng);
            labels[i as usize] = self._pts_to_labels[rnd_loc as usize];
        });
    }

    pub fn load_label_map(
        &mut self,
        map_reader: &mut dyn std::io::BufRead,
    ) -> std::collections::HashMap<String, LabelT> {
        let mut string_to_int_mp = std::collections::HashMap::new();
        let mut line = String::new();
        while map_reader.read_line(&mut line).unwrap() > 0 {
            let mut parts = line.trim().split('\t');
            if let (Some(label_str), Some(token)) = (parts.next(), parts.next()) {
                let token_as_num = token.parse::<LabelT>().unwrap();
                string_to_int_mp.insert(label_str.to_string(), token_as_num);
            }
            line.clear();
        }
        string_to_int_mp
    }

    pub fn get_converted_label(&self, filter_label: &str) -> LabelT {
        if let Some(label) = self._label_map.get(filter_label) {
            return *label;
        }
        if self._use_universal_label {
            return self._universal_filter_label;
        }
        eprintln!("Unable to find label in the Label Map");
        panic!("Label not found");
    }

    pub fn reset_stream_for_reading(&self, infile: &mut dyn std::io::BufRead) {
        infile.consume(infile.buffer().len());
    }

    pub fn get_label_file_metadata(&self, file_content: &str) -> (u32, u32) {
        let mut num_pts = 0;
        let mut num_total_labels = 0;

        for line in file_content.lines() {
            num_pts += 1;
            num_total_labels += line.split(',').count();
        }

        println!(
            "Labels file metadata: num_points: {}, #total_labels: {}",
            num_pts, num_total_labels
        );
        (num_pts, num_total_labels)
    }

    pub fn point_has_label(&self, point_id: u32, label_id: LabelT) -> bool {
        let start_vec = self._pts_to_label_offsets[point_id as usize];
        let num_lbls = self._pts_to_label_counts[point_id as usize];
        self._pts_to_labels[start_vec as usize..(start_vec + num_lbls) as usize].contains(&label_id)
    }

    pub fn parse_label_file(&mut self, infile: &mut dyn std::io::BufRead) -> usize {
        let mut buffer = String::new();
        infile.read_to_string(&mut buffer).unwrap();

        let (num_pts_in_label_file, num_total_labels) = self.get_label_file_metadata(&buffer);

        self._pts_to_label_offsets = vec![0; num_pts_in_label_file as usize];
        self._pts_to_label_counts = vec![0; num_pts_in_label_file as usize];
        self._pts_to_labels = vec![LabelT::default(); num_total_labels as usize];
        let mut labels_seen_so_far = 0;

        for (line_cnt, line) in buffer.lines().enumerate() {
            self._pts_to_label_offsets[line_cnt] = labels_seen_so_far;
            let mut num_lbls_in_cur_pt = 0;

            for label_str in line.split(',') {
                let token_as_num = label_str.parse::<LabelT>().unwrap();
                self._pts_to_labels[labels_seen_so_far] = token_as_num;
                labels_seen_so_far += 1;
                num_lbls_in_cur_pt += 1;
            }

            self._pts_to_label_counts[line_cnt] = num_lbls_in_cur_pt;
            if num_lbls_in_cur_pt == 0 {
                println!("No label found for point {}", line_cnt);
                panic!("Invalid label file");
            }
        }

        self.reset_stream_for_reading(infile);
        buffer.lines().count()
    }

    pub fn set_universal_label(&mut self, label: LabelT) {
        self._use_universal_label = true;
        self._universal_filter_label = label;
    }

    pub fn load(&mut self, num_threads: u32, index_prefix: &str) -> i32 {
        let pq_table_bin = format!("{}_pq_pivots.bin", index_prefix);
        let pq_compressed_vectors = format!("{}_pq_compressed.bin", index_prefix);
        let disk_index_file = format!("{}_disk.index", index_prefix);
        self.load_from_separate_paths(
            num_threads,
            &disk_index_file,
            &pq_table_bin,
            &pq_compressed_vectors,
        )
    }

    pub fn load_from_separate_paths(
        &mut self,
        num_threads: u32,
        index_filepath: &str,
        pivots_filepath: &str,
        compressed_filepath: &str,
    ) -> i32 {
        let pq_table_bin = pivots_filepath.to_string();
        let pq_compressed_vectors = compressed_filepath.to_string();
        let disk_index_file = index_filepath.to_string();
        let medoids_file = format!("{}_medoids.bin", disk_index_file);
        let centroids_file = format!("{}_centroids.bin", disk_index_file);
        let labels_file = format!("{}_labels.txt", disk_index_file);
        let labels_to_medoids = format!("{}_labels_to_medoids.txt", disk_index_file);
        let dummy_map_file = format!("{}_dummy_map.txt", disk_index_file);
        let labels_map_file = format!("{}_labels_map.txt", disk_index_file);
        let univ_label_file = format!("{}_universal_label.txt", disk_index_file);
        let disk_pq_pivots_path = format!("{}_pq_pivots.bin", disk_index_file);
        let norm_file = format!("{}_max_base_norm.bin", disk_index_file);

        let mut num_pts_in_label_file = 0;
        let (pq_file_num_centroids, pq_file_dim) = {
            #[cfg(feature = "exec_env_ols")]
            {
                diskann::get_bin_metadata(files, &pq_table_bin, METADATA_SIZE)
            }
            #[cfg(not(feature = "exec_env_ols"))]
            {
                diskann::get_bin_metadata(&pq_table_bin, METADATA_SIZE)
            }
        };

        if pq_file_num_centroids != 256 {
            eprintln!("Error. Number of PQ centroids is not 256. Exiting.");
            return -1;
        }

        self._data_dim = pq_file_dim;
        self._disk_bytes_per_point = self._data_dim * std::mem::size_of::<T>();
        self._aligned_dim = ROUND_UP(pq_file_dim, 8);

        let (npts_u64, nchunks_u64) = {
            #[cfg(feature = "exec_env_ols")]
            {
                diskann::load_bin::<u8>(files, &pq_compressed_vectors, &mut self.data)
            }
            #[cfg(not(feature = "exec_env_ols"))]
            {
                diskann::load_bin::<u8>(&pq_compressed_vectors, &mut self.data)
            }
        };

        self._num_points = npts_u64;
        self._n_chunks = nchunks_u64;

        // Load labels if exists

        if std::path::Path::new(&labels_file).exists() {
            let mut infile = std::fs::File::open(&labels_file).unwrap();
            self.parse_label_file(&mut infile, &mut num_pts_in_label_file);
            assert_eq!(num_pts_in_label_file, self._num_points);

            let mut map_reader = std::fs::File::open(&labels_map_file).unwrap();
            self._label_map = self.load_label_map(&mut map_reader);

            if std::path::Path::new(&labels_to_medoids).exists() {
                let mut medoid_stream = std::fs::File::open(&labels_to_medoids).unwrap();
                self.load_labels_to_medoids(&mut medoid_stream);
            }

            if std::path::Path::new(&univ_label_file).exists() {
                let mut univ_reader = std::fs::File::open(&univ_label_file).unwrap();
                self.load_universal_label(&mut univ_reader);
            }

            if std::path::Path::new(&dummy_map_file).exists() {
                let mut dummy_reader = std::fs::File::open(&dummy_map_file).unwrap();
                self.load_dummy_map(&mut dummy_reader);
            }
        }

        // Load PQ centroids

        self._pq_table
            .load_pq_centroid_bin(&pq_table_bin, nchunks_u64);

        // Load disk PQ if exists
        if { std::path::Path::new(&disk_pq_pivots_path).exists() } {
            self._use_disk_index_pq = true;
            self._disk_pq_table
                .load_pq_centroid_bin(&disk_pq_pivots_path, 0);

            self._disk_pq_n_chunks = self._disk_pq_table.get_num_chunks();
            self._disk_bytes_per_point = self._disk_pq_n_chunks * std::mem::size_of::<u8>();
        }

        // Load medoids and centroids

        if std::path::Path::new(&medoids_file).exists() {
            let (medoids, num_medoids, _) = diskann::load_bin::<u32>(&medoids_file);
            self._medoids = medoids;
            self._num_medoids = num_medoids;

            if std::path::Path::new(&centroids_file).exists() {
                let (centroid_data, num_centroids, aligned_dim) =
                    diskann::load_aligned_bin::<f32>(&centroids_file);
                self._centroid_data = centroid_data;
            } else {
                self.use_medoids_data_as_centroids();
            }
        } else {
            self._num_medoids = 1;
            self._medoids = vec![0];
            self.use_medoids_data_as_centroids();
        }

        // Load max base norm if exists and metric is inner product
        if self.metric == diskann::Metric::INNER_PRODUCT {
            if std::path::Path::new(&norm_file).exists() {
                let (norm_val, _, _) = diskann::load_bin::<f32>(&norm_file);
                self._max_base_norm = norm_val[0];
            }
        }

        // Setup thread data and return
        self.setup_thread_data(num_threads);
        self._max_nthreads = num_threads;
        0
    }

    // First overload of cached_beam_search
    pub fn cached_beam_search_1(
        &self,
        query1: &[T],
        k_search: u64,
        l_search: u64,
        indices: &mut [u64],
        distances: Option<&mut [f32]>,
        beam_width: u64,
        use_reorder_data: bool,
        stats: Option<&mut QueryStats>,
    ) {
        self.cached_beam_search_4(
            query1,
            k_search,
            l_search,
            indices,
            distances,
            beam_width,
            false,
            0,
            u32::MAX,
            use_reorder_data,
            stats,
        );
    }

    // Second overload of cached_beam_search
    pub fn cached_beam_search_2(
        &self,
        query1: &[T],
        k_search: u64,
        l_search: u64,
        indices: &mut [u64],
        distances: Option<&mut [f32]>,
        beam_width: u64,
        use_filter: bool,
        filter_label: LabelT,
        use_reorder_data: bool,
        stats: Option<&mut QueryStats>,
    ) {
        self.cached_beam_search_4(
            query1,
            k_search,
            l_search,
            indices,
            distances,
            beam_width,
            use_filter,
            filter_label,
            u32::MAX,
            use_reorder_data,
            stats,
        );
    }

    // Third overload of cached_beam_search
    pub fn cached_beam_search_3(
        &self,
        query1: &[T],
        k_search: u64,
        l_search: u64,
        indices: &mut [u64],
        distances: Option<&mut [f32]>,
        beam_width: u64,
        io_limit: u32,
        use_reorder_data: bool,
        stats: Option<&mut QueryStats>,
    ) {
        let dummy_filter = 0;
        self.cached_beam_search_4(
            query1,
            k_search,
            l_search,
            indices,
            distances,
            beam_width,
            false,
            dummy_filter,
            io_limit,
            use_reorder_data,
            stats,
        );
    }

    // Fourth overload of cached_beam_search (main implementation)
    pub fn cached_beam_search_4(
        &self,
        query1: &[T],
        k_search: u64,
        l_search: u64,
        indices: &mut [u64],
        distances: Option<&mut [f32]>,
        beam_width: u64,
        use_filter: bool,
        filter_label: LabelT,
        io_limit: u32,
        use_reorder_data: bool,
        stats: Option<&mut QueryStats>,
    ) {
        // Calculate number of sectors per node
        let num_sector_per_nodes = (self._max_node_len + SECTOR_LEN - 1) / SECTOR_LEN;
        if beam_width > num_sector_per_nodes as u64 * MAX_N_SECTOR_READS as u64 {
            panic!("Beamwidth can not be higher than MAX_N_SECTOR_READS");
        }

        // Initialize scratch space
        let mut data = self._thread_data.scratch_space();
        let ctx = &mut data.ctx;
        let query_scratch = &mut data.scratch;
        let pq_query_scratch = query_scratch.pq_scratch();

        // Reset query scratch
        query_scratch.reset();

        // Copy query to aligned memory
        let mut query_norm = 0.0;
        let aligned_query_T = query_scratch.aligned_query_T();
        let query_float = pq_query_scratch.aligned_query_float;
        let query_rotated = pq_query_scratch.rotated_query;

        // Normalize query based on metric
        if self.metric == diskann::Metric::INNER_PRODUCT || self.metric == diskann::Metric::COSINE {
            let inherent_dim = if self.metric == diskann::Metric::COSINE {
                self._data_dim
            } else {
                self._data_dim - 1
            };

            for i in 0..inherent_dim {
                aligned_query_T[i] = query1[i];
                query_norm += query1[i] * query1[i];
            }

            if self.metric == diskann::Metric::INNER_PRODUCT {
                aligned_query_T[self._data_dim - 1] = 0.0;
            }

            query_norm = query_norm.sqrt();

            for i in 0..inherent_dim {
                aligned_query_T[i] = (aligned_query_T[i] / query_norm) as T;
            }

            pq_query_scratch.initialize(self._data_dim, aligned_query_T);
        } else {
            for i in 0..self._data_dim {
                aligned_query_T[i] = query1[i];
            }
            pq_query_scratch.initialize(self._data_dim, aligned_query_T);
        }

        // Prefetch data buffer
        let data_buf = query_scratch.coord_scratch;
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                data_buf.as_ptr() as *const i8,
                std::arch::x86_64::_MM_HINT_T1,
            )
        };

        // Initialize sector scratch
        let sector_scratch = query_scratch.sector_scratch;
        let mut sector_scratch_idx = query_scratch.sector_idx;
        let num_sectors_per_node = if self._nnodes_per_sector > 0 {
            1
        } else {
            (self._max_node_len + SECTOR_LEN - 1) / SECTOR_LEN
        };

        // Preprocess query for PQ table
        self._pq_table.preprocess_query(query_rotated);
        let pq_dists = pq_query_scratch.aligned_pqtable_dist_scratch;
        self._pq_table
            .populate_chunk_distances(query_rotated, pq_dists);

        // Initialize distance scratch
        let dist_scratch = pq_query_scratch.aligned_dist_scratch;
        let pq_coord_scratch = pq_query_scratch.aligned_pq_coord_scratch;

        // Lambda to compute distances in PQ space
        let compute_dists = |ids: &[u32], n_ids: u64, dists_out: &mut [f32]| {
            diskann::aggregate_coords(ids, n_ids, self.data, self._n_chunks, pq_coord_scratch);
            diskann::pq_dist_lookup(pq_coord_scratch, n_ids, self._n_chunks, pq_dists, dists_out);
        };

        let mut query_timer = Instant::now();
        let mut io_timer = Instant::now();
        let mut cpu_timer = Instant::now();

        let visited = &mut query_scratch.visited;
        let retset = &mut query_scratch.retset;
        retset.reserve(l_search as usize);
        let full_retset = &mut query_scratch.full_retset;

        // Find best medoid
        let mut best_medoid = 0;
        let mut best_dist = f32::MAX;

        if !use_filter {
            for cur_m in 0..self._num_medoids {
                let cur_expanded_dist = self._dist_cmp_float.compare(
                    query_float,
                    &self._centroid_data[self._aligned_dim * cur_m..],
                    self._aligned_dim as u32,
                );
                if cur_expanded_dist < best_dist {
                    best_medoid = self._medoids[cur_m];
                    best_dist = cur_expanded_dist;
                }
            }
        } else {
            if let Some(medoid_ids) = self._filter_to_medoid_ids.get(&filter_label) {
                for cur_m in 0..medoid_ids.len() {
                    compute_dists(&[medoid_ids[cur_m]], 1, dist_scratch);
                    let cur_expanded_dist = dist_scratch[0];
                    if cur_expanded_dist < best_dist {
                        best_medoid = medoid_ids[cur_m];
                        best_dist = cur_expanded_dist;
                    }
                }
            } else {
                panic!("Cannot find medoid for specified filter.");
            }
        }

        compute_dists(&[best_medoid], 1, dist_scratch);
        retset.insert(Neighbor::new(best_medoid, dist_scratch[0]));
        visited.insert(best_medoid);

        let mut cmps = 0;
        let mut hops = 0;
        let mut num_ios = 0;

        // Main search loop
        while retset.has_unexpanded_node() && num_ios < io_limit {
            // Clear iteration state
            let mut frontier = Vec::with_capacity(2 * beam_width as usize);
            let mut frontier_nhoods = Vec::with_capacity(2 * beam_width as usize);
            let mut frontier_read_reqs = Vec::with_capacity(2 * beam_width as usize);
            let mut cached_nhoods = Vec::with_capacity(2 * beam_width as usize);
            sector_scratch_idx = 0;

            // Find new beam
            let mut num_seen = 0;
            while retset.has_unexpanded_node()
                && frontier.len() < beam_width as usize
                && num_seen < beam_width as usize
            {
                let nbr = retset.closest_unexpanded();
                num_seen += 1;
                if let Some(iter) = self._nhood_cache.get(&nbr.id) {
                    cached_nhoods.push((nbr.id, iter.clone()));
                    if let Some(stats) = stats {
                        stats.n_cache_hits += 1;
                    }
                } else {
                    frontier.push(nbr.id);
                }
                if self._count_visited_nodes {
                    self._node_visit_counter[nbr.id as usize]
                        .1
                        .fetch_add(1, Ordering::Relaxed);
                }
            }

            // Read nhoods of frontier ids
            if !frontier.is_empty() {
                if let Some(stats) = stats {
                    stats.n_hops += 1;
                }
                for id in frontier.iter() {
                    let fnhood = (
                        *id,
                        &mut sector_scratch
                            [num_sectors_per_node * sector_scratch_idx * SECTOR_LEN..],
                    );
                    sector_scratch_idx += 1;
                    frontier_nhoods.push(fnhood);
                    frontier_read_reqs.push(AlignedRead::new(
                        self.get_node_sector(*id as usize) * SECTOR_LEN,
                        num_sectors_per_node * SECTOR_LEN,
                        fnhood.1,
                    ));
                    if let Some(stats) = stats {
                        stats.n_4k += 1;
                        stats.n_ios += 1;
                    }
                    num_ios += 1;
                }
                io_timer = Instant::now();
                self.reader.read(&mut frontier_read_reqs, ctx);
                if let Some(stats) = stats {
                    stats.io_us += io_timer.elapsed().as_micros() as f32;
                }
            }

            // Process cached nhoods
            for (id, (nnbrs, node_nbrs)) in cached_nhoods.iter() {
                let node_fp_coords_copy = self._coord_cache.get(id).unwrap();
                let cur_expanded_dist = if !self._use_disk_index_pq {
                    self._dist_cmp.compare(
                        aligned_query_T,
                        node_fp_coords_copy,
                        self._aligned_dim as u32,
                    )
                } else {
                    if self.metric == diskann::Metric::INNER_PRODUCT {
                        self._disk_pq_table
                            .inner_product(query_float, node_fp_coords_copy)
                    } else {
                        self._disk_pq_table
                            .l2_distance(query_float, node_fp_coords_copy)
                    }
                };
                full_retset.push(Neighbor::new(*id, cur_expanded_dist));

                // Compute node_nbrs <-> query dists in PQ space
                cpu_timer = Instant::now();
                compute_dists(node_nbrs, *nnbrs as u64, dist_scratch);
                if let Some(stats) = stats {
                    stats.n_cmps += *nnbrs;
                    stats.cpu_us += cpu_timer.elapsed().as_micros() as f32;
                }

                // Process prefetched nhood
                for m in 0..*nnbrs {
                    let id = node_nbrs[m as usize];
                    if visited.insert(id) {
                        if !use_filter && self._dummy_pts.contains(&id) {
                            continue;
                        }
                        if use_filter
                            && !self.point_has_label(id, filter_label)
                            && (!self._use_universal_label
                                || !self.point_has_label(id, self._universal_filter_label))
                        {
                            continue;
                        }
                        cmps += 1;
                        let dist = dist_scratch[m as usize];
                        retset.insert(Neighbor::new(id, dist));
                    }
                }
            }

            // Process frontier nhoods
            for (id, node_disk_buf) in frontier_nhoods.iter() {
                let node_buf = self.offset_to_node_nhood(node_disk_buf);
                let nnbrs = *node_buf as u64;
                let node_fp_coords = self.offset_to_node_coords(node_disk_buf);
                data_buf.copy_from_slice(node_fp_coords);
                let cur_expanded_dist = if !self._use_disk_index_pq {
                    self._dist_cmp
                        .compare(aligned_query_T, data_buf, self._aligned_dim as u32)
                } else {
                    if self.metric == diskann::Metric::INNER_PRODUCT {
                        self._disk_pq_table.inner_product(query_float, data_buf)
                    } else {
                        self._disk_pq_table.l2_distance(query_float, data_buf)
                    }
                };
                full_retset.push(Neighbor::new(*id, cur_expanded_dist));
                let node_nbrs = &node_buf[1..];

                // Compute node_nbrs <-> query dist in PQ space
                cpu_timer = Instant::now();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if let Some(stats) = stats {
                    stats.n_cmps += nnbrs as u32;
                    stats.cpu_us += cpu_timer.elapsed().as_micros() as f32;
                }

                // Process prefetched nhood
                for m in 0..nnbrs {
                    let id = node_nbrs[m as usize];
                    if visited.insert(id) {
                        if !use_filter && self._dummy_pts.contains(&id) {
                            continue;
                        }
                        if use_filter
                            && !self.point_has_label(id, filter_label)
                            && (!self._use_universal_label
                                || !self.point_has_label(id, self._universal_filter_label))
                        {
                            continue;
                        }
                        cmps += 1;
                        let dist = dist_scratch[m as usize];
                        retset.insert(Neighbor::new(id, dist));
                    }
                }
            }

            hops += 1;
        }

        // Re-sort by distance
        full_retset.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if use_reorder_data {
            if !self._reorder_data_exists {
                panic!("Requested use of reordering data which does not exist in index file");
            }

            let mut vec_read_reqs = Vec::new();
            if full_retset.len() > k_search as usize * FULL_PRECISION_REORDER_MULTIPLIER {
                full_retset.truncate(k_search as usize * FULL_PRECISION_REORDER_MULTIPLIER);
            }

            for i in 0..full_retset.len() {
                vec_read_reqs.push(AlignedRead::new(
                    self.vector_sector_no(full_retset[i].id as usize) * SECTOR_LEN,
                    SECTOR_LEN,
                    &mut sector_scratch[i * SECTOR_LEN..],
                ));
                if let Some(stats) = stats {
                    stats.n_4k += 1;
                    stats.n_ios += 1;
                }
            }

            io_timer = Instant::now();
            self.reader.read(&mut vec_read_reqs, ctx);
            if let Some(stats) = stats {
                stats.io_us += io_timer.elapsed().as_micros() as f32;
            }

            for i in 0..full_retset.len() {
                let id = full_retset[i].id;
                let location =
                    &sector_scratch[i * SECTOR_LEN + self.vector_sector_offset(id as usize)..];
                full_retset[i].distance =
                    self._dist_cmp
                        .compare(aligned_query_T, location, self._data_dim as u32);
            }

            full_retset.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        // Copy k_search values
        for i in 0..k_search as usize {
            indices[i] = full_retset[i].id as u64;
            let key = full_retset[i].id;
            if self._dummy_pts.contains(&key) {
                indices[i] = self._dummy_to_real_map[&key] as u64;
            }

            if let Some(distances) = distances {
                distances[i] = full_retset[i].distance;
                if self.metric == diskann::Metric::INNER_PRODUCT {
                    distances[i] = -distances[i];
                    if self._max_base_norm != 0.0 {
                        distances[i] *= self._max_base_norm * query_norm;
                    }
                }
            }
        }

        if let Some(stats) = stats {
            stats.total_us = query_timer.elapsed().as_micros() as f32;
        }
    }

    // Range search implementation
    pub fn range_search(
        &self,
        query1: &[T],
        range: f64,
        min_l_search: u64,
        max_l_search: u64,
        indices: &mut Vec<u64>,
        distances: &mut Vec<f32>,
        min_beam_width: u64,
        stats: Option<&mut QueryStats>,
    ) -> u32 {
        let mut res_count = 0;
        let mut stop_flag = false;
        let mut l_search = min_l_search as u32;

        while !stop_flag {
            indices.resize(l_search as usize, 0);
            distances.resize(l_search as usize, f32::MAX);

            let mut cur_bw = if min_beam_width > (l_search as u64 / 5) {
                min_beam_width
            } else {
                l_search as u64 / 5
            };
            cur_bw = if cur_bw > 100 { 100 } else { cur_bw };

            self.cached_beam_search_1(
                query1,
                l_search as u64,
                l_search as u64,
                indices,
                Some(distances),
                cur_bw,
                false,
                stats,
            );

            for i in 0..l_search {
                if distances[i as usize] > range as f32 {
                    res_count = i;
                    break;
                } else if i == l_search - 1 {
                    res_count = l_search;
                }
            }

            if res_count < (l_search / 2) {
                stop_flag = true;
            }

            l_search *= 2;
            if l_search > max_l_search as u32 {
                stop_flag = true;
            }
        }

        indices.resize(res_count as usize, 0);
        distances.resize(res_count as usize, 0.0);
        res_count
    }

    pub fn get_data_dim(&self) -> u64 {
        self._data_dim as u64
    }

    pub fn get_metric(&self) -> () {
        self.metric
    }

    pub fn get_pq_vector(&self, vid: u64) -> Vec<u8> {
        let start = vid as usize * self._n_chunks;
        let end = start + self._n_chunks;
        self.data[start..end].to_vec()
    }

    pub fn get_num_points(&self) -> u64 {
        self._num_points
    }
}
