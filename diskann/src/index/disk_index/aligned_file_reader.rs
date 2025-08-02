use crate::model::IOContext;

pub struct AlignedRead {
    pub offset: usize,
    pub len: usize,
    pub buf: *mut u8,
}

impl AlignedRead {
    pub fn new(offset: usize, len: usize, buf: *mut u8) -> Self {
        Self { offset, len, buf }
    }
}

pub trait AlignedFileReader {
    // Returns the thread-specific context
    // Returns (io_context_t)(-1) if thread is not registered
    fn get_ctx(&mut self) -> IOContext;

    // Register thread-id for a context
    fn register_thread(&mut self);
    // De-register thread-id for a context
    fn deregister_thread(&mut self);
    // De-register all threads
    fn deregister_all_threads(&mut self);

    // Open file (blocking call)
    fn open(&mut self, fname: &str);
    // Close file (blocking call)
    fn close(&mut self);

    // Process batch of aligned requests in parallel (blocking call)
    fn read(&mut self, read_reqs: &mut Vec<AlignedRead>, ctx: &mut IOContext);
}
