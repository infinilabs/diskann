pub unsafe fn read_file_to_slice<T>(
    _file_handle: &crate::FileHandle,
    _buffer_slice: &mut [T],
    _overlapped: *mut std::ffi::c_void,
    _offset: u64,
) -> std::io::Result<bool> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Not supported",
    ))
}
pub unsafe fn get_io_completion_status(
    _completion_port: &crate::IOCompletionPort,
    _lp_number_of_bytes: &mut u32,
    _lp_completion_key: &mut usize,
    _lp_overlapped: *mut *mut std::ffi::c_void,
    _dw_milliseconds: u32,
) -> std::io::Result<bool> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Not supported",
    ))
}
