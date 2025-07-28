// TODO: Caller code is very callous about copying IOContext objects
// all over the place. MUST verify that it won't cause leaks/logical
// errors.
// Because of such callous copying, we have to use ptr->atomic instead
// of atomic, as atomic is not copyable.

enum Status
    {
        READ_WAIT,
        READ_SUCCESS,
        READ_FAILED,
        PROCESS_COMPLETE
    }

struct IOContext {
    
    std::shared_ptr<ANNIndex::IDiskPriorityIO> m_pDiskIO = nullptr;
    std::shared_ptr<std::vector<ANNIndex::AsyncReadRequest>> m_pRequests;
    std::shared_ptr<std::vector<Status>> m_pRequestsStatus;

    // waitonaddress on this memory to wait for IO completion signal
    // reader should signal this memory after IO completion
    // TODO: WindowsAlignedFileReader can be modified to take advantage of this
    //   and can largely share code with the file reader for Bing.
    mutable volatile long m_completeCount = 0;

    IOContext()
        : m_pRequestsStatus(new std::vector<Status>()), m_pRequests(new std::vector<ANNIndex::AsyncReadRequest>())
    {
        (*m_pRequestsStatus).reserve(MAX_IO_DEPTH);
        (*m_pRequests).reserve(MAX_IO_DEPTH);
    }
}
