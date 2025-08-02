/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

cfg_if::cfg_if! {
    if #[cfg(target_os = "windows")] {
        pub mod perf;
        pub use perf::{get_process_cycle_time, get_process_handle};

        pub mod file_io;
        pub use file_io::{get_queued_completion_status, read_file_to_slice};

        pub mod file_handle;
        pub use file_handle::FileHandle;

        pub mod io_completion_port;
        pub use io_completion_port::IOCompletionPort;
    } else {
        // For non-Windows platforms, we'll use simplified implementations
        // In a production environment, you would implement platform-specific optimizations
        pub mod perf_generic;
        pub use perf_generic::{get_process_cycle_time, get_process_handle};

        pub mod file_io_generic;
        pub use file_io_generic::{read_file_to_slice, get_io_completion_status};

        pub mod file_handle_generic;
        pub use file_handle_generic::FileHandle;

        pub mod io_completion_port_generic;
        pub use io_completion_port_generic::IOCompletionPort;
    }
}
