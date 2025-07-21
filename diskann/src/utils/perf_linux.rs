// MIT License
//
// Copyright (C) INFINI Labs & INFINI LIMITED. <hello@infini.ltd>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// Get current process handle.
#[cfg(target_os = "linux")]
use libc::{getpid, sysconf, _SC_CLK_TCK};

/// Linux implementation to get process CPU time
///
/// This function retrieves the combined user and system CPU time for the current process,
/// similar to Windows' QueryProcessCycleTime functionality.
/// Returns the time in nanoseconds, or None if the operation fails.
pub fn get_process_cycle_time(_process_handle: Option<usize>) -> Option<u64> {
    // In Linux, we can directly read process statistics from /proc/[pid]/stat file
    let pid = unsafe { getpid() };
    let stat_path = format!("/proc/{}/stat", pid);

    if let Ok(contents) = std::fs::read_to_string(&stat_path) {
        // Split the stat file contents into whitespace-separated parts
        let parts: Vec<&str> = contents.split_whitespace().collect();

        // The stat file contains at least 44 fields (varies by kernel version)
        // We need at least 15 fields to access utime and stime
        if parts.len() >= 15 {
            // utime is field 14 (0-based index 13) - user CPU time in clock ticks
            let utime: u64 = parts[13].parse().unwrap_or(0);

            // stime is field 15 (0-based index 14) - system CPU time in clock ticks
            let stime: u64 = parts[14].parse().unwrap_or(0);

            // Get system clock ticks per second (usually 100 on Linux)
            let clock_ticks = unsafe { sysconf(_SC_CLK_TCK) } as u64;

            // Convert ticks to nanoseconds and return sum of user+system time
            // This approximates Windows' cycle time concept
            return Some((utime + stime) * (1_000_000_000 / clock_ticks));
        }
    }

    // Return None if we couldn't read or parse the stat file
    None
}

/// Gets a process "handle" for Linux
///
/// In Linux, we don't need explicit process handles like Windows.
/// This simply returns the current process ID as a pseudo-handle.
pub fn get_process_handle() -> Option<usize> {
    // On Linux, we just use the PID as our "handle"
    Some(unsafe { getpid() } as usize)
}
