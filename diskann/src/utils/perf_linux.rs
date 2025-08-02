/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use std::time::Duration;

#[cfg(target_os = "linux")]
use libc::{getpid, sysconf, _SC_CLK_TCK};

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn getpid() -> i32 {
    // Fallback for non-Linux platforms
    0
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn sysconf(_name: i32) -> i64 {
    // Fallback for non-Linux platforms
    100
}

#[cfg(not(target_os = "linux"))]
const _SC_CLK_TCK: i32 = 0;

/// Linux implementation to get process CPU time
#[cfg(target_os = "linux")]
pub fn get_process_cycle_time() -> Duration {
    // In Linux, we can directly read process statistics from /proc/[pid]/stat file
    let pid = unsafe { getpid() };
    let stat_path = format!("/proc/{}/stat", pid);
    
    if let Ok(contents) = std::fs::read_to_string(stat_path) {
        if let Some(utime_str) = contents.split_whitespace().nth(13) {
            if let Ok(utime) = utime_str.parse::<u64>() {
                // Get system clock ticks per second (usually 100 on Linux)
                let clock_ticks_per_sec = unsafe { sysconf(_SC_CLK_TCK) };
                let duration_secs = utime as f64 / clock_ticks_per_sec as f64;
                return Duration::from_secs_f64(duration_secs);
            }
        }
    }
    
    Duration::from_secs(0)
}

#[cfg(not(target_os = "linux"))]
pub fn get_process_cycle_time() -> Duration {
    // Fallback for non-Linux platforms
    Duration::from_secs(0)
}

/// Gets a process "handle" for Linux
#[cfg(target_os = "linux")]
pub fn get_process_handle() -> Option<usize> {
    // In Linux, we don't need explicit process handles like Windows.
    // We can use the PID directly for most operations.
    // On Linux, we just use the PID as our "handle"
    Some(unsafe { getpid() } as usize)
}

#[cfg(not(target_os = "linux"))]
pub fn get_process_handle() -> Option<usize> {
    // Fallback for non-Linux platforms
    Some(0)
}
