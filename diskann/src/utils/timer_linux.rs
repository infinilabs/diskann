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

pub use super::perf_linux::*;

use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Timer {
    check_point: Instant,
    pid: Option<usize>,
    cycles: Option<u64>,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    pub fn new() -> Timer {
        let pid = get_process_handle();
        let cycles = get_process_cycle_time(pid);
        Timer {
            check_point: Instant::now(),
            pid,
            cycles,
        }
    }

    pub fn reset(&mut self) {
        self.check_point = Instant::now();
        self.cycles = get_process_cycle_time(self.pid);
    }

    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.check_point)
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    pub fn elapsed_gcycles(&self) -> f32 {
        let cur_cycles = get_process_cycle_time(self.pid);
        if let (Some(cur_cycles), Some(cycles)) = (cur_cycles, self.cycles) {
            let spent_cycles =
                ((cur_cycles - cycles) as f64 * 1.0f64) / (1024 * 1024 * 1024) as f64;
            return spent_cycles as f32;
        }

        0.0
    }

    pub fn elapsed_seconds_for_step(&self, step: &str) -> String {
        format!(
            "Time for {}: {:.3} seconds, {:.3}B cycles",
            step,
            self.elapsed_seconds(),
            self.elapsed_gcycles()
        )
    }
}

#[cfg(test)]
mod timer_tests {
    use super::*;
    use std::{thread, time};

    #[test]
    fn test_new() {
        let timer = Timer::new();
        assert!(timer.check_point.elapsed().as_secs() < 1);
        if cfg!(windows) {
            assert!(timer.pid.is_some());
            assert!(timer.cycles.is_some());
        } else {
            assert!(timer.pid.is_none());
            assert!(timer.cycles.is_none());
        }
    }

    #[test]
    fn test_reset() {
        let mut timer = Timer::new();
        thread::sleep(time::Duration::from_millis(100));
        timer.reset();
        assert!(timer.check_point.elapsed().as_millis() < 10);
    }

    #[test]
    fn test_elapsed() {
        let timer = Timer::new();
        thread::sleep(time::Duration::from_millis(100));
        assert!(timer.elapsed().as_millis() > 100);
        assert!(timer.elapsed_seconds() > 0.1);
    }
}
