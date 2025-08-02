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

use std::time::{Duration, Instant};

use crate::utils::perf_linux::{get_process_cycle_time, get_process_handle};

pub struct Timer {
    pub check_point: Instant,
    pub pid: Option<usize>,
    pub cycles: Option<u64>,
    pub start_time: Option<Instant>,
    pub elapsed: Option<Duration>,
    pub cycle_diff: Option<u64>,
    pub step_name: Option<String>,
}

impl Timer {
    pub fn new() -> Timer {
        let pid = get_process_handle();
        let cycles = get_process_cycle_time();
        Timer {
            check_point: Instant::now(),
            pid,
            cycles: Some(cycles.as_nanos() as u64),
            start_time: None,
            elapsed: None,
            cycle_diff: None,
            step_name: None,
        }
    }

    pub fn reset(&mut self) {
        self.check_point = Instant::now();
        let cycles = get_process_cycle_time();
        self.cycles = Some(cycles.as_nanos() as u64);
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        let cycles = get_process_cycle_time();
        self.cycles = Some(cycles.as_nanos() as u64);
    }

    pub fn stop(&mut self) {
        if let Some(start_time) = self.start_time {
            self.elapsed = Some(start_time.elapsed());
            let cur_cycles = get_process_cycle_time();
            if let Some(cycles) = self.cycles {
                self.cycle_diff = Some(cur_cycles.as_nanos() as u64 - cycles);
            }
        }
    }

    pub fn elapsed(&self) -> Option<Duration> {
        self.elapsed
    }

    pub fn cycle_diff(&self) -> Option<u64> {
        self.cycle_diff
    }

    pub fn elapsed_seconds_for_step(&self, step_name: &str) -> String {
        if let Some(elapsed) = self.elapsed {
            format!("{}: {:.3}s", step_name, elapsed.as_secs_f32())
        } else {
            format!("{}: No timing data", step_name)
        }
    }

    pub fn elapsed_gcycles(&self) -> f32 {
        self.cycle_diff.unwrap_or(0) as f32
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
        assert!(timer.elapsed().is_some());
        assert!(timer.elapsed().unwrap().as_millis() > 100);
    }
}
