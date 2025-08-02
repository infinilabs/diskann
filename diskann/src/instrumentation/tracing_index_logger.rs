/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{info, instrument};

use crate::common::ANNResult;
use crate::utils::Timer;

pub struct TracingIndexLogger {
    items_processed: AtomicUsize,
    timer: Timer,
    range: usize,
}

impl TracingIndexLogger {
    pub fn new(range: usize) -> Self {
        Self {
            items_processed: AtomicUsize::new(0),
            timer: Timer::new(),
            range,
        }
    }

    #[instrument(skip(self), fields(range = self.range))]
    pub fn vertex_processed(&self) -> ANNResult<()> {
        let count = self.items_processed.fetch_add(1, Ordering::Relaxed);

        if count % 100_000 == 0 {
            let percentage_complete = (100_f32 * count as f32) / (self.range as f32);
            let time_spent = self.timer.elapsed().unwrap_or_default().as_secs_f32();
            let g_cycles = self.timer.elapsed_gcycles();

            info!(
                percentage_complete = percentage_complete,
                time_spent_in_seconds = time_spent,
                g_cycles_spent = g_cycles,
                vertices_processed = count,
                "Index construction progress"
            );
        }

        Ok(())
    }
}

// Alias for backward compatibility
pub type IndexLogger = TracingIndexLogger;
