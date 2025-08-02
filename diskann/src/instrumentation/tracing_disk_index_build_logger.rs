/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use tracing::{info, instrument};

use crate::{common::ANNResult, utils::Timer};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiskIndexConstructionCheckpoint {
    None = 0,
    PqConstruction = 1,
    InmemIndexBuild = 2,
    DiskLayout = 3,
}

impl From<i32> for DiskIndexConstructionCheckpoint {
    fn from(value: i32) -> Self {
        match value {
            0 => DiskIndexConstructionCheckpoint::None,
            1 => DiskIndexConstructionCheckpoint::PqConstruction,
            2 => DiskIndexConstructionCheckpoint::InmemIndexBuild,
            3 => DiskIndexConstructionCheckpoint::DiskLayout,
            _ => DiskIndexConstructionCheckpoint::None,
        }
    }
}

impl From<DiskIndexConstructionCheckpoint> for i32 {
    fn from(checkpoint: DiskIndexConstructionCheckpoint) -> Self {
        checkpoint as i32
    }
}

pub struct TracingDiskIndexBuildLogger {
    timer: Timer,
    checkpoint: DiskIndexConstructionCheckpoint,
}

impl TracingDiskIndexBuildLogger {
    pub fn new(checkpoint: DiskIndexConstructionCheckpoint) -> Self {
        Self {
            timer: Timer::new(),
            checkpoint,
        }
    }

    #[instrument(skip(self), fields(checkpoint = ?self.checkpoint))]
    pub fn log_checkpoint(
        &mut self,
        next_checkpoint: DiskIndexConstructionCheckpoint,
    ) -> ANNResult<()> {
        if self.checkpoint == DiskIndexConstructionCheckpoint::None {
            return Ok(());
        }

        let time_spent = self.timer.elapsed().unwrap_or_default().as_secs_f32();
        let g_cycles = self.timer.elapsed_gcycles();

        info!(
            checkpoint = ?self.checkpoint,
            time_spent_in_seconds = time_spent,
            g_cycles_spent = g_cycles,
            next_checkpoint = ?next_checkpoint,
            "Disk index construction checkpoint"
        );

        self.checkpoint = next_checkpoint;
        self.timer.reset();
        Ok(())
    }
}

// Alias for backward compatibility
pub type DiskIndexBuildLogger = TracingDiskIndexBuildLogger;
