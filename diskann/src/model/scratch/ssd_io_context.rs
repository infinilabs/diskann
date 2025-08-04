/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![allow(dead_code)] // Todo: Remove this when the disk index query code is complete.
use std::fs::File;
use std::path::Path;

use crate::common::{ANNError, ANNResult};

/// I/O context for SSD operations
#[derive(Debug)]
pub struct IOContext {
    /// File handle for I/O operations
    pub file_handle: Option<File>,
    /// File path
    pub file_path: String,
    /// Current position in file
    pub position: u64,
    /// Buffer for I/O operations
    pub buffer: Vec<u8>,
}

impl Default for IOContext {
    fn default() -> Self {
        Self {
            file_handle: None,
            file_path: String::new(),
            position: 0,
            buffer: Vec::new(),
        }
    }
}

impl IOContext {
    /// Create a new I/O context
    pub fn new(file_path: &str) -> ANNResult<Self> {
        let path = Path::new(file_path);
        if !path.exists() {
            return Err(ANNError::log_io_error(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", file_path),
            )));
        }

        Ok(Self {
            file_handle: None,
            file_path: file_path.to_string(),
            position: 0,
            buffer: Vec::new(),
        })
    }

    /// Open the file handle
    pub fn open_file(&mut self) -> ANNResult<()> {
        if self.file_handle.is_none() {
            let file = File::open(&self.file_path)?;
            self.file_handle = Some(file);
        }
        Ok(())
    }

    /// Close the file handle
    pub fn close_file(&mut self) {
        self.file_handle = None;
    }

    /// Read data from file at specified offset
    pub fn read_at(&mut self, offset: u64, buffer: &mut [u8]) -> ANNResult<usize> {
        self.open_file()?;

        if let Some(ref mut file) = self.file_handle {
            use std::io::{Read, Seek, SeekFrom};

            file.seek(SeekFrom::Start(offset))?;
            let bytes_read = file.read(buffer)?;
            self.position = offset + bytes_read as u64;

            Ok(bytes_read)
        } else {
            Err(ANNError::log_io_error(std::io::Error::new(
                std::io::ErrorKind::Other,
                "File handle not available",
            )))
        }
    }

    /// Read aligned data from file
    pub fn read_aligned(
        &mut self,
        offset: u64,
        buffer: &mut [u8],
        alignment: usize,
    ) -> ANNResult<usize> {
        // Ensure offset is aligned
        if offset % alignment as u64 != 0 {
            return Err(ANNError::log_io_error(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Offset {} is not aligned to {}", offset, alignment),
            )));
        }

        // Ensure buffer size is aligned
        if buffer.len() % alignment != 0 {
            return Err(ANNError::log_io_error(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Buffer size {} is not aligned to {}",
                    buffer.len(),
                    alignment
                ),
            )));
        }

        self.read_at(offset, buffer)
    }

    /// Get the current file position
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Set the file position
    pub fn set_position(&mut self, position: u64) {
        self.position = position;
    }

    /// Get the file path
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    /// Check if file is open
    pub fn is_open(&self) -> bool {
        self.file_handle.is_some()
    }

    /// Get file size
    pub fn file_size(&self) -> ANNResult<u64> {
        if let Some(ref file) = self.file_handle {
            let metadata = file.metadata()?;
            Ok(metadata.len())
        } else {
            // Try to get size without opening file
            let metadata = std::fs::metadata(&self.file_path)?;
            Ok(metadata.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_io_context_creation() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");

        // Create a test file
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        let ctx = IOContext::new(file_path.to_str().unwrap());
        assert!(ctx.is_ok());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.file_path(), file_path.to_str().unwrap());
        assert_eq!(ctx.position(), 0);
        assert!(!ctx.is_open());
    }

    #[test]
    fn test_io_context_file_not_found() {
        let ctx = IOContext::new("nonexistent_file");
        assert!(ctx.is_err());
    }

    #[test]
    fn test_io_context_read_operations() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");

        // Create a test file with some data
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"test data for reading").unwrap();
        drop(file);

        let mut ctx = IOContext::new(file_path.to_str().unwrap()).unwrap();

        // Test reading
        let mut buffer = vec![0u8; 4];
        let bytes_read = ctx.read_at(0, &mut buffer).unwrap();

        assert_eq!(bytes_read, 4);
        assert_eq!(&buffer, b"test");
        assert_eq!(ctx.position(), 4);
        assert!(ctx.is_open());
    }

    #[test]
    fn test_io_context_aligned_read() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");

        // Create a test file with aligned data
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"test data for aligned reading").unwrap();
        drop(file);

        let mut ctx = IOContext::new(file_path.to_str().unwrap()).unwrap();

        // Test aligned reading
        let mut buffer = vec![0u8; 8];
        let bytes_read = ctx.read_aligned(0, &mut buffer, 8).unwrap();

        assert_eq!(bytes_read, 8);
        assert_eq!(&buffer, b"test dat");
    }

    #[test]
    fn test_io_context_file_size() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_file");

        // Create a test file
        let test_data = b"test data for size checking";
        let mut file = File::create(&file_path).unwrap();
        file.write_all(test_data).unwrap();
        drop(file);

        let ctx = IOContext::new(file_path.to_str().unwrap()).unwrap();
        let size = ctx.file_size().unwrap();

        assert_eq!(size, test_data.len() as u64);
    }
}
