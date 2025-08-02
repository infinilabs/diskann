/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use std::env;

extern crate prost_build;

fn main() {
    println!(
        "cargo:warning=cfg!(target_os): {}",
        if cfg!(target_os = "linux") {
            "linux"
        } else {
            "not linux"
        }
    );
    println!(
        "cargo:warning=std::env::consts::OS: {}",
        std::env::consts::OS
    );

    println!("cargo:rerun-if-changed=src/indexlog.proto");

    // Try to find protoc in PATH first
    if let Ok(protoc_path) = which::which("protoc") {
        println!("cargo:warning=Found protoc at: {:?}", protoc_path);
        env::set_var("PROTOC", protoc_path);
    } else {
        println!("cargo:warning=protoc not found in PATH, trying platform-specific methods");

        if cfg!(target_os = "linux") {
            // Linux: try to use system protoc
            if let Ok(_) = std::process::Command::new("protoc")
                .arg("--version")
                .output()
            {
                // protoc is available, use it
                println!("cargo:warning=Using system protoc on Linux");
            } else {
                // Try to install protoc if not available
                println!("cargo:warning=protoc not found, please install protobuf-compiler");
                println!("cargo:warning=On Ubuntu/Debian: sudo apt-get install protobuf-compiler");
                println!("cargo:warning=On CentOS/RHEL: sudo yum install protobuf-compiler");
            }
        } else if cfg!(target_os = "macos") {
            // macOS: try to use Homebrew protoc
            if let Ok(_) = std::process::Command::new("protoc")
                .arg("--version")
                .output()
            {
                println!("cargo:warning=Using system protoc on macOS");
            } else {
                println!("cargo:warning=protoc not found, please install via Homebrew:");
                println!("cargo:warning=brew install protobuf");
            }
        } else if cfg!(target_os = "windows") {
            // Windows: try vcpkg, but don't fail if not available
            #[cfg(target_os = "windows")]
            {
                // Only try vcpkg on Windows
                if let Ok(_) = std::process::Command::new("protoc")
                    .arg("--version")
                    .output()
                {
                    println!("cargo:warning=Using system protoc on Windows");
                } else {
                    println!("cargo:warning=protoc not found, please install protobuf");
                    println!("cargo:warning=You can download from: https://github.com/protocolbuffers/protobuf/releases");
                }
            }
        }
    }

    // Try to compile protobuf, but don't fail if protoc is not available
    match prost_build::compile_protos(&["src/indexlog.proto"], &["src/"]) {
        Ok(_) => println!("cargo:warning=Successfully compiled protobuf"),
        Err(e) => {
            println!("cargo:warning=Failed to compile protobuf: {}", e);
            println!("cargo:warning=This is not critical for basic functionality");
            // Don't fail the build, just warn
        }
    }
}
