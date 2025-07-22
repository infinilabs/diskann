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
    if cfg!(target_os = "linux") {
        /*
        let protobuf_path = protopkg.link_paths[0].parent().unwrap();
        let protobuf_bin_path = protobuf_path
            .join("tools")
            .join("protobuf")
            .join("protoc.exe")
            .to_str()
            .unwrap()
            .to_string();
        env::set_var("PROTOC", protobuf_bin_path);

        let protobuf_inc_path = protobuf_path
            .join("include")
            .join("google")
            .join("protobuf")
            .to_str()
            .unwrap()
            .to_string();
        env::set_var("PROTOC_INCLUDE", protobuf_inc_path);
        */

        prost_build::compile_protos(&["src/indexlog.proto"], &["src/"]).unwrap();
    } else {
        let protopkg = vcpkg::find_package("protobuf").unwrap();
        let protobuf_path = protopkg.link_paths[0].parent().unwrap();

        let protobuf_bin_path = protobuf_path
            .join("tools")
            .join("protobuf")
            .join("protoc.exe")
            .to_str()
            .unwrap()
            .to_string();
        env::set_var("PROTOC", protobuf_bin_path);

        let protobuf_inc_path = protobuf_path
            .join("include")
            .join("google")
            .join("protobuf")
            .to_str()
            .unwrap()
            .to_string();
        env::set_var("PROTOC_INCLUDE", protobuf_inc_path);

        prost_build::compile_protos(&["src/indexlog.proto"], &["src/"]).unwrap();
    }
}
