[![DiskANN Paper](https://img.shields.io/badge/Paper-NeurIPS%3A_DiskANN-blue)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Arxiv%3A_Fresh--DiskANN-blue)](https://arxiv.org/abs/2105.09613)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Filtered--DiskANN-blue)](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)

# DiskANN in Rust

DiskANN is a suite of scalable, accurate, cost-effective approximate nearest neighbor (ANN) search algorithms for large-scale vector search. 
It uses a graph-based index (Vamana) to manage very large datasets with high recall and low latency. 
For example, DiskANN can index, store, and query a billion-point dataset on a single machine with 64GB RAM and an SSD, achieving high recall and low query latency.  
The INFINI Labs DiskANN project provides a pure Rust implementation of these ideas. This crate was originally forked from Microsoft’s partial Rust port (the DiskANN/rust folder) and extended to be complete. 
In particular, it implements the previously-missing disk-based query functionality, enabling full end-to-end DiskANN indexing and search in Rust.

# Features

Key features:
- Pure Rust implementation – no C/C++ dependencies; leverages Rust’s memory safety and concurrency.
- Disk-based indexing and querying – supports very large datasets that do not fit in memory.
- High recall and low latency – optimized for fast approximate nearest neighbor search.
- Parallelism – leverages Rust’s concurrency features for efficient indexing and querying.


# Installation

Add the diskann crate to your project by including it in Cargo.toml. For example:
```
[dependencies]
diskann = "0.1"
```

# Usage
```rust
```

# Contributing

Contributions and bug reports are welcome! Please open issues or pull requests on the GitHub repository. We follow the standard Rust community conventions.

# Notice

The code is based on ideas from the [DiskANN](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf), [Fresh-DiskANN](https://arxiv.org/abs/2105.09613) and the [Filtered-DiskANN](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf) papers with further improvements.

This project forked off from [Microsoft/DiskANN](https://github.com/microsoft/DiskANN/tree/main/rust) and [code for NSG](https://github.com/ZJULearning/nsg) .

# License

This project is released under the MIT License. See the LICENSE file for details.

References: DiskANN was introduced in [Jayaram et al., NeurIPS 2019] as a disk-based ANN solution. This crate’s design follows the DiskANN methodology, adapted for Rust with support for modern Rust features (parallelism, memory mapping, etc.)
