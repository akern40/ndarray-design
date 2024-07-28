# Design for Rust's `ndarray` Library

As `ndarray` evolves, it's helpful to have a place to test out new ideas and concepts without being overwhelmed by the size and complexity of `ndarray`'s codebase.

This repository is currently exploring a redesign of `ndarray`'s core data structure (formally `ArrayBase`) to allow for an API centered around Rust's auto-deref capabilities.