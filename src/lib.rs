//! FastSLIC implementation in Rust.
//!
//! This crate provides an improved version of SLIC superpixel-based image segmentation named
//! FastSLIC.
//!
//! This crate does not implement as many features as the original, notably Preemptive SLIC
//! (PreSLIC) and LSC. But implements some improvements for big images, where it can have marginally
//! better performance.
//!
//! The following example describes how to process image in packed RGB24 (RGB 8bit) format
//! (default for image crate):
//!
//! ```rust
//! use fast_slic_rust::arrays::LABImage;
//! use fast_slic_rust::common::*;
//! use fast_slic_rust::slic::{iterate, Clusters};
//!
//! fn main(){
//!     // Open image
//!     let dimg = image::open("test/data/aerial.jpg").unwrap();
//!     // convert it to packed RGB24
//!     let img = dimg.as_rgb8().unwrap().as_raw();
//!     let width = dimg.width() as usize;
//!     let height = dimg.height() as usize;
//!     // convert image to packed Lab with padding byte
//!     let image = LABImage::from_srgb(img.as_slice(), width, height);
//!     // create config with defaults
//!     let mut config = Config::default();
//!     // override subsample stride to 5
//!     config.subsample_stride = 5;
//!     // initialize clusters (it's possible to write custom initializer to have e.g. ROI)
//!     let mut clusters = Clusters::initialize_clusters(&image, &config);
//!     // make the computation
//!     iterate(&image, &config, &mut clusters)
//! }
//! ```
//!
//! It's also possible to write your custom `iterate()` function and get the distances map out of
//! the algorithm which can be useful for post-processing.
//!
//! This crate has also benchmarks and tests. It's strongly recommended to use this in release
//! build. This library uses unsafe code and uses `assume!` macro to avoid boundary checks in
//! hot-loops in release builds.
//!
//! There may be some not so perfect things. The worst is possibility of rayon deadlocking, because
//! using `std:sync::Barrier` as a mean of synchronization. I don't have time now for researching
//! alternatives or doing huge rewrites. One way of avoiding this can be creating new thread pool
//! for this crate. Discussion or PR about this is welcome.
//!
//! Also, there may be changes in API. It's possible, that I find some better alternative to
//! `arrays` or `atomic_arrays` which I made for this crate.
//!
//! Note: There may be problems in builds for non-x86 platforms.
//!

pub mod arrays;
pub mod assign;
pub mod atomic_arrays;
pub mod cielab;
pub mod cluster;
pub mod common;
pub mod conectivity;
pub mod slic;
