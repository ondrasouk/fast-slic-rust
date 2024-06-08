# fast-slic-rust

FastSLIC implementation in Rust.

This crate provides an improved version of SLIC superpixel-based image segmentation named FastSLIC.

Original implementation written in C++ is here https://github.com/Algy/fast-slic.

This crate does not implement as many features as the original, notably Preemptive SLIC (PreSLIC) and LSC.
But implements some improvements for big images, where it can have better performance.

Also, I can't guarantee perfectly same output as a original or other reference implementations.
It hugely depends on order of the assignments, which cluster takes the pixel, which hase same distance (5D) from the
cluster, first. So there is even differences between single-threaded assignment output and the multithreaded output.

## Performance

### Processing 4K (4096x2160) image On AMD Ryzen 9 5850X (16c/32t desktop Zen 3 with 4 x DDR4 3200 MT/s):

| Row subsamplig stride | Fast-slic-rust [ms] | Fast-slic [ms] | Speedup |
|-----------------------|---------------------|----------------|---------|
| 1 (no subsampling)    | 69.45               | 122.79         | 1.768   |
| 2                     | 58.49               | 92.78          | 1.586   |
| 3                     | 51.11               | 82.40          | 1.612   |
| 4                     | 46.91               | 75.26          | 1.604   |
| 5                     | 44.44               | 71.80          | 1.616   |

Scikit-image implementation runs for 5272 ms.

### Processing 4K (4096x2160) image On AMD Ryzen 5 5500U (6c/12t notebook Zen 2 with 2 x LPDDR4 3200 MT/s):

| Row subsamplig stride | Fast-slic-rust [ms] | Fast-slic [ms] | Speedup |
|-----------------------|---------------------|----------------|---------|
| 1 (no subsampling)    | 167.56              | 306.04         | 1.826   |
| 2                     | 131.22              | 206.56         | 1.574   |
| 3                     | 109.75              | 172.81         | 1.574   |
| 4                     | 99.31               | 152.82         | 1.539   |
| 5                     | 91.91               | 127.48         | 1.387   |

Scikit-image implementation runs for 8876 ms.

### Notes:

Tested with image `test/data/aerial.jpg`, 2000 clusters, enforced connectivity, without LAB conversion, 10 iterations,
0.25 min. size factor, compactness 10 and default settings for this crate.

Original FastSLIC C++ implementation had preemption turned off and reported duration of color conversion was
subtracted from the `slic.slic_model.last_timing_report` total time.

Scikit-image implementation CPU times also includes color conversion, and it's a completely single-threaded
implementation. It's there mainly for a reference.

## Example

The following example describes how to process image in packed RGB24 (RGB 8bit) format (default for image crate):

```rust
use fast_slic_rust::arrays::LABImage;
use fast_slic_rust::common::*;
use fast_slic_rust::slic::{iterate, Clusters};

fn main(){
    // Open image
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    // convert it to packed RGB24
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    // convert image to packed Lab with padding byte
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    // create config with defaults
    let mut config = Config::default();
    // override subsample stride to 5
    config.subsample_stride = 5;
    // initialize clusters (it's possible to write custom initializer to have e.g. ROI)
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    // make the computation
    iterate(&image, &config, &mut clusters)
}
```

It’s also possible to write your custom iterate() function and get the distances map out of the algorithm which can be useful for post-processing.

This crate has also benchmarks and tests. It’s strongly recommended to use this in release build. This library uses unsafe code and uses assume! macro to avoid boundary checks in hot-loops in release builds.

## Notes

- There may be some not so perfect things. The worst is possibility of rayon deadlocking, because using `std:sync::Barrier` as a mean of synchronization. I don’t have time now for researching alternatives or doing huge rewrites. One way of avoiding this can be creating new thread pool for this crate. Discussion or PR about this is welcome.
- Also, there may be changes in API. It’s possible, that I find some better alternative to arrays or atomic_arrays which I made for this crate.
- There may be problems in builds for non-x86 platforms.

## Licence

This work is licenced under MIT licence.

This is part of my diploma thesis (written in Czech) which can be accessed here https://www.vut.cz/studenti/zav-prace/detail/159072. It's only one part of my diploma thesis.

It is very likely that there will be a publication in English, but for now you can cite my diploma thesis.

Image `tests/data/aerial.jpg` is frame number 986 from Netflix Aerial video sequence, which is licenced by https://creativecommons.org/licenses/by-nc-nd/4.0/.

## Contributions

Contributions are welcome. Also tips on API, etc.
