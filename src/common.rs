use std::ops::Range;

/// Type of spatial distance, if it's Manhattan distance ([wikipedia](https://en.wikipedia.org/wiki/Taxicab_geometry))
/// or Euclidean distance ([wikipedia](https://en.wikipedia.org/wiki/Euclidean_distance)).
///
/// There is almost no penalty for using Euclidean distance for spatial distance other than LUT
/// can take longer to compute (it's still bellow noise level).
///
/// There is no support for Euclidean distance in color space.
///
/// The color distance is accelerated by specialized instructions for computing Sum of Absolute
/// Differences (SAD). On other hand Euclidean distance needs square root which is slow.
#[derive(Clone)]
pub enum DistanceMetric {
    /// Manhattan distance for spatial and color space.
    Manhattan,
    /// Euclidean distance for spatial and Manhattan for color space.
    RealDistManhattanColor,
}

/// Changes between parallelization schemas.
#[derive(Clone, PartialEq, Debug, Copy)]
pub enum AssignThreadingStrategy {
    /// No threading - used only for correctness checks and very small images.
    SingleThread,
    /// Use as small tiles as possible. Used automatically for smaller images, when there is more
    /// available `rayon::current_num_threads()` than number of minimal-sized tiles.
    ///
    /// For larger images, there can be overhead by doing small tasks on different places of the
    /// image.
    FineGrained,
    /// This mode works for large images by splitting image by number of available threads
    /// to equally sized tiles to eliminate overhead and maximize CPU cache usage.
    ///
    /// Should be in most cases faster, and it's the default.
    ///
    /// If you have some intensive work on background `FineGrained` mode can be better choice
    /// because of work stealing.
    CoreDistributed,
    /// This mode removes the tiled threading scheme, which is only really needed for
    /// `AssignSweepBy::Cluster`. Incompatible with `AssignSweepBy::Cluster`.
    ///
    /// This mode should be the fastest, so it's the default.
    RowBased,
    /// Same as `AssignSweepBy::RowBased`, but after every completed image row, the partial update
    /// step is done to reuse local cache. This was an experiment, but it proved, that local
    /// accumulator values gets evicted from caches during assignment making the update very slow.
    ///
    /// This severely regresses performance for large images.
    RowBasedFusedUpdate,
}

/// Changes between `assign_clusters_by_cluster()` and `assign_clusters_by_row()`, where the
/// difference is in outer loop. If we iterate through clusters in tile and by its rows or
/// iterate over rows of a tile and by the clusters.
#[derive(Clone, PartialEq, Debug, Copy)]
pub enum AssignSweepBy {
    /// This uses `Cluster` for _subsample stride_ = 1 and `Row` for other cases.
    Auto,
    /// With this `assign_clusters_by_row()` is always used. Should be better for very large tiles
    /// or CPUs with small caches.
    Row,
    /// With this `assign_clusters_by_cluster()` is always used. Should be better for smaller tiles.
    /// Does not support _subsample stride_ other than 1 (no subsampling). May change in the future.
    Cluster,
}

/// Main config for the processing.
///
/// To get the best performance on your hardware you can run benchmarks using
/// `cargo criterion --bench bench` and adjust it for your hardware.
///
/// It's recommended to customize it for your scenario (image size, number of clusters)
/// to adjust `assign_threading_strategy` and `assign_sweep_by`, but defaults should be good
/// enough (expect maximally around 10% speedup).

#[derive(Clone)]
pub struct Config {
    /// Number of clusters.This directly influences search region size (_S_)
    ///
    /// Which is calculated using: _S = sqrt((image width * height) / num_of_clusters)_
    ///
    /// Generally for good performance _S_ should be somewhere between 20 (more clusters) and 50 (fewer clusters).
    pub num_of_clusters: u16,
    /// How many iterations is done. N+1 iterations.
    ///
    /// Currently, there is no support for preemptive exit.
    pub max_iterations: u16,
    /// This is used to compute the spatial distance LUTs.
    /// Higher means more compact superpixels -> this is about trading color accuracy for locality.
    ///
    /// This setting does not affect performance.
    pub compactness: f32,
    /// This is for Connected-component labeling ([wikipedia](https://en.wikipedia.org/wiki/Connected-component_labeling)).
    ///
    /// It's used for eliminating small disconnected regions.
    pub min_size_factor: f32,
    /// By how much we advance on image row in assignment. Non-zero, 1 is no row subsampling.
    /// This is mainly about trading segmentation quality for speed.
    ///
    /// Higher means lower quality. Sane number is 3 (almost no quality loss for 40% speedup) to 7.
    pub subsample_stride: u8,
    /// Type of spatial distance, if it's Manhattan distance ([wikipedia](https://en.wikipedia.org/wiki/Taxicab_geometry))
    /// or Euclidean distance ([wikipedia](https://en.wikipedia.org/wiki/Euclidean_distance)).
    ///
    /// There is almost no penalty for using Euclidean distance for spatial distance other than LUT
    /// can take longer to compute (it's still bellow noise level).
    ///
    /// There is no support for Euclidean distance in color space. Computing it would be very slow.
    pub distance_metric: DistanceMetric,
    /// Threading strategy for assign step. It's not strictly enforced and with small images can be
    /// even limited to single thread processing.
    pub assign_threading_strategy: AssignThreadingStrategy,
    /// Modes for trading complexity for better cache hits. For large images (4K+ or FHD single core)
    /// row-based assignment should be faster by 10%+.
    pub assign_sweep_by: AssignSweepBy,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            num_of_clusters: 2000,
            max_iterations: 10,
            compactness: 10f32,
            min_size_factor: 0.25,
            subsample_stride: 3,
            distance_metric: DistanceMetric::RealDistManhattanColor,
            assign_threading_strategy: AssignThreadingStrategy::RowBased,
            assign_sweep_by: AssignSweepBy::Auto,
        }
    }
}

pub(crate) fn split_length_to_ranges(length: usize, splits: usize) -> Vec<Range<usize>> {
    let chunk_size = length / splits;
    let rem = length % splits;
    (0..splits)
        .scan((rem, 0usize), |(r, acc), _split| {
            let mut size = chunk_size;
            if *r > 0 {
                *r -= 1;
                size += 1;
            }
            let out = (*acc, *acc + size);
            *acc += size;
            Some(out.0..out.1)
        })
        .collect()
}
