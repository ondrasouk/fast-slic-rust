use crate::arrays::{Array2D, LABImage};
use crate::atomic_arrays::{AtomicArray2D, AtomicSubArray2D};
use crate::cluster::Cluster;
use crate::common::{split_length_to_ranges, AssignSweepBy, AssignThreadingStrategy, Config};
use crate::slic::Clusters;
use gcdx::gcdx;
use multiversion::multiversion;
use multiversion::target::selected_target;
use rayon::current_num_threads;
use rayon::prelude::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::array;
use std::mem::align_of;
use std::ops::Range;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};
use std::sync::{Arc, Barrier};

#[multiversion(targets = "simd")]
pub fn fused_assign_update(
    image: &LABImage,
    config: &Config,
    clusters: &mut Clusters,
    min_distances: &mut AtomicArray2D<AtomicU16>,
    spatial_distance_lut: &Array2D<u16>,
    search_region_size: u16,
    subsample_start: u8,
) {
    assert!(
        config.subsample_stride > 0,
        "Subsample stride must be higher than zero. How you want to advance by zero?"
    );
    if config.subsample_stride == 1 {
        min_distances.fill(u16::MAX);
    } else {
        (subsample_start as usize..min_distances.height)
            .step_by(config.subsample_stride as usize)
            .for_each(|row| {
                min_distances
                    .get_row(row)
                    .iter()
                    .for_each(|el| el.store(u16::MAX, Ordering::Relaxed))
            });
    }
    let num_threads = current_num_threads();

    clusters
        .clusters
        .iter_mut()
        .for_each(|c| c.update_coords(image, search_region_size));

    #[cfg(test)]
    println!("AssignThreadingStrategy::RowBasedFusedUpdate");

    let num_cluster_members: Vec<AtomicU32> =
        Vec::from_iter((0..config.num_of_clusters).map(|_| AtomicU32::new(0)));
    let cluster_acc_vec: Vec<[AtomicU32; 5]> =
        Vec::from_iter((0..config.num_of_clusters).map(|_| array::from_fn(|_| AtomicU32::new(0))));

    let ranges = split_length_to_ranges(image.height, num_threads);

    ranges.into_par_iter().for_each(|rows| {
        let tile_assignment = clusters.assignments.get_subarray(
            0,
            rows.start,
            clusters.assignments.width,
            rows.len(),
        );
        let tile_min_distances =
            min_distances.get_subarray(0, rows.start, min_distances.width, rows.len());

        let mut update_clusters: Vec<&Cluster> = clusters.clusters.iter().filter(|c| {
            (c.bottom > rows.start as u16) && (c.top < rows.end as u16)
        }).collect();
        // Note: The threading schema ensures, that the atomic operations are not necessary...
        //  Atomics are there used to avoid using unsafe or splitting the 2D array to regions
        //  with every line as slice... this was my first attempt, and it was horribly slow since
        //  it generated on one thread thousands mutable which were then distributed to the threads...

        // Note: Also "global" here means that the coordinates are in the image's scope,
        //  "local" in the tile's context scope.
        update_clusters.sort_unstable_by_key(|c| c.y);

        let mut num_cluster_members_local: Vec<u32> = vec![0; config.num_of_clusters as usize];
        let mut cluster_acc_local: Vec<[u32; 5]> =
            vec![[0, 0, 0, 0, 0]; config.num_of_clusters as usize];

        let stride = config.subsample_stride as usize;
        let (subsampling_global_tile_start, subsampling_local_tile_start) = subsampling_tile_start(subsample_start as usize, stride, tile_assignment.y);

        let search_region_size_2 = search_region_size * 2;
        let search_region_size_2_1 = search_region_size_2 + 1;

        let image_width_divisible_16: bool = image.width % 16 == 0;

        let mut skip_processed_clusters = 0;
        // Iterate over rows of image
        for (local_row, global_row) in (subsampling_local_tile_start..)
            .zip(subsampling_global_tile_start..(tile_assignment.y + tile_assignment.height))
            .step_by(config.subsample_stride as usize)
        {
            for cluster in update_clusters.iter().skip(skip_processed_clusters) {
                if cluster.bottom <= global_row as u16 {
                    skip_processed_clusters += 1;
                    continue;
                } else if cluster.top > global_row as u16 {
                    break;
                }

                let cluster_color = [cluster.l, cluster.a, cluster.b, 0];

                let global_top = cluster.top as usize;
                let global_left = cluster.left as usize;
                let global_right = cluster.right as usize;

                let (local_left, local_right) = tile_assignment
                    .get_local_index_left_right_index(global_left, global_right);

                let image_row = image.get_row_part(global_row, global_left, global_right);
                let assignments_row = tile_assignment.get_row_part(local_row, local_left, local_right);
                let min_distances_row = tile_min_distances.get_row_part(local_row, local_left, local_right);

                let lut_row = global_row - global_top;
                let dist_row = spatial_distance_lut.get_row_part(
                    lut_row,
                    cluster.lut_left as usize,
                    cluster.lut_right as usize,
                );
                debug_assert!(assignments_row.len() <= search_region_size_2_1 as usize, "The assignments_row slice should not be wider ({}) than 2*search_region_size+1 ({search_region_size_2_1}) (width of distance LUT)!", assignments_row.len());
                debug_assert_eq!(assignments_row.len(), min_distances_row.len());
                debug_assert_eq!(
                    assignments_row.len(),
                    dist_row.len(),
                    "Info: 2*search_region_size+1 = {search_region_size_2_1}; cluster: {:?}",
                    cluster
                );
                debug_assert_eq!(assignments_row.len() * 4, image_row.len());

                if selected_target!().supports_feature_str("avx2") {
                    unsafe {
                        if image_width_divisible_16 {
                            assign_row_avx2::<false>(
                                image_row,
                                dist_row,
                                &cluster_color,
                                cluster.number,
                                min_distances_row,
                                assignments_row,
                            );
                        } else {
                            assign_row_avx2::<true>(
                                image_row,
                                dist_row,
                                &cluster_color,
                                cluster.number,
                                min_distances_row,
                                assignments_row,
                            );
                        }
                    }
                } else {
                    assign_row_generic(
                        image_row,
                        dist_row,
                        &cluster_color,
                        cluster.number,
                        min_distances_row,
                        assignments_row,
                    );
                }
            }
            let image_row = image.get_row(global_row);
            let assignments_row = tile_assignment.get_row(global_row);
            for (column, (pixel, assignment)) in
            image_row.chunks_exact(4).zip(assignments_row).enumerate()
            {
                let cluster_n = assignment.load(Ordering::Relaxed);
                if cluster_n == 0xFFFF {
                    continue;
                }
                num_cluster_members_local[cluster_n as usize] += 1;
                cluster_acc_local[cluster_n as usize][0] += global_row as u32;
                cluster_acc_local[cluster_n as usize][1] += column as u32;
                cluster_acc_local[cluster_n as usize][2] += pixel[0] as u32;
                cluster_acc_local[cluster_n as usize][3] += pixel[1] as u32;
                cluster_acc_local[cluster_n as usize][4] += pixel[2] as u32;
            }
        }
        for (cluster_n, num_members) in num_cluster_members_local
            .into_iter()
            .enumerate()
            .filter(|(_, x)| *x != 0)
        {
            num_cluster_members[cluster_n].fetch_add(num_members, Ordering::Relaxed);
            cluster_acc_vec[cluster_n][0]
                .fetch_add(cluster_acc_local[cluster_n][0], Ordering::Relaxed);
            cluster_acc_vec[cluster_n][1]
                .fetch_add(cluster_acc_local[cluster_n][1], Ordering::Relaxed);
            cluster_acc_vec[cluster_n][2]
                .fetch_add(cluster_acc_local[cluster_n][2], Ordering::Relaxed);
            cluster_acc_vec[cluster_n][3]
                .fetch_add(cluster_acc_local[cluster_n][3], Ordering::Relaxed);
            cluster_acc_vec[cluster_n][4]
                .fetch_add(cluster_acc_local[cluster_n][4], Ordering::Relaxed);
        }
    });

    for cluster in clusters.clusters.iter_mut() {
        let cluster_num = cluster.number as usize;
        let cluster_members = num_cluster_members[cluster_num].load(Ordering::Relaxed);
        if cluster_members == 0 {
            continue;
        }
        let cluster_members_half = cluster_members / 2;
        cluster.num_members = cluster_members;
        let next_x = ((cluster_acc_vec[cluster_num][1].load(Ordering::Relaxed)
            + cluster_members_half)
            / cluster_members) as u16;
        let next_y = ((cluster_acc_vec[cluster_num][0].load(Ordering::Relaxed)
            + cluster_members_half)
            / cluster_members) as u16;
        debug_assert!(
            next_x < image.width as u16,
            "{:?} trying to update x which is out of bounds - x={next_x} acc_x={:?}",
            cluster,
            cluster_acc_vec[cluster_num][1].load(Ordering::Relaxed)
        );
        debug_assert!(
            next_y < image.height as u16,
            "{:?} trying to update y which is out of bounds - y={next_y} acc_y={:?}",
            cluster,
            cluster_acc_vec[cluster_num][0].load(Ordering::Relaxed)
        );
        cluster.y = next_y;
        cluster.x = next_x;
        cluster.l = ((cluster_acc_vec[cluster_num][2].load(Ordering::Relaxed)
            + cluster_members_half)
            / cluster_members) as u8;
        cluster.a = ((cluster_acc_vec[cluster_num][3].load(Ordering::Relaxed)
            + cluster_members_half)
            / cluster_members) as u8;
        cluster.b = ((cluster_acc_vec[cluster_num][4].load(Ordering::Relaxed)
            + cluster_members_half)
            / cluster_members) as u8;
    }
}

/// This function implements the assign step in SLIC algorithm.
pub fn assign(
    image: &LABImage,
    config: &Config,
    clusters: &mut Clusters,
    min_distances: &mut AtomicArray2D<AtomicU16>,
    spatial_distance_lut: &Array2D<u16>,
    search_region_size: u16,
    subsample_start: u8,
) {
    assert!(
        config.subsample_stride > 0,
        "Subsample stride must be higher than zero. How you want to advance by zero?"
    );
    if config.subsample_stride == 1 {
        min_distances.fill(u16::MAX);
    } else {
        (subsample_start as usize..min_distances.height)
            .step_by(config.subsample_stride as usize)
            .for_each(|row| {
                min_distances
                    .get_row(row)
                    .iter()
                    .for_each(|el| el.store(u16::MAX, Ordering::Relaxed))
            });
    }
    let num_threads = current_num_threads();

    clusters
        .clusters
        .iter_mut()
        .for_each(|c| c.update_coords(image, search_region_size));

    if (config.assign_threading_strategy == AssignThreadingStrategy::RowBased
        || config.assign_threading_strategy == AssignThreadingStrategy::RowBasedFusedUpdate)
        && num_threads > 1
        && config.assign_sweep_by != AssignSweepBy::Cluster
    {
        // This threading model does not use tiles. It enables huge speedups.
        #[cfg(test)]
        println!("AssignThreadingStrategy::RowBased");

        let ranges = split_length_to_ranges(image.height, num_threads);

        ranges.into_par_iter().for_each(|rows| {
            let tile_assignment = clusters.assignments.get_subarray(
                0,
                rows.start,
                clusters.assignments.width,
                rows.len(),
            );
            let tile_min_distances =
                min_distances.get_subarray(0, rows.start, min_distances.width, rows.len());

            let mut update_clusters: Vec<&Cluster> = clusters
                .clusters
                .iter()
                .filter(|c| (c.bottom > rows.start as u16) && (c.top < rows.end as u16))
                .collect();

            assign_clusters_by_row(
                &tile_assignment,
                &tile_min_distances,
                &mut update_clusters,
                image,
                config,
                subsample_start,
                spatial_distance_lut,
                search_region_size,
            );
        });
        return;
    }

    let min_tile_size_half: u16 = search_region_size + 16;
    let min_tile_size: u16 = 2 * min_tile_size_half;
    let cell_w = (image.width as u16).div_ceil(min_tile_size);
    let cell_h = (image.height as u16).div_ceil(min_tile_size);

    if config.assign_threading_strategy == AssignThreadingStrategy::SingleThread
        || num_threads < 2
        || cell_h <= 4
        || cell_w <= 4
    {
        #[cfg(test)]
        println!("AssignThreadingStrategy::SingleThread");
        // The threading is not possible, and we fall back to single-thread.
        let tile_assignment = clusters.assignments.get_subarray(
            0,
            0,
            clusters.assignments.width,
            clusters.assignments.height,
        );
        let tile_min_distances =
            min_distances.get_subarray(0, 0, min_distances.width, min_distances.height);
        // assign_clusters_by_row is faster always for very large images (less cache misses)
        let assign_clusters = if config.assign_sweep_by == AssignSweepBy::Auto
            || config.assign_sweep_by == AssignSweepBy::Row
            || config.subsample_stride != 1
        {
            assign_clusters_by_row
        } else {
            assign_clusters_by_cluster
        };
        assign_clusters(
            &tile_assignment,
            &tile_min_distances,
            &mut clusters.clusters.iter().collect::<Vec<&Cluster>>(),
            image,
            config,
            subsample_start,
            spatial_distance_lut,
            search_region_size,
        );
        return;
    }
    let assign_clusters = if config.assign_sweep_by == AssignSweepBy::Row
        || (config.assign_sweep_by == AssignSweepBy::Auto && config.subsample_stride != 1)
    {
        assign_clusters_by_row
    } else {
        assign_clusters_by_cluster
    };

    let mut used_threads_num = num_threads;
    let optimal_tile_num = num_threads * 4;
    let mut possible_threads_y: Vec<usize> = (1..((image.height) / (2 * min_tile_size) as usize))
        .filter_map(|y| gcdx(&[used_threads_num, y]))
        .filter(|y| {
            (*y < 2 * cell_w as usize)
                && (image.width / (used_threads_num / y) > 2 * min_tile_size as usize)
        })
        .collect();
    if possible_threads_y.is_empty() {
        // try again without one thread since the speedup is huge...
        used_threads_num -= 1;
        possible_threads_y = (1..((image.height) / (2 * min_tile_size) as usize))
            .filter_map(|y| gcdx(&[used_threads_num, y]))
            .filter(|y| {
                (*y < 2 * cell_w as usize)
                    && (image.width / (used_threads_num / y) > 2 * min_tile_size as usize)
            })
            .collect();
    }

    if config.assign_threading_strategy == AssignThreadingStrategy::FineGrained
        || optimal_tile_num >= (cell_w * cell_h) as usize
        || possible_threads_y.is_empty()
    {
        #[cfg(test)]
        println!("AssignThreadingStrategy::FineGrained");
        let tile_size = (min_tile_size, min_tile_size);
        let tile_size_half = (min_tile_size / 2, min_tile_size / 2);

        let clusters_tiles: Vec<(usize, usize)> = clusters
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    i,
                    ((c.y / tile_size.0) * cell_w + (c.x / tile_size.1)) as usize,
                )
            })
            .collect();

        // FIXME: when number of tiles in horizontal or vertical dimension is lower than 4
        //        It happens, that there is only one active tile row or column it fails on assert...
        //        Probably calculation of tile's context sizes is not prepared for this.
        for phase in 0..4u16 {
            let start_x = phase % 2;
            let start_y = phase / 2;
            let width = clusters.assignments.width as u16;
            let height = clusters.assignments.height as u16;
            let tiles_x: Vec<u16> = (start_x..cell_w).step_by(2).collect();
            let tiles_y: Vec<u16> = (start_y..cell_h).step_by(2).collect();
            let tiles_x_p: Vec<u16> = tiles_x.iter().map(|tile_x| *tile_x * tile_size.0).collect();
            let tiles_y_p: Vec<u16> = tiles_y.iter().map(|tile_y| *tile_y * tile_size.1).collect();
            let x_left_s: Vec<u16> = tiles_x
                .iter()
                .zip(&tiles_x_p)
                .map(|(tile_x, tile_x_p)| {
                    if *tile_x == 0 {
                        0
                    } else {
                        *tile_x_p - tile_size_half.0
                    }
                })
                .collect();
            let y_top_s: Vec<u16> = tiles_y
                .iter()
                .zip(&tiles_y_p)
                .map(|(tile_y, tile_y_p)| {
                    if *tile_y == 0 {
                        0
                    } else {
                        *tile_y_p - tile_size_half.1
                    }
                })
                .collect();
            let x_sizes: Vec<u16> = x_left_s
                .iter()
                .map(|x_point| {
                    if x_point + 2 * tile_size.0 >= width {
                        width - x_point
                    } else if *x_point == 0 {
                        tile_size.0 + tile_size_half.0
                    } else {
                        tile_size.0 + 2 * tile_size_half.0
                    }
                })
                .collect();
            let y_sizes: Vec<u16> = y_top_s
                .iter()
                .map(|y_point| {
                    if y_point + 2 * tile_size.1 >= height {
                        height - y_point
                    } else if *y_point == 0 {
                        tile_size.1 + tile_size_half.1
                    } else {
                        tile_size.1 + 2 * tile_size_half.1
                    }
                })
                .collect();
            debug_assert!(
                x_left_s
                    .iter()
                    .zip(&x_sizes)
                    .any(|(x_left, x_size)| *x_left + *x_size < width),
                "phase: {phase}, {:?}, {:?}",
                x_left_s,
                x_sizes
            );
            debug_assert!(y_top_s
                .iter()
                .zip(&y_sizes)
                .any(|(y_top, y_size)| *y_top + *y_size < height));
            let x_sizes_usize: Vec<usize> = x_sizes.iter().map(|i| *i as usize).collect();
            let y_sizes_usize: Vec<usize> = y_sizes.iter().map(|i| *i as usize).collect();

            let tiles_assignment = clusters.assignments.split_to_tiles(
                x_left_s[0] as usize,
                y_top_s[0] as usize,
                &x_sizes_usize,
                &y_sizes_usize,
            );
            let tiles_min_distances = min_distances.split_to_tiles(
                x_left_s[0] as usize,
                y_top_s[0] as usize,
                &x_sizes_usize,
                &y_sizes_usize,
            );
            let tile_coordinates: Vec<(u16, u16)> = tiles_y_p
                .iter()
                .flat_map(|y| tiles_x_p.iter().map(move |x| (*x, *y)))
                .collect();

            let tile_numbers: Vec<usize> = tiles_y
                .iter()
                .flat_map(|y| tiles_x.iter().map(move |x| (y * cell_w + x) as usize))
                .collect();

            (
                tile_numbers,
                tiles_min_distances,
                tiles_assignment,
                tile_coordinates,
            )
                .into_par_iter()
                .for_each(
                    |(tile_number, tile_min_distances, tile_assignment, tile_coordinate)| {
                        debug_assert!(
                            tile_assignment.x <= tile_coordinate.0 as usize,
                            "{:?}",
                            tile_assignment
                        );
                        debug_assert!(
                            tile_assignment.y <= tile_coordinate.1 as usize,
                            "{:?}",
                            tile_assignment
                        );
                        debug_assert!(
                            tile_assignment.x + tile_assignment.width > tile_coordinate.0 as usize,
                            "{:?} (tile_size={:?}) is not in {:?}",
                            tile_coordinate,
                            tile_size,
                            tile_assignment
                        );
                        debug_assert!(
                            tile_assignment.y + tile_assignment.height > tile_coordinate.1 as usize,
                            "{:?} (tile_size={:?}) is not in {:?}",
                            tile_coordinate,
                            tile_size,
                            tile_assignment
                        );
                        let mut update_clusters: Vec<&Cluster> = clusters_tiles
                            .iter()
                            .filter(|(_, cluster_tile_num)| *cluster_tile_num == tile_number)
                            .map(|(ind, _)| &clusters.clusters[*ind])
                            .collect();
                        assign_clusters(
                            &tile_assignment,
                            &tile_min_distances,
                            &mut update_clusters,
                            image,
                            config,
                            subsample_start,
                            spatial_distance_lut,
                            search_region_size,
                        );
                    },
                );
        }
        return;
    }

    #[cfg(test)]
    println!("AssignThreadingStrategy::CoreDistributed");
    // Here we try to adjust the size of the tile to have optimal number of tiles.
    let mut tile_aspect_ratio_min = 0f32;
    let mut threads_y = 1;
    let mut threads_x = 1;
    for threads_yp in possible_threads_y {
        let threads_xp = used_threads_num / threads_yp;
        let tile_aspect_ratio =
            (image.width / threads_xp) as f32 / (image.height / threads_yp) as f32;
        if (1f32 - (1f32 / tile_aspect_ratio)).abs() < (1f32 - (1f32 / tile_aspect_ratio_min)).abs()
        {
            tile_aspect_ratio_min = tile_aspect_ratio;
            threads_y = threads_yp;
            threads_x = threads_xp;
        }
    }

    let tile_groups_ranges_x = split_length_to_ranges(image.width, threads_x);
    let tile_groups_ranges_y = split_length_to_ranges(image.height, threads_y);

    let mut tile_groups_ranges: Vec<(Range<usize>, Range<usize>)> =
        Vec::with_capacity(tile_groups_ranges_x.len() * tile_groups_ranges_y.len());
    for tile_groups_range_y in &tile_groups_ranges_y {
        for tile_groups_ranges_x in &tile_groups_ranges_x {
            tile_groups_ranges.push((tile_groups_ranges_x.clone(), tile_groups_range_y.clone()));
        }
    }

    debug_assert_eq!(used_threads_num, tile_groups_ranges.len());

    // This is seriously problematic, since it can cause deadlock, when the panic occurs on some
    // thread or even if not all threads in thread pool are available.
    let barrier_assign = Arc::new(Barrier::new(used_threads_num));

    rayon::scope(|s| {
        let clusters_c = clusters.clusters.as_slice();
        let assignment_c = &clusters.assignments;
        let min_distances_ref = &min_distances;
        tile_groups_ranges
            .iter()
            .for_each(|(group_range_x, group_range_y)| {
                s.spawn(|_| {
                    let barrier_assign_c = Arc::clone(&barrier_assign);
                    let middle_x = group_range_x.start + group_range_x.len() / 2;
                    let middle_y = group_range_y.start + group_range_y.len() / 2;
                    let tiles_range_x =
                        [group_range_x.start..middle_x, middle_x..group_range_x.end];
                    let tiles_range_y =
                        [group_range_y.start..middle_y, middle_y..group_range_y.end];

                    let tiles_ranges = [
                        (&tiles_range_x[0], &tiles_range_y[0]),
                        (&tiles_range_x[1], &tiles_range_y[0]),
                        (&tiles_range_x[1], &tiles_range_y[1]),
                        (&tiles_range_x[0], &tiles_range_y[1]),
                    ];
                    // Phases but in-thread
                    for tile_ranges in tiles_ranges.iter() {
                        let tile_coordinate =
                            (tile_ranges.0.start as u16, tile_ranges.1.start as u16);
                        let tile_size = (tile_ranges.0.len() as u16, tile_ranges.1.len() as u16);
                        // tile context size
                        let x = tile_ranges
                            .0
                            .start
                            .saturating_sub(min_tile_size_half as usize);
                        let y = tile_ranges
                            .1
                            .start
                            .saturating_sub(min_tile_size_half as usize);
                        let width =
                            (tile_ranges.0.len() + (min_tile_size as usize)).min(image.width - x);
                        let height =
                            (tile_ranges.1.len() + (min_tile_size as usize)).min(image.height - y);

                        let tile_assignment = assignment_c.get_subarray(x, y, width, height);
                        let tile_min_distances =
                            min_distances_ref.get_subarray(x, y, width, height);
                        debug_assert!(
                            tile_assignment.x <= tile_coordinate.0 as usize,
                            "{:?}",
                            tile_assignment
                        );
                        debug_assert!(
                            tile_assignment.y <= tile_coordinate.1 as usize,
                            "{:?}",
                            tile_assignment
                        );
                        debug_assert!(
                            tile_assignment.x + tile_assignment.width > tile_coordinate.0 as usize,
                            "{:?} (tile_size={:?}) is not in {:?}",
                            tile_coordinate,
                            tile_size,
                            tile_assignment
                        );
                        debug_assert!(
                            tile_assignment.y + tile_assignment.height > tile_coordinate.1 as usize,
                            "{:?} (tile_size={:?}) is not in {:?}",
                            tile_coordinate,
                            tile_size,
                            tile_assignment
                        );
                        let mut update_clusters: Vec<&Cluster> = clusters_c
                            .iter()
                            .filter(|c| {
                                (c.y >= tile_coordinate.1)
                                    && (c.x >= tile_coordinate.0)
                                    && (c.y < (tile_coordinate.1 + tile_size.1))
                                    && (c.x < (tile_coordinate.0 + tile_size.0))
                            })
                            .collect();

                        barrier_assign_c.wait();

                        assign_clusters(
                            &tile_assignment,
                            &tile_min_distances,
                            &mut update_clusters,
                            image,
                            config,
                            subsample_start,
                            spatial_distance_lut,
                            search_region_size,
                        );
                    }
                });
            });
    });
}

#[multiversion(targets = "simd")]
fn assign_clusters_by_cluster(
    assignments: &AtomicSubArray2D<AtomicU16>,
    min_distances: &AtomicSubArray2D<AtomicU16>,
    update_clusters: &mut [&Cluster],
    image: &LABImage,
    config: &Config,
    _subsample_global_start: u8,
    spatial_distance_lut: &Array2D<u16>,
    _search_region_size: u16,
) {
    // Note: The threading schema ensures, that the atomic operations are not necessary...
    //  Atomics are there used to avoid using unsafe or splitting the 2D array to regions
    //  with every line as slice... this was my first attempt, and it was horribly slow since
    //  it generated on one thread thousands mutable which were then distributed to the threads...

    // Note: Also "global" here means that the coordinates are in the image's scope,
    //  "local" in the tile's context scope.
    if config.subsample_stride != 1 {
        println!("assign_clusters_by_cluster does not support row subsampling. Expect lower performance.")
    }
    let image_width_divisible_16: bool = image.width % 16 == 0;

    for cluster in update_clusters {
        let global_top = cluster.top as usize;
        let global_bottom = cluster.bottom as usize;
        let global_left = cluster.left as usize;
        let global_right = cluster.right as usize;

        let (local_left, local_top) =
            assignments.get_local_index_from_full_array_index(global_left, global_top);
        let (local_right, _local_bottom) =
            assignments.get_local_index_from_full_array_index(global_right, global_bottom);

        let cluster_color = [cluster.l, cluster.a, cluster.b, 0];
        for row in 0..(global_bottom - global_top) {
            let image_row = image.get_row_part(row + global_top, global_left, global_right);
            let assignments_row =
                assignments.get_row_part(row + local_top, local_left, local_right);
            let min_distances_row =
                min_distances.get_row_part(row + local_top, local_left, local_right);
            let dist_row = spatial_distance_lut.get_row_part(
                row,
                cluster.lut_left as usize,
                cluster.lut_right as usize,
            );

            if selected_target!().supports_feature_str("avx2") {
                unsafe {
                    if image_width_divisible_16 {
                        assign_row_avx2::<false>(
                            image_row,
                            dist_row,
                            &cluster_color,
                            cluster.number,
                            min_distances_row,
                            assignments_row,
                        );
                    } else {
                        assign_row_avx2::<true>(
                            image_row,
                            dist_row,
                            &cluster_color,
                            cluster.number,
                            min_distances_row,
                            assignments_row,
                        );
                    }
                }
            } else {
                assign_row_generic(
                    image_row,
                    dist_row,
                    &cluster_color,
                    cluster.number,
                    min_distances_row,
                    assignments_row,
                );
            }
        }
    }
}

#[multiversion(targets = "simd")]
fn assign_clusters_by_row(
    assignments: &AtomicSubArray2D<AtomicU16>,
    min_distances: &AtomicSubArray2D<AtomicU16>,
    update_clusters: &mut [&Cluster],
    image: &LABImage,
    config: &Config,
    subsample_global_start: u8,
    spatial_distance_lut: &Array2D<u16>,
    search_region_size: u16,
) {
    // Note: The threading schema ensures, that the atomic operations are not necessary...
    //  Atomics are there used to avoid using unsafe or splitting the 2D array to regions
    //  with every line as slice... this was my first attempt, and it was horribly slow since
    //  it generated on one thread thousands mutable which were then distributed to the threads...

    // Note: Also "global" here means that the coordinates are in the image's scope,
    //  "local" in the tile's context scope.
    update_clusters.sort_unstable_by_key(|c| c.y);

    let stride = config.subsample_stride as usize;
    let (subsampling_global_tile_start, subsampling_local_tile_start) =
        subsampling_tile_start(subsample_global_start as usize, stride, assignments.y);
    let search_region_size_2 = search_region_size * 2;
    let search_region_size_2_1 = search_region_size_2 + 1;

    let image_width_divisible_16: bool = image.width % 16 == 0;

    let mut skip_processed_clusters = 0;
    // Iterate over rows of image
    for (local_row, global_row) in (subsampling_local_tile_start..)
        .zip(subsampling_global_tile_start..(assignments.y + assignments.height))
        .step_by(config.subsample_stride as usize)
    {
        for cluster in update_clusters.iter().skip(skip_processed_clusters) {
            if cluster.bottom <= global_row as u16 {
                skip_processed_clusters += 1;
                continue;
            } else if cluster.top > global_row as u16 {
                break;
            }

            let cluster_color = [cluster.l, cluster.a, cluster.b, 0];

            let global_top = cluster.top as usize;
            let global_left = cluster.left as usize;
            let global_right = cluster.right as usize;

            let (local_left, local_right) =
                assignments.get_local_index_left_right_index(global_left, global_right);

            let image_row = image.get_row_part(global_row, global_left, global_right);
            let assignments_row = assignments.get_row_part(local_row, local_left, local_right);
            let min_distances_row = min_distances.get_row_part(local_row, local_left, local_right);

            let lut_row = global_row - global_top;
            let dist_row = spatial_distance_lut.get_row_part(
                lut_row,
                cluster.lut_left as usize,
                cluster.lut_right as usize,
            );
            debug_assert!(assignments_row.len() <= search_region_size_2_1 as usize, "The assignments_row slice should not be wider ({}) than 2*search_region_size+1 ({search_region_size_2_1}) (width of distance LUT)!", assignments_row.len());
            debug_assert_eq!(assignments_row.len(), min_distances_row.len());
            debug_assert_eq!(
                assignments_row.len(),
                dist_row.len(),
                "Info: 2*search_region_size+1 = {search_region_size_2_1}; cluster: {:?}",
                cluster
            );
            debug_assert_eq!(assignments_row.len() * 4, image_row.len());

            if selected_target!().supports_feature_str("avx2") {
                unsafe {
                    if image_width_divisible_16 {
                        assign_row_avx2::<false>(
                            image_row,
                            dist_row,
                            &cluster_color,
                            cluster.number,
                            min_distances_row,
                            assignments_row,
                        );
                    } else {
                        assign_row_avx2::<true>(
                            image_row,
                            dist_row,
                            &cluster_color,
                            cluster.number,
                            min_distances_row,
                            assignments_row,
                        );
                    }
                }
            } else {
                assign_row_generic(
                    image_row,
                    dist_row,
                    &cluster_color,
                    cluster.number,
                    min_distances_row,
                    assignments_row,
                );
            }
        }
    }
}

#[inline(always)]
fn assign_row_generic(
    image_row: &[u8],
    dist_row: &[u16],
    cluster_color: &[u8; 4],
    cluster_number: u16,
    min_dist_row: &[AtomicU16],
    assign_row: &[AtomicU16],
) {
    for (((pixel, dist_lut), min_dist), assign) in image_row
        .chunks_exact(4)
        .zip(dist_row)
        .zip(min_dist_row)
        .zip(assign_row)
    {
        let l = pixel[0];
        let a = pixel[1];
        let b = pixel[2];
        let color_dist = l.abs_diff(cluster_color[0]) as u16
            + a.abs_diff(cluster_color[1]) as u16
            + b.abs_diff(cluster_color[2]) as u16;
        let dist = color_dist + dist_lut;
        if dist < min_dist.load(Ordering::Relaxed) {
            min_dist.store(dist, Ordering::Relaxed);
            assign.store(cluster_number, Ordering::Relaxed);
        }
    }
}

// The USE_UNALIGNED_GENERIC is used if image width is not a multiple of 16 to avoid data races
#[inline(always)]
unsafe fn assign_row_avx2<const USE_UNALIGNED_GENERIC: bool>(
    image_row: &[u8],
    dist_row: &[u16],
    cluster_color: &[u8; 4],
    cluster_number: u16,
    min_dist_row: &[AtomicU16],
    assign_row: &[AtomicU16],
) {
    let (min_dist_pre, min_dist_mid, min_dist_suf) = min_dist_row.align_to::<__m256i>();
    let (assign_pre, assign_mid, assign_suf) = assign_row.align_to::<__m256i>();
    let p_image: *const u8 = &image_row[min_dist_pre.len() * 4];
    debug_assert_eq!(min_dist_pre.len(), assign_pre.len());
    debug_assert_eq!(p_image.align_offset(align_of::<__m256i>()), 0);

    let color_i32: i32 = i32::from_ne_bytes([
        cluster_color[0],
        cluster_color[1],
        cluster_color[2],
        cluster_color[3],
    ]);

    let mut p_image_v: *const __m256i = p_image as *const __m256i;
    let mut p_dist_row: *const u16 = &dist_row[min_dist_pre.len()];

    // compute the unaligned prefix part
    if USE_UNALIGNED_GENERIC {
        let image_pre = &image_row[..(min_dist_pre.len() * 4)];
        assign_row_generic(
            image_pre,
            dist_row,
            cluster_color,
            cluster_number,
            min_dist_pre,
            assign_pre,
        );
    } else if min_dist_pre.len() > 8 {
        assign_primitive_avx2::<true>(
            p_image_v.sub(2),
            p_dist_row.sub(16),
            (&min_dist_mid[0] as *const __m256i).sub(1),
            (&assign_mid[0] as *const __m256i).sub(1),
            color_i32,
            cluster_number,
            16 - (assign_pre.len() as i8),
        );
    } else if !min_dist_pre.is_empty() {
        assign_primitive_avx2_half::<true>(
            p_image_v.sub(1),
            p_dist_row.sub(8),
            min_dist_mid.as_ptr().cast::<__m128i>().sub(1),
            assign_mid.as_ptr().cast::<__m128i>().sub(1),
            color_i32,
            cluster_number,
            8 - (assign_pre.len() as i8),
        );
    }

    // AVX2 main part
    for (min_dist, assign) in min_dist_mid.iter().zip(assign_mid) {
        assign_primitive_avx2::<false>(
            p_image_v,
            p_dist_row,
            min_dist,
            assign,
            color_i32,
            cluster_number,
            0,
        );
        p_image_v = p_image_v.add(2);
        p_dist_row = p_dist_row.add(16);
    }

    // compute the unaligned suffix part
    if USE_UNALIGNED_GENERIC {
        let image_suf = &image_row[image_row.len() - (min_dist_suf.len() * 4)..];
        assign_row_generic(
            image_suf,
            &dist_row[dist_row.len() - min_dist_suf.len()..],
            cluster_color,
            cluster_number,
            min_dist_suf,
            assign_suf,
        );
    } else if min_dist_suf.len() > 8 {
        assign_primitive_avx2::<true>(
            p_image_v,
            p_dist_row,
            min_dist_suf.as_ptr().cast::<__m256i>(),
            assign_suf.as_ptr().cast::<__m256i>(),
            color_i32,
            cluster_number,
            (min_dist_suf.len() as i8) - 16,
        );
    } else if !min_dist_suf.is_empty() {
        assign_primitive_avx2_half::<true>(
            p_image_v,
            p_dist_row,
            min_dist_suf.as_ptr().cast::<__m128i>(),
            assign_suf.as_ptr().cast::<__m128i>(),
            color_i32,
            cluster_number,
            (min_dist_suf.len() as i8) - 8,
        );
    }

    #[inline(always)]
    unsafe fn assign_primitive_avx2<const USE_MASK: bool>(
        p_image_v: *const __m256i,
        p_dist_lut: *const u16,
        min_dist: *const __m256i,
        assign: *const __m256i,
        color_i32: i32,
        cluster_number: u16,
        mask_num: i8,
    ) {
        let image1 = _mm256_load_si256(p_image_v);
        let image2 = _mm256_load_si256(p_image_v.add(1));

        // we permute this to get SADs in the 128bit lanes where we want it
        let image12_high = _mm256_permute2x128_si256::<0b00100000>(image1, image2);
        let image12_low = _mm256_permute2x128_si256::<0b00110001>(image1, image2);

        let color_vec = _mm256_set1_epi32(color_i32);

        // mpsadbw
        // IMM8[4:3] - setting quadruplets in first 128bits in b
        // IMM8[1:0] - setting quadruplets in second 128bits in b
        // IMM8[2] - setting quadruplets in second 128bits in b
        // IMM8[5] - setting quadruplets in second 128bits in b
        //
        // inner two SADs of quadruplets in 128 lanes [SAD1,0,0,0,SAD2,0,0,0,SAD5,0,0,0,SAD6,0,0,0]
        let diff1inner = _mm256_and_si256(
            _mm256_mpsadbw_epu8::<0b100100>(image12_high, color_vec),
            _mm256_set1_epi64x(0x000000000000FFFFu64 as i64),
        );
        let diff2inner = _mm256_and_si256(
            _mm256_mpsadbw_epu8::<0b100100>(image12_low, color_vec),
            _mm256_set1_epi64x(0x000000000000FFFFu64 as i64),
        );
        // SADs of 8xu8 [SAD0+1,0,0,0,SAD2+3,0,0,0,SAD4+5,0,0,0,SAD6+7,0,0,0]
        let diff1sum = _mm256_sad_epu8(image12_high, color_vec);
        let diff2sum = _mm256_sad_epu8(image12_low, color_vec);
        // outer SADs [SAD0,0,0,0,SAD3,0,0,0,SAD4,0,0,0,SAD7,0,0,0]
        let diff1outer = _mm256_sub_epi16(diff1sum, diff1inner);
        let diff2outer = _mm256_sub_epi16(diff2sum, diff2inner);
        // all SADs, but needs a shuffle in low 64bits of 128bit lanes
        // Now we have (16 SADs)
        // [SAD0,SAD1,   0,   0,SAD3,SAD2,   0,   0,SAD8,SAD9,    0,    0,SAD10,SAD11,    0,    0]
        // [   0,   0,SAD4,SAD5,   0,   0,SAD7,SAD6,   0,   0,SAD12,SAD13,    0,    0,SAD15,SAD14]
        let diff1or = _mm256_or_si256(diff1outer, _mm256_bslli_epi128::<2>(diff1inner));
        let diff2or = _mm256_bslli_epi128::<4>(_mm256_or_si256(
            diff2outer,
            _mm256_bslli_epi128::<2>(diff2inner),
        ));
        // [0,1,4,5,3,2,7,6,8,9,12,13,10,11,15,14]
        let diff12 = _mm256_or_si256(diff1or, diff2or);
        // [0,1,4,5,2,3,6,7,8,9,12,13,11,10,14,15]
        let diff12shuffle = _mm256_shufflehi_epi16::<0b10110001>(diff12);
        // final shuffle
        let cdiff_v = _mm256_shuffle_epi32::<0b11011000>(diff12shuffle);

        let dist_v: __m256i = if USE_MASK {
            let mut k = mask_num;
            let m: [i16; 16] = array::from_fn(|i| {
                if k == 0 {
                    0
                } else if k > 0 {
                    k -= 1;
                    -1
                } else if (16 - i) <= k.unsigned_abs() as usize {
                    -1
                } else {
                    0
                }
            });
            let d = _mm256_add_epi16(_mm256_loadu_si256(p_dist_lut.cast::<__m256i>()), cdiff_v);
            _mm256_or_si256(
                d,
                _mm256_set_epi16(
                    // m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11],
                    // m[12], m[13], m[14], m[15],
                    m[15], m[14], m[13], m[12], m[11], m[10], m[9], m[8], m[7], m[6], m[5], m[4],
                    m[3], m[2], m[1], m[0],
                ),
            )
        } else {
            // There is unaligned load from the distance LUT...
            _mm256_add_epi16(_mm256_loadu_si256(p_dist_lut.cast::<__m256i>()), cdiff_v)
        };
        let old_min_dist_v = _mm256_load_si256(min_dist);
        let store_v = _mm256_min_epu16(old_min_dist_v, dist_v);
        _mm256_store_si256(min_dist.cast_mut(), store_v);
        let mask_k = _mm256_cmpeq_epi16(old_min_dist_v, store_v);

        let old_assign_v = _mm256_load_si256(assign);

        let store_v = _mm256_blendv_epi8(
            _mm256_set1_epi16(cluster_number as i16),
            old_assign_v,
            mask_k,
        );
        // SAFETY: At the top of the assign_cluster function is a note about this.
        //  Basically on x86 all operations have ordering so the change makes its way to other
        //  threads eventually and if it is cache-aligned always consistently even without
        //  atomics. On other platforms this can be problematic...
        //  Also, the threading schema ensures, that no collisions occurs, which can be a pretty
        //  big overhead when false sharing occurs.
        _mm256_store_si256(assign.cast_mut(), store_v);
    }

    #[inline(always)]
    unsafe fn assign_primitive_avx2_half<const USE_MASK: bool>(
        p_image_v: *const __m256i,
        p_dist_lut: *const u16,
        min_dist: *const __m128i,
        assign: *const __m128i,
        color_i32: i32,
        cluster_number: u16,
        mask_num: i8,
    ) {
        let image1 = _mm256_load_si256(p_image_v);

        let color_vec = _mm256_set1_epi32(color_i32);

        // mpsadbw
        // IMM8[4:3] - setting quadruplets in first 128bits in b
        // IMM8[1:0] - setting quadruplets in second 128bits in b
        // IMM8[2] - setting quadruplets in second 128bits in b
        // IMM8[5] - setting quadruplets in second 128bits in b
        //
        // inner two SADs of quadruplets in 128 lanes [SAD1,0,0,0,SAD2,0,0,0,SAD5,0,0,0,SAD6,0,0,0]
        let diff1inner = _mm256_and_si256(
            _mm256_mpsadbw_epu8::<0b100100>(image1, color_vec),
            _mm256_set1_epi64x(0x000000000000FFFFu64 as i64),
        );
        // SADs of 8xu8 [SAD0+1,0,0,0,SAD2+3,0,0,0,SAD4+5,0,0,0,SAD6+7,0,0,0]
        let diff1sum = _mm256_sad_epu8(image1, color_vec);
        // outer SADs [SAD0,0,0,0,SAD3,0,0,0,SAD4,0,0,0,SAD7,0,0,0]
        let diff1outer = _mm256_sub_epi16(diff1sum, diff1inner);
        // all SADs, but needs a shuffle in high 64bits of 128bit lanes
        // Now we have (8 SADs)
        // [SAD0,SAD1,   0,   0,SAD3,SAD2,   0,   0,SAD4,SAD5,    0,    0,SAD7,SAD6,    0,    0]
        let diff1 = _mm256_or_si256(diff1outer, _mm256_bslli_epi128::<2>(diff1inner));
        // [SAD0,SAD1,0,0,SAD2,SAD3,0,0,SAD4,SAD5,0,0,SAD6,SAD7,0,0]
        let diff1shuffle1 = _mm256_shufflehi_epi16::<0b11110001>(diff1);
        // [0,0,SAD0,SAD1,SAD2,SAD3,0,0,0,0,SAD4,SAD5,SAD6,SAD7,0,0]
        let diff1shuffle2 = _mm256_shufflelo_epi16::<0b01001111>(diff1shuffle1);

        // extract high and low 128 bits
        // [0,0,SAD0,SAD1,SAD2,SAD3,0,0]
        let diff1high = _mm256_extracti128_si256::<0b0>(diff1shuffle2);
        // [0,0,SAD4,SAD5,SAD6,SAD7,0,0]
        let diff1low = _mm256_extracti128_si256::<0b1>(diff1shuffle2);
        // [SAD0,SAD1,SAD2,SAD3,0,0,0,0]
        let diff1high_shift = _mm_bsrli_si128::<4>(diff1high);
        // [0,0,0,0,SAD4,SAD5,SAD6,SAD7]
        let diff1low_shift = _mm_bslli_si128::<4>(diff1low);
        // OR together
        let cdiff_v = _mm_or_si128(diff1high_shift, diff1low_shift);

        let dist_v: __m128i = if USE_MASK {
            let mut k = mask_num;
            let m: [i16; 8] = array::from_fn(|i| {
                if k == 0 {
                    0
                } else if k > 0 {
                    k -= 1;
                    -1
                } else if (8 - i) <= k.unsigned_abs() as usize {
                    -1
                } else {
                    0
                }
            });
            let d = _mm_add_epi16(_mm_loadu_si128(p_dist_lut.cast::<__m128i>()), cdiff_v);
            _mm_or_si128(
                d,
                _mm_set_epi16(m[7], m[6], m[5], m[4], m[3], m[2], m[1], m[0]),
            )
        } else {
            // There is unaligned load from the distance LUT...
            _mm_add_epi16(_mm_loadu_si128(p_dist_lut.cast::<__m128i>()), cdiff_v)
        };
        let old_min_dist_v = _mm_load_si128(min_dist);
        let store_v = _mm_min_epu16(old_min_dist_v, dist_v);
        _mm_store_si128(min_dist.cast_mut(), store_v);
        let mask_k = _mm_cmpeq_epi16(old_min_dist_v, store_v);

        let old_assign_v = _mm_load_si128(assign);

        let store_v = _mm_blendv_epi8(_mm_set1_epi16(cluster_number as i16), old_assign_v, mask_k);
        // SAFETY: At the top of the assign_cluster function is a note about this.
        //  Basically on x86 all operations have ordering so the change makes its way to other
        //  threads eventually and if it is cache-aligned always consistently even without
        //  atomics. On other platforms this can be problematic...
        //  Also, the threading schema ensures, that no collisions occurs, which can be a pretty
        //  big overhead when false sharing occurs.
        _mm_store_si128(assign.cast_mut(), store_v);
    }
}

#[inline(always)]
fn subsampling_tile_start(
    subsample_global_start: usize,
    stride: usize,
    tile_start: usize,
) -> (usize, usize) {
    let mut subsampling_global_tile_start =
        (tile_start + subsample_global_start).div_ceil(stride) * stride + subsample_global_start;
    let mut subsampling_local_tile_start = subsampling_global_tile_start - tile_start;
    if subsampling_local_tile_start >= stride {
        subsampling_local_tile_start %= stride;
        subsampling_global_tile_start = subsampling_local_tile_start + tile_start;
    }

    debug_assert_eq!(
        subsampling_global_tile_start % stride,
        subsample_global_start,
        "{subsampling_global_tile_start} % {stride} == {subsample_global_start}"
    );
    debug_assert!(
        subsampling_local_tile_start < stride,
        "{subsampling_local_tile_start} < {stride}"
    );
    debug_assert!(
        subsampling_global_tile_start >= tile_start,
        "{subsampling_global_tile_start} > {tile_start}"
    );
    debug_assert!(
        subsampling_global_tile_start >= subsampling_local_tile_start,
        "{subsampling_global_tile_start} >= {subsampling_local_tile_start}"
    );

    (subsampling_global_tile_start, subsampling_local_tile_start)
}

#[cfg(test)]
mod tests {
    use crate::arrays::LABImage;
    use crate::assign::assign;
    use crate::atomic_arrays::AtomicArray2D;
    use crate::common::{AssignSweepBy, AssignThreadingStrategy, Config, DistanceMetric};
    use crate::slic::{compute_spatial_path, Clusters};
    use std::sync::atomic::{AtomicU16, Ordering};
    use std::{panic, process};

    #[test]
    fn assign_test() {
        let orig_hook = panic::take_hook();
        panic::set_hook(Box::new(move |panic_info| {
            // invoke the default handler and exit the process
            orig_hook(panic_info);
            process::exit(1);
        }));
        // This can be a very slow test on slower HW...
        let mut config = Config::default();
        config.subsample_stride = 1;
        for width in (200..1920usize).step_by(9) {
            for height in (200..1080usize).step_by(9) {
                if 3 * width < height || 3 * height < width {
                    continue;
                }
                config.num_of_clusters = (width * height / 1000) as u16;
                let search_region_size =
                    ((width * height) as f32 / config.num_of_clusters as f32).sqrt() as u16;
                println!("Testing size: {width}x{height}, search_region_size: {search_region_size}, num. of clusters: {}", config.num_of_clusters);
                let image = LABImage::from_srgb(&vec![0u8; width * height * 3], width, height);
                let mut clusters = Clusters::initialize_clusters(&image, &config);
                let mut min_distances: AtomicArray2D<AtomicU16> =
                    AtomicArray2D::from_fill(0, width, height);
                let spatial_distance_lut = compute_spatial_path(&config, &search_region_size);
                assign(
                    &image,
                    &config,
                    &mut clusters,
                    &mut min_distances,
                    &spatial_distance_lut,
                    search_region_size,
                    0,
                );
                let unassigned: Vec<usize> = clusters
                    .assignments
                    .data
                    .iter()
                    .enumerate()
                    .filter(|(_i, x)| x.load(Ordering::Relaxed) == 0xFFFF)
                    .map(|(i, _x)| i)
                    .collect();
                if !unassigned.is_empty() {
                    crate::slic::save_assignment_as_image(
                        &clusters.assignments,
                        config.num_of_clusters,
                        "test-assign.png",
                        false,
                    );
                }
                assert!(
                    unassigned.is_empty(),
                    "Unassigned pixels: {:?}",
                    unassigned
                        .iter()
                        .map(|ind| { clusters.assignments.get_x_y_index(*ind) })
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn assign_implementations_test() {
        let dimg = image::open("test/data/aerial.jpg").unwrap();
        let img = dimg.as_rgb8().unwrap().as_raw();
        let width = dimg.width() as usize;
        let height = dimg.height() as usize;
        let conv_img = LABImage::from_srgb(img.as_slice(), width, height);
        let mut config = Config::default();
        config.distance_metric = DistanceMetric::Manhattan;
        config.subsample_stride = 3;
        let search_region_size = ((conv_img.width * conv_img.height) as f32
            / config.num_of_clusters as f32)
            .sqrt() as u16;
        let spatial_distance_lut = compute_spatial_path(&config, &search_region_size);

        let mut min_distances_ref =
            AtomicArray2D::from_fill(0xFFFFu16, conv_img.width, conv_img.height);
        let mut clusters_ref = Clusters::initialize_clusters(&conv_img, &config);
        config.assign_threading_strategy = AssignThreadingStrategy::SingleThread;
        config.assign_sweep_by = AssignSweepBy::Cluster;
        let subsample_rem = 0;
        assign(
            &conv_img,
            &config,
            &mut clusters_ref,
            &mut min_distances_ref,
            &spatial_distance_lut,
            search_region_size,
            subsample_rem,
        );

        let mut min_distances_1 =
            AtomicArray2D::from_fill(0xFFFFu16, conv_img.width, conv_img.height);
        let mut clusters_1 = Clusters::initialize_clusters(&conv_img, &config);
        config.assign_threading_strategy = AssignThreadingStrategy::FineGrained;
        config.assign_sweep_by = AssignSweepBy::Auto;
        let subsample_rem = 0;
        assign(
            &conv_img,
            &config,
            &mut clusters_1,
            &mut min_distances_1,
            &spatial_distance_lut,
            search_region_size,
            subsample_rem,
        );

        let mut min_distances_2 =
            AtomicArray2D::from_fill(0xFFFFu16, conv_img.width, conv_img.height);
        let mut clusters_2 = Clusters::initialize_clusters(&conv_img, &config);
        config.assign_threading_strategy = AssignThreadingStrategy::CoreDistributed;
        config.assign_sweep_by = AssignSweepBy::Auto;
        let subsample_rem = 0;
        assign(
            &conv_img,
            &config,
            &mut clusters_2,
            &mut min_distances_2,
            &spatial_distance_lut,
            search_region_size,
            subsample_rem,
        );

        let mut min_distances_3 =
            AtomicArray2D::from_fill(0xFFFFu16, conv_img.width, conv_img.height);
        let mut clusters_3 = Clusters::initialize_clusters(&conv_img, &config);
        config.assign_threading_strategy = AssignThreadingStrategy::RowBased;
        config.assign_sweep_by = AssignSweepBy::Row;
        let subsample_rem = 0;
        assign(
            &conv_img,
            &config,
            &mut clusters_3,
            &mut min_distances_3,
            &spatial_distance_lut,
            search_region_size,
            subsample_rem,
        );

        crate::slic::save_assignment_as_image(
            &clusters_ref.assignments,
            config.num_of_clusters,
            "assign-single-thread.png",
            true,
        );
        crate::slic::save_assignment_as_image(
            &min_distances_ref,
            0x00FF,
            "min_dists-single-thread.png",
            false,
        );

        crate::slic::save_assignment_as_image(
            &clusters_1.assignments,
            config.num_of_clusters,
            "assign-fine-grained.png",
            true,
        );
        crate::slic::save_assignment_as_image(
            &min_distances_1,
            0x00FF,
            "min_dists-fine-grained.png",
            false,
        );

        crate::slic::save_assignment_as_image(
            &clusters_2.assignments,
            config.num_of_clusters,
            "assign-distributed.png",
            true,
        );
        crate::slic::save_assignment_as_image(
            &min_distances_2,
            0x00FF,
            "min_dists-distributed.png",
            false,
        );

        crate::slic::save_assignment_as_image(
            &clusters_3.assignments,
            config.num_of_clusters,
            "assign-row-based.png",
            true,
        );
        crate::slic::save_assignment_as_image(
            &min_distances_3,
            0x00FF,
            "min_dists-row-based.png",
            false,
        );

        for (pixel_1, pixel_2) in min_distances_1.data.iter().zip(min_distances_2.data.iter()) {
            assert_eq!(
                pixel_1.load(Ordering::Relaxed),
                pixel_2.load(Ordering::Relaxed)
            )
        }
        for (pixel_ref, pixel_1) in min_distances_ref
            .data
            .iter()
            .zip(min_distances_1.data.iter())
        {
            assert_eq!(
                pixel_ref.load(Ordering::Relaxed),
                pixel_1.load(Ordering::Relaxed)
            )
        }
        for (pixel_ref, pixel_2) in min_distances_ref
            .data
            .iter()
            .zip(min_distances_2.data.iter())
        {
            assert_eq!(
                pixel_ref.load(Ordering::Relaxed),
                pixel_2.load(Ordering::Relaxed)
            )
        }
        for (pixel_ref, pixel_3) in min_distances_ref
            .data
            .iter()
            .zip(min_distances_3.data.iter())
        {
            assert_eq!(
                pixel_ref.load(Ordering::Relaxed),
                pixel_3.load(Ordering::Relaxed)
            )
        }
        for (pixel_ref, pixel_3) in clusters_ref
            .assignments
            .data
            .iter()
            .zip(clusters_3.assignments.data.iter())
        {
            assert_eq!(
                pixel_ref.load(Ordering::Relaxed),
                pixel_3.load(Ordering::Relaxed)
            )
        }
    }
}
