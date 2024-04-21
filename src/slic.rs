use crate::arrays::{Array2D, LABImage};
use crate::assign::{assign, fused_assign_update};
use crate::atomic_arrays::AtomicArray2D;
use crate::cielab::tables::OUTPUT_SHIFT;
use crate::cluster::Cluster;
use crate::common::{AssignThreadingStrategy, Config, DistanceMetric};
use crate::conectivity::enforce_connectivity;
use multiversion::multiversion;
use rayon::current_num_threads;
use std::array;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};

/// Convenient struct for passing values around.
pub struct Clusters {
    /// For every pixel in image this stores to which cluster it belongs (see `Cluster.number`).
    pub assignments: AtomicArray2D<AtomicU16>,
    pub clusters: Vec<Cluster>,
}

impl Clusters {
    /// Default initialize clusters function.
    ///
    /// For custom implementations the needed filled fields in new cluster are `x`, `y`, `l`, `a`,
    /// `b` and unique `number` used as an identification.
    pub fn initialize_clusters(image: &LABImage, config: &Config) -> Clusters {
        assert!(config.num_of_clusters > 1);
        let mut clusters = Clusters {
            assignments: AtomicArray2D::from_fill(0xFFFFu16, image.width, image.height),
            clusters: Vec::with_capacity(config.num_of_clusters as usize),
        };
        let n_y = (config.num_of_clusters as f32).sqrt() as u16;
        let mut n_xs: Vec<u16> = vec![config.num_of_clusters / n_y; n_y as usize];
        let mut remainder = config.num_of_clusters % n_y;
        let mut row = 0;
        while remainder > 0 {
            n_xs[row] += 1;
            row += 2;
            if row >= n_y as usize {
                row = 1;
            }
            remainder -= 1;
        }
        let h = image.height.div_ceil(n_y as usize);
        let mut acc_k: usize = 0;
        for i in (0..image.height).step_by(h) {
            let w = image
                .width
                .div_ceil(n_xs[std::cmp::min(i / h, (n_y - 1) as usize)] as usize);
            for j in (0..image.width).step_by(w) {
                if acc_k >= config.num_of_clusters as usize {
                    break;
                }
                let center_y = (i + h / 2).clamp(0, image.height - 1) as u16;
                let center_x = (j + w / 2).clamp(0, image.width - 1) as u16;
                let p = image.get_pixel(center_x as usize, center_y as usize);
                clusters.clusters.push(Cluster {
                    x: center_x,
                    y: center_y,
                    l: p[0],
                    a: p[1],
                    b: p[2],
                    number: acc_k as u16,
                    ..Cluster::default()
                });
                acc_k += 1;
            }
        }
        while acc_k < config.num_of_clusters as usize {
            let center_y = image.height as u16 / 2;
            let center_x = image.width as u16 / 2;
            let p = image.get_pixel(center_x as usize, center_y as usize);
            clusters.clusters.push(Cluster {
                x: center_x,
                y: center_y,
                l: p[0],
                a: p[1],
                b: p[2],
                number: acc_k as u16,
                ..Cluster::default()
            });
            acc_k += 1;
        }
        debug_assert_eq!(clusters.clusters.len(), config.num_of_clusters as usize);
        clusters
    }
}

#[cfg(test)]
pub(crate) fn save_assignment_as_image(
    assignments: &AtomicArray2D<AtomicU16>,
    max_assign: u16,
    path: &str,
    imgl8: bool,
) {
    use image::{save_buffer, ColorType};
    let buf: Vec<u16> = assignments
        .data
        .iter()
        .map(|x| x.load(Ordering::Relaxed))
        .collect();
    println!(
        "avg: {}",
        buf.iter().map(|x| *x as u64).sum::<u64>() / (buf.len() as u64)
    );
    if imgl8 {
        let buf_u8: Vec<u8> = buf.iter().map(|x| *x as u8).collect();
        save_buffer(
            path,
            buf_u8.as_slice(),
            assignments.width as u32,
            assignments.height as u32,
            ColorType::L8,
        )
        .unwrap()
    } else {
        let buf_u8: Vec<u8> = buf
            .iter()
            .map(|x| {
                let norm = (*x as f64) / max_assign as f64;
                let v = (norm * u16::MAX as f64) as u16;
                let b = v.to_le_bytes();
                [b[0], b[1]]
            })
            .flatten()
            .collect();
        save_buffer(
            path,
            buf_u8.as_slice(),
            assignments.width as u32,
            assignments.height as u32,
            ColorType::L16,
        )
        .unwrap()
    }
}

/// This function is the main loop.
///
/// The steps are generally:
/// - N iterations
///     - assign
///     - update
/// - full assign (_subsample stride_ = 1)
/// - update
/// - enforce_connectivity (CCA)
///
/// The subsample_start increments in every iteration, so the rows of the image is every
/// iteration different (after `subsample_stride` iterations it starts over).
/// Without this row subsampling does not work.
pub fn iterate(image: &LABImage, config: &Config, clusters: &mut Clusters) {
    let search_region_size =
        ((image.width * image.height) as f32 / config.num_of_clusters as f32).sqrt() as u16;
    let spatial_distance_lut = compute_spatial_path(config, &search_region_size);
    let mut min_distances = AtomicArray2D::from_fill(0xFFFFu16, image.width, image.height);
    let mut subsaple_start = 0;
    #[allow(unused)]
    for i in 0..config.max_iterations {
        if config.assign_threading_strategy == AssignThreadingStrategy::RowBasedFusedUpdate {
            fused_assign_update(
                image,
                config,
                clusters,
                &mut min_distances,
                &spatial_distance_lut,
                search_region_size,
                subsaple_start,
            );
        } else {
            assign(
                image,
                config,
                clusters,
                &mut min_distances,
                &spatial_distance_lut,
                search_region_size,
                subsaple_start,
            );
            update(clusters, image, config, subsaple_start);
        }
        #[cfg(test)]
        save_assignment_as_image(
            &clusters.assignments,
            config.num_of_clusters,
            format!("iter-{i}.png").as_str(),
            false,
        );
        #[cfg(test)]
        save_assignment_as_image(
            &min_distances,
            0x00FF,
            format!("min_dists-{i}.png").as_str(),
            false,
        );
        subsaple_start = (subsaple_start + 1) % config.subsample_stride;
    }
    let mut no_subsample_config: Config = config.clone();
    no_subsample_config.subsample_stride = 1;
    assign(
        image,
        &no_subsample_config,
        clusters,
        &mut min_distances,
        &spatial_distance_lut,
        search_region_size,
        0,
    );
    #[cfg(test)]
    save_assignment_as_image(
        &clusters.assignments,
        config.num_of_clusters,
        format!("iter-final.png").as_str(),
        true,
    );
    #[cfg(test)]
    save_assignment_as_image(
        &min_distances,
        0x00FF,
        format!("min_dists-final.png").as_str(),
        false,
    );
    #[cfg(test)]
    save_assignment_as_image(
        &clusters.assignments,
        config.num_of_clusters,
        format!("iter-final-f.png").as_str(),
        false,
    );

    enforce_connectivity(clusters, image, config, search_region_size);
    #[cfg(test)]
    save_assignment_as_image(
        &clusters.assignments,
        config.num_of_clusters,
        format!("iter-cca.png").as_str(),
        true,
    );
    #[cfg(test)]
    save_assignment_as_image(
        &clusters.assignments,
        config.num_of_clusters,
        format!("iter-cca-f.png").as_str(),
        false,
    );
}

/// This function computes LUT for spatial distances.
pub fn compute_spatial_path(config: &Config, search_region_size: &u16) -> Array2D<u16> {
    let coef =
        (1f32 / (*search_region_size as f32 / config.compactness)) * (1 << OUTPUT_SHIFT) as f32;
    let search_region_size_2 = 2 * search_region_size + 1;
    let lut_size = search_region_size_2 as usize;
    let mut spatial_distance_lut: Array2D<u16> = Array2D::from_fill(0xFFFFu16, lut_size, lut_size);
    match config.distance_metric {
        DistanceMetric::Manhattan => {
            for i in 0..search_region_size_2 {
                for j in 0..search_region_size_2 {
                    spatial_distance_lut[(i as usize, j as usize)] = (coef
                        * (search_region_size.abs_diff(i) + search_region_size.abs_diff(j)) as f32)
                        as u16;
                }
            }
        }
        DistanceMetric::RealDistManhattanColor => {
            for i in 0..search_region_size_2 {
                for j in 0..search_region_size_2 {
                    spatial_distance_lut[(i as usize, j as usize)] = (coef
                        * (search_region_size.abs_diff(i) as f32)
                            .hypot(search_region_size.abs_diff(j) as f32))
                        as u16;
                }
            }
        }
    }
    spatial_distance_lut
}

/// This function does the update step.
///
/// Instead of median, average is used for the performance.
#[multiversion(targets = "simd")]
pub fn update(clusters: &mut Clusters, image: &LABImage, config: &Config, subsample_start: u8) {
    let num_cluster_members: Vec<AtomicU32> =
        Vec::from_iter((0..config.num_of_clusters).map(|_| AtomicU32::new(0)));
    let cluster_acc_vec: Vec<[AtomicU32; 5]> =
        Vec::from_iter((0..config.num_of_clusters).map(|_| array::from_fn(|_| AtomicU32::new(0))));
    let num_threads = current_num_threads();
    let rows_v: Vec<usize> = (0..image.height)
        .skip(subsample_start as usize)
        .step_by(config.subsample_stride as usize)
        .collect();
    let chunk_size = image.height.div_ceil(num_threads);

    fn update_part_generic(
        rows: &[usize],
        image: &LABImage,
        assignments: &AtomicArray2D<AtomicU16>,
        config: &Config,
        num_cluster_members: &[AtomicU32],
        cluster_acc_vec: &[[AtomicU32; 5]],
    ) {
        let mut num_cluster_members_local: Vec<u32> = vec![0; config.num_of_clusters as usize];
        let mut cluster_acc_local: Vec<[u32; 5]> =
            vec![[0, 0, 0, 0, 0]; config.num_of_clusters as usize];
        for row in rows {
            let image_row = image.get_row(*row);
            let assignments_row = assignments.get_row(*row);
            for (column, (pixel, assignment)) in
                image_row.chunks_exact(4).zip(assignments_row).enumerate()
            {
                let cluster_n = assignment.load(Ordering::Relaxed);
                if cluster_n == 0xFFFF {
                    continue;
                }
                num_cluster_members_local[cluster_n as usize] += 1;
                cluster_acc_local[cluster_n as usize][0] += (*row) as u32;
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
    }

    rayon::scope(|s| {
        let assignments = &clusters.assignments;
        for rows in rows_v.chunks(chunk_size) {
            s.spawn(|_| {
                update_part_generic(
                    rows,
                    image,
                    assignments,
                    config,
                    &num_cluster_members,
                    &cluster_acc_vec,
                )
            })
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

#[cfg(test)]
mod tests {
    use super::{iterate, Clusters};
    use crate::arrays::LABImage;
    use crate::common::{AssignThreadingStrategy, Config, DistanceMetric};
    #[test]
    fn clusters_test() {
        // NOTE: The threading scheme introduces small differences, because clusters are not assigned
        //  in the same order. When the distance of the pixel from two clusters is equal, then
        //  the cluster which claimed the pixel first can be different on the tile boundaries.
        //  Due to this pixel-perfect assignments are not possible without doing boundaries
        //  single-threaded.
        //  The minimal distances must be the same for all threading schemas, and we can test it.
        //  This is in a test assign_implementations_test. This is a test for development, where
        //  before doing any changes you save your output and then compare to it.
        let dimg = image::open("test/data/aerial.jpg").unwrap();
        let img = dimg.as_rgb8().unwrap().as_raw();
        let width = dimg.width() as usize;
        let height = dimg.height() as usize;
        let conv_img = LABImage::from_srgb(img.as_slice(), width, height);
        let mut config = Config::default();
        config.distance_metric = DistanceMetric::Manhattan;
        config.subsample_stride = 5;
        config.assign_threading_strategy = AssignThreadingStrategy::RowBased;
        let mut clusters = Clusters::initialize_clusters(&conv_img, &config);
        iterate(&conv_img, &config, &mut clusters);
        let dimg_ref = image::open("iter-cca-ref.png").unwrap();
        let img_ref = dimg_ref.as_luma8().unwrap().as_raw();
        let dimg_test = image::open("iter-cca.png").unwrap();
        let img_test = dimg_test.as_luma8().unwrap().as_raw();
        for (p_ref, p_test) in img_ref.iter().zip(img_test) {
            debug_assert_eq!(p_ref, p_test);
        }
    }
}
