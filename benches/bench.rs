use criterion::*;
use fast_slic_rust::arrays::LABImage;
use fast_slic_rust::assign::{assign, fused_assign_update};
use fast_slic_rust::atomic_arrays::AtomicArray2D;
use fast_slic_rust::common::{AssignSweepBy, AssignThreadingStrategy, Config};
use fast_slic_rust::conectivity::{assign_disjoint_set, enforce_connectivity};
use fast_slic_rust::slic::{compute_spatial_path, iterate, update, Clusters};
use image::imageops::FilterType;
use std::time::Duration;

fn bench_lab_image_from_rgb(c: &mut Criterion) {
    sas::init();
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    c.bench_function("rgb_to_lab", |b| {
        b.iter(|| {
            let _ = black_box(LABImage::from_srgb(img.as_slice(), width, height));
        });
    });
}

fn bench_assign(c: &mut Criterion) {
    sas::init();
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let search_region_size =
        ((image.width * image.height) as f32 / config.num_of_clusters as f32).sqrt() as u16;
    let spatial_distance_lut = compute_spatial_path(&config, &search_region_size);
    let mut min_distances = AtomicArray2D::from_fill(0xFFFFu16, image.width, image.height);
    let subsample_rem = 0;
    let mut group = c.benchmark_group("SLIC assign");
    let subsample_strides = 1..6;
    let threading_strategies = [
        AssignThreadingStrategy::SingleThread,
        AssignThreadingStrategy::FineGrained,
        AssignThreadingStrategy::CoreDistributed,
        AssignThreadingStrategy::RowBased,
        AssignThreadingStrategy::RowBasedFusedUpdate,
    ];
    let assign_sweeps = [AssignSweepBy::Row, AssignSweepBy::Cluster];
    for threading_strategy in threading_strategies {
        for assign_sweep in assign_sweeps {
            if assign_sweep == AssignSweepBy::Cluster {
                if (threading_strategy == AssignThreadingStrategy::RowBased
                    || threading_strategy == AssignThreadingStrategy::RowBasedFusedUpdate)
                    && assign_sweep == AssignSweepBy::Cluster
                {
                    continue;
                }
                group.bench_with_input(
                    BenchmarkId::new(
                        "assign",
                        format!("{:?}::{:?}", threading_strategy, assign_sweep),
                    ),
                    &(threading_strategy, assign_sweep),
                    |b, &(threading_strategy, assign_sweep)| {
                        config.subsample_stride = 1;
                        config.assign_threading_strategy = threading_strategy;
                        config.assign_sweep_by = assign_sweep;
                        b.iter(|| {
                            let _ = black_box(assign(
                                &image,
                                &config,
                                &mut clusters,
                                &mut min_distances,
                                &spatial_distance_lut,
                                search_region_size,
                                subsample_rem,
                            ));
                        });
                    },
                );
            } else if threading_strategy == AssignThreadingStrategy::RowBasedFusedUpdate {
                group.bench_with_input(
                    BenchmarkId::new(
                        "assign",
                        format!("{:?}::{:?}", threading_strategy, assign_sweep),
                    ),
                    &(threading_strategy, assign_sweep),
                    |b, &(threading_strategy, assign_sweep)| {
                        config.subsample_stride = 1;
                        config.assign_threading_strategy = threading_strategy;
                        config.assign_sweep_by = assign_sweep;
                        b.iter(|| {
                            let _ = black_box(fused_assign_update(
                                &image,
                                &config,
                                &mut clusters,
                                &mut min_distances,
                                &spatial_distance_lut,
                                search_region_size,
                                subsample_rem,
                            ));
                        });
                    },
                );
            } else {
                for subsample_stride in subsample_strides.clone() {
                    group.bench_with_input(
                        BenchmarkId::new(
                            "assign",
                            format!(
                                "{:?}::{:?}::stride={subsample_stride}",
                                threading_strategy, assign_sweep
                            ),
                        ),
                        &(threading_strategy, assign_sweep, subsample_stride),
                        |b, &(threading_strategy, assign_sweep, subsample_stride)| {
                            config.subsample_stride = subsample_stride;
                            config.assign_threading_strategy = threading_strategy;
                            config.assign_sweep_by = assign_sweep;
                            b.iter(|| {
                                let _ = black_box(assign(
                                    &image,
                                    &config,
                                    &mut clusters,
                                    &mut min_distances,
                                    &spatial_distance_lut,
                                    search_region_size,
                                    subsample_rem,
                                ));
                            });
                        },
                    );
                }
            }
        }
    }
}

fn bench_update(c: &mut Criterion) {
    sas::init();
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.subsample_stride = 1;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let search_region_size =
        ((image.width * image.height) as f32 / config.num_of_clusters as f32).sqrt() as u16;
    let spatial_distance_lut = compute_spatial_path(&config, &search_region_size);
    let mut min_distances = AtomicArray2D::from_fill(0xFFFFu16, image.width, image.height);
    let subsample_rem = 0;
    assign(
        &image,
        &config,
        &mut clusters,
        &mut min_distances,
        &spatial_distance_lut,
        search_region_size,
        subsample_rem,
    );
    c.bench_function("update", |b| {
        b.iter(|| {
            let _ = black_box(update(&mut clusters, &image, &config, subsample_rem));
        });
    });
}

fn bench_connectivity(c: &mut Criterion) {
    sas::init();
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.subsample_stride = 1;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let search_region_size =
        ((image.width * image.height) as f32 / config.num_of_clusters as f32).sqrt() as u16;
    let spatial_distance_lut = compute_spatial_path(&config, &search_region_size);
    let mut min_distances = AtomicArray2D::from_fill(0xFFFFu16, image.width, image.height);
    let subsample_rem = 0;
    for _ in 0..11 {
        assign(
            &image,
            &config,
            &mut clusters,
            &mut min_distances,
            &spatial_distance_lut,
            search_region_size,
            subsample_rem,
        );
        update(&mut clusters, &image, &config, subsample_rem);
    }
    c.bench_function("enforce_connectivity", |b| {
        b.iter(|| {
            let _ = black_box(enforce_connectivity(
                &mut clusters,
                &image,
                &config,
                search_region_size,
            ));
        });
    });
}

fn bench_assign_disjoint_set(c: &mut Criterion) {
    sas::init();
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.subsample_stride = 1;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let search_region_size =
        ((image.width * image.height) as f32 / config.num_of_clusters as f32).sqrt() as u16;
    let spatial_distance_lut = compute_spatial_path(&config, &search_region_size);
    let mut min_distances = AtomicArray2D::from_fill(0xFFFFu16, image.width, image.height);
    let subsample_rem = 0;
    for _ in 0..11 {
        assign(
            &image,
            &config,
            &mut clusters,
            &mut min_distances,
            &spatial_distance_lut,
            search_region_size,
            subsample_rem,
        );
        update(&mut clusters, &image, &config, subsample_rem);
    }
    c.bench_function("assign_disjoint_set", |b| {
        b.iter(|| {
            let _ = black_box(assign_disjoint_set(&clusters.assignments));
        });
    });
}

fn bench_disjoint_set_flatten(c: &mut Criterion) {
    sas::init();
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.subsample_stride = 1;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let search_region_size =
        ((image.width * image.height) as f32 / config.num_of_clusters as f32).sqrt() as u16;
    let spatial_distance_lut = compute_spatial_path(&config, &search_region_size);
    let mut min_distances = AtomicArray2D::from_fill(0xFFFFu16, image.width, image.height);
    let subsample_rem = 0;
    for _ in 0..11 {
        assign(
            &image,
            &config,
            &mut clusters,
            &mut min_distances,
            &spatial_distance_lut,
            search_region_size,
            subsample_rem,
        );
        update(&mut clusters, &image, &config, subsample_rem);
    }
    c.bench_function("assign_disjoint_set_flatten", |b| {
        b.iter(|| {
            let _ = black_box(assign_disjoint_set(&clusters.assignments).flatten());
        });
    });
}

fn bench_slic_image(c: &mut Criterion) {
    sas::init();
    let dimg = image::open("test/data/aerial.jpg").unwrap();
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.assign_threading_strategy = AssignThreadingStrategy::SingleThread;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let mut group = c.benchmark_group("SLIC iterate 4k");
    let threading_strategies = [
        AssignThreadingStrategy::SingleThread,
        AssignThreadingStrategy::FineGrained,
        AssignThreadingStrategy::CoreDistributed,
        AssignThreadingStrategy::RowBased,
        AssignThreadingStrategy::RowBasedFusedUpdate,
    ];
    let subsample_strides = 1..6;
    for threading_strategy in threading_strategies {
        for subsample_stride in subsample_strides.clone() {
            group.bench_with_input(
                BenchmarkId::new(
                    "bench_slic_image",
                    format!("{:?}-stride={:?}", threading_strategy, subsample_stride),
                ),
                &(threading_strategy, subsample_stride),
                |b, &(threading_strategy, subsample_stride)| {
                    config.subsample_stride = subsample_stride;
                    config.assign_threading_strategy = threading_strategy;
                    b.iter(|| {
                        let _ = black_box(iterate(&image, &config, &mut clusters));
                    });
                },
            );
        }
    }
}

fn bench_slic_image_qhd(c: &mut Criterion) {
    sas::init();
    let dimg_full = image::open("test/data/aerial.jpg").unwrap();
    let dimg = dimg_full
        .crop_imm(0, 0, 3840, 2160)
        .resize(2560, 1440, FilterType::CatmullRom);
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.assign_threading_strategy = AssignThreadingStrategy::SingleThread;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let mut group = c.benchmark_group("SLIC iterate QHD");
    let threading_strategies = [
        AssignThreadingStrategy::SingleThread,
        AssignThreadingStrategy::FineGrained,
        AssignThreadingStrategy::CoreDistributed,
        AssignThreadingStrategy::RowBased,
        AssignThreadingStrategy::RowBasedFusedUpdate,
    ];
    let subsample_strides = 1..6;
    for threading_strategy in threading_strategies {
        for subsample_stride in subsample_strides.clone() {
            group.bench_with_input(
                BenchmarkId::new(
                    "bench_slic_image",
                    format!("{:?}-stride={:?}", threading_strategy, subsample_stride),
                ),
                &(threading_strategy, subsample_stride),
                |b, &(threading_strategy, subsample_stride)| {
                    config.subsample_stride = subsample_stride;
                    config.assign_threading_strategy = threading_strategy;
                    b.iter(|| {
                        let _ = black_box(iterate(&image, &config, &mut clusters));
                    });
                },
            );
        }
    }
}

fn bench_slic_image_fhd(c: &mut Criterion) {
    sas::init();
    let dimg_full = image::open("test/data/aerial.jpg").unwrap();
    let dimg = dimg_full
        .crop_imm(0, 0, 3840, 2160)
        .resize(1920, 1080, FilterType::CatmullRom);
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.assign_threading_strategy = AssignThreadingStrategy::SingleThread;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let mut group = c.benchmark_group("SLIC iterate FHD");
    let threading_strategies = [
        AssignThreadingStrategy::SingleThread,
        AssignThreadingStrategy::FineGrained,
        AssignThreadingStrategy::CoreDistributed,
        AssignThreadingStrategy::RowBased,
        AssignThreadingStrategy::RowBasedFusedUpdate,
    ];
    let subsample_strides = 1..6;
    for threading_strategy in threading_strategies {
        for subsample_stride in subsample_strides.clone() {
            group.bench_with_input(
                BenchmarkId::new(
                    "bench_slic_image",
                    format!("{:?}-stride={:?}", threading_strategy, subsample_stride),
                ),
                &(threading_strategy, subsample_stride),
                |b, &(threading_strategy, subsample_stride)| {
                    config.subsample_stride = subsample_stride;
                    config.assign_threading_strategy = threading_strategy;
                    b.iter(|| {
                        let _ = black_box(iterate(&image, &config, &mut clusters));
                    });
                },
            );
        }
    }
}

fn bench_slic_image_hd(c: &mut Criterion) {
    sas::init();
    let dimg_full = image::open("test/data/aerial.jpg").unwrap();
    let dimg = dimg_full
        .crop_imm(0, 0, 3840, 2160)
        .resize(1280, 720, FilterType::CatmullRom);
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.assign_threading_strategy = AssignThreadingStrategy::SingleThread;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let mut group = c.benchmark_group("SLIC iterate HD");
    let threading_strategies = [
        AssignThreadingStrategy::SingleThread,
        AssignThreadingStrategy::FineGrained,
        AssignThreadingStrategy::CoreDistributed,
        AssignThreadingStrategy::RowBased,
        AssignThreadingStrategy::RowBasedFusedUpdate,
    ];
    let subsample_strides = 1..6;
    for threading_strategy in threading_strategies {
        for subsample_stride in subsample_strides.clone() {
            group.bench_with_input(
                BenchmarkId::new(
                    "bench_slic_image",
                    format!("{:?}-stride={:?}", threading_strategy, subsample_stride),
                ),
                &(threading_strategy, subsample_stride),
                |b, &(threading_strategy, subsample_stride)| {
                    config.subsample_stride = subsample_stride;
                    config.assign_threading_strategy = threading_strategy;
                    b.iter(|| {
                        let _ = black_box(iterate(&image, &config, &mut clusters));
                    });
                },
            );
        }
    }
}

fn bench_slic_image_sd(c: &mut Criterion) {
    sas::init();
    let dimg_full = image::open("test/data/aerial.jpg").unwrap();
    let dimg = dimg_full
        .crop_imm(0, 0, 3840, 2160)
        .resize(960, 540, FilterType::CatmullRom);
    let img = dimg.as_rgb8().unwrap().as_raw();
    let width = dimg.width() as usize;
    let height = dimg.height() as usize;
    let image = LABImage::from_srgb(img.as_slice(), width, height);
    let mut config = Config::default();
    config.assign_threading_strategy = AssignThreadingStrategy::SingleThread;
    let mut clusters = Clusters::initialize_clusters(&image, &config);
    let mut group = c.benchmark_group("SLIC iterate SD");
    let threading_strategies = [
        AssignThreadingStrategy::SingleThread,
        AssignThreadingStrategy::FineGrained,
        AssignThreadingStrategy::CoreDistributed,
        AssignThreadingStrategy::RowBased,
        AssignThreadingStrategy::RowBasedFusedUpdate,
    ];
    let subsample_strides = 1..6;
    for threading_strategy in threading_strategies {
        for subsample_stride in subsample_strides.clone() {
            group.bench_with_input(
                BenchmarkId::new(
                    "bench_slic_image",
                    format!("{:?}-stride={:?}", threading_strategy, subsample_stride),
                ),
                &(threading_strategy, subsample_stride),
                |b, &(threading_strategy, subsample_stride)| {
                    config.subsample_stride = subsample_stride;
                    config.assign_threading_strategy = threading_strategy;
                    b.iter(|| {
                        let _ = black_box(iterate(&image, &config, &mut clusters));
                    });
                },
            );
        }
    }
}

criterion_group!(name = benches;
config = Criterion::default().measurement_time(Duration::from_secs(30)).warm_up_time(Duration::from_secs(10));
targets = bench_lab_image_from_rgb);
criterion_group!(name = benches1;
config = Criterion::default().measurement_time(Duration::from_secs(30)).warm_up_time(Duration::from_secs(10));
targets = bench_assign);
criterion_group!(name = benches2;
config = Criterion::default().measurement_time(Duration::from_secs(30)).warm_up_time(Duration::from_secs(10));
targets = bench_update);
criterion_group!(name = benches3;
config = Criterion::default().measurement_time(Duration::from_secs(30)).warm_up_time(Duration::from_secs(10));
targets = bench_connectivity);
criterion_group!(name = benches4;
config = Criterion::default().measurement_time(Duration::from_secs(30)).warm_up_time(Duration::from_secs(10));
targets = bench_assign_disjoint_set);
criterion_group!(name = benches5;
config = Criterion::default().measurement_time(Duration::from_secs(30)).warm_up_time(Duration::from_secs(10));
targets = bench_disjoint_set_flatten);
criterion_group!(name = benches6;
config = Criterion::default().measurement_time(Duration::from_secs(30)).warm_up_time(Duration::from_secs(10));
targets = bench_slic_image, bench_slic_image_qhd, bench_slic_image_fhd, bench_slic_image_hd, bench_slic_image_sd);
criterion_main!(benches, benches1, benches2, benches3, benches4, benches5, benches6);
