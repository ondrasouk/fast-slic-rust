use crate::arrays::LABImage;
use crate::atomic_arrays::AtomicArray2D;
use crate::common::{split_length_to_ranges, Config};
use crate::slic::Clusters;
use assume::assume;
use multiversion::multiversion;
use rayon::current_num_threads;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};
use std::sync::{Arc, Barrier, RwLock};

#[derive(Debug)]
pub struct ComponentSet {
    num_components: u32, // 0xFFFFFFFF as uninit
    component_assignment: Vec<AtomicU32>,
    num_component_members: Vec<AtomicU32>,
    component_leaders: Vec<AtomicU32>,
}

pub struct DisjointSet {
    parents: Vec<AtomicU32>,
}

impl DisjointSet {
    pub fn new(size: u32) -> Self {
        assert!(size > 0, "Size must be larger than zero.");
        assert!(size < u32::MAX, "Size must be smaller than {}", u32::MAX);
        DisjointSet {
            parents: (0..size).map(AtomicU32::new).collect(),
        }
    }

    #[inline]
    pub fn merge(&self, node_i: u32, node_j: u32) {
        // NOTE: This is probably not the most thread-safe, so keep in mind, that the range between
        //  two nodes should be blocked for other threads, but it would add some overhead to check
        //  this... also as it is used in this code, there can't be any race conditions
        let mut root_x = node_i as usize;
        let mut root_y = node_j as usize;
        let mut parent_x = self.parents[root_x].load(Ordering::Relaxed);
        let mut parent_y = self.parents[root_y].load(Ordering::Relaxed);
        while parent_x != parent_y {
            if parent_x > parent_y {
                // NOTE: assume is there used since this is very hot function, and it yielded 68%
                //  improvement in performance of this function since we can be sure, that every
                //  element in self.parents is lower than the length.
                assume!(unsafe: root_x < self.parents.len(), "root: {root_x} > {}", self.parents.len());
                if root_x as u32 == parent_x {
                    self.parents[root_x].store(parent_y, Ordering::Relaxed);
                    break;
                }
                let z = parent_x as usize;
                parent_x = self.parents[z].load(Ordering::Relaxed);
                assume!(unsafe: (parent_x as usize) < self.parents.len(), "root: {parent_x} > {}", self.parents.len());
                self.parents[root_x].store(parent_y, Ordering::Relaxed);
                root_x = z;
                assume!(unsafe: root_x < self.parents.len(), "root: {root_x} > {}", self.parents.len());
            } else {
                assume!(unsafe: root_y < self.parents.len(), "root: {root_y} > {}", self.parents.len());
                if root_y as u32 == parent_y {
                    self.parents[root_y].store(parent_x, Ordering::Relaxed);
                    break;
                }
                let z = parent_y as usize;
                parent_y = self.parents[z].load(Ordering::Relaxed);
                assume!(unsafe: (parent_y as usize) < self.parents.len(), "root: {parent_y} > {}", self.parents.len());
                self.parents[root_y].store(parent_x, Ordering::Relaxed);
                root_y = z;
                assume!(unsafe: root_y < self.parents.len(), "root: {root_y} > {}", self.parents.len());
            }
        }
    }

    pub fn flatten(&self) -> ComponentSet {
        let result = ComponentSet {
            num_components: 0,
            component_assignment: Vec::from_iter(
                (0..self.parents.len()).map(|_| AtomicU32::new(u32::MAX)),
            ),
            num_component_members: vec![],
            component_leaders: vec![],
        };
        let num_threads = current_num_threads();

        let root_sizes: Arc<Vec<AtomicU32>> =
            Arc::new(Vec::from_iter((0..num_threads).map(|_| AtomicU32::new(0))));

        let parent_splits = split_length_to_ranges(self.parents.len(), num_threads);

        let barrier = Arc::new(Barrier::new(num_threads));

        let result_ref = Arc::new(RwLock::new(result));
        rayon::scope(|s| {
            for (i, parent_range) in parent_splits.into_iter().enumerate() {
                let barrier_c = Arc::clone(&barrier);
                let root_sizes_c = Arc::clone(&root_sizes);
                let result_ref_c = Arc::clone(&result_ref);
                s.spawn(move |_| {
                    let mut local_roots = Vec::with_capacity(
                        self.parents.len().checked_div(num_threads / 2).unwrap_or(1),
                    );
                    // First, rename leading nodes
                    (parent_range.start as u32..)
                        .zip(&self.parents[parent_range.clone()])
                        .filter(|(i, val)| val.load(Ordering::Relaxed) == *i)
                        .for_each(|(i, _)| local_roots.push(AtomicU32::new(i)));
                    barrier_c.wait();

                    // OpenMP model is slightly different, so instead of using offsets computed in "single" section we compute offset on every thread
                    root_sizes_c[i].store(local_roots.len() as u32, Ordering::Relaxed);
                    barrier_c.wait();

                    let local_root_offset: u32 = root_sizes_c[0..i]
                        .iter()
                        .map(|x| x.load(Ordering::Relaxed))
                        .sum();
                    (0..).zip(local_roots).for_each(|(component_counter, i)| {
                        result_ref_c.read().unwrap().component_assignment
                            [i.load(Ordering::Relaxed) as usize]
                            .store(local_root_offset + component_counter, Ordering::Relaxed)
                    });

                    // Second, allocate info arrays - this can be improved by using spin-locks probably
                    let barrier_info_arrays_r = barrier_c.wait();
                    if barrier_info_arrays_r.is_leader() {
                        let num_components: u32 =
                            root_sizes_c.iter().map(|x| x.load(Ordering::Relaxed)).sum();
                        result_ref_c.write().unwrap().num_components = num_components;
                        result_ref_c
                            .write()
                            .unwrap()
                            .num_component_members
                            .resize_with(num_components as usize, || AtomicU32::new(0));
                        result_ref_c
                            .write()
                            .unwrap()
                            .component_leaders
                            .resize_with(num_components as usize, || AtomicU32::new(0));
                    }
                    barrier_c.wait();
                    let mut local_num_component_members =
                        vec![0; result_ref_c.read().unwrap().num_components as usize];
                    let component_assignment = &result_ref_c.read().unwrap().component_assignment;
                    let component_leaders = &result_ref_c.read().unwrap().component_leaders;
                    (parent_range.start..)
                        .zip(&self.parents[parent_range])
                        .for_each(|(i, parent_a)| {
                            let mut parent = parent_a.load(Ordering::Relaxed) as usize;
                            assume!(unsafe: parent < self.parents.len(), "parent: {parent} > {}", self.parents.len());
                            assume!(unsafe: parent < component_assignment.len(), "parent: {parent} > {}", component_assignment.len());
                            assume!(unsafe: i < component_assignment.len(), "i: {i} > {}", component_assignment.len());
                            if parent < i {
                                let mut component_no =
                                    component_assignment[parent].load(Ordering::Relaxed);
                                // NOTE: from original implementation:
                                //  In case that parent crosses over thread boundaries, it could be possible
                                //  that component_no is not assigned. If so, search for the value of it walking through tree upward.
                                while component_no == u32::MAX {
                                    parent = self.parents[parent].load(Ordering::Relaxed) as usize;
                                    assume!(unsafe: parent < self.parents.len(), "parent: {parent} > {}", self.parents.len());
                                    assume!(unsafe: parent < component_assignment.len(), "parent: {parent} > {}", component_assignment.len());
                                    component_no =
                                        component_assignment[parent].load(Ordering::Relaxed);
                                }
                                assume!(unsafe: (component_no as usize) < local_num_component_members.len(), "component_no: {component_no} > {}", local_num_component_members.len());
                                component_assignment[i].store(component_no, Ordering::Relaxed);
                                local_num_component_members[component_no as usize] += 1;
                            } else {
                                let component_no =
                                    component_assignment[i].load(Ordering::Relaxed) as usize;
                                assume!(unsafe: component_no < component_leaders.len(), "component_no: {component_no} > {}", component_leaders.len());
                                assume!(unsafe: component_no < local_num_component_members.len(), "component_no: {component_no} > {}", local_num_component_members.len());
                                component_leaders[component_no].store(i as u32, Ordering::Relaxed);
                                local_num_component_members[component_no] += 1;
                            }
                        });
                    let num_component_members = &result_ref_c.read().unwrap().num_component_members;
                    local_num_component_members
                        .into_iter()
                        .zip(num_component_members)
                        .filter(|(x, _)| *x > 0)
                        .for_each(|(x, d)| {
                            d.fetch_add(x, Ordering::Relaxed);
                        });
                });
            }
        });
        let result_rwlock: RwLock<ComponentSet> =
            Arc::<RwLock<ComponentSet>>::try_unwrap(result_ref).unwrap();
        result_rwlock.into_inner().unwrap()
    }
}

#[multiversion(targets = "simd")]
pub fn assign_disjoint_set(assignments: &AtomicArray2D<AtomicU16>) -> DisjointSet {
    let num_threads: usize = current_num_threads();
    let cc_set = DisjointSet::new(assignments.data.len() as u32);
    let vsplit_ranges = split_length_to_ranges(assignments.height, num_threads);
    let mut seam_ys: Vec<u32> = vsplit_ranges.iter().map(|r| r.end as u32).collect();
    seam_ys.pop(); // remove the end (bottom line of the image)

    #[multiversion(targets = "simd")]
    fn assign_disjoint_set_thread(
        assignments: &AtomicArray2D<AtomicU16>,
        cc_set: &DisjointSet,
        range: Range<usize>,
    ) {
        let mut row_iter = range.into_iter();
        let row_num = row_iter.next().unwrap();
        let cluster_row = assignments.get_row(row_num);
        let mut cluster_column_iter = cluster_row.iter();
        let mut left_cluster_no = cluster_column_iter.next().unwrap().load(Ordering::Relaxed);
        let mut index = (assignments.width * row_num) as u32;
        for cluster_no in cluster_column_iter {
            index += 1;
            if left_cluster_no == cluster_no.load(Ordering::Relaxed) {
                cc_set.merge(index - 1, index)
            } else {
                left_cluster_no = cluster_no.load(Ordering::Relaxed);
            }
        }

        for row_num in row_iter {
            let mut index_u = assignments.get_index(0, row_num);
            let mut index_up_u = assignments.get_index(0, row_num - 1);
            if assignments.data[index_u].load(Ordering::Relaxed)
                == assignments.data[index_up_u].load(Ordering::Relaxed)
            {
                cc_set.merge(index_up_u as u32, index_u as u32);
            }
            left_cluster_no = assignments.data[index_u].load(Ordering::Relaxed);
            index_u += 1;
            index_up_u += 1;

            for column_num in 1..assignments.width {
                debug_assert_eq!(assignments.get_index(column_num, row_num), index_u);
                debug_assert_eq!(assignments.get_index(column_num, row_num - 1), index_up_u);
                let cluster_no = assignments.data[index_u].load(Ordering::Relaxed);
                let cluster_up_no = assignments.data[index_up_u].load(Ordering::Relaxed);

                if cluster_no == left_cluster_no {
                    cc_set.merge((index_u - 1) as u32, index_u as u32);
                    if cluster_up_no == cluster_no {
                        cc_set.merge((index_u - 1) as u32, index_up_u as u32);
                    }
                } else if cluster_up_no == cluster_no {
                    cc_set.merge(index_u as u32, index_up_u as u32);
                }

                index_u += 1;
                index_up_u += 1;
                left_cluster_no = cluster_no;
            }
        }
    }

    vsplit_ranges
        .into_par_iter()
        .for_each(|r| assign_disjoint_set_thread(assignments, &cc_set, r));

    seam_ys.iter().for_each(|y| {
        let row_index = assignments.get_index(0, *y as usize);
        (0..assignments.width).for_each(|x| {
            let index_c = row_index + x;
            let index_up = index_c - assignments.width;
            let cluster_no = assignments.data[index_c].load(Ordering::Relaxed);
            let cluster_up_no = assignments.data[index_up].load(Ordering::Relaxed);
            if cluster_no == cluster_up_no {
                cc_set.merge(index_c as u32, index_up as u32);
            }
        })
    });
    cc_set
}

/// This function implements the CCA step.
pub fn enforce_connectivity(
    clusters: &mut Clusters,
    image: &LABImage,
    config: &Config,
    search_region_size: u16,
) {
    let min_threshold =
        ((search_region_size * search_region_size) as f32 * config.min_size_factor).round() as u32;

    let disjoint_set = assign_disjoint_set(&clusters.assignments);

    let cc_set = disjoint_set.flatten();

    let num_components = cc_set.num_components;
    let mut substitute = vec![u16::MAX; num_components as usize];
    // get large segments
    let mut comps: Vec<u32> = cc_set
        .num_component_members
        .iter()
        .enumerate()
        .filter(|(_, x)| x.load(Ordering::Relaxed) >= min_threshold)
        .map(|(i, _)| i as u32)
        .collect();
    // limit the number of large segments (most of the time this will not be true)
    if (config.num_of_clusters as usize) < comps.len() {
        comps.sort_by(|a, b| {
            cc_set.num_component_members[*b as usize]
                .load(Ordering::Relaxed)
                .cmp(&cc_set.num_component_members[*a as usize].load(Ordering::Relaxed))
        });
        comps.truncate(config.num_of_clusters as usize);
    }
    comps.sort_by(|a, b| {
        cc_set.component_leaders[*a as usize]
            .load(Ordering::Relaxed)
            .cmp(&cc_set.component_leaders[*b as usize].load(Ordering::Relaxed))
    });

    // Substitute step

    (0u16..)
        .zip(comps)
        .for_each(|(label, component_no)| substitute[component_no as usize] = label);
    if num_components > 0 && substitute[0] == u16::MAX {
        substitute[0] = 0;
    }
    for component_no in 0..num_components as usize {
        if substitute[component_no] != u16::MAX {
            continue;
        }
        let leader_index = cc_set.component_leaders[component_no].load(Ordering::Relaxed) as usize;
        let subs_label: u16 = if (leader_index % image.width) > 0 {
            substitute
                [cc_set.component_assignment[leader_index - 1].load(Ordering::Relaxed) as usize]
        } else {
            substitute[cc_set.component_assignment[leader_index - image.width]
                .load(Ordering::Relaxed) as usize]
        };
        debug_assert!(subs_label != u16::MAX, "leader {leader_index}");
        substitute[component_no] = subs_label;
    }

    // Relabeling
    let output_chunks_ranges =
        split_length_to_ranges(clusters.assignments.data.len(), current_num_threads());
    output_chunks_ranges.into_par_iter().for_each(|r| {
        (r.start..).zip(&clusters.assignments.data[r.clone()]).for_each(|(i, label)| {
            assume!(unsafe: i < cc_set.component_assignment.len(), "i: {i} > {}", cc_set.component_assignment.len());
            let component_assignment = cc_set.component_assignment[i].load(Ordering::Relaxed) as usize;
            assume!(unsafe: component_assignment < substitute.len(), "i: {i}, component_assignment: {} > {}", component_assignment, substitute.len());
            label.store(substitute[component_assignment], Ordering::Relaxed)
        })
    });
}
