use aligned_vec::{AVec, ConstAlign};
use std::fmt;
use std::ops::{Index, IndexMut};

const ALIGN: usize = 64;

pub struct AtomicArray2D<T: Sync + Send> {
    pub data: AVec<T, ConstAlign<ALIGN>>,
    pub width: usize,
    pub height: usize,
}

impl<T: Sync + Send> AtomicArray2D<T> {
    pub fn from_fill<U: Copy + Clone + Into<T>>(value: U, width: usize, height: usize) -> Self {
        let data: AVec<T, ConstAlign<ALIGN>> =
            AVec::from_iter(ALIGN, (0..width * height).map(|_| value.into()));
        Self {
            width,
            height,
            data,
        }
    }

    pub fn from_slice<U: Clone + Copy + Into<T>>(
        source: &[U],
        width: usize,
        height: usize,
    ) -> Self {
        assert_eq!(width * height, source.len());
        let data: AVec<T, ConstAlign<ALIGN>> =
            AVec::from_iter(ALIGN, source.iter().map(|y| (*y).into()));
        Self {
            width,
            height,
            data,
        }
    }

    pub fn fill<U: Clone + Copy + Into<T>>(&mut self, value: U) {
        self.data.fill_with(|| value.into())
    }

    #[inline(always)]
    pub fn get_row(&self, row: usize) -> &[T] {
        &self.data[self.width * row..self.width * row + self.width]
    }

    #[inline(always)]
    pub fn get_index(&self, x: usize, y: usize) -> usize {
        debug_assert!(
            self.width > x,
            "Index ({x}, {y}) is out of bounds ({}, {})",
            self.width,
            self.height
        );
        debug_assert!(
            self.height > y,
            "Index ({x}, {y}) is out of bounds ({}, {})",
            self.width,
            self.height
        );
        self.width * y + x
    }

    pub fn get_x_y_index(&self, ind: usize) -> (usize, usize) {
        debug_assert!(ind < self.data.len());
        let y = ind / self.width;
        let x = ind % self.width;
        (x, y)
    }

    pub fn split_to_tiles(
        &mut self,
        x_start: usize,
        y_start: usize,
        x_sizes: &[usize],
        y_sizes: &[usize],
    ) -> Vec<AtomicSubArray2D<T>> {
        assert!(
            x_start + x_sizes.iter().sum::<usize>() <= self.width,
            "{x_start} + {:?} < {}",
            x_sizes,
            self.width
        );
        assert!(
            y_start + y_sizes.iter().sum::<usize>() <= self.height,
            "{y_start} + {:?} < {}",
            y_sizes,
            self.height
        );
        let num_tiles = x_sizes.len() * y_sizes.len();
        let mut tiles: Vec<AtomicSubArray2D<T>> = Vec::with_capacity(num_tiles);
        let mut y = y_start;
        for y_size in y_sizes {
            let mut x = x_start;
            for x_size in x_sizes {
                tiles.push(AtomicSubArray2D {
                    x,
                    y,
                    width: *x_size,
                    height: *y_size,
                    data: self,
                });
                x += x_size;
            }
            y += y_size;
        }
        tiles
    }

    pub fn get_subarray(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> AtomicSubArray2D<T> {
        AtomicSubArray2D {
            x,
            y,
            width,
            height,
            data: self,
        }
    }
}

impl<T: Sync + Send> fmt::Debug for AtomicArray2D<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .debug_struct("AtomicArray2D")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("data", &"Omitted...")
            .finish()
    }
}

impl<T: Sync + Send> Index<(usize, usize)> for AtomicArray2D<T> {
    type Output = T;
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[self.get_index(x, y)]
    }
}
impl<T: Sync + Send> IndexMut<(usize, usize)> for AtomicArray2D<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        let idx = self.get_index(x, y);
        &mut self.data[idx]
    }
}

#[derive(Debug)]
pub struct AtomicSubArray2D<'a, T: Sync + Send> {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    data: &'a AtomicArray2D<T>,
}

impl<'a, T: Sync + Send> AtomicSubArray2D<'a, T> {
    pub fn get_index(&self, x: usize, y: usize) -> (usize, usize) {
        debug_assert!(self.width > x);
        debug_assert!(self.height > y);
        (self.x + x, self.y + y)
    }

    #[inline(always)]
    pub fn get_row(&self, row: usize) -> &[T] {
        &self.data.data[self.width * row..self.width * row + self.width]
    }

    #[inline(always)]
    pub fn get_row_part(&self, row: usize, left: usize, right: usize) -> &'a [T] {
        &self.data.data[self.data.get_index(self.x + left, self.y + row)
            ..self.data.get_index(self.x + right, self.y + row) + 1]
    }

    #[inline(always)]
    pub fn get_local_index_from_full_array_index(
        &self,
        x_full: usize,
        y_full: usize,
    ) -> (usize, usize) {
        debug_assert!(
            (self.x <= x_full) & (self.y <= y_full),
            "Ouf-of-bounds x index {x_full} >= {} and y index {y_full} >= {}",
            self.x,
            self.y
        );
        debug_assert!(
            self.x <= x_full,
            "Ouf-of-bounds x index {x_full} >= {}",
            self.x
        );
        debug_assert!(
            self.y <= y_full,
            "Ouf-of-bounds y index {y_full} >= {}",
            self.y
        );
        let x = x_full - self.x;
        let y = y_full - self.y;
        debug_assert!(
            self.width >= x,
            "Ouf-of-bounds x index {x} (full {x_full}) < {} (full {})",
            self.width,
            self.x + self.width
        );
        debug_assert!(
            self.height >= y,
            "Ouf-of-bounds y index {y} (full {y_full}) < {} (full {})",
            self.height,
            self.y + self.height
        );
        (x, y)
    }

    #[inline(always)]
    pub fn get_local_index_left_right_index(
        &self,
        left_full: usize,
        right_full: usize,
    ) -> (usize, usize) {
        debug_assert!(left_full < right_full);
        debug_assert!(left_full < self.width + self.x);
        debug_assert!(right_full < self.width + self.x);
        debug_assert!(left_full >= self.x);
        debug_assert!(right_full >= self.x);
        let left = left_full - self.x;
        let right = right_full - self.x;
        (left, right)
    }
}

impl<'a, T: Sync + Send> Index<(usize, usize)> for AtomicSubArray2D<'a, T> {
    type Output = T;
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[self.get_index(x, y)]
    }
}

#[cfg(test)]
mod tests {
    use crate::atomic_arrays::{AtomicArray2D, AtomicSubArray2D};
    use std::sync::atomic::{AtomicU16, Ordering};

    #[test]
    fn test_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AtomicArray2D<AtomicU16>>();
    }

    #[test]
    fn test_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<AtomicArray2D<AtomicU16>>();
    }

    #[test]
    fn atomic_array_test() {
        let mut data: AtomicArray2D<AtomicU16> = AtomicArray2D::from_fill(0, 1920, 1080);
        data.fill(0xFFFF);
        data.fill(0);
        assert_eq!((155, 560), data.get_x_y_index(data.get_index(155, 560)));

        let d: AtomicArray2D<AtomicU16> = AtomicArray2D::from_slice(&[0, 1, 2, 3, 4, 5], 3, 2);
        rayon::scope(|s| {
            s.spawn(|_| {
                d[(1, 1)].store(54, Ordering::Relaxed);
            });
            s.spawn(|_| {
                d[(0, 0)].store(99, Ordering::Relaxed);
            });
        });
        assert_eq!(
            d.data
                .iter()
                .map(|v| v.load(Ordering::Relaxed))
                .collect::<Vec<u16>>(),
            [99, 1, 2, 3, 54, 5]
        );
        let sub = AtomicSubArray2D {
            x: 1,
            y: 1,
            width: 2,
            height: 1,
            data: &d,
        };
        let sub2 = AtomicSubArray2D {
            x: 0,
            y: 0,
            width: 2,
            height: 2,
            data: &d,
        };

        rayon::scope(|s| {
            s.spawn(|_| {
                sub[(1, 0)].store(69, Ordering::Relaxed);
            });
            s.spawn(|_| {
                sub2[(1, 1)].store(79, Ordering::Relaxed);
            });
        });
        assert_eq!(
            d.data
                .iter()
                .map(|v| v.load(Ordering::Relaxed))
                .collect::<Vec<u16>>(),
            [99, 1, 2, 3, 79, 69]
        );
    }

    #[test]
    fn atomic_array_split_to_subarrays_test() {
        let width = 1920;
        let height = 1080;
        let mut data: AtomicArray2D<AtomicU16> = AtomicArray2D::from_fill(0, width, height);
        data.fill(0xFFFF);
        let x_sizes = vec![192; 10];
        let y_sizes = vec![108; 10];
        let subarrs = data.split_to_tiles(0, 0, &x_sizes, &y_sizes);
        for subarr in subarrs {
            for row_num in 0..subarr.height {
                subarr
                    .get_row_part(row_num, 0, subarr.width - 1)
                    .iter()
                    .for_each(|x| x.store(0, Ordering::Relaxed))
            }
        }
        let unassigned: Vec<usize> = data
            .data
            .iter()
            .enumerate()
            .filter(|(_i, x)| x.load(Ordering::Relaxed) == 0xFFFF)
            .map(|(i, _x)| i)
            .collect();
        assert!(
            unassigned.is_empty(),
            "Unassigned pixels: {:?}",
            unassigned
                .iter()
                .map(|ind| { data.get_x_y_index(*ind) })
                .collect::<Vec<_>>()
        );
    }
}
