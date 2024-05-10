use crate::cielab::srgb_to_cielab_pixel;
use aligned_vec::{AVec, ConstAlign};
use rayon::current_num_threads;
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

const ALIGN: usize = 64;

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    IndicesOutOfBounds(usize, usize),
    IndexOutOfBounds(usize),
    DimensionMismatch,
    NotEnoughElements,
}
impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IndicesOutOfBounds(row, column) => {
                write!(f, "indices ({row}, {column}) out of bounds")
            }
            Error::IndexOutOfBounds(index) => write!(f, "index {index} out of bounds"),
            Error::DimensionMismatch => write!(f, "dimension mismatch"),
            Error::NotEnoughElements => write!(f, "not enough elements"),
        }
    }
}

#[derive(Debug)]
pub struct Array2D<T> {
    pub data: AVec<T, ConstAlign<ALIGN>>,
    pub width: usize,
    pub height: usize,
}

impl<T> Array2D<T> {
    pub fn from_slice(data: &[T], width: usize, height: usize) -> Result<Self, Error>
    where
        T: Clone,
    {
        if data.len() != width * height {
            return Err(Error::DimensionMismatch);
        }
        Ok(Self {
            width,
            height,
            data: AVec::from_slice(ALIGN, data),
        })
    }

    pub fn from_fill(value: T, width: usize, height: usize) -> Self
    where
        T: Clone + Copy,
    {
        let data: AVec<T, ConstAlign<ALIGN>> =
            AVec::from_iter(ALIGN, (0..width * height).map(|_| value));
        Self {
            width,
            height,
            data,
        }
    }

    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value)
    }
    pub fn get_row(&self, row: usize) -> &[T] {
        debug_assert!(row < self.height);
        &self.data[(self.width * row)..(self.width * row + self.width)]
    }
    pub fn get_row_mut(&mut self, row: usize) -> &mut [T] {
        debug_assert!(row < self.height);
        &mut self.data[(self.width * row)..(self.width * row + self.width)]
    }
    #[inline(always)]
    pub fn get_row_part(&self, row: usize, left: usize, right: usize) -> &[T] {
        debug_assert!(
            row < self.height,
            "Out-of-bounds row {row} < {}",
            self.height
        );
        debug_assert!(
            left < self.width,
            "Out-of-bounds left {left} < {}",
            self.width
        );
        debug_assert!(
            right < self.width,
            "Out-of-bounds right {right} < {}",
            self.width
        );
        &self.data[(self.width * row + left)..(self.width * row + right) + 1]
    }
    pub fn get_row_part_mut(&mut self, row: usize, left: usize, right: usize) -> &mut [T] {
        debug_assert!(row < self.height);
        debug_assert!(left < self.width);
        debug_assert!(right < self.width);
        &mut self.data[(self.width * row + left)..(self.width * row + right) + 1]
    }
    pub fn get_index(&self, x: usize, y: usize) -> usize {
        debug_assert!(self.width > x);
        debug_assert!(self.height > y);
        self.width * y + x
    }
    pub fn split_to_tiles_mut<'a>(
        &'a mut self,
        x_sizes: &[usize],
        y_sizes: &[usize],
    ) -> &'a mut [&'a mut [&'a mut [T]]] {
        assert_eq!(x_sizes.iter().sum::<usize>(), self.width);
        assert_eq!(y_sizes.iter().sum::<usize>(), self.height);
        let num_tiles = x_sizes.len() * y_sizes.len();
        let mut data_slice = self.data.as_mut_slice();
        let mut tiles_data_vec: Vec<&'a mut [&'a mut [T]]> = Vec::with_capacity(num_tiles);
        for y_size in y_sizes {
            let mut tiles_data_part_line_v: Vec<Vec<&'a mut [T]>> = (0..x_sizes.len())
                .map(|_| Vec::with_capacity(*y_size))
                .collect();
            for _ in 0..*y_size {
                for (i, x_size) in x_sizes.iter().enumerate() {
                    let (chunk, rest) = data_slice.split_at_mut(*x_size);
                    tiles_data_part_line_v[i].push(chunk);
                    data_slice = rest;
                }
            }
            tiles_data_vec.append(
                &mut tiles_data_part_line_v
                    .into_iter()
                    .map(|t| t.leak())
                    .collect(),
            );
        }
        tiles_data_vec.leak()
    }
    pub fn split_to_tiles_subarrays<'a>(
        &'a mut self,
        x_start: usize,
        y_start: usize,
        x_sizes: &[usize],
        y_sizes: &[usize],
    ) -> &'a mut [SubArray2DRef<'a, T>] {
        assert!(
            x_start + x_sizes.iter().sum::<usize>() <= self.width,
            "Wrong sum of sizes in width: {x_start}+{} <= {}",
            x_sizes.iter().sum::<usize>(),
            self.width
        );
        assert!(
            y_start + y_sizes.iter().sum::<usize>() <= self.height,
            "Wrong sum of sizes in length: {y_start}+{} <= {}",
            y_sizes.iter().sum::<usize>(),
            self.height
        );
        let x_remainder = self.width - x_sizes.iter().sum::<usize>() - x_start;
        let num_tiles = x_sizes.len() * y_sizes.len();
        let mut data_slice = self.data.as_mut_slice();
        if y_start != 0 {
            data_slice = &mut data_slice[self.width * y_start..];
        }
        let mut tiles_data_vec: Vec<SubArray2DRef<'a, T>> = Vec::with_capacity(num_tiles);
        let mut y_p: usize = y_start;
        for y_size in y_sizes {
            let mut tiles_data_part_line_v: Vec<Vec<&'a mut [T]>> = (0..x_sizes.len())
                .map(|_| Vec::with_capacity(*y_size))
                .collect();
            for _ in 0..*y_size {
                if x_start != 0 {
                    let (_chunk, rest) = data_slice.split_at_mut(x_start);
                    data_slice = rest;
                }
                for (i, x_size) in x_sizes.iter().enumerate() {
                    let (chunk, rest) = data_slice.split_at_mut(*x_size);
                    tiles_data_part_line_v[i].push(chunk);
                    data_slice = rest;
                }
                if x_remainder != 0 {
                    let (_chunk, rest) = data_slice.split_at_mut(x_remainder);
                    data_slice = rest;
                }
            }
            tiles_data_vec.append(
                &mut tiles_data_part_line_v
                    .into_iter()
                    .zip(x_sizes)
                    .scan(x_start, |x_p, (t, x_size)| {
                        let arr = Some(SubArray2DRef {
                            x: *x_p,
                            y: y_p,
                            width: *x_size,
                            height: *y_size,
                            data_ref: t.leak(),
                        });
                        *x_p += *x_size;
                        arr
                    })
                    .collect(),
            );
            y_p += y_size;
        }
        debug_assert_eq!(
            data_slice.len(),
            (self.height - y_sizes.iter().sum::<usize>() - y_start) * self.width
        );
        tiles_data_vec.leak()
    }
}
impl<T> Index<(usize, usize)> for Array2D<T> {
    type Output = T;
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[self.get_index(x, y)]
    }
}
impl<T> IndexMut<(usize, usize)> for Array2D<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        let idx = self.get_index(x, y);
        &mut self.data[idx]
    }
}
pub struct SubArray2DRef<'a, T> {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub data_ref: &'a mut [&'a mut [T]],
}

impl<'a, T> SubArray2DRef<'a, T> {
    pub fn get_local_index_from_full_array_index(
        &self,
        x_full: usize,
        y_full: usize,
    ) -> (usize, usize) {
        debug_assert!(
            self.x <= x_full,
            "Ouf-of-bounds x index {x_full} <= {}",
            self.x
        );
        debug_assert!(
            self.y <= y_full,
            "Ouf-of-bounds y index {y_full} <= {}",
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
    pub fn get_full_array_index_from_local_index(&self, x: usize, y: usize) -> (usize, usize) {
        debug_assert!(self.width > x);
        debug_assert!(self.height > y);
        (x + self.x, y + self.y)
    }
    #[inline(always)]
    pub fn get_row_part(&self, row: usize, left: usize, right: usize) -> &[T] {
        &self.data_ref[row][left..right]
    }
    #[inline(always)]
    pub fn get_row_part_mut(&mut self, row: usize, left: usize, right: usize) -> &mut [T] {
        &mut self.data_ref[row][left..right]
    }
}
#[inline(always)]
fn avec_fill_zeros<T: Sized>(size: usize) -> AVec<T, ConstAlign<ALIGN>> {
    let size_of = std::mem::size_of::<T>();
    let size_bytes = match size.checked_mul(size_of) {
        Some(size_bytes) => size_bytes,
        None => panic!(
            "Number of elements {} overflowed u64 when size is in bytes. Can't allocate!",
            size
        ),
    };
    let will_overflow = size_bytes > usize::MAX - (ALIGN - 1);
    let is_invalid_alloc = usize::BITS < 64 && size_bytes > isize::MAX as usize;
    if will_overflow || is_invalid_alloc {
        panic!(
            "Number of elements {} of {} bytes can't be allocated!",
            size, size_bytes
        )
    }
    let layout = std::alloc::Layout::from_size_align(size_bytes, ALIGN)
        .expect("Creation of layout failed. Check alignment size (must be power of two)!");
    let ptr_b = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr_b.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let ptr = unsafe { std::ptr::NonNull::new_unchecked(ptr_b as *mut T) };
    unsafe { AVec::from_raw_parts(ptr.as_ptr(), ALIGN, size, size) }
}

pub struct LABImage {
    pub lab_data: AVec<u8, ConstAlign<ALIGN>>,
    pub width: usize,
    pub height: usize,
}

impl LABImage {
    pub fn from_srgb(rgb_image: &[u8], width: usize, height: usize) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert_eq!(rgb_image.len(), width * height * 3);
        let num_threads = current_num_threads();
        let chunk_size_ideal = rgb_image.len() / num_threads;
        let chunk_size = chunk_size_ideal - chunk_size_ideal % 3;
        let chunk_size_output = (chunk_size / 3) * 4;
        assert_eq!(chunk_size % 3, 0);
        let mut lab_output: AVec<u8, ConstAlign<ALIGN>> = avec_fill_zeros(width * height * 4);
        debug_assert_eq!(lab_output.iter().rev().step_by(4).sum::<u8>(), 0);
        rayon::scope(|s| {
            let mut rgb_input: &[u8] = rgb_image;
            let mut data_output: &mut [u8] = &mut lab_output;
            for _ in 0..num_threads {
                let (chunk_in, rest_in) = rgb_input.split_at(chunk_size);
                rgb_input = rest_in;
                let (chunk_out, rest_out) = data_output.split_at_mut(chunk_size_output);
                data_output = rest_out;
                s.spawn(|_| {
                    for i in 0..(chunk_out.len() / 4) {
                        chunk_out[i * 4..i * 4 + 3].copy_from_slice(
                            srgb_to_cielab_pixel(&chunk_in[i * 3..i * 3 + 3]).as_slice(),
                        )
                    }
                });
            }
        });
        Self {
            width,
            height,
            lab_data: lab_output,
        }
    }

    pub fn from_raw_slice(lab_data: &[u8], width: usize, height: usize) -> Self {
        assert!(width > 0);
        assert!(height > 0);
        assert_eq!(lab_data.len(), width * height * 4);

        let lab_output = AVec::from_slice(ALIGN, lab_data);

        Self {
            width,
            height,
            lab_data: lab_output,
        }
    }

    pub fn from_iter<I>(lab_iter: I, width: usize, height: usize) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        assert!(width > 0);
        assert!(height > 0);

        let lab_output = AVec::from_iter(ALIGN, lab_iter);

        assert_eq!(lab_output.len(), width * height * 4);

        Self {
            width,
            height,
            lab_data: lab_output,
        }
    }

    #[inline(always)]
    pub fn get_row(&self, row: usize) -> &[u8] {
        debug_assert!(row < self.height);
        &self.lab_data[(self.width * 4 * row)..(self.width * 4 * row + self.width * 4)]
    }
    #[inline(always)]
    pub fn get_index(&self, x: usize, y: usize) -> usize {
        debug_assert!(self.width > x);
        debug_assert!(self.height > y);
        self.width * y * 4 + x * 4
    }
    #[inline(always)]
    pub fn get_pixel(&self, x: usize, y: usize) -> &[u8] {
        let idx = self.get_index(x, y);
        &self.lab_data[idx..idx + 3]
    }
    #[inline(always)]
    pub fn get_row_part(&self, row: usize, left: usize, right: usize) -> &[u8] {
        debug_assert!(row < self.height);
        debug_assert!(left < self.width);
        debug_assert!(right <= self.width);
        &self.lab_data[(4 * self.width * row + 4 * left)..(4 * self.width * row + 4 * right) + 4]
    }
}
impl Index<(usize, usize)> for LABImage {
    type Output = [u8];
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let idx = self.get_index(x, y);
        &self.lab_data[idx..idx + 3]
    }
}

#[cfg(test)]
mod tests {
    use super::{Array2D, LABImage};
    #[test]
    fn srgb_to_cielab_test() {
        let dimg = image::open("test/data/aerial.jpg").unwrap();
        let img = dimg.as_rgb8().unwrap().as_raw();
        let width = dimg.width() as usize;
        let height = dimg.height() as usize;
        let conv_img = LABImage::from_srgb(img.as_slice(), width, height);
        for i in 0..conv_img.lab_data.len() / 4 {
            assert_eq!(conv_img.lab_data[3 + i * 4], 0);
        }
    }

    #[test]
    fn srgb_to_cielab_iter_test() {
        let dimg = image::open("test/data/aerial.jpg").unwrap();
        let img = dimg.as_rgb8().unwrap().as_raw();
        let width = dimg.width() as usize;
        let height = dimg.height() as usize;
        let convert_iter = img.chunks_exact(3).flat_map(|p| {
            let lab = crate::cielab::srgb_to_cielab_pixel(p);
            [lab[0], lab[1], lab[2], 0]
        });
        let conv_img = LABImage::from_iter(convert_iter, width, height);
        for i in 0..conv_img.lab_data.len() / 4 {
            assert_eq!(conv_img.lab_data[3 + i * 4], 0);
        }
    }

    #[test]
    fn labimage_get_row_test() {
        let dimg = image::open("test/data/aerial.jpg").unwrap();
        let img = dimg.as_rgb8().unwrap().as_raw();
        let width = dimg.width() as usize;
        let height = dimg.height() as usize;
        let conv_img = LABImage::from_srgb(img.as_slice(), width, height);
        assert_eq!(conv_img.get_row(0).len(), width * 4);
        let left = 10;
        let right = 24;
        assert_eq!(
            conv_img.get_row_part(0, left, right).len(),
            (right - left + 1) * 4
        );
    }

    #[test]
    fn array2d_split_tiles() {
        let width = 1920;
        let height = 1080;
        let mut arr = Array2D::from_fill(0u16, width, height);
        let tile_div = 10;
        let num_tiles = tile_div * tile_div;
        let split_size_x = width / tile_div;
        let split_size_y = height / tile_div;
        let splits_x: Vec<_> = vec![split_size_x; tile_div];
        let splits_y: Vec<_> = vec![split_size_y; tile_div];
        let tiles = arr.split_to_tiles_mut(&splits_x, &splits_y);
        let mut acc: usize = 0;
        debug_assert_eq!(tiles.len(), num_tiles, "wrong returned number of tiles");
        for tile in tiles {
            debug_assert_eq!(tile.len(), split_size_y, "wrong height of tiles");
            for row in tile.iter() {
                debug_assert_eq!(row.len(), split_size_x, "wrong width of tiles");
                acc += row.len();
            }
        }
        debug_assert_eq!(acc, width * height);
    }

    #[test]
    fn array2d_split_tiles_subarray_exact() {
        let width = 1920;
        let height = 1080;
        let mut arr = Array2D::from_fill(0u16, width, height);
        let tile_div = 10;
        let num_tiles = tile_div * tile_div;
        let split_size_x = width / tile_div;
        let split_size_y = height / tile_div;
        let splits_x: Vec<_> = vec![split_size_x; tile_div];
        let splits_y: Vec<_> = vec![split_size_y; tile_div];
        let tiles = arr.split_to_tiles_subarrays(0, 0, &splits_x, &splits_y);
        debug_assert_eq!(tiles.len(), num_tiles, "wrong returned number of tiles");
        for tile in tiles {
            debug_assert_eq!(tile.width, split_size_x, "wrong width of tiles");
            debug_assert_eq!(tile.height, split_size_y, "wrong height of tiles");
        }
    }
}
