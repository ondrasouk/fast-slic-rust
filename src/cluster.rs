use crate::arrays::LABImage;

/// Struct of SLIC cluster/superpixel.
///
/// Fields `x`, `y`, `l`, `a`, `b`, `num_members` are updated by `slic::update()`. To get final
/// values after `enforce_connectivity()` (CCA) run update again.
#[derive(Debug)]
pub struct Cluster {
    /// x position of center (number of column (starting from 0))
    pub x: u16,
    /// y position of center
    pub y: u16,
    /// Average L color of cluster
    pub l: u8,
    /// Average a color of cluster
    pub a: u8,
    /// Average b color of cluster
    pub b: u8,
    /// Number of cluster (used in assignment)
    pub number: u16,
    /// Number of pixels inside cluster
    pub num_members: u32,
    // This field and fields bellow can have old values which are dependent on search region size.
    //
    // Updated in `assign()` using `Cluster::update_coords()` and made outdated in `update()` step.
    pub(crate) top: u16,
    pub(crate) bottom: u16,
    pub(crate) left: u16,
    pub(crate) right: u16,
    pub(crate) lut_left: u16,
    pub(crate) lut_right: u16,
}
impl Default for Cluster {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            l: 0,
            a: 0,
            b: 0,
            number: 0xFFFF,
            num_members: 0,
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
            lut_left: 0,
            lut_right: 0,
        }
    }
}
impl Cluster {
    pub(crate) fn is_boundary(&self, image: &LABImage, search_region_size: u16) -> bool {
        self.y.checked_sub(search_region_size).is_none()
            || self.x.checked_sub(search_region_size).is_none()
            || (self.y + search_region_size) >= image.height as u16
            || (self.x + search_region_size) >= image.width as u16
    }

    #[inline(always)]
    pub(crate) fn top(&self, search_region_size: u16) -> usize {
        self.y.saturating_sub(search_region_size) as usize
    }

    #[inline(always)]
    pub(crate) fn bottom(&self, image: &LABImage, search_region_size: u16) -> usize {
        let bottom = (self.y + search_region_size + 1).min(image.height as u16) as usize;
        debug_assert!(bottom <= image.height);
        bottom
    }

    #[inline(always)]
    pub(crate) fn left(&self, search_region_size: u16) -> usize {
        self.x.saturating_sub(search_region_size) as usize
    }

    #[inline(always)]
    pub(crate) fn right(&self, image: &LABImage, search_region_size: u16) -> usize {
        let right: usize = (self.x + search_region_size).min((image.width - 1) as u16) as usize;
        debug_assert!(right < image.width);
        right
    }

    #[inline(always)]
    pub(crate) fn get_left_right_spatial_distance_lut(
        &self,
        image: &LABImage,
        search_region_size: u16,
    ) -> (usize, usize) {
        let s = search_region_size as i32;
        let left: usize = search_region_size.saturating_sub(self.x) as usize;
        let right: usize = (image.width as u16 - (self.x.saturating_sub(search_region_size)) - 1)
            .min(2 * search_region_size)
            .min(image.width as u16 - 1) as usize;
        debug_assert!(
            left == 0 || self.is_boundary(image, search_region_size),
            "Even through it's not boundary: left={left} == 0 (s={s}, self.x={}, image.width={})",
            self.x,
            image.width
        );
        debug_assert!(
            right == 2 * search_region_size as usize || self.is_boundary(image, search_region_size),
            "Even through it's not boundary: right={right} == {} (s={s}, self.x={}, image.width={})",
            2 * s + 1,
            self.x,
            image.width
        );
        debug_assert!(right <= 2 * s as usize + 1, "{right} <= 2*{s}+1");
        debug_assert!(left < right, "{left} < {right}; s={s}");
        (left, right)
    }

    pub(crate) fn update_coords(&mut self, image: &LABImage, search_region_size: u16) {
        self.top = self.top(search_region_size) as u16;
        self.bottom = self.bottom(image, search_region_size) as u16;
        self.left = self.left(search_region_size) as u16;
        self.right = self.right(image, search_region_size) as u16;
        let (lut_left, lut_right) =
            self.get_left_right_spatial_distance_lut(image, search_region_size);
        (self.lut_left, self.lut_right) = (lut_left as u16, lut_right as u16);
    }
}
