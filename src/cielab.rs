use assume::assume;
use tables::{C_B, LAB_SHIFT, LAB_TBL, OUTPUT_SHIFT, SRGB_GAMMA_TBL, SRGB_SHIFT};
pub(crate) mod tables {
    use static_init::dynamic;
    pub const SRGB_SHIFT: u8 = 13;
    const SRGB_MAX: usize = 1 << SRGB_SHIFT;
    pub const OUTPUT_SHIFT: u8 = 1;
    pub const LAB_SHIFT: u8 = 16;
    pub const C_B: [u32; 9] = [28440, 24656, 12442, 13938, 46868, 4730, 1164, 7175, 57202];
    #[dynamic(65535)]
    pub static SRGB_GAMMA_TBL: [u32; 256] =
        core::array::from_fn(|i| (calculate_xyz_nonlin(i as u8) * SRGB_MAX as f32) as u32);
    #[dynamic(65535)]
    pub static LAB_TBL: [i32; SRGB_MAX + 1] = core::array::from_fn(|i| {
        (calculate_lab_nonlin(i as f32 / SRGB_MAX as f32) * SRGB_MAX as f32).round() as i32
    });
    fn calculate_xyz_nonlin(a: u8) -> f32 {
        let v: f64 = a as f64 / 255.0;
        if v <= 0.04045 {
            return (v / 12.92) as f32;
        }
        ((v + 0.055) / 1.055).powf(2.4) as f32
    }

    fn calculate_lab_nonlin(a: f32) -> f32 {
        debug_assert!(a >= 0.0);
        if a < 0.008856 {
            return 7.787 * a + 0.137931;
        }
        a.powf(0.333333)
    }
}

/// Convert pixel in RGB24 to Lab24. This is an approximation.
///
/// The output range is for:
///  - L - from 0 to 200
///  - a - from 0 to 255 (some values are clamped)
///  - b - from 0 to 255 (some values are clamped)
#[inline(always)]
pub fn srgb_to_cielab_pixel(rgb: &[u8]) -> [u8; 3] {
    let sr = unsafe { SRGB_GAMMA_TBL[rgb[0] as usize] };
    let sg = unsafe { SRGB_GAMMA_TBL[rgb[1] as usize] };
    let sb = unsafe { SRGB_GAMMA_TBL[rgb[2] as usize] };
    let xr = (C_B[0] * sr + C_B[1] * sg + C_B[2] * sb) >> LAB_SHIFT;
    let yr = (C_B[3] * sr + C_B[4] * sg + C_B[5] * sb) >> LAB_SHIFT;
    let zr = (C_B[6] * sr + C_B[7] * sg + C_B[8] * sb) >> LAB_SHIFT;
    assume!(unsafe: xr < LAB_TBL.len() as u32);
    assume!(unsafe: yr < LAB_TBL.len() as u32);
    assume!(unsafe: zr < LAB_TBL.len() as u32);
    let fx = unsafe { LAB_TBL[xr as usize] };
    let fy = unsafe { LAB_TBL[yr as usize] };
    let fz = unsafe { LAB_TBL[zr as usize] };
    let ciel = 116 * fy - (16 << SRGB_SHIFT);
    let ciea = 500 * (fx - fy) + (128 << SRGB_SHIFT);
    let cieb = 200 * (fy - fz) + (128 << SRGB_SHIFT);
    assume!(unsafe: ciel >= 0);
    assume!(unsafe: ciea >= 0);
    assume!(unsafe: cieb >= 0);
    let l: u8 = (ciel as u32 >> (SRGB_SHIFT - OUTPUT_SHIFT)) as u8;
    let a: u8 = ((ciea >> (SRGB_SHIFT - OUTPUT_SHIFT)) - (64 << OUTPUT_SHIFT)).clamp(0, 255) as u8;
    let b: u8 = ((cieb >> (SRGB_SHIFT - OUTPUT_SHIFT)) - (64 << OUTPUT_SHIFT)).clamp(0, 255) as u8;
    [l, a, b]
}

#[cfg(test)]
mod tests {
    use super::srgb_to_cielab_pixel;
    use super::tables::{C_B, LAB_TBL};
    #[test]
    fn c_b_table_test() {
        println!("{:?}", C_B.as_slice());
    }

    #[test]
    fn lab_table_test() {
        for i in unsafe { LAB_TBL.into_iter() } {
            assert!(i > 0);
        }
    }

    #[test]
    fn srgb_to_cielab_pix_test() {
        let mut l: u8;
        let mut a: u8;
        let mut bb: u8;
        let mut l_min: u8 = 255;
        let mut l_max: u8 = 0;
        let mut a_min: u8 = 255;
        let mut a_max: u8 = 0;
        let mut bb_min: u8 = 255;
        let mut bb_max: u8 = 0;
        for r in 0..255u8 {
            for g in 0..255u8 {
                for b in 0..255u8 {
                    [l, a, bb] = srgb_to_cielab_pixel(&[r, g, b]);
                    if l > l_max {
                        l_max = l;
                    };
                    if l < l_min {
                        l_min = l;
                    };
                    if a > a_max {
                        a_max = a;
                    };
                    if a < a_min {
                        a_min = a;
                    };
                    if bb > bb_max {
                        bb_max = bb;
                    };
                    if bb < bb_min {
                        bb_min = bb;
                    };
                }
            }
        }
        println!("L: {}..{}", l_min, l_max);
        println!("A: {}..{}", a_min, a_max);
        println!("B: {}..{}", bb_min, bb_max);
    }
}
