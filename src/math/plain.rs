use std::ops::{Add, Sub, Mul, Div, Neg};
use core::arch::x86_64::*;
use std::mem::transmute;

#[derive(Clone, Copy, Debug)]
pub struct Vector(__m128);
impl Sub for Vector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output { unsafe { Vector(_mm_sub_ps(self.0, rhs.0)) } }
}
impl Add for Vector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { unsafe { Vector(_mm_add_ps(self.0, rhs.0)) } }
}
impl Mul<f32> for Vector {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output { unsafe { Vector(_mm_mul_ps(self.0, _mm_set1_ps(rhs))) } }
}
impl Div<f32> for Vector {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output { unsafe { Vector(_mm_div_ps(self.0, _mm_set1_ps(rhs))) } }
}
impl Neg for Vector {
    type Output = Self;
    fn neg(self) -> Self::Output { unsafe { Vector(_mm_xor_ps(self.0, _mm_set1_ps(transmute(0x80000000 as u32)))) } }
}
impl PartialEq for Vector {
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { _mm_test_all_ones(transmute(_mm_cmp_ps(self.0, rhs.0, _CMP_EQ_OQ))) == 1 }
    }
}
impl Default for Vector {
    fn default() -> Self { unsafe { Vector(_mm_setzero_ps()) } }
}
impl Vector {
    pub fn new(x: f32, y: f32, z: f32) -> Self { unsafe { Vector(_mm_set_ps(0.0, z, y, x)) } }
    pub fn dot(self, rhs: Self) -> f32 { unsafe { transmute(_mm_extract_ps(_mm_dp_ps(self.0, rhs.0, 0xf1), 0)) } }
    pub fn cross(self, rhs: Self) -> Self {
        unsafe {
            let a = _mm_shuffle_ps(rhs.0, rhs.0, 0b11001001);
            let b = _mm_shuffle_ps(self.0, self.0, 0b11001001);
            let res = _mm_sub_ps(_mm_mul_ps(self.0, a), _mm_mul_ps(b, rhs.0));
            Vector(_mm_shuffle_ps(res, res, 0b11001001))
        }
    }
    pub fn mag_sqr(self) -> f32 { self.dot(self) }
    pub fn mag(self) -> f32 { self.mag_sqr().sqrt() }
    pub fn sgn(self) -> Self { self / self.mag() }
}



#[derive(Clone, Copy, Debug)]
pub struct Point(__m128);
impl PartialEq for Point {
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { _mm_test_all_ones(transmute(_mm_cmp_ps(self.0, rhs.0, _CMP_EQ_OQ))) == 1 }
    }
}
impl Default for Point {
    fn default() -> Self { unsafe { Point(_mm_setzero_ps()) } }
}
impl Point {
    pub fn new(x: f32, y: f32, z: f32) -> Self { unsafe { Point(_mm_set_ps(0.0, z, y, x)) } }
    pub fn affine(self, rhs: Vector) -> Self { unsafe { Point(_mm_add_ps(self.0, rhs.0)) } }
}




#[cfg(test)]
mod test {
    use super::{Point, Vector};
    #[test]
    fn test_eq() {
        assert_eq!(Vector::new(1.0, 1.0, 1.0), Vector::new(1.0, 1.0, 1.0));
        assert!(Vector::new(0.0, 1.0, 1.0) != Vector::new(1.0, 1.0, 1.0));
    }
    #[test]
    fn test_vec_arith() {
        let a = Vector::new(2.0, 4.0, 6.0);
        let b = Vector::new(1.0, 2.0, 3.0);
        assert_eq!(a + b, Vector::new(3.0, 6.0, 9.0));
        assert_eq!(a - b, Vector::new(1.0, 2.0, 3.0));
        assert_eq!(a * 2.0, Vector::new(4.0, 8.0, 12.0));
        assert_eq!(a / 2.0, Vector::new(1.0, 2.0, 3.0));
        assert_eq!(-a, Vector::new(-2.0, -4.0, -6.0));
    }
    #[test]
    fn test_vec_dot() {
        assert_eq!(Vector::new(1.0, 2.0, 3.0).dot(Vector::new(4.0, 5.0, 6.0)), 32.0);
    }
    #[test]
    fn test_vec_cross() {
        assert_eq!(Vector::new(1.0, 2.0, 3.0).cross(Vector::new(4.0, 5.0, 6.0)), Vector::new(-3.0, 6.0, -3.0));
        assert_eq!(Vector::new(1.0, 0.0, 0.0).cross(Vector::new(0.0, 1.0, 0.0)), Vector::new(0.0, 0.0, 1.0));
        assert_eq!(Vector::new(0.0, 1.0, 0.0).cross(Vector::new(0.0, 0.0, 1.0)), Vector::new(1.0, 0.0, 0.0));
        assert_eq!(Vector::new(0.0, 0.0, 1.0).cross(Vector::new(1.0, 0.0, 0.0)), Vector::new(0.0, 1.0, 0.0));
        assert_eq!(Vector::new(1.0, 2.0, 3.0).cross(Vector::new(-4.0, -5.0, -6.0)), Vector::new(3.0, -6.0, 3.0));
    }
    #[test]
    fn test_pt_affine() {
        assert_eq!(Point::new(1.0, 2.0, 3.0).affine(Vector::new(1.0, 2.0, 3.0)), Point::new(2.0, 4.0, 6.0));
    }
}
