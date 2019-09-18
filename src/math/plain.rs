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
    pub fn dot(self, rhs: Self) -> f32 { unsafe { _mm_cvtss_f32(_mm_dp_ps(self.0, rhs.0, 0xf1)) } }
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
        unsafe { _mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) == 0x0f }
    }
    fn ne(&self, rhs: &Self) -> bool {
        unsafe { _mm_movemask_ps(_mm_cmpneq_ps(self.0, rhs.0)) == 0x0f }
    }
}
impl Default for Point {
    fn default() -> Self { unsafe { Point(_mm_setzero_ps()) } }
}
impl Point {
    pub fn new(x: f32, y: f32, z: f32) -> Self { unsafe { Point(_mm_set_ps(0.0, z, y, x)) } }
    pub fn affine(self, rhs: Vector) -> Self { unsafe { Point(_mm_add_ps(self.0, rhs.0)) } }
}



#[derive(Clone, Copy, Debug)]
pub struct Transform(__m128, __m128, __m128); // Row-major, 3x4 matrix, left-mul for transform.
impl Mul<Vector> for Transform {
    type Output = Vector;
    fn mul(self, rhs: Vector) -> Vector {
        unsafe {
            let c = _mm_insert_ps(rhs.0, _mm_set1_ps(1.0), 0x00);
            Vector(_mm_or_ps(_mm_or_ps(_mm_dp_ps(self.0, c, 0xf1), _mm_dp_ps(self.1, c, 0xf2)), _mm_dp_ps(self.2, c, 0xf4)))
        }
    }
}
impl Mul<Transform> for Transform {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let z = _mm_set_ps(1.0, 0.0, 0.0, 0.0);
            let c1c2lo = _mm_unpacklo_ps(rhs.0, rhs.1);
            let c1c2hi = _mm_unpacklo_ps(rhs.2, z);
            let c3c4lo = _mm_unpackhi_ps(rhs.0, rhs.1);
            let c3c4hi = _mm_unpackhi_ps(rhs.2, z);
            let c1 = _mm_movelh_ps(c1c2lo, c1c2hi);
            let c2 = _mm_movehl_ps(c1c2hi, c1c2lo);
            let c3 = _mm_movelh_ps(c3c4lo, c3c4hi);
            let c4 = _mm_movehl_ps(c3c4hi, c3c4lo);
            let r1 = _mm_or_ps(_mm_or_ps(_mm_dp_ps(self.0, c1, 0xf1), _mm_dp_ps(self.0, c2, 0xf2)), _mm_or_ps(_mm_dp_ps(self.0, c3, 0xf4), _mm_dp_ps(self.0, c4, 0xf8)));
            let r2 = _mm_or_ps(_mm_or_ps(_mm_dp_ps(self.1, c1, 0xf1), _mm_dp_ps(self.1, c2, 0xf2)), _mm_or_ps(_mm_dp_ps(self.1, c3, 0xf4), _mm_dp_ps(self.1, c4, 0xf8)));
            let r3 = _mm_or_ps(_mm_or_ps(_mm_dp_ps(self.2, c1, 0xf1), _mm_dp_ps(self.2, c2, 0xf2)), _mm_or_ps(_mm_dp_ps(self.2, c3, 0xf4), _mm_dp_ps(self.2, c4, 0xf8)));
            Transform(r1, r2, r3)
        }
    }
}
impl PartialEq for Transform {
    fn eq(&self, rhs: &Transform) -> bool {
        unsafe {
            _mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & _mm_movemask_ps(_mm_cmpeq_ps(self.1, rhs.1)) & _mm_movemask_ps(_mm_cmpeq_ps(self.2, rhs.2)) == 0x0f
        }
    }
}
impl Transform {
    pub fn eye() -> Transform {
        unsafe {
            Transform(
                _mm_set_ps(0.0, 0.0, 0.0, 1.0),
                _mm_set_ps(0.0, 0.0, 1.0, 0.0),
                _mm_set_ps(0.0, 1.0, 0.0, 0.0),
            )
        }
    }
}



#[cfg(test)]
mod test {
    use super::{*};
    #[test]
    fn test_eq() {
        assert_eq!(Vector::new(1.0, 1.0, 1.0), Vector::new(1.0, 1.0, 1.0));
        assert!(Vector::new(0.0, 1.0, 1.0) != Vector::new(1.0, 1.0, 1.0));
        assert!(Vector::new(1.0, 0.0, 1.0) != Vector::new(1.0, 1.0, 1.0));
        assert!(Vector::new(1.0, 1.0, 0.0) != Vector::new(1.0, 1.0, 1.0));
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
    #[test]
    fn test_trans_identity() {
        assert_eq!(Transform::eye() * Transform::eye(), Transform::eye());
    }
    #[test]
    fn test_trans() {
        assert_eq!(Transform::eye() * Vector::new(1.0, 1.0, 1.0), Vector::new(1.0, 1.0, 1.0));
    }
}
