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
impl Sub for Point {
    type Output = Vector;
    fn sub(self, rhs: Self) -> Vector { unsafe { Vector(_mm_sub_ps(self.0, rhs.0)) } }
}
impl Add<Vector> for Point {
    type Output = Point;
    fn add(self, rhs: Vector) -> Point { unsafe { Point(_mm_add_ps(self.0, rhs.0)) } }
}
impl Default for Point {
    fn default() -> Self { unsafe { Point(_mm_setzero_ps()) } }
}
impl Point {
    pub fn new(x: f32, y: f32, z: f32) -> Self { unsafe { Point(_mm_set_ps(1.0, z, y, x)) } }
}



trait PointSet: AsRef<[Point]> {
    fn get_bounds(&self) -> (Point, Point) {
        let pts = self.as_ref();
        let n = pts.len();
        unsafe {
            let (mut min, mut max) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            for i in (0..n).step_by(2) {
                let x = _mm256_insertf128_ps(_mm256_castps128_ps256(pts[i].0), pts[i + 1].0, 1);
                min = _mm256_min_ps(min, x);
                max = _mm256_max_ps(max, x);
            }
            let mut min = _mm_min_ps(_mm256_extractf128_ps(min, 0), _mm256_extractf128_ps(min, 1));
            let mut max = _mm_max_ps(_mm256_extractf128_ps(max, 0), _mm256_extractf128_ps(max, 1));
            if n & 1 == 1 {
                let last = pts[n - 1].0;
                min = _mm_min_ps(min, last);
                max = _mm_max_ps(max, last);
            }
            (Point(min), Point(max))
        }
    }
}
impl<T> PointSet for T where T: AsRef<[Point]> {}



/// 4x4 column-major matrix, this can do no algebra and should be only used as a
/// data container. Left-mul for transformation in GLSL code.
pub struct Matrix(__m128, __m128, __m128, __m128);
trait ToMatrix {
    // Convert a linear operator to Vulkan-compatible column-major matrix,
    // stored in contigeous memory.
    fn to_matrix(&self) -> Matrix;
}

/// 3x4 row-major matrix, left-mul to apply transform. Less memory and
/// instructions per multiplication is needed on host side.
#[derive(Clone, Copy, Debug)]
pub struct Transform(__m128, __m128, __m128);
impl Mul<Transform> for Transform {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            let Matrix(x, y, z, w) = self.to_matrix();
            let r1 = _mm_or_ps(_mm_or_ps(_mm_dp_ps(rhs.0, x, 0xf1), _mm_dp_ps(rhs.0, y, 0xf2)), _mm_or_ps(_mm_dp_ps(rhs.0, z, 0xf4), _mm_dp_ps(rhs.0, w, 0xf8)));
            let r2 = _mm_or_ps(_mm_or_ps(_mm_dp_ps(rhs.1, x, 0xf1), _mm_dp_ps(rhs.1, y, 0xf2)), _mm_or_ps(_mm_dp_ps(rhs.1, z, 0xf4), _mm_dp_ps(rhs.1, w, 0xf8)));
            let r3 = _mm_or_ps(_mm_or_ps(_mm_dp_ps(rhs.2, x, 0xf1), _mm_dp_ps(rhs.2, y, 0xf2)), _mm_or_ps(_mm_dp_ps(rhs.2, z, 0xf4), _mm_dp_ps(rhs.2, w, 0xf8)));
            Transform(r1, r2, r3)
        }
    }
}
impl Mul<Vector> for Transform {
    type Output = Vector;
    fn mul(self, rhs: Vector) -> Vector {
        unsafe {
            let c = _mm_or_ps(rhs.0, _mm_set_ps(1.0, 0.0, 0.0, 0.0));
            Vector(_mm_or_ps(_mm_or_ps(_mm_dp_ps(self.0, c, 0xf1), _mm_dp_ps(self.1, c, 0xf2)), _mm_dp_ps(self.2, c, 0xf4)))
        }
    }
}
impl PartialEq for Transform {
    fn eq(&self, rhs: &Transform) -> bool {
        unsafe {
            _mm_movemask_ps(_mm_cmpeq_ps(self.0, rhs.0)) & _mm_movemask_ps(_mm_cmpeq_ps(self.1, rhs.1)) & _mm_movemask_ps(_mm_cmpeq_ps(self.2, rhs.2)) == 0x0f
        }
    }
    fn ne(&self, rhs: &Transform) -> bool {
        unsafe {
            _mm_movemask_ps(_mm_cmpneq_ps(self.0, rhs.0)) & _mm_movemask_ps(_mm_cmpneq_ps(self.1, rhs.1)) & _mm_movemask_ps(_mm_cmpneq_ps(self.2, rhs.2)) == 0x0f
        }
    }
}
impl Transform {
    #[inline]
    pub fn new(x: Vector, y: Vector, z: Vector, shift: Vector) -> Transform {
        unsafe {
            let r1r2lo = _mm_unpacklo_ps(x.0, y.0);
            let r1r2hi = _mm_unpacklo_ps(z.0, shift.0);
            let r3r4lo = _mm_unpackhi_ps(x.0, y.0);
            let r3r4hi = _mm_unpackhi_ps(z.0, shift.0);
            let r1 = _mm_movelh_ps(r1r2lo, r1r2hi);
            let r2 = _mm_movehl_ps(r1r2hi, r1r2lo);
            let r3 = _mm_movelh_ps(r3r4lo, r3r4hi);
            Transform(r1, r2, r3)
        }
    }
    /// Make a lookat matrix from a camera to a target to be looked at. The
    /// camera is rolled away from the +y direction, by a positive angle counter-
    /// clockwise or a negative angle counter-clockwise.
    #[inline]
    pub fn lookat(offset: Point, lookat: Point, roll: f32) -> Transform {
        let (sin, cos) = roll.sin_cos();
        let up = Vector::new(sin, cos, 0.0);
        let n = (offset - lookat).sgn();
        let u = up.cross(n).sgn();
        let v = n.cross(u);
        Transform::new(u, v, n, Point::default() - offset)
    }
    #[inline]
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
impl ToMatrix for Transform {
    fn to_matrix(&self) -> Matrix {
        unsafe {
            let z = _mm_set_ps(1.0, 0.0, 0.0, 0.0);
            let c1c2lo = _mm_unpacklo_ps(self.0, self.1);
            let c1c2hi = _mm_unpacklo_ps(self.2, z);
            let c3c4lo = _mm_unpackhi_ps(self.0, self.1);
            let c3c4hi = _mm_unpackhi_ps(self.2, z);
            let c1 = _mm_movelh_ps(c1c2lo, c1c2hi);
            let c2 = _mm_movehl_ps(c1c2hi, c1c2lo);
            let c3 = _mm_movelh_ps(c3c4lo, c3c4hi);
            let c4 = _mm_movehl_ps(c3c4hi, c3c4lo);
            Matrix(c1, c2, c3, c4)
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
        let eye = Transform::eye();
        let ones = Vector::new(1.0, 1.0, 1.0);
        assert_eq!(eye * eye, eye);
        let vec = Vector::new(1.0, 2.0, 3.0);
        let trans = Transform::new(vec, vec, vec, ones);
        assert_eq!(eye * trans, Transform::new(vec, vec, vec, ones));
    }
    #[test]
    fn test_trans() {
        let eye = Transform::eye();
        let ones = Vector::new(1.0, 1.0, 1.0);
        assert_eq!(eye * ones, ones);
        let vec = Vector::new(1.0, 2.0, 3.0);
        let trans = Transform::new(vec, vec, vec, vec);
        assert_eq!(trans * ones, Vector::new(4.0, 8.0, 12.0));
    }
}
