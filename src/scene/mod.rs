mod world;

/// Use the 'cover' sizing strategy that ensure the rendered area covers the
/// entire viewport.
struct PerspectiveCamera {
    /// Aspect-ratio from width by height.
    screen: (u32, u32),
    /// Field of view in angle. Normal human vision is limited to `pi / 6`
    /// radians vertically.
    ///
    /// See https://en.wikipedia.org/wiki/Human_eye#Field_of_view.
    fov: f32,
    clip_range: f32,
}
impl ToMatrix for PerspectiveCamera {
    fn to_matrix(&self) -> Matrix {
        let tan = (self.fov / 2.0).tan();
        let (near, far) = self.clip_range;
        let scale = (tan * near).recip();
        let (screen_w, screen_h) = self.screen;
        let (m11, m22) = if screen_w < screen_h {
            // Normalize portrait coordinates.
            (scale, scale * (screen_w / screen_h))
        } else {
            // Normalize landscape coordinates.
            (scale * (screen_h / screen_w), scale)
        };
        // Unit depth.
        let m33 = (near - far).recip();
        // Depth offset for near plain, which is the origin of our frustum.
        let m34 = near * m33;
        // Keep a copy of the actual depth.
        let m43 = -1.0;
        Matrix(
            _mm_set_ps(0.0, 0.0, 0.0, m11),
            _mm_set_ps(0.0, 0.0, m22, 0.0),
            _mm_set_ps(m34, m33, 0.0, 0.0),
            _mm_set_ps(0.0, m43, 0.0, 0.0),
        )
    }
}
