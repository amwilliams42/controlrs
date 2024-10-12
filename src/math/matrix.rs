//! Matrix operations for 3x3 matrices

/// A 3x3 matrix struct
#[derive(Debug, Clone, Copy)]
pub struct Matrix3x3 {
    pub data: [[f32; 3]; 3],
}

impl Matrix3x3 {
    /// Create a new Matrix3x3
    #[inline]
    pub fn new(data: [[f32; 3]; 3]) -> Self {
        Matrix3x3 { data }
    }

    // TODO: Implement matrix operations
}