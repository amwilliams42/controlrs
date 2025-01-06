//! Math module for controlrs
//! 
//! This module provides basic mathematical operations optimized for embedded systems.

mod vector;
mod matrix;
mod quaternion; 
pub mod dare;

pub use vector::Vector3;
pub use matrix::Matrix3x3;
pub use quaternion::Quaternion;
pub use dare::solve_dare_sda;

/// Constant for π
pub const PI: f32 = 3.14159265358979323846;

/// Convert degrees to radians
#[inline]
pub fn deg_to_rad(deg: f32) -> f32 {
    deg * PI / 180.0
}

/// Convert radians to degrees
#[inline]
pub fn rad_to_deg(rad: f32) -> f32 {
    rad * 180.0 / PI
}

/// Fast approximation of sine function
pub fn fast_sin(x: f32) -> f32 {
    // Implementation of fast sine approximation
    // This is a placeholder and should be replaced with an actual fast approximation
    x.sin()
}

/// Fast approximation of cosine function
pub fn fast_cos(x: f32) -> f32 {
    // Implementation of fast cosine approximation
    // This is a placeholder and should be replaced with an actual fast approximation
    x.cos()
}