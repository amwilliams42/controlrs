//! Quaternion operations for 3D representation and rotations
//! 
//! Quaternions are an extension of the complex number system, here represented
//!  as q = w + xi + yj + zk, where:
//! - w is the scalar part
//! - x, y, z form the vector part
//! - i, j, k are imaginary units satisfying i^2 = j^2 = k^2 = ijk = -1

use super::Vector3;
use core::ops::{Add,Sub,Mul};

#[derive(Debug, Clone, Copy)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32, 
    pub y: f32,
    pub z: f32,

}
impl Quaternion {
    ///Create a Quaternion
    ///
    /// A quaternion is not, by default, normalized. If you need a unit 
    /// quaterion, call normalize()
    /// 
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Quaternion{ w, x, y, z }
    }

    /// Compute the norm of the quaternion
    /// 
    /// The magnitude (square norm) of a quaternion q = (w, x, y, z) is:
    /// |q| = sqrt(w^2 + x^2 + y^2 + z^2)
    pub fn magnitude(&self) -> f32 {
        (self.w * self.w + 
         self.x * self.x + 
         self.y * self.y + 
         self.z * self.z).sqrt()
    }

    /// Normalize the quaternion
    /// 
    /// A unit quaternion has a magnitude of 1 and is used to represent
    /// a pure rotation without scaling.
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        Quaternion {
            w: self.w / mag,
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    /// Compute the conjugate of the quaternion
    /// 
    /// The conjugate of a quaternion q = (w, x, y, z) is q* = (w, -x, -y, -z).
    /// For unit quaternions, the conjugate is equal to the inverse.
    pub fn conjugate(&self) -> Self {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
    /// Quaternion multiplication
    /// 
    /// For quaternions q1 = (w1, x1, y1, z1) and q2 = (w2, x2, y2, z2):
    /// q1 * q2 = (
    ///     w1w2 - x1x2 - y1y2 - z1z2,
    ///     w1x2 + x1w2 + y1z2 - z1y2,
    ///     w1y2 - x1z2 + y1w2 + z1x2,
    ///     w1z2 + x1y2 - y1x2 + z1w2
    /// )
    /// 
    /// Quaternion multiplication is not commutative: in general, q1 * q2 ≠ q2 * q1
    fn mul_quaternions(a: &Quaternion, b: &Quaternion) -> Quaternion {
        Quaternion {
            w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        }
    }
    /// Rotate a vector by this quaternion
    /// 
    /// This method works correctly whether the quaternion is normalized or not.
    /// However, if you're performing multiple rotations, it's more efficient
    /// to normalize the quaternion once before doing the rotations.
    /// 
    /// # Mathematical Background
    /// To rotate a vector v by a quaternion q:
    /// 1. Convert v to a quaternion p = (0, v_x, v_y, v_z)
    /// 2. Compute the rotation: p' = q * p * q^(-1)
    /// 3. The rotated vector is the vector part of p'
    /// 
    /// For non-unit quaternions, we use q^(-1) instead of q* (conjugate).
    /// 
    /// # Performance Note
    /// This method computes the quaternion's magnitude and performs division,
    /// which may be computationally expensive. For better performance in
    /// tight loops, consider normalizing the quaternion once before
    /// performing multiple rotations.
    pub fn rotate_vector(&self, v: &Vector3) -> Vector3 {
        let q = Quaternion::new(0.0, v.x, v.y, v.z);
        let mag_squared = self.magnitude().powi(2);
        
        if mag_squared.abs() < 1e-10 {
            // Handle the case of a near-zero quaternion
            // Return the original vector or handle as appropriate for your use case
            *v
        } else {
            let result = self * &q * &self.conjugate();
            Vector3::new(
                result.x / mag_squared,
                result.y / mag_squared,
                result.z / mag_squared
            )
        }
    }

    /// Efficiently rotate multiple vectors by this quaternion
    /// 
    /// This method normalizes the quaternion once and then performs
    /// efficient rotations on multiple vectors.
    /// 
    /// # Arguments
    /// * `vectors` - A slice of Vector3 to be rotated
    /// 
    /// # Returns
    /// A Vec of rotated Vector3
    pub fn rotate_vectors(&self, vectors: &[Vector3]) -> Vec<Vector3> {
        let normalized = self.normalize();
        vectors.iter().map(|v| normalized.rotate_vector_normalized(v)).collect()
    }

    /// Rotate a vector by a normalized quaternion
    /// 
    /// # Safety
    /// This method assumes the quaternion is already normalized.
    /// Use only if you're certain the quaternion is a unit quaternion.
    #[inline]
    fn rotate_vector_normalized(&self, v: &Vector3) -> Vector3 {
        let q = Quaternion::new(0.0, v.x, v.y, v.z);
        let rotated = self * &q * &self.conjugate();
        Vector3::new(rotated.x, rotated.y, rotated.z)
    }


    /// Create a quaternion from an axis-angle representation
    /// 
    /// # Arguments
    /// * `axis` - A unit vector representing the axis of rotation
    /// * `angle` - The angle of rotation in radians
    /// 
    /// For a rotation of angle θ around axis (x, y, z), the quaternion is:
    /// q = (cos(θ/2), x*sin(θ/2), y*sin(θ/2), z*sin(θ/2))
    pub fn from_axis_angle(axis: &Vector3, angle: f32) -> Self {
        let half_angle = angle * 0.5;
        let sin_half_angle = half_angle.sin();

        Quaternion{
            w: half_angle.cos(),
            x: axis.x * sin_half_angle,
            y: axis.y * sin_half_angle,
            z: axis.z * sin_half_angle,
        }
    }

    /// Convert quaternion to axis-angle representation
    /// 
    /// # Returns
    /// A tuple (axis, angle) where:
    /// * `axis` is a Vector3 representing the axis of rotation
    /// * `angle` is the rotation angle in radians
    /// 
    /// For a quaternion q = (w, x, y, z):
    /// - angle θ = 2 * acos(w)
    /// - axis = (x, y, z) / sin(θ/2)
    /// 
    /// Note: For very small rotations, the calculation of the rotation can
    /// become unstable due to division by a very small number. In practice
    /// such small rotations are often indistinguishable from 0 rotation.
    /// A unit vector with 0 rotation (here, (1,0,0)) is chosen, this choice 
    /// is arbitrary.
    pub fn to_axis_angle(&self) -> (Vector3, f32) {
        let scale = (1.0 - self.w * self.w).sqrt();
        if scale.abs() <= 1e-6 {
            // For very small rotations, return a default axis and zero angle
            (Vector3::new(1.0, 0.0, 0.0), 0.0)
        } else {
            let angle = 2.0 * self.w.acos();
            let axis = Vector3::new(self.x / scale, self.y / scale, self.z / scale);
            (axis, angle)
        }
    }

}

// Implement multiplication for Quaternion

// Implement &Quaternion * &Quaternion
impl Mul<&Quaternion> for &Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: &Quaternion) -> Quaternion {
        Quaternion::mul_quaternions(self, rhs)
    }
}

// Implement Quaternion * &Quaternion
impl Mul<&Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: &Quaternion) -> Quaternion {
        Quaternion::mul_quaternions(&self, rhs)
    }
}

// Implement &Quaternion * Quaternion
impl Mul<Quaternion> for &Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Quaternion {
        Quaternion::mul_quaternions(self, &rhs)
    }
}

// Implement Quaternion * Quaternion
impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Quaternion {
        Quaternion::mul_quaternions(&self, &rhs)
    }
}


// Implement addition for Quaternion
impl Add for Quaternion {
    type Output = Quaternion;

    /// Quaternion addition
    /// 
    /// Quaternions are added component-wise. Note that the sum of two
    /// unit quaternions is generally not a unit quaternion.
    fn add(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

// Implement subtraction for Quaternion
impl Sub for Quaternion {
    type Output = Quaternion;

    /// Quaternion subtraction
    /// 
    /// Quaternions are subtracted component-wise. Note that the difference of two
    /// unit quaternions is generally not a unit quaternion.
    fn sub(self, rhs: Quaternion) -> Quaternion {
        Quaternion {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl PartialEq for Quaternion {
    fn eq(&self, other: &Self) -> bool {
        (self.w - other.w).abs() < f32::EPSILON &&
        (self.x - other.x).abs() < f32::EPSILON &&
        (self.y - other.y).abs() < f32::EPSILON &&
        (self.z - other.z).abs() < f32::EPSILON
    }
}