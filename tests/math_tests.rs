use controlrs::math::*;
use std::f32::consts::PI;

const EPSILON: f32 = 1e-6;

fn assert_vector_eq(v1: Vector3, v2: Vector3) {
    assert!((v1.x - v2.x).abs() < EPSILON, "x components differ: {} != {}", v1.x, v2.x);
    assert!((v1.y - v2.y).abs() < EPSILON, "y components differ: {} != {}", v1.y, v2.y);
    assert!((v1.z - v2.z).abs() < EPSILON, "z components differ: {} != {}", v1.z, v2.z);
}

fn assert_quaternion_eq(q1: Quaternion, q2: Quaternion) {
    assert!((q1.w - q2.w).abs() < EPSILON, "w components differ: {} != {}", q1.w, q2.w);
    assert!((q1.x - q2.x).abs() < EPSILON, "x components differ: {} != {}", q1.x, q2.x);
    assert!((q1.y - q2.y).abs() < EPSILON, "y components differ: {} != {}", q1.y, q2.y);
    assert!((q1.z - q2.z).abs() < EPSILON, "z components differ: {} != {}", q1.z, q2.z);
}

#[test]
fn test_vector_creation() {
    let v = Vector3::new(1.0, 2.0, 3.0);
    assert_vector_eq(v, Vector3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_vector_magnitude() {
    let v = Vector3::new(3.0, 4.0, 0.0);
    assert!((v.magnitude() - 5.0).abs() < EPSILON);
}

#[test]
fn test_vector_normalize() {
    let v = Vector3::new(3.0, 4.0, 0.0);
    let normalized = v.normalize();
    assert!((normalized.magnitude() - 1.0).abs() < EPSILON);
    assert_vector_eq(normalized, Vector3::new(0.6, 0.8, 0.0));
}

#[test]
fn test_vector_dot_product() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    assert!((v1.dot(&v2) - 32.0).abs() < EPSILON);
}

#[test]
fn test_vector_cross_product() {
    let v1 = Vector3::new(1.0, 0.0, 0.0);
    let v2 = Vector3::new(0.0, 1.0, 0.0);
    let cross = v1.cross(&v2);
    assert_vector_eq(cross, Vector3::new(0.0, 0.0, 1.0));
}

#[test]
fn test_vector_addition() {
    let v1 = Vector3::new(1.0, 2.0, 3.0);
    let v2 = Vector3::new(4.0, 5.0, 6.0);
    let sum = v1 + v2;
    assert_vector_eq(sum, Vector3::new(5.0, 7.0, 9.0));
}

#[test]
fn test_vector_subtraction() {
    let v1 = Vector3::new(4.0, 5.0, 6.0);
    let v2 = Vector3::new(1.0, 2.0, 3.0);
    let diff = v1 - v2;
    assert_vector_eq(diff, Vector3::new(3.0, 3.0, 3.0));
}

#[test]
fn test_vector_scalar_multiplication() {
    let v = Vector3::new(1.0, 2.0, 3.0);
    let scaled = v * 2.0;
    assert_vector_eq(scaled, Vector3::new(2.0, 4.0, 6.0));
}

#[test]
fn test_vector_scalar_division() {
    let v = Vector3::new(2.0, 4.0, 6.0);
    let divided = v / 2.0;
    assert_vector_eq(divided, Vector3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_quaternion_creation() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    assert_quaternion_eq(q, Quaternion::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_quaternion_magnitude() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    assert!((q.magnitude() - 30.0_f32.sqrt()).abs() < EPSILON);
}

#[test]
fn test_quaternion_normalize() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let normalized = q.normalize();
    assert!((normalized.magnitude() - 1.0).abs() < EPSILON);
}

#[test]
fn test_quaternion_conjugate() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let conjugate = q.conjugate();
    assert_quaternion_eq(conjugate, Quaternion::new(1.0, -2.0, -3.0, -4.0));
}

#[test]
fn test_quaternion_multiplication() {
    let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let q2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    let result = &q1 * &q2;
    assert_quaternion_eq(result, Quaternion::new(-60.0, 12.0, 30.0, 24.0));
}

#[test]
fn test_quaternion_rotate_vector() {
    let axis = Vector3::new(0.0, 1.0, 0.0);
    let angle = PI / 2.0; // 90 degrees
    let q = Quaternion::from_axis_angle(&axis, angle);
    let v = Vector3::new(1.0, 0.0, 0.0);
    let rotated = q.rotate_vector(&v);
    assert_vector_eq(rotated, Vector3::new(0.0, 0.0, -1.0));
}

#[test]
fn test_quaternion_from_axis_angle() {
    let axis = Vector3::new(0.0, 1.0, 0.0);
    let angle = PI / 4.0; // 45 degrees
    let q = Quaternion::from_axis_angle(&axis, angle);
    
    let expected_w = (PI / 8.0).cos();
    let expected_y = (PI / 8.0).sin();
    
    assert!((q.w - expected_w).abs() < EPSILON, "w component: expected {}, got {}", expected_w, q.w);
    assert!((q.x - 0.0).abs() < EPSILON, "x component: expected 0, got {}", q.x);
    assert!((q.y - expected_y).abs() < EPSILON, "y component: expected {}, got {}", expected_y, q.y);
    assert!((q.z - 0.0).abs() < EPSILON, "z component: expected 0, got {}", q.z);
}

#[test]
fn test_quaternion_to_axis_angle() {
    let original_axis = Vector3::new(0.0, 1.0, 0.0);
    let original_angle = PI / 4.0; // 45 degrees
    let q = Quaternion::from_axis_angle(&original_axis, original_angle);
    let (axis, angle) = q.to_axis_angle();
    assert_vector_eq(axis, Vector3::new(0.0, 1.0, 0.0));
    assert!((angle - PI / 4.0).abs() < EPSILON);
}