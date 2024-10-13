use controlrs::control::{LQRController, LQRType, Horizon};
use nalgebra::{self as na};
use approx::assert_relative_eq;

#[test]
fn test_continuous_infinite_horizon() {
    // System: Simple harmonic oscillator
    // x' = [0  1] x + [0] u
    //      [0 -1]     [1]
    let a = na::DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, -1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    
    // Cost matrices: Equal weight on position and velocity states, and control input
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[1.0]);

    let controller = LQRController::new(a, b, q, r, LQRType::ContinuousTime, Horizon::Infinite, None)
        .expect("Failed to create LQR controller");

    let k = controller.get_feedback_gain(0);
    
    // Expected values:
    // K ≈ [1.0  1.7321]
    // This gain will make the closed-loop system stable with poles at -0.8660 ± 0.5i
    assert_relative_eq!(k[(0, 0)], 1.0, epsilon = 1e-6);
    assert_relative_eq!(k[(0, 1)], 1.7321, epsilon = 1e-4);
}

#[test]
fn test_discrete_infinite_horizon() {
    // System: Discrete-time double integrator
    let a = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[0.1]);

    let controller = LQRController::new(a.clone(), b.clone(), q, r, LQRType::DiscreteTime, Horizon::Infinite, None)
        .expect("Failed to create LQR controller");

    let k = controller.get_feedback_gain(0);

    // Check dimensions
    assert_eq!(k.nrows(), 1, "K matrix should have 1 row");
    assert_eq!(k.ncols(), 2, "K matrix should have 2 columns");

    // Check that K values are reasonable (non-zero)
    assert!(k[(0, 0)].abs() > 1e-6, "K[0,0] should not be zero");
    assert!(k[(0, 1)].abs() > 1e-6, "K[0,1] should not be zero");

    // Compute closed-loop matrix
    let closed_loop_a = &a - &b * k;

    // Check stability
    let eigenvalues = closed_loop_a.complex_eigenvalues();
    for eigenvalue in eigenvalues.iter() {
        assert!(eigenvalue.norm() < 1.0, "Closed-loop system should be stable (eigenvalue magnitude < 1)");
    }
}

#[test]
fn test_lqr_dynamic_system() {
    // Mass-spring-damper system
    // x' = [0    1] x + [0]
    //      [-k/m -c/m]   [1/m] u
    // where k = 2, m = 1, c = 0.5
    let a = na::DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0, -0.5]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    
    // Cost matrices
    let q = na::DMatrix::from_diagonal(&na::DVector::from_vec(vec![1.0, 0.1]));
    let r = na::DMatrix::from_element(1, 1, 0.05);

    let controller = LQRController::new(a.clone(), b.clone(), q, r, LQRType::ContinuousTime, Horizon::Infinite, None)
        .expect("Failed to create LQR controller");

    let k = controller.get_feedback_gain(0);

    // Check dimensions
    assert_eq!(k.nrows(), 1, "K matrix should have 1 row");
    assert_eq!(k.ncols(), 2, "K matrix should have 2 columns");

    // Check that K values are reasonable (non-zero)
    assert!(k[(0, 0)].abs() > 1e-6, "K[0,0] should not be zero");
    assert!(k[(0, 1)].abs() > 1e-6, "K[0,1] should not be zero");

    // Compute and check closed-loop stability
    let closed_loop_a = &a - &b * k;
    let eigenvalues = closed_loop_a.complex_eigenvalues();

    for eigenvalue in eigenvalues.iter() {
        assert!(eigenvalue.re < 0.0, "Closed-loop system should be stable (negative real parts of eigenvalues)");
    }
}

#[test]
fn test_discrete_finite_horizon() {
    // System: Discrete-time double integrator (same as in discrete infinite horizon test)
    let a = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[0.1]);
    
    // Terminal cost: Higher weight on final state
    let f = na::DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 10.0]);

    let controller = LQRController::new(a, b, q, r, LQRType::DiscreteTime, Horizon::Finite(10), Some(f))
        .expect("Failed to create LQR controller");

    let k_start = controller.get_feedback_gain(0);
    let k_end = controller.get_feedback_gain(9);
    
    // As with the continuous case, start and end gains should differ
    assert!(k_start != k_end, "Start and end gains should be different for finite horizon");
}

#[test]
fn test_compute_control() {
    // Using the continuous-time harmonic oscillator system
    let a = na::DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, -1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[1.0]);

    let controller = LQRController::new(a, b, q, r, LQRType::ContinuousTime, Horizon::Infinite, None)
        .expect("Failed to create LQR controller");

    let state = na::DVector::from_column_slice(&[1.0, 2.0]);
    let control = controller.compute_control(&state, 0);

    // The control should be a 1x1 vector
    // u = -Kx = -[1.0  1.7321] * [1.0; 2.0] ≈ -4.4642
    assert_eq!(control.nrows(), 1);
    assert_eq!(control.ncols(), 1);
    assert_relative_eq!(control[0], -4.4642, epsilon = 1e-4);
}

#[test]
fn test_integrate() {
    // Using the continuous-time harmonic oscillator system
    let a = na::DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, -1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[1.0]);

    let controller = LQRController::new(a, b, q, r, LQRType::ContinuousTime, Horizon::Infinite, None)
        .expect("Failed to create LQR controller");

    let state = na::DVector::from_column_slice(&[1.0, 2.0]);
    let new_state = controller.integrate(&state, 0.1, 0);

    // The new state should be different from the original state
    // The system should move towards the origin (0, 0)
    assert!(new_state != state, "State should change after integration");
    assert!(new_state.norm() < state.norm(), "System should move towards the origin");
}

#[test]
fn test_error_handling() {
    // Using the continuous-time harmonic oscillator system
    let a = na::DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, -1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[0.0]); // Singular R matrix

    // Attempting to create an LQR controller with a singular R matrix should fail
    let result = LQRController::new(a, b, q, r, LQRType::ContinuousTime, Horizon::Infinite, None);
    assert!(result.is_err(), "Should error with singular R matrix");
}

#[test]
fn test_finite_horizon_without_f() {
    // Using the continuous-time harmonic oscillator system
    let a = na::DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, -1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[1.0]);

    // Attempting to create a finite-horizon LQR controller without providing F should fail
    let result = LQRController::new(a, b, q, r, LQRType::ContinuousTime, Horizon::Finite(10), None);
    assert!(result.is_err(), "Should error when F is not provided for finite horizon");
}