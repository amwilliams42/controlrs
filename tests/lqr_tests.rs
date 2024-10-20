#[cfg(test)]
mod tests {
    use nalgebra::{OMatrix, OVector, U2, U4};
    use controlrs::control::LQRController;

    /// Test LQR creation with valid system matrices.
    #[test]
    fn test_lqr_creation() {
        let a = OMatrix::<f64, U4, U4>::identity();
        let b = OMatrix::<f64, U4, U2>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let q = OMatrix::<f64, U4, U4>::identity() * 10.0;
        let r = OMatrix::<f64, U2, U2>::identity();

        let epsilon = 1e-9;
        let lqr = LQRController::new(a, b, q, r, epsilon, None);
        assert!(lqr.is_ok(), "LQR creation should succeed with valid matrices.");
    }

    /// Test that the control input is computed correctly for simple state and target.
    #[test]
fn test_lqr_control_input() {
    let a = OMatrix::<f64, U4, U4>::new(
        1.0, 0.1, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.1,
        0.0, 0.0, 1.0, 0.1,
        0.0, 0.0, 0.0, 1.0,
    );

    let b = OMatrix::<f64, U4, U2>::new(
        0.0, 0.0,
        1.0, 0.0,
        0.0, 0.0,
        0.0, 1.0,
    );

    let q = OMatrix::<f64, U4, U4>::identity() * 10.0;
    let r = OMatrix::<f64, U2, U2>::identity();

    let epsilon = 1e-9;
    let lqr = LQRController::new(a, b, q, r, epsilon, None).unwrap();

    // Define a non-trivial state and target
    let state = OVector::<f64, U4>::new(5.0, -2.0, 3.0, 0.5);
    let target = OVector::<f64, U4>::new(1.0, 1.0, 1.0, 1.0);

    // Compute control input
    let control = lqr.compute_control(&state, &target);

    // Ensure the control input is non-zero
    assert!(
        control.norm() > 1e-3,
        "Control input should be non-zero for this state and target."
    );
}

#[test]
fn test_recompute_gain() {
    let a = OMatrix::<f64, U4, U4>::identity();
    let b = OMatrix::<f64, U4, U2>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let q = OMatrix::<f64, U4, U4>::identity() * 10.0;
    let r = OMatrix::<f64, U2, U2>::identity();

    let epsilon = 1e-9;
    let mut lqr = LQRController::new(a, b, q, r, epsilon, None).unwrap();

    // Modify the cost matrices
    lqr.q *= 1.1;
    lqr.r *= 0.9;

    // Recompute the gain matrix
    let result = lqr.recompute_gain(epsilon, None);
    assert!(result.is_ok(), "Gain matrix recomputation should succeed.");
}

    /// Test that LQR handles convergence issues gracefully by limiting iterations.
    #[test]
    fn test_lqr_convergence_limit() {
        let a = OMatrix::<f64, U4, U4>::identity();
        let b = OMatrix::<f64, U4, U2>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let q = OMatrix::<f64, U4, U4>::identity() * 10.0;
        let r = OMatrix::<f64, U2, U2>::identity();

        let epsilon = 1e-9;
        let max_iterations = Some(1); // Force premature exit

        let result = LQRController::new(a, b, q, r, epsilon, max_iterations);
        assert!(
            result.is_err(),
            "LQR creation should fail if max iterations are too low for convergence."
        );
    }
}
