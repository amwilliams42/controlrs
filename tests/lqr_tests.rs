#[cfg(test)]
mod tests {
    use nalgebra::{OMatrix, OVector, U1, U2, U3};
    use controlrs::control::lqr::{LQRController, LQRType, Horizon};

    // Helper function to create a simple 2x2 continuous-time LQR controller.
    fn create_continuous_lqr() -> LQRController<f64, U2, U1> {
        let a = OMatrix::<f64, U2, U2>::new(0.0, 1.0, 0.0, 0.0); // Simple double integrator
        let b = OMatrix::<f64, U2, U1>::new(0.0, 1.0); // Control input
        let q = OMatrix::<f64, U2, U2>::identity(); // State cost
        let r = OMatrix::<f64, U1, U1>::identity() * 0.1; // Control cost

        LQRController::new(a, b, q, r, LQRType::ContinuousTime, Horizon::Infinite, None)
            .expect("Failed to create continuous-time LQR controller")
    }

    // Helper function to create a 3x1 discrete-time LQR controller with a finite horizon.
    fn create_discrete_finite_horizon_lqr() -> LQRController<f64, U3, U1> {
        let a = OMatrix::<f64, U3, U3>::new(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0);
        let b = OMatrix::<f64, U3, U1>::new(0.0, 0.0, 1.0);
        let q = OMatrix::<f64, U3, U3>::identity();
        let r = OMatrix::<f64, U1, U1>::identity() * 0.01;
        let f = OMatrix::<f64, U3, U3>::identity(); // Terminal cost

        LQRController::new(
            a,
            b,
            q,
            r,
            LQRType::DiscreteTime,
            Horizon::Finite(10),
            Some(f),
        )
        .expect("Failed to create discrete-time finite-horizon LQR controller")
    }

    #[test]
    fn test_continuous_infinite_horizon_lqr_gain() {
        let lqr = create_continuous_lqr();
        let k = lqr.get_feedback_gain(0);
        // Check the dimensions of the gain matrix K (should be 1x2)
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn test_compute_control_input_continuous() {
        let lqr = create_continuous_lqr();
        let state = OVector::<f64, U2>::new(1.0, 0.0); // Initial state: [1, 0]
        let control = lqr.compute_control(&state, 0);
        // Verify the control input's dimension
        assert_eq!(control.nrows(), 1);
        assert_eq!(control.ncols(), 1);
    }

    #[test]
    fn test_discrete_finite_horizon_lqr_gain_sequence() {
        let lqr = create_discrete_finite_horizon_lqr();
        let k1 = lqr.get_feedback_gain(0);
        let k2 = lqr.get_feedback_gain(5);
        let k_final = lqr.get_feedback_gain(9);

        // Check the dimensions of the first gain matrix (should be 1x3)
        assert_eq!(k1.nrows(), 1);
        assert_eq!(k1.ncols(), 3);

        // Verify that the final gain matrix is well-formed
        assert_eq!(k_final.nrows(), 1);
        assert_eq!(k_final.ncols(), 3);
    }

    #[test]
    fn test_discrete_finite_horizon_control_input() {
        let lqr = create_discrete_finite_horizon_lqr();
        let state = OVector::<f64, U3>::new(1.0, 0.0, 0.0); // Initial state: [1, 0, 0]
        let control = lqr.compute_control(&state, 0);

        // Verify control input dimension
        assert_eq!(control.nrows(), 1);
        assert_eq!(control.ncols(), 1);
    }

    #[test]
    fn test_discrete_finite_horizon_control_evolution() {
        let lqr = create_discrete_finite_horizon_lqr();
        let mut state = OVector::<f64, U3>::new(1.0, 0.0, 0.0); // Initial state

        // Simulate the evolution of the state over 10 steps
        for t in 0..10 {
            let control = lqr.compute_control(&state, t);
            state = lqr.integrate(&state, 1.0, t); // Discrete-time evolution
            println!("Step {}: State = {:?}", t, state);
        }
    }

    #[test]
    fn test_lqr_infinite_vs_finite_horizon_behavior() {
        let continuous_lqr = create_continuous_lqr();
        let discrete_finite_lqr = create_discrete_finite_horizon_lqr();

        let initial_state = OVector::<f64, U2>::new(1.0, 1.0);
        let control_inf = continuous_lqr.compute_control(&initial_state, 0);

        let initial_state_finite = OVector::<f64, U3>::new(1.0, 0.0, 0.0);
        let control_finite = discrete_finite_lqr.compute_control(&initial_state_finite, 0);

        println!("Infinite Horizon Control: {:?}", control_inf);
        println!("Finite Horizon Control: {:?}", control_finite);
    }
}
