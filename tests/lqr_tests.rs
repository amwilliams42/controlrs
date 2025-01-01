#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix2, Matrix2x1, Matrix1};
    use controlrs::{control::ControlSystem, LQRController};
    use num_traits::Float; // Add this line to import LqrController

    #[test]
    fn test_double_integrator_control() {
        // System: double integrator
        let a = Matrix2::new(
            1.0, 1.0,
            0.0, 1.0
        );
        let b = Matrix2x1::new(0.5, 1.0);
        let q = Matrix2::new(
            1.0, 0.0,
            0.0, 1.0
        );
        let r = Matrix1::new(1.0);

        // Create controller
        let mut lqr = LQRController::new(a, b, q, r, 1e-10).unwrap();

        // Initial state [position = 1.0, velocity = 0.0]
        let mut state = Matrix2x1::new(1.0, 0.0);
        let dt = 0.01;

        // Run system for a few steps
        for _ in 0..100 {
            let u = lqr.step(state, dt);
            
            // Verify control is finite and reasonable
            assert!(u[0].is_finite());
            assert!(u[0].abs() < 10.0);  // Should not be too aggressive

            // Update state (simulate system)
            state = &lqr.a * &state + &lqr.b * u;
            
            // System should be moving towards zero
            assert!(state.norm() < 1.0);
        }

        // Final state should be close to zero
        assert_relative_eq!(state.norm(), 0.0, epsilon = 1e-2);
    }

    #[test]
    fn test_reference_tracking() {
        let a = Matrix2::new(
            1.0, 1.0,
            0.0, 1.0
        );
        let b = Matrix2x1::new(0.5, 1.0);
        let q = Matrix2::new(
            10.0, 0.0,   // Higher weight on position
            0.0, 1.0
        );
        let r = Matrix1::new(0.1);  // Lower weight on control effort

        let mut lqr = LQRController::new(a, b, q, r, 1e-10).unwrap();

        // Set reference state [position = 2.0, velocity = 0.0]
        let reference = Matrix2x1::new(2.0, 0.0);
        lqr.set_reference(reference.clone());

        // Start from origin
        let mut state = Matrix2x1::new(0.0, 0.0);
        let dt = 0.01;

        // Run system
        for _ in 0..200 {
            let u = lqr.step(state, dt);
            state = &lqr.a * &state + &lqr.b * u;
        }

        // Final state should be close to reference
        assert_relative_eq!((state - reference).norm(), 0.0, epsilon = 1e-1);
    }

    #[test]
    fn test_unstable_system_stabilization() {
        // Unstable system
        let a = Matrix2::new(
            2.0, 1.0,
            0.0, 2.0
        );
        let b = Matrix2x1::new(1.0, 1.0);
        let q = Matrix2::identity();
        let r = Matrix1::new(1.0);

        let mut lqr = LQRController::new(a, b, q, r, 1e-10).unwrap();

        // Start from non-zero state
        let mut state = Matrix2x1::new(1.0, 1.0);
        let dt = 0.01;

        let initial_norm = state.norm();
        
        // Run system
        for _ in 0..100 {
            let u = lqr.step(state, dt);
            state = &lqr.a * &state + &lqr.b * u;
        }

        // State should be closer to zero than it started
        assert!(state.norm() < initial_norm);
    }

    #[test]
    fn test_control_constraints() {
        let a = Matrix2::new(
            1.0, 1.0,
            0.0, 1.0
        );
        let b = Matrix2x1::new(0.5, 1.0);
        let q = Matrix2::new(
            100.0, 0.0,  // Very high weight on position
            0.0, 1.0
        );
        let r = Matrix1::new(0.01);  // Very low weight on control effort

        let mut lqr = LQRController::new(a, b, q, r, 1e-10).unwrap();

        // Large initial error
        let state = Matrix2x1::new(10.0, 0.0);
        let dt = 0.01;

        // Control should be large but finite
        let u = lqr.step(state, dt);
        assert!(u[0].is_finite());
        
        // Verify gain matrix properties
        let k: nalgebra::Matrix<f64, nalgebra::U1, nalgebra::U2, nalgebra::ArrayStorage<f64, 1, 2>> = lqr.get_gain();
        assert!(k.norm() > 1.0);  // Should be significant due to high Q/low R
        assert!(k.norm().is_finite());
    }
}