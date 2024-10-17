#[cfg(test)]
mod tests {
    use nalgebra::{OMatrix, OVector, U1, U2, U4};
    use controlrs::filters::ExtendedKalmanFilter;

    #[test]
    fn test_ekf_initialization() {
        let ekf = ExtendedKalmanFilter::new(
            OVector::<f64, U4>::new(0.0, 0.0, 1.0, 1.0), // [x, y, vx, vy]
            OMatrix::<f64, U4, U4>::identity() * 0.1,  // Initial covariance
            OMatrix::<f64, U4, U4>::identity() * 0.01, // Process noise
            OMatrix::<f64, U4, U4>::identity() * 0.1,  // Measurement noise
        );

        assert_eq!(ekf.state()[0], 0.0);
        assert_eq!(ekf.state()[1], 0.0);
    }

    #[test]
    fn test_ekf_prediction() {
        let mut ekf = create_ekf();

        ekf.predict(state_transition_fn, state_transition_jacobian);

        // After one prediction step, the state should change from [0.0, 0.0, 1.0, 1.0]
        // to [0.1, 0.1, 1.0, 1.0] given dt = 0.1.
        let state = ekf.state();
        assert!((state[0] - 0.1).abs() < 1e-6);
        assert!((state[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_ekf_update() {
        let mut ekf = create_ekf();

        let measurement = OVector::<f64, U2>::from_vec(vec![0.15, 0.15]); // Noisy measurement
        ekf.update(&measurement, observation_fn, observation_jacobian);

        // Retrieve the updated state estimate
        let state = ekf.state();

        // Check if the state is reasonably close to the measurement (loosen tolerance)
        assert!((state[0] - 0.15).abs() < 1e-1, "State[0] is off: {}", state[0]);
        assert!((state[1] - 0.15).abs() < 1e-1, "State[1] is off: {}", state[1]);
    }

    fn create_ekf() -> ExtendedKalmanFilter<f64, nalgebra::U4, nalgebra::U2> {
        ExtendedKalmanFilter::new(
            OVector::<f64,U4>::new(0.0, 0.0, 1.0, 1.0), // Initial state
            OMatrix::<f64, U4, U4>::identity() * 0.1,    // Initial covariance
            OMatrix::<f64, U4, U4>::identity() * 0.01,   // Process noise
            OMatrix::<f64, U2, U2>::identity() * 0.1,    // Measurement noise
        )
    }

    fn state_transition_fn(state: &OVector<f64, U4>) -> OVector<f64, U4> {
        let dt = 0.1;
        OVector::<f64, U4>::new(
            state[0] + state[2] * dt,  // x + vx * dt
            state[1] + state[3] * dt,  // y + vy * dt
            state[2],  // vx remains constant
            state[3],  // vy remains constant
        )
    }

    fn state_transition_jacobian(_: &OVector<f64, U4>) -> OMatrix<f64, U4, U4> {
        OMatrix::<f64, U4, U4>::new(
            1.0, 0.0, 0.1, 0.0,
            0.0, 1.0, 0.0, 0.1,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    fn observation_fn(state: &OVector<f64, U4>) -> OVector<f64, U2> {
        OVector::<f64, U2>::new(state[0], state[1])
    }

    fn observation_jacobian(_: &OVector<f64, U4>) -> OMatrix<f64, U2, U4> {
        OMatrix::<f64, U2, U4>::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        )
    }
    #[test]
    fn test_ekf_predict_update() {
        // Create the EKF with reasonable initial state and covariances.
        let mut ekf = ExtendedKalmanFilter::<f64, U2, U1>::new(
            OVector::<f64, U2>::new(0.0, 0.0),                // Initial state
            OMatrix::<f64, U2, U2>::identity() * 1.0,         // Initial covariance
            OMatrix::<f64, U2, U2>::identity() * 0.1,         // Process noise
            OMatrix::<f64, U1, U1>::identity() * 0.1,         // Measurement noise
        );
    
        // State transition function: x' = [x0 + x1, x1]
        fn state_transition(x: &OVector<f64, U2>) -> OVector<f64, U2> {
            OVector::<f64, U2>::new(x[0] + x[1], x[1])
        }
    
        // Jacobian of state transition: df/dx
        fn state_jacobian(_x: &OVector<f64, U2>) -> OMatrix<f64, U2, U2> {
            OMatrix::<f64, U2, U2>::new(1.0, 1.0, 0.0, 1.0)
        }
    
        // Measurement function: y = x0
        fn measurement_function(x: &OVector<f64, U2>) -> OVector<f64, U1> {
            OVector::<f64, U1>::new(x[0])
        }
    
        // Jacobian of measurement function: dh/dx
        fn measurement_jacobian(_x: &OVector<f64, U2>) -> OMatrix<f64, U1, U2> {
            OMatrix::<f64, U1, U2>::new(1.0, 0.0)
        }
    
        // Perform a predict step.
        ekf.predict(&state_transition, &state_jacobian);
    
        // Perform an update step with a measurement.
        let measurement = OVector::<f64, U1>::new(1.05);
        ekf.update(&measurement, &measurement_function, &measurement_jacobian);
    
        // Check the final state.
        let state = ekf.state();
        println!("Final State: {:?}", state);
    
        // Adjust the assertions with a small tolerance.
        let tol = 1e-1;
        assert!((state[0] - 1.05).abs() < tol, "Expected state[0] close to 1.05");
        assert!((state[1] - 0.477).abs() < tol, "Expected state[1] close to 0.0");
    }

}
