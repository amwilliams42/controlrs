use proptest::prelude::*;
use controlrs::control::PIDController;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use controlrs::control::ControlSystem;
    use nalgebra::ComplexField;

    // Helper function to create a default PID controller for testing
    fn create_test_pid() -> PIDController<f64> {
        PIDController::new(1.0, 0.1, 0.01, 0.0)
    }

    #[test]
    fn test_proportional_only() {
        let mut pid = PIDController::new(1.0, 0.0, 0.0, 2.0);
        
        // Error should be 2.0 (setpoint) - 1.0 (measurement) = 1.0
        let output = pid.step(1.0, 0.1);
        
        // With Kp = 1.0, output should equal error
        assert_relative_eq!(output, 1.0);
    }

    #[test]
    fn test_integral_only() {
        let mut pid = PIDController::new(0.0, 1.0, 0.0, 1.0);
        let dt = 0.1;
        
        // With setpoint = 1.0 and measurement = 0.0:
        // Error is constant at 1.0
        // Integral accumulates: error * dt each step
        // Output = Ki * integral
        
        let output1 = pid.step(0.0, dt); // integral = 0.1, output = 0.1
        assert_relative_eq!(output1, 0.1);
        
        let output2 = pid.step(0.0, dt); // integral = 0.2, output = 0.2
        assert_relative_eq!(output2, 0.2);
        
        let output3 = pid.step(0.0, dt); // integral = 0.3, output = 0.3
        assert_relative_eq!(output3, 0.3);
        
        // For verification, add a few more steps
        let output4 = pid.step(0.0, dt); // integral = 0.4, output = 0.4
        assert_relative_eq!(output4, 0.4);
    }

    #[test]
    fn test_derivative_only() {
        let mut pid = PIDController::new(0.0, 0.0, 1.0, 0.0);
        let dt = 0.1;
        
        // First step (derivative undefined, should return 0)
        let output1 = pid.step(0.0, dt);
        assert_relative_eq!(output1, 0.0);
        
        // Error going from 0 to 1 in 0.1 seconds = rate of 10
        let output2 = pid.step(1.0, dt);
        assert_relative_eq!(output2, -10.0);
    }

    #[test]
    fn test_zero_dt() {
        let mut pid = create_test_pid();
        
        // Should not panic or produce NaN
        let output = pid.step(1.0, 0.0);
        assert!(!output.is_nan());
    }

    #[test]
    fn test_negative_dt() {
        let mut pid = create_test_pid();
        
        // Controller should work with negative dt, though this isn't physically meaningful
        let output = pid.step(1.0, -0.1);
        assert!(!output.is_nan());
    }

    #[test]
    fn test_steady_state() {
        let mut pid = PIDController::new(1.0, 0.1, 0.0, 1.0);
        let dt = 0.1;
        
        // Run until we reach approximate steady state
        let mut prev_output = 0.0;
        let mut steady_state_count = 0;
        
        for _ in 0..100 {
            let output = pid.step(1.0, dt);
            
            if (output - prev_output).abs() < 1e-6 {
                steady_state_count += 1;
                if steady_state_count > 5 {
                    // We've reached steady state
                    assert_relative_eq!(output, 0.0, epsilon = 1e-6);
                    return;
                }
            } else {
                steady_state_count = 0;
            }
            
            prev_output = output;
        }
        
        panic!("Failed to reach steady state");
    }

    #[test]
    fn test_with_different_number_types() {
        // Test with f32
        let mut pid_f32 = PIDController::<f32>::new(1.0, 0.1, 0.01, 0.0);
        let output_f32 = pid_f32.step(0.5, 0.1);
        assert!(!output_f32.is_nan());

        // Test with f64
        let mut pid_f64 = PIDController::<f64>::new(1.0, 0.1, 0.01, 0.0);
        let output_f64 = pid_f64.step(0.5, 0.1);
        assert!(!output_f64.is_nan());
    }

    #[test]
    fn test_numerical_stability() {
        let mut pid = create_test_pid();
        
        // Test with very small numbers
        let output_small = pid.step(1e-10, 1e-10);
        assert!(!output_small.is_nan());
        
        // Test with very large numbers
        let output_large = pid.step(1e10, 1e10);
        assert!(!output_large.is_nan());
    }

    // Property-based tests using proptest
    #[cfg(test)]
    mod proptest {
        use super::*;

        proptest! {
            #[test]
            fn test_pid_properties(
                kp in -100.0..100.0f64,
                ki in -100.0..100.0f64,
                kd in -100.0..100.0f64,
                setpoint in -100.0..100.0f64,
                measurement in -100.0..100.0f64,
                dt in 0.0001..1.0f64
            ) {
                let mut pid = PIDController::new(kp, ki, kd, setpoint);
                let output = pid.step(measurement, dt);
                
                // Output should not be NaN
                prop_assert!(!output.is_nan());
                
                // Output should be finite
                prop_assert!(output.is_finite());
                
                // If error is zero and no prior state, output should be zero
                if measurement == setpoint && pid.integral == 0.0 && pid.previous_error == 0.0 {
                    prop_assert_eq!(output, 0.0);
                }
            }
        }
    }
}