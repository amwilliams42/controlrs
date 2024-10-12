use controlrs::control::PIDController;

#[test]
fn test_pid_controller_basic_functionality() {
    let mut controller = PIDController::builder(0.5, 0.01, 0.1, 10.0)
        .with_output_limits(-100.0, 100.0)
        .build();

    let mut measurement = 0.0;
    let dt = 0.1;
    let max_iterations = 200;
    let tolerance = 0.1; // 1% of setpoint

    for iteration in 0..max_iterations {
        let output = controller.update(measurement, dt);
        measurement += output * dt; // More realistic system response

        // Check if we've converged
        if (measurement - controller.setpoint).abs() < tolerance {
            println!("Converged after {} iterations", iteration);
            return; // Test passes if we converge within max_iterations
        }
    }

    panic!("Did not converge within {} iterations. Final value: {}", max_iterations, measurement);
}

#[test]
fn test_pid_term_limits() {
    let mut controller = PIDController::builder(10.0, 1.0, 1.0, 100.0)
        .with_p_limits(-30.0, 30.0)
        .with_i_limits(-20.0, 20.0)
        .with_d_limits(-10.0, 10.0)
        .build();

    let output = controller.update(0.0, 0.1);
    
    assert!((output - 60.0).abs() < 0.001);
}

#[test]
fn test_pid_integral_windup_prevention() {
    let mut controller = PIDController::builder(1.0, 1.0, 0.0, 100.0)
        .with_i_limits(-10.0, 10.0)
        .build();

    for _ in 0..100 {
        controller.update(0.0, 0.1);
    }

    assert!((controller.integral - 10.0).abs() < 0.001);
}

#[test]
fn test_pid_derivative_on_measurement() {
    let mut controller = PIDController::builder(1.0, 0.0, 1.0, 10.0).build();

    let output1 = controller.update(0.0, 0.1);
    let output2 = controller.update(5.0, 0.1);

    assert!(output2 < output1);
}

#[test]
fn test_direct_field_access() {
    let mut controller = PIDController::builder(1.0, 0.1, 0.05, 10.0).build();
    
    assert_eq!(controller.kp, 1.0);
    assert_eq!(controller.ki, 0.1);
    assert_eq!(controller.kd, 0.05);
    assert_eq!(controller.setpoint, 10.0);

    controller.setpoint = 15.0;
    assert_eq!(controller.setpoint, 15.0);

    controller.update(5.0, 0.1);
    assert_eq!(controller.previous_measurement, 5.0);
}