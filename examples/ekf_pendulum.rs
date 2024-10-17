use nalgebra::{OMatrix, OVector, U2};
use controlrs::ExtendedKalmanFilter;
use plotters::prelude::*;
use std::f64::consts::PI;

/// Simulates and tracks a pendulum using an EKF.
fn main() {
    // Pendulum parameters
    let dt = 0.1; // Time step
    let g = 9.81; // Gravitational acceleration
    let l = 1.0; // Pendulum length
    let b = 0.1; // Damping factor

    // Initialize EKF
    let mut ekf = ExtendedKalmanFilter::new(
        OVector::<f64, U2>::from_vec(vec![PI / 4.0, 0.0]), // Initial state: 45 degrees, 0 velocity
        OMatrix::<f64, U2, U2>::identity() * 0.1, // Initial covariance
        OMatrix::<f64, U2, U2>::identity() * 0.01, // Process noise
        OMatrix::<f64, U2, U2>::identity() * 0.1, // Measurement noise
    );

    // Simulated measurements (noisy angle readings)
    let mut measurements = vec![];
    let mut true_states = vec![];
    let mut estimated_states = vec![];

    let mut state = OVector::<f64, U2>::from_vec(vec![PI / 4.0, 0.0]); // Initial true state

    for _ in 0..100 {
        // Simulate next state using the non-linear dynamics
        state = pendulum_dynamics(&state, dt, g, l, b);
        true_states.push(state.clone());

        // Generate a noisy measurement (only angle is observed)
        let noisy_measurement = OVector::<f64, U2>::from_vec(vec![state[0] + noise(), 0.0]);
        measurements.push(noisy_measurement[0]);

        // EKF predict and update steps
        ekf.predict(|x| pendulum_dynamics(x, dt, g, l, b), pendulum_jacobian);
        ekf.update(&noisy_measurement, observation_fn, observation_jacobian);

        // Store estimated state for plotting
        estimated_states.push(ekf.state().clone());
    }

    // Plot results
    plot_results(&measurements, &true_states, &estimated_states).unwrap();
}

/// Non-linear pendulum dynamics with simple Euler integration.
fn pendulum_dynamics(
    state: &OVector<f64, U2>,
    dt: f64,
    g: f64,
    l: f64,
    b: f64,
) -> OVector<f64, U2> {
    let theta = state[0];
    let theta_dot = state[1];

    let theta_ddot = -g / l * theta.sin() - b * theta_dot;

    OVector::<f64, U2>::from_vec(vec![
        theta + theta_dot * dt, // Integrate angle
        theta_dot + theta_ddot * dt, // Integrate velocity
    ])
}

/// Jacobian of the pendulum dynamics (linearized).
fn pendulum_jacobian(_: &OVector<f64, U2>) -> OMatrix<f64, U2, U2> {
    OMatrix::<f64, U2, U2>::from_row_slice(&[
        1.0, 0.1, // Partial derivatives wrt state variables
        -0.981, 0.9,
    ])
}

/// Observation function: only measures the angle (theta).
fn observation_fn(state: &OVector<f64, U2>) -> OVector<f64, U2> {
    OVector::<f64, U2>::from_vec(vec![state[0], 0.0])
}

/// Jacobian of the observation function.
fn observation_jacobian(_: &OVector<f64, U2>) -> OMatrix<f64, U2, U2> {
    OMatrix::<f64, U2, U2>::from_row_slice(&[
        1.0, 0.0,
        0.0, 0.0,
    ])
}

/// Generates small Gaussian noise for the measurements.
fn noise() -> f64 {
    let mut rng = rand::thread_rng();
    rand::Rng::gen_range(&mut rng, -0.05..0.05)
}

/// Plots the results using `plotters`.
fn plot_results(
    measurements: &[f64],
    true_states: &[OVector<f64, U2>],
    estimated_states: &[OVector<f64, U2>],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("pendulum_tracking.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Pendulum Tracking with EKF", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..100, -PI..PI)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..).zip(true_states.iter()).map(|(x, state)| (x, state[0])),
            &BLUE,
        ))?
        .label("True Angle")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            (0..).zip(estimated_states.iter()).map(|(x, state)| (x, state[0])),
            &RED,
        ))?
        .label("Estimated Angle")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            (0..).zip(measurements.iter()).map(|(x, &meas)| (x, meas)),
            &GREEN,
        ))?
        .label("Measurements")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));


    Ok(())
}
