use full_palette::PURPLE;
use nalgebra::{OMatrix, OVector, U2, U4};
use plotters::prelude::*;
use controlrs::control::LQRController; 
/// Discrete-time simulation of a drone following a 2D Lissajous curve using LQR.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // System matrices for 2D state: [x, vx, y, vy]^T
    let a = OMatrix::<f64, U4, U4>::new(
        1.0, 1.0, 0.0, 0.0,  // Position x, velocity vx
        0.0, 1.0, 0.0, 0.0,  // Velocity vx
        0.0, 0.0, 1.0, 1.0,  // Position y, velocity vy
        0.0, 0.0, 0.0, 1.0,  // Velocity vy
    );

    let b = OMatrix::<f64, U4, U2>::new(
        0.5, 0.0,  // Control input for x
        1.0, 0.0,  // Control input for vx
        0.0, 0.5,  // Control input for y
        0.0, 1.0,  // Control input for vy
    );

    // Cost matrices
    let q = OMatrix::<f64, U4, U4>::identity() * 10.0;  // Penalize state error
    let r = OMatrix::<f64, U2, U2>::identity();         // Penalize control effort

    // LQR controller setup
    let epsilon = 1e-9;
    let lqr = LQRController::new(a, b, q, r, epsilon, None)?;

    // Simulation parameters
    let steps = 500;  // Number of time steps
    let dt = 0.05;    // Time step size
    let mut state = OVector::<f64, U4>::new(0.0, 0.0, 0.0, 0.0);  // [x, vx, y, vy]

    // Lissajous curve parameters
    let amplitude_x = 10.0;
    let amplitude_y = 5.0;
    let frequency_a = 1.0;
    let frequency_b = 0.5;
    let phase_shift = std::f64::consts::PI / 4.0;

    // Storage for plotting
    let mut actual_positions = vec![];
    let mut target_positions = vec![];

    // Simulation loop: Track the 2D Lissajous curve path.
    for t in 0..steps {
        let time = t as f64 * dt;

        // Lissajous curve: x(t) = A * sin(a * t + delta), y(t) = B * sin(b * t)
        let target_x = amplitude_x * (frequency_a * time + phase_shift).sin();
        let target_y = amplitude_y * (frequency_b * time).sin();
        let target = OVector::<f64, U4>::new(target_x, 0.0, target_y, 0.0);  // [x, vx, y, vy]

        // Compute the control input and update the state.
        let control = lqr.compute_control(&state, &target);
        state = a * state + b * control;

        // Store data for plotting.
        actual_positions.push((state[0], state[2]));  // (x, y)
        target_positions.push((target_x, target_y));  // (x, y)
    }

    // Plot results.
    plot_results(&actual_positions, &target_positions)?;

    Ok(())
}

/// Plots the 2D trajectory: Actual vs Target Path.
fn plot_results(
    actual_positions: &[(f64, f64)],
    target_positions: &[(f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new("lqr_2d_lissajous.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .caption("2D Lissajous Curve: Actual vs Target Path", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(-12.0..12.0, -6.0..6.0)?;

    chart.configure_mesh().draw()?;

    // Plot actual path (red).
    chart.draw_series(LineSeries::new(
        actual_positions.iter().cloned(),
        &RED,
    ))?
    .label("Actual Path")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    // Plot target path (green).
    chart.draw_series(LineSeries::new(
        target_positions.iter().cloned(),
        &GREEN,
    ))?
    .label("Target Path")
    .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    root_area.present()?;
    println!("Results saved to 'lqr_2d_lissajous.png'");

    Ok(())
}