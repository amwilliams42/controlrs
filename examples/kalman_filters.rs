use controlrs::filters::KalmanFilter;
use nalgebra as na;
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the system
    let dt = 0.1;
    let num_steps = 100;
    
    // State transition matrix (constant velocity model)
    let f = na::Matrix4::new(
        1.0, 0.0, dt, 0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    // Observation matrix (we can only measure position, not velocity)
    let h = na::Matrix2x4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    );

    // Process noise covariance
    let q = na::Matrix4::new(
        0.01, 0.0, 0.0, 0.0,
        0.0, 0.01, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.0,
        0.0, 0.0, 0.0, 0.1,
    );

    // Measurement noise covariance
    let r = na::Matrix2::new(
        0.1, 0.0,
        0.0, 0.1,
    );

    // Initial state estimate
    let x0 = na::Vector4::new(0.0, 0.0, 1.0, 1.0);

    // Initial estimate covariance
    let p0 = na::Matrix4::identity();

    // Create Kalman filter
    let mut kf = KalmanFilter::new(x0, p0, f, h, q, r);

    // Generate true path (a straight line with some noise)
    let mut rng = thread_rng();
    let noise_dist = Normal::new(0.0, 0.1).unwrap();

    let mut true_positions = Vec::new();
    let mut measured_positions = Vec::new();
    let mut estimated_positions = Vec::new();
    let mut deviations = Vec::new();

    for i in 0..num_steps {
        let t = i as f64 * dt;
        let true_x = t + noise_dist.sample(&mut rng);
        let true_y = t + noise_dist.sample(&mut rng);
        true_positions.push((true_x, true_y));

        // Generate noisy measurement
        let measured_x = true_x + noise_dist.sample(&mut rng);
        let measured_y = true_y + noise_dist.sample(&mut rng);
        measured_positions.push((measured_x, measured_y));

        // Update Kalman filter
        let measurement = na::Vector2::new(measured_x, measured_y);
        kf.predict();
        kf.update(&measurement);

        // Get estimated position
        let state = kf.state();
        let estimated_x = state[0];
        let estimated_y = state[1];
        estimated_positions.push((estimated_x, estimated_y));

        // Calculate deviation
        let deviation = ((estimated_x - true_x).powi(2) + (estimated_y - true_y).powi(2)).sqrt();
        deviations.push(deviation);
    }

    // Plot results
    let root = BitMapBackend::new("line_following_kalman.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Line Following with Kalman Filter", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..10.0, 0.0..10.0)?;

    chart.configure_mesh().draw()?;

    // Plot true path
    chart.draw_series(LineSeries::new(
        true_positions.iter().map(|&(x, y)| (x, y)),
        &RED,
    ))?
    .label("True Path")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot measured positions
    chart.draw_series(PointSeries::of_element(
        measured_positions.iter().map(|&(x, y)| (x, y)),
        2,
        &BLUE,
        &|c, s, st| {
            return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
        },
    ))?
    .label("Measurements")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot estimated path
    chart.draw_series(LineSeries::new(
        estimated_positions.iter().map(|&(x, y)| (x, y)),
        &GREEN,
    ))?
    .label("Estimated Path")
    .legend(|(x, y)| PathElement::new(vec![(x + 20, y), (x + 40, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    // Plot deviation over time
    let root = BitMapBackend::new("deviation_over_time.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Estimation Deviation Over Time", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..num_steps, 0.0f64..1.0f64)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        deviations.iter().enumerate().map(|(i, &d)| (i, d)),
        &BLUE,
    ))?
    .label("Deviation")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("Average deviation: {}", deviations.iter().sum::<f64>() / deviations.len() as f64);
    println!("Max deviation: {}", deviations.iter().cloned().fold(0./0., f64::max));

    Ok(())
}