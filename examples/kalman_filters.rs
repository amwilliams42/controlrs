use nalgebra::{OMatrix, OVector, U2};
use controlrs::KalmanFilter;

fn main() {
    // Define the time step (dt).
    let dt = 1.0; // 1 second

    // Initial state: [position, velocity] = [0.0, 50.0]
    let initial_state = OVector::<f64, U2>::from_vec(vec![0.0, 50.0]);

    // Initial state covariance matrix (uncertainty in initial estimates).
    let initial_covariance = OMatrix::<f64, U2, U2>::identity() * 1000.0;

    // State transition matrix (models how the state evolves over time).
    let state_transition = OMatrix::<f64, U2, U2>::from_row_slice(&[
        1.0, dt,  // Position = Position + Velocity * dt
        0.0, 1.0, // Velocity remains constant
    ]);

    // Observation matrix (how we measure the state: we only observe position).
    let observation_matrix = OMatrix::<f64, U2, U2>::from_row_slice(&[
        1.0, 0.0, // We observe position directly
        0.0, 0.0, // We don't observe velocity
    ]);

    // Process noise covariance (models uncertainty in the process model).
    let process_noise = OMatrix::<f64, U2, U2>::identity() * 0.1;

    // Measurement noise covariance (models uncertainty in the measurements).
    let measurement_noise = OMatrix::<f64, U2, U2>::identity() * 10.0;

    // Create the Kalman filter with the initial parameters.
    let mut kf = KalmanFilter::new(
        initial_state,
        initial_covariance,
        state_transition,
        observation_matrix,
        process_noise,
        measurement_noise,
    );

    // Simulated noisy measurements of position.
    let measurements = vec![0.0, 52.0, 110.0, 160.0, 210.0, 300.0];

    // Run the Kalman filter with the simulated measurements.
    for (i, &measurement) in measurements.iter().enumerate() {
        // Predict the next state.
        kf.predict();

        // Create a measurement vector from the noisy position measurement.
        let measurement_vector = OVector::<f64, U2>::from_vec(vec![measurement, 0.0]);

        // Update the Kalman filter with the measurement.
        kf.update(&measurement_vector);

        // Retrieve the current state estimate (position and velocity).
        let state = kf.state();
        println!(
            "Step {}: Position = {:.2}, Velocity = {:.2}",
            i, state[0], state[1]
        );
    }
}
