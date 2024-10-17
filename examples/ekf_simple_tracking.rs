use nalgebra::{OMatrix, OVector, U2, U4};
use controlrs::ExtendedKalmanFilter;

fn main() {
    let mut ekf = ExtendedKalmanFilter::new(
        OVector::<f64, U4>::from_column_slice(&[0.0, 0.0, 1.0, 1.0]),  // [x, y, vx, vy]
        OMatrix::<f64, U4, U4>::identity() * 0.1,  // Initial covariance
        OMatrix::<f64, U4, U4>::identity() * 0.01, // Process noise
        OMatrix::<f64, U2, U2>::identity() * 0.1,  // Measurement noise
    );

    let measurements = vec![
        OVector::<f64, U2>::from_column_slice(&[0.1, 0.1]),
        OVector::<f64, U2>::from_column_slice(&[0.2, 0.2]),
        OVector::<f64, U2>::from_column_slice(&[0.4, 0.3]),
    ];

    for measurement in measurements {
        ekf.predict(state_transition_fn, state_transition_jacobian);
        ekf.update(&measurement, observation_fn, observation_jacobian);

        println!("Estimated state: {:?}", ekf.state());
    }
}

fn state_transition_fn(state: &OVector<f64, U4>) -> OVector<f64, U4> {
    let dt = 0.1;
    OVector::<f64,U4>::from_vec( vec![
        state[0] + state[2] * dt,  // x + vx * dt
        state[1] + state[3] * dt,  // y + vy * dt
        state[2],  // vx remains constant
        state[3],  // vy remains constant
    ])
}

fn state_transition_jacobian(_: &OVector<f64, U4>) -> OMatrix<f64, U4, U4> {
    OMatrix::<f64, U4, U4>::from_row_slice(&[
        1.0, 0.0, 0.1, 0.0,
        0.0, 1.0, 0.0, 0.1,
        0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 0.1,
    ])
}

fn observation_fn(state: &OVector<f64, U4>) -> OVector<f64, U2> {
    OVector::<f64, U2> ::from_vec(vec![state[0], state[1]])
}

fn observation_jacobian(_: &OVector<f64, U4>) -> OMatrix<f64, U2, U4> {
    OMatrix::<f64, U2, U4>::from_row_slice(&[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ])
}
