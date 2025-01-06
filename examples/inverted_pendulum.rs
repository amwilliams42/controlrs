use controlrs::control::LQRController;
use nalgebra::{Matrix1, Matrix2, Vector2};
use plotters::prelude::*;

pub trait SystemDynamics {
    type State;
    type Control;

    fn propogate(&self, state: 
        &Self::State, 
        control: &Self::Control,
        dt: f64) 
        -> Self::State;
}
pub trait Controller<S, C> {
    fn compute(&self, state: &S) -> C;
}

// Inverted Pendulum Model

pub struct InvertedPendulum {
    mass: f64,
    length: f64,
    damping: f64,
    gravity: f64,
}

impl SystemDynamics for InvertedPendulum {
    type State = Vector2<f64>;
    type Control = f64;

    fn propogate(&self, state: 
            &Self::State, control: 
            &Self::Control, 
            dt: f64) 
            -> Self::State {
        let theta = state[0];
        let theta_dot = state[1];

        // Acceleration due to Gravity
        let theta_ddot = (self.gravity * theta.sin() - 
            self.damping * theta_dot + control) /
            (self.mass * self.length.powi(2));

        // Euler Integration
        Vector2::new(
            theta + theta_dot * dt,
            theta_dot + theta_ddot * dt
        )
    }
}

fn main() {
    let pendulum = InvertedPendulum {
        mass: 5.0,
        length: 1.0,
        damping: 0.75,
        gravity: 9.81,
    };

    let dt = 0.001;

    let a = Matrix2::new(0.0, 1.0, 9.81, 0.0);
    let b = Vector2::new(0.0, 1.0);
    let q = Matrix2::new(100.0, 0.0, 0.0, 10.0);
    let r = Matrix1::from_element(0.0001);

    let lqr = LQRController::new(a, b, q, r, 1e-6, None).unwrap();

    let mut state = Vector2::new(1.0, 0.0);
    let target = Vector2::new(0.0, 0.0);

    let root = BitMapBackend::new("inverted_pendulum.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Inverted Pendulum", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..100.0, -2.0..2.0).unwrap();


    chart.configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Angle (rad)")
        .draw().unwrap();

    let mut data = Vec::new();

    for t in 0..100000 {
        let control = lqr.compute_control(&state, &target);
        state = pendulum.propogate(&state, &control[0], dt);
        data.push((t as f64 * dt, state[0]));
    }

    chart.draw_series(LineSeries::new(data, &RED)).unwrap();
}