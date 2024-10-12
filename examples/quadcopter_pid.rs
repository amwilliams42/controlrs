use controlrs::control::PIDController;
use plotters::prelude::*;
use rand::Rng;

const G: f32 = 9.81; // Acceleration due to gravity (m/s^2)

struct Quadcopter {
    mass: f32,      // kg
    altitude: f32,  // m
    velocity: f32,  // m/s
    thrust: f32,    // N
}

impl Quadcopter {
    fn new(mass: f32, initial_altitude: f32) -> Self {
        Quadcopter {
            mass,
            altitude: initial_altitude,
            velocity: 0.0,
            thrust: mass * G, // Initial thrust to hover
        }
    }

    fn update(&mut self, dt: f32) {
        let acceleration = (self.thrust - self.mass * G) / self.mass;
        self.velocity += acceleration * dt;
        self.altitude += self.velocity * dt;

        // Ensure altitude doesn't go below 0
        if self.altitude < 0.0 {
            self.altitude = 0.0;
            self.velocity = 0.0;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let simulation_time = 30.0; // Total simulation time in seconds
    let dt = 0.01; // Time step in seconds

    let setpoint_altitude = 10.0; // Target altitude in meters
    let quadcopter_mass = 1.0; // Mass in kg

    // PID controller for altitude
    let mut altitude_controller = PIDController::builder(1.0, 0.1, 0.5, setpoint_altitude)
        .with_output_limits(-5.0, 5.0) // Limits on velocity setpoint
        .build();

    // PID controller for velocity
    let mut velocity_controller = PIDController::builder(5.0, 0.1, 1.0, 0.0)
        .with_output_limits(0.0, quadcopter_mass * G * 2.0) // Limits on thrust
        .build();

    let mut quadcopter = Quadcopter::new(quadcopter_mass, 0.0);
    let mut time_data = Vec::new();
    let mut altitude_data = Vec::new();
    let mut velocity_data = Vec::new();
    let mut thrust_data = Vec::new();

    for t in (0..=(simulation_time as i32 * 100)).map(|t| t as f32 * 0.01) {
        // Altitude control loop
        let altitude_error = altitude_controller.update(quadcopter.altitude, dt);
        velocity_controller.setpoint = altitude_error;

        // Velocity control loop
        let thrust_adjustment = velocity_controller.update(quadcopter.velocity, dt);
        quadcopter.thrust = thrust_adjustment;

        // Add some random disturbance
        let disturbance = rng.gen_range(-0.5..0.5);
        quadcopter.thrust += disturbance;

        // Simulate wind gust at t=15s
        if t >= 15.0 && t < 15.5 {
            quadcopter.thrust -= 2.0;
        }

        quadcopter.update(dt);

        time_data.push(t);
        altitude_data.push(quadcopter.altitude);
        velocity_data.push(quadcopter.velocity);
        thrust_data.push(quadcopter.thrust);
    }

    // Plotting
    let root = BitMapBackend::new("quadcopter_altitude_control.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Quadcopter Altitude Control", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..simulation_time, -2f32..15f32)?;

    chart.configure_mesh().draw()?;

    // Plot altitude
    chart
        .draw_series(LineSeries::new(
            time_data.iter().zip(altitude_data.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))?
        .label("Altitude (m)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot velocity
    chart
        .draw_series(LineSeries::new(
            time_data.iter().zip(velocity_data.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?
        .label("Velocity (m/s)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot thrust
    chart
        .draw_series(LineSeries::new(
            time_data.iter().zip(thrust_data.iter()).map(|(&x, &y)| (x, y / quadcopter_mass / G)),
            &GREEN,
        ))?
        .label("Thrust (G)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // Plot setpoint
    chart
        .draw_series(LineSeries::new(
            time_data.iter().map(|&x| (x, setpoint_altitude)),
            &BLACK,
        ))?
        .label("Setpoint")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("The plot has been saved as quadcopter_altitude_control.png");

    Ok(())
}