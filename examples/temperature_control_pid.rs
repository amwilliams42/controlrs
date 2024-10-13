use controlrs::control::PIDController;
use plotters::prelude::*;
use std::collections::VecDeque;

struct ThermalSystem {
    temperature: f32,
    ambient_temp: f32,
    heat_capacity: f32,
    heat_loss_factor: f32,
    delay_buffer: VecDeque<f32>,
}

impl ThermalSystem {
    fn new(initial_temp: f32, ambient_temp: f32, delay_seconds: f32, dt: f32) -> Self {
        let delay_steps = (delay_seconds / dt) as usize;
        ThermalSystem {
            temperature: initial_temp,
            ambient_temp,
            heat_capacity: 10.0,    // Very low heat capacity for faster response
            heat_loss_factor: 0.02, // Low heat loss for more pronounced effects
            delay_buffer: VecDeque::from(vec![0.0; delay_steps]),
        }
    }

    fn update(&mut self, heat_input: f32, dt: f32) -> f32 {
        // Apply delayed heat input
        let delayed_input = self.delay_buffer.pop_front().unwrap_or(0.0);
        self.delay_buffer.push_back(heat_input);

        let heat_loss = (self.temperature - self.ambient_temp) * self.heat_loss_factor;
        let temp_change = (delayed_input - heat_loss) / self.heat_capacity;
        self.temperature += temp_change * dt;
        self.temperature
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let setpoint = 50.0; // Target temperature in Celsius
    let simulation_time = 6000.0; // 10 minutes
    let dt = 0.1; // Time step in seconds

    // Adjusted PID parameters for more oscillation
    let mut controller = PIDController::builder(10.0, 0.01, 5.0, setpoint)
        .with_output_limits(0.0, 1000.0) // Heater power limits in Watts
        .build();

    let mut system = ThermalSystem::new(20.0, 20.0, 5.0, dt); // 5 second delay
    let mut time_temp_data = Vec::new();
    let mut time_output_data = Vec::new();

    for t in (0..=(simulation_time as i32 * 10)).map(|t| t as f32 * 0.1) {
        let current_temp = system.temperature;
        let control_output = controller.update(current_temp, dt);
        
        let new_temp = system.update(control_output, dt);

        time_temp_data.push((t, new_temp));
        time_output_data.push((t, control_output));

        // Simulate disturbances
        if t == 200.0 {
            system.temperature -= 15.0; // Stronger sudden cooling
        } else if t == 400.0 {
            system.ambient_temp += 10.0; // Larger change in ambient temperature
        }
    }

    // Plotting
    let root = BitMapBackend::new("oscillatory_temperature_control_revised.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Oscillatory Temperature Control Simulation", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .right_y_label_area_size(40)
        .build_cartesian_2d(0f32..simulation_time, 0f32..80f32)?
        .set_secondary_coord(0f32..simulation_time, 0f32..1000f32);

    chart.configure_mesh().draw()?;

    // Draw temperature data
    chart
        .draw_series(LineSeries::new(time_temp_data, &RED))?
        .label("Temperature")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Draw setpoint
    chart
        .draw_series(LineSeries::new(
            (0..=(simulation_time as i32)).map(|x| (x as f32, setpoint)),
            &BLUE,
        ))?
        .label("Setpoint")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw control output
    chart
        .draw_secondary_series(LineSeries::new(time_output_data, &GREEN))?
        .label("Control Output")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    println!("The plot has been saved as oscillatory_temperature_control_revised.png");

    Ok(())
}