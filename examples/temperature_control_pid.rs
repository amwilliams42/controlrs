use std::time::Instant;

use controlrs::{control::ControlSystem, PIDController};

struct HeatingSystem {
    current_temp: f64,
    thermal_mass: f64,
    heat_loss_rate: f64,
}

impl HeatingSystem {
    fn new() -> Self {
        HeatingSystem {
            current_temp: 20.0,
            thermal_mass: 100.0,
            heat_loss_rate: 0.1,
        }
    }
    fn apply_heat(&mut self, power: f64, dt: f64) {

        let heat_loss = self.heat_loss_rate * (self.current_temp - 20.0);
        let delta_temp = (power - heat_loss) / self.thermal_mass * dt;
        self.current_temp += delta_temp;
    }
}

fn simulate_temperature_control(time_step: f64, total_time: f64) -> Vec<(f64, f64, f64)> {
    let mut system = HeatingSystem::new();
    let mut controller = PIDController::new(
        1.0, 
        0.05, 
        100.0, 
        60.0);
    let steps = (total_time / time_step) as usize;
    let mut results = Vec::with_capacity(steps);

    for i in 0..steps {
        let time = i as f64 * time_step;
        let current_temp = system.current_temp;
        
        let power = controller.step(current_temp, time_step);
        let clamped_power = power.clamp(0.0, 1000.0);
        

        system.apply_heat(clamped_power, time_step);
        results.push((time, current_temp, clamped_power));
    }

    results
}

fn main() {
    let results = simulate_temperature_control(0.1, 100.0);

    for (time, temp, power) in results {
        println!("Time: {:.1} Temp: {:.2} Power: {:.2}", time, temp, power);
    }
}