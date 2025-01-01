use nalgebra::RealField;
use num_traits::Signed;

/*
# Overview

A PID controller continuously calculates an error value as the difference between a desired setpoint and a measured process variable. The controller attempts to minimize the error by adjusting the process control inputs. The PID controller algorithm involves three separate parameters: the proportional, integral, and derivative values, denoted as `kp`, `ki`, and `kd` respectively.

- **Proportional (P)**: The proportional term produces an output value that is proportional to the current error value. It is calculated by multiplying the error by a constant `kp`.
- **Integral (I)**: The integral term is concerned with the accumulation of past errors. If the error has been present for a long time, the integral term will accumulate a significant value. It is calculated by multiplying the accumulated error by a constant `ki`.
- **Derivative (D)**: The derivative term is a prediction of future error, based on its rate of change. It is calculated by multiplying the rate of change of the error by a constant `kd`.

# Usage

To use the PID controller, create an instance of `PIDController` with the desired gains and setpoint. The controller can then be used to compute the control output based on the current process variable.

# Example

```rust
use pid::PIDController;
use pid::Number;

let mut pid = PIDController::new(1.0, 0.1, 0.01, 10.0);
let control_output = pid.update(5.0, 1.0);
```
*/
use crate::Number;
use super::ControlSystem;


#[derive(Debug, Clone, Copy)]
pub struct PIDController<N: Number>{
    pub kp: N,                 // Proportional Gain
    pub ki: N,                 // Integral Gain
    pub kd: N,                 // Derivative Gain
    pub setpoint: N,           // Desired setpoint
    pub integral: N,           // Accumulated Integral term
    pub previous_error: N,     // error from previous update
    pub prev_filtered_derivative: N, // Previous filtered derivative

    pub integral_limit: Option<N>, // Integral windup limit
    pub derivative_filter_coefficient: N, // Derivative filter
    pub max_rate: Option<N>,       // Maximum rate of change
    pub prev_output: N,            // Previous output

    pub deadband: Option<N>,               // Deadband
}

impl<N: Number> PIDController<N> {
    pub fn new(kp: N, ki: N, kd: N, setpoint: N) -> Self {
        PIDController {
            kp,
            ki,
            kd,
            setpoint,
            integral: N::zero(),
            previous_error: N::zero(),
            prev_filtered_derivative: N::zero(),
            integral_limit: None,
            derivative_filter_coefficient: N::one(),
            max_rate: None,
            prev_output: N::zero(),
            deadband: None,

        }
    }

    pub fn with_integral_limit(mut self, limit: N) -> Self {
        self.integral_limit = Some(limit);
        self
    }

    pub fn with_derivative_filter(mut self, coefficient: N) -> Self {
        self.derivative_filter_coefficient = coefficient;
        self
    }

    pub fn with_rate_limit(mut self, limit: N) -> Self {
        self.max_rate = Some(limit);
        self
    }

    pub fn with_deadband(mut self, deadband: N) -> Self {
        self.deadband = Some(deadband);
        self
    }

    pub fn set_gains(&mut self, kp: N, ki: N, kd: N) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }

    pub fn set_setpoint(&mut self, setpoint: N) {
        self.setpoint = setpoint;
    }
}

impl<N: Number> ControlSystem for PIDController<N> 
where N: Number{
    type Input = N;

    type Output = N;

    type State = (N, N); // (integral, previous_error)

    type Time = N;

    fn step(&mut self, input: Self::Input, dt: Self::Time) -> Self::Output {
        let error = self.setpoint - input;

        // Apply deadband
        let error = if let Some(deadband) = self.deadband {
            if Signed::abs(&error) < deadband {
                N::zero()
            } else {
                error
            }
        } else {
            error
        };

        // Update Integral term with windup limit
        self.integral += error * dt;
        if let Some(limit) = self.integral_limit {
            self.integral = RealField::clamp(self.integral, -limit, limit);
        }

        // Calculate Derivative term
        let derivative = (error - self.previous_error) / dt;

        let filtered_derivative = 
            self.derivative_filter_coefficient * 
            derivative + 
            (N::one() - self.derivative_filter_coefficient) *
            self.prev_filtered_derivative;

        self.prev_filtered_derivative = filtered_derivative;
        self.previous_error = error;

        // PID output
        let mut output = self.kp * error + self.ki * self.integral + self.kd * derivative;

        if let Some(max_rate) = self.max_rate {
            let rate = (output - self.prev_output) / dt;
            if Signed::abs(&rate) > max_rate {
                output = self.prev_output + max_rate * Signed::signum(&rate) * dt;
            }
            output = self.prev_output + RealField::clamp(rate, -max_rate, max_rate) * dt;
        }
        self.prev_output = output;
        output
    }

    fn reset(&mut self) {
        self.integral = N::zero();
        self.previous_error = N::zero();
        self.prev_filtered_derivative = N::zero();
        self.prev_output = N::zero();
    }

    fn get_state(&self) -> Self::State {
            (self.integral, self.previous_error)
        }
}