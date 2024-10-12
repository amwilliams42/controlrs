//! Proportional Integral Derivative controller implementation
//!
//! 

type OptValue = Option<(f32,f32)>;


#[derive(Debug, Clone, Copy)]
pub struct PIDController {
    pub kp: f32,                 // Proportional Gain
    pub ki: f32,                 // Integral Gain
    pub kd: f32,                 // Derivative Gain
    pub setpoint: f32,           // Desired setpoint
    pub integral: f32,           // Accumulated Integral term
    pub previous_error: f32,         // error from previous update
    pub previous_measurement: f32,   // measurement from prev update
    pub output_limits: OptValue, // Optional, limits on output
    pub p_limit: OptValue,       // Optional, limits on p
    pub d_limit: OptValue,       // Optional, limits on d
    pub i_limit: OptValue,       // Optional, limits on i
}

/// PID controller builder
///
/// This struct is used to configure and create a PIDController instance.
pub struct PIDControllerBuilder {
    kp: f32,
    ki: f32,
    kd: f32,
    setpoint: f32,
    output_limits: Option<(f32, f32)>,
    p_limit: Option<(f32, f32)>,
    i_limit: Option<(f32, f32)>,
    d_limit: Option<(f32, f32)>,
}

impl PIDControllerBuilder {
    /// Create a new PID controller builder
    ///
    /// # Arguments
    ///
    /// * `kp` - Proportional gain
    /// * `ki` - Integral gain
    /// * `kd` - Derivative gain
    /// * `setpoint` - Desired setpoint
    pub fn new(kp: f32, ki: f32, kd: f32, setpoint: f32) -> Self {
        PIDControllerBuilder {
            kp,
            ki,
            kd,
            setpoint,
            output_limits: None,
            p_limit: None,
            i_limit: None,
            d_limit: None,
        }
    }
    pub fn with_output_limits(mut self, min:f32, max: f32) -> Self {
        self.output_limits = Some((min,max));
        self
    }
    pub fn with_p_limits(mut self, min: f32, max: f32) -> Self {
        self.p_limit = Some((min, max));
        self
    }
    pub fn with_i_limits(mut self, min: f32, max: f32) -> Self {
        self.i_limit = Some((min, max));
        self
    }
    pub fn with_d_limits(mut self, min: f32, max: f32) -> Self {
        self.d_limit = Some((min, max));
        self
    }

    /// Build the PID controller
    pub fn build(self) -> PIDController {
        PIDController {
            kp: self.kp,
            ki: self.ki,
            kd: self.kd,
            setpoint: self.setpoint,
            integral: 0.0,
            previous_error: 0.0,
            previous_measurement: 0.0,
            output_limits: self.output_limits,
            p_limit: self.p_limit,
            i_limit: self.i_limit,
            d_limit: self.d_limit,
        }
    }
}

impl PIDController {
    /// Convenience function for a PID controller builder
    pub fn builder(
        kp: f32, 
        ki: f32, 
        kd: f32, 
        setpoint: f32) 
        -> PIDControllerBuilder {
            PIDControllerBuilder::new(kp,ki,kd,setpoint)
    }

    pub fn update(&mut self, measurement: f32, dt: f32) -> f32 {
        let error = self.setpoint - measurement;

        let mut p_term = self.kp * error;
        if let Some((min,max)) = self.p_limit {
            p_term = p_term.clamp(min, max);
        }

        self.integral += error * dt;
        let mut i_term = self.ki * self.integral;
        if let Some((min,max)) = self.i_limit {
            i_term = i_term.clamp(min, max);
            self.integral = i_term / self.ki;
        }

        let d_input = (measurement - self.previous_measurement) / dt;
        // Negative because we want to counteract the change
        let mut d_term = -self.kd * d_input; 
        if let Some((min, max)) = self.d_limit {
            d_term = d_term.clamp(min, max);
        }

        let mut output = p_term + i_term + d_term;

        if let Some((min, max)) = self.output_limits {
            output = output.clamp(min, max)
        }

        self.previous_error = error;
        self.previous_measurement = measurement;

        output
    }
}