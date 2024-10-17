use nalgebra::{DefaultAllocator, Dim, DimName, OMatrix, OVector};
use nalgebra::allocator::Allocator;
use crate::Number;

/// Extended Kalman Filter (EKF) implementation.
///
/// # Generics:
/// - `T`: Scalar type (e.g., `f64`).
/// - `R`: Dimension of the state vector.
/// - `C`: Dimension of the measurement vector.
///
/// The EKF maintains an estimate of the system's state over time and updates
/// the estimate based on noisy measurements.
///
/// # Example:
/// ```
/// use nalgebra::{OMatrix, OVector, U2, U1};
/// use controlrs::ExtendedKalmanFilter;
///
/// // Define system dimensions.
/// let mut ekf = ExtendedKalmanFilter::<f64, U2, U1>::new(
///     OVector::<f64, U2>::new(0.0, 0.0),  // Initial state
///     OMatrix::<f64, U2, U2>::identity(), // Initial covariance
///     OMatrix::<f64, U2, U2>::identity(), // Process noise covariance
///     OMatrix::<f64, U1, U1>::identity() * 0.1, // Measurement noise covariance
/// );
///
/// // Define a state transition and measurement function.
/// fn state_transition(x: &OVector<f64, U2>) -> OVector<f64, U2> {
///     OVector::<f64, U2>::new(x[0] + x[1], x[1])
/// }
/// // Define the Jacobian of the state transition function.
/// fn state_jacobian(_x: &OVector<f64, U2>) -> OMatrix<f64, U2, U2> {
///     OMatrix::<f64, U2, U2>::new(1.0, 1.0, 0.0, 1.0)
/// }
/// fn measurement_function(x: &OVector<f64, U2>) -> OVector<f64, U1> {
///     OVector::<f64, U1>::new(x[0])
/// }
/// // Define the Jacobian of the measurement function.
/// fn measurement_jacobian(_x: &OVector<f64, U2>) -> OMatrix<f64, U1, U2> {
///     OMatrix::<f64, U1, U2>::new(1.0, 0.0)
/// }
/// 
/// // Predict step.
/// ekf.predict(&state_transition, &state_jacobian);
///
/// // Update step with a noisy measurement
/// let measurement = OVector::<f64, U1>::new(1.05);
/// ekf.update(&measurement, &measurement_function, &measurement_jacobian);
///
/// // Check the state estimate.
/// let state = ekf.state();
/// assert!((state[0] - 1.05).abs() < 1e-1 );
/// ```
pub struct ExtendedKalmanFilter<T,R,OC>
where
    T: Number,
    R: Dim + DimName,
    OC: Dim,
    DefaultAllocator: Allocator<R>
        + Allocator<OC>
        + Allocator<R,R>
        + Allocator<OC, R>
        + Allocator<OC, OC>
        + Allocator<R, OC>
{
    x: OVector<T, R>, //State Estimate
    p: OMatrix<T, R, R>, //Estimated Covariance
    q: OMatrix<T, R, R>, // Process Noise Covariance
    r: OMatrix<T, OC, OC> // Measurement Noise Covariance
}

impl <T, R, OC> ExtendedKalmanFilter<T, R, OC> 
where
    T: Number,
    R: Dim + DimName,
    OC: Dim,
    DefaultAllocator: Allocator<R>
        + Allocator<OC>
        + Allocator<R,R>
        + Allocator<OC, R>
        + Allocator<OC, OC>
        + Allocator<R, OC>
{
    /// Creates a new Extended Kalman Filter (EKF) with the given initial parameters.
    ///
    /// # Parameters
    /// - `initial_state`: The initial estimate of the system state.
    /// - `initial_estimate_covariance`: The initial state covariance matrix.
    /// - `process_noise_covariance`: Process noise covariance matrix.
    /// - `measurement_noise_covariance`: Measurement noise covariance matrix.
    pub fn new(
        initial_state: OVector<T, R>,
        initial_estimated_covariance: OMatrix<T, R, R>,
        process_noise_covariance: OMatrix<T, R, R>,
        measurement_noise_covariance: OMatrix<T, OC, OC>
    ) -> Self{
        Self { 
            x: initial_state, 
            p: initial_estimated_covariance, 
            q: process_noise_covariance, 
            r: measurement_noise_covariance }
    } 
    /// Performs the predict step of the EKF.
    ///
    /// Uses the non-linear state transition function to predict the next state
    /// and updates the covariance matrix using the Jacobian of the state transition.
    ///
    /// # Parameters
    /// - `state_transition_fn`: Non-linear function for state transition.
    /// - `state_transition_jacobian`: Jacobian matrix of the state transition function.
    pub fn predict<F, J>(
        &mut self,
        state_transition_fn: F,
        state_transition_jacobian: J,
    ) where 
        F: Fn(&OVector<T, R>) -> OVector<T, R>, //Nonlinear state transition
        J: Fn(&OVector<T, R>) -> OMatrix<T, R, R>, // Jacobian of state transition
    {
        self.x = state_transition_fn(&self.x);

        let f_jacobian = state_transition_jacobian(&self.x);

        // Debugging: Print intermediate values
        println!("After predict:");
        println!("State: {:?}", self.x);
        println!("Covariance: {:?}", self.p);
        self.p = &f_jacobian * &self.p * f_jacobian.transpose() + &self.q 
    }
    /// Performs the update step of the EKF with a new measurement.
    ///
    /// Uses the non-linear observation function and its Jacobian to update
    /// the state estimate and covariance matrix.
    ///
    /// # Parameters
    /// - `measurement`: The new measurement vector.
    /// - `observation_fn`: Non-linear function for the observation model.
    /// - `observation_jacobian`: Jacobian matrix of the observation function.

    pub fn update<H, J>(
        &mut self,
        measurement: &OVector<T, OC>,
        observation_fn: H,
        observation_jacobian: J,
    ) where 
        H: Fn(&OVector<T, R>) -> OVector<T, OC>, //Nonlinear state transition
        J: Fn(&OVector<T, R>) -> OMatrix<T, OC, R>, // Jacobian of state transition
    {
        // predict measurement
        let predicted_measurement = observation_fn(&self.x);

        let innovation = measurement - predicted_measurement;

        // Linearize w Jacobian
        let h_jacobian = observation_jacobian(&self.x);

        // innovation covariance
        let s = &h_jacobian * &self.p * h_jacobian.transpose() + &self.r;

        // Kalman gain
        let k = &self.p * h_jacobian.transpose() * s.clone().try_inverse().expect("Matrix Inversion Failed");

         // Debugging: Print intermediate values for better clarity.
        println!("Innovation: {:?}", innovation);
        println!("Kalman Gain: {:?}", k);

        // Update state
        self.x += &k * innovation;

        // Update estimated covariance
        let i = OMatrix::<T, R, R>::identity();
        self.p = (&i - &k * &h_jacobian) * &self.p;

        // Debugging: Print intermediate values
        // Debugging: Print updated state and covariance.
        println!("After update:");
        println!("State: {:?}", self.x);
        println!("Covariance: {:?}", self.p);

    }
    /// Get the current state estimate
    pub fn state(&self) -> &OVector<T, R> {
        &self.x
    }

    /// Get the current estimate covariance
    pub fn covariance(&self) -> &OMatrix<T, R, R> {
        &self.p
    }
    

}