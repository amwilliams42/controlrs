use nalgebra::{RealField, ComplexField};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

pub mod math;
pub mod control;
pub mod filters;

pub use math::solve_dare_sda;
pub use control::{
    PIDController,
    LQRController,};
pub use filters::kalman::KalmanFilter;
pub use filters::ekf::ExtendedKalmanFilter;

pub trait Number:
    Sized + Copy + Debug + 
    Float + FromPrimitive + ToPrimitive + 
    RealField + ComplexField
{}

impl<N> Number for N
where
    N: Sized + Copy + Debug +
    Float + FromPrimitive + ToPrimitive +
    RealField + ComplexField
{}

