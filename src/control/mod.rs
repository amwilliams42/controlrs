// This module provides implementations of various control algorithms.

pub mod pid;
pub mod lqr;

pub use pid::PIDController;
pub use lqr::LQRController;

use crate::Number;

pub trait ControlSystem {
    type Input;
    type Output;
    type State;

    type Time: Number;

    fn step(&mut self, input: Self::Input, dt: Self::Time) -> Self::Output;
    fn reset(&mut self);
    fn get_state(&self) -> Self::State;
}