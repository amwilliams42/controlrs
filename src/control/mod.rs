//! This module provides implementations of various control algorithms.

mod pid;
mod lqr;

pub use pid::PIDController;
pub use lqr::{LQRController,LQRType,Horizon};