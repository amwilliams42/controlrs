//! This module provides implementations of various control algorithms.

pub mod pid;
pub mod lqr;

pub use pid::PIDController;
pub use lqr::LQRController;