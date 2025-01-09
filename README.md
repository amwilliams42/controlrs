# Controlrs: A Rust Guidance, Control, and Navigation Library

This library provides a collection of tools for guidance, control, and navigation in Rust. It includes implementations of various control algorithms, state estimation filters, and mathematical utilities to support robotics and control system applications.

## Features

- **Control Systems**
  - [`PIDController`](src/control/pid.rs): A full PID controller implementation with:
    - Builder pattern for configuration
    - Anti-windup protection
    - Configurable limits for P, I, D terms and output
  - [`LQRController`](src/control/lqr.rs): Linear Quadratic Regulator with:
    - Support for discrete-time systems
    - Uses Structured Doubling Algorithm for DARE (Discrete Algebraic Riccati Equation) solver

- **State Estimation**
  - [`KalmanFilter`](src/filters/kalman.rs): Standard Kalman Filter for linear systems
  - [`ExtendedKalmanFilter`](src/filters/ekf.rs): Extended Kalman Filter for nonlinear systems

## Example Use

See [Examples](examples/) for implementation details

# Features/Todo

## Linear Control
- [X] State space systems
- [X] PID control
- [X] Linear Quadratic Regulator (LQR)
- [X] Kalman Filter
- [ ] Linear Quadratic Gaussian (LQG)
- [ ] Linear H-infinity control

## Nonlinear Control
- [ ] Nonlinear system dynamics
- [ ] Sliding mode control
- [ ] Adaptive control
- [ ] Backstepping control
- [ ] Model Predictive Control (MPC)

## Reachability Analysis
- [ ] Level set methods
- [ ] Hamilton-Jacobi equation solvers
- [ ] Fast marching methods
- [ ] Forward reachable sets
- [ ] Backward reachable sets
- [ ] Safety verification

## Optimization and Planning
- [ ] Trajectory optimization
- [ ] Path planning with obstacles
- [ ] Optimal control synthesis
- [ ] Value function computation
- [ ] Numerical optimization routines
- [ ] Collision avoidance