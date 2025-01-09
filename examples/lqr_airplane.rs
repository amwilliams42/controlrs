use nalgebra::{OMatrix, OVector, RealField, U1, U12, U4};

pub struct AircraftSimulation<f64> {
    state: OVector<f64, U12>,
    mass: f64,
    ix: f64,
    iy: f64,
    iz: f64,
    ixz: f64,
}

impl AircraftSimulation<f64> {
    pub fn new() -> Self {
        AircraftSimulation {
            state: OVector::<f64, U12>::zeros(),
            mass: 1000.0,
            ix: 1500.0,
            iy: 3000.0,
            iz: 3500.0,
            ixz: 150.0,
        }
    }

} 