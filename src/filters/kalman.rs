use nalgebra::{DefaultAllocator, Dim, DimName, OMatrix, OVector};
use nalgebra::allocator::Allocator;
use crate::Number;

pub struct KalmanFilter<T, R, OC>
where
    T: Number,
    R: Dim + DimName,
    OC: Dim,
    DefaultAllocator: Allocator<R> + Allocator<OC> + Allocator<R, R> + Allocator<OC, R> + Allocator<OC, OC>,
{
    x: OVector<T, R>,
    f: OMatrix<T, R, R>,
    h: OMatrix<T, OC, R>,
    p: OMatrix<T, R, R>,
    q: OMatrix<T, R, R>,
    r: OMatrix<T, OC, OC>,
}

impl<T, R, OC> KalmanFilter<T, R, OC>
where
    T: Number,
    R: Dim + DimName,
    OC: Dim,
    DefaultAllocator: Allocator<R> + Allocator<OC> + Allocator<R, R> + Allocator<OC, R> + Allocator<OC, OC> + Allocator<R,OC>,
{
    pub fn new(
        initial_state: OVector<T, R>,
        initial_estimate_covariance: OMatrix<T, R, R>,
        state_transition: OMatrix<T, R, R>,
        observation: OMatrix<T, OC, R>,
        process_noise_covariance: OMatrix<T, R, R>,
        measurement_noise_covariance: OMatrix<T, OC, OC>,
    ) -> Self {
        KalmanFilter {
            x: initial_state,
            f: state_transition,
            h: observation,
            p: initial_estimate_covariance,
            q: process_noise_covariance,
            r: measurement_noise_covariance,
        }
    }

    pub fn predict(&mut self) {
        self.x = &self.f * &self.x;
        self.p = &self.f * &self.p * self.f.transpose() + &self.q;
    }

    pub fn update(&mut self, measurement: &OVector<T, OC>) {
        let y = measurement - &self.h * &self.x;
        let s = &self.h * &self.p * self.h.transpose() + &self.r;
        let k = &self.p * self.h.transpose() * s.try_inverse().expect("Matrix inversion failed");
        
        self.x += &k * y;
        let i = OMatrix::<T, R, R>::identity();
        self.p = (&i - &k * &self.h) * &self.p;
    }

    pub fn state(&self) -> &OVector<T, R> {
        &self.x
    }

    pub fn covariance(&self) -> &OMatrix<T, R, R> {
        &self.p
    }
}