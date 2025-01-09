use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimMin, DimName, OMatrix, OVector, U1};
use crate::{solve_dare_sda, Number};

use super::ControlSystem;

pub struct LQRController<T, R, C> 
where 
    T: Number, 
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<R, R>
        + Allocator<R, C>
        + Allocator<C, R>
        + Allocator<C, C>
        + Allocator<R>
        + Allocator<C>
{
    // State Transition Matrix A.
    pub a: OMatrix<T, R, R>,
    // Control Matrix B.
    pub b: OMatrix<T, R, C>,
    // State cost matrix Q
    pub q: OMatrix<T, R, R>,
    // Control Cost Matrix R
    pub r: OMatrix<T, C, C>,
    // Feedback Gain Matrix K
    pub k: OMatrix<T, C, R>,

    state: OVector<T, R>,
    reference: OVector<T, R>,
}

    impl <T, R, C> LQRController<T, R, C>
    where
        T: Number,
        R: Dim + DimName + DimMin<R, Output = R>,
        C: Dim + DimName,
        DefaultAllocator: Allocator<R, R>
            + Allocator<R, C>
            + Allocator<C, R>
            + Allocator<C, C>
            + Allocator<R>
            + Allocator<C>
{
    /// Creates a new LQR controller by solving the DARE using the structured doubling algorithm (SDA).
    ///
    /// # Parameters
    /// - `a`: State transition matrix.
    /// - `b`: Control matrix.
    /// - `q`: State cost matrix.
    /// - `r`: Control cost matrix.
    /// - `epsilon`: Convergence tolerance for the DARE solver.
    /// - `max_iterations`: Optional maximum number of iterations for the SDA.
    ///
    /// # Returns
    /// - `Ok(LQRController)` if the gain matrix K is successfully computed.
    /// - `Err(&str)` if the DARE solver fails to converge or a matrix inversion fails.
    pub fn new(
        a: OMatrix<T, R, R>,
        b: OMatrix<T, R, C>,
        q: OMatrix<T, R, R>,
        r: OMatrix<T, C, C>,
        epsilon: T,
    ) -> Result<Self, &'static str> {

        // Solve the DARE using the SDA algorithm.
        let p = solve_dare_sda(&a, &b, &q, &r, epsilon, None)?;

        // Calculate optimal feedback gain K = (R + B'PB)^(-1)B'PA
        let btpb = &b.transpose() * &p * &b;
        let btpa = &b.transpose() * &p * &a;
        let k = (&r + &btpb).try_inverse()
            .ok_or("Failed to invert R + B'PB")?
            * btpa;
        Ok(Self {
            a,
            b,
            q,
            r,
            k,
            state: OVector::zeros_generic(R::name(), U1::name()),
            reference: OVector::zeros_generic(R::name(), U1::name()),
        })
    }

    pub fn set_reference(&mut self, reference: OVector<T, R>) {
        self.reference = reference;
    }

    pub fn get_gain(&self) -> OMatrix<T, C, R> {
        self.k.clone()
    }
}

impl<T, R, C> ControlSystem for LQRController<T, R, C> 
where
    T: Number,
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<R, R> 
        + Allocator<R, C>
        + Allocator<C, R>
        + Allocator<C, C>
        + Allocator<R, U1>
        + Allocator<C>

{
    type Input = OVector<T, R>;

    type Output = OVector<T, C>;

    type State = OVector<T, R>;

    type Time = T;

    fn step(&mut self, state: Self::Input, _dt: Self::Time) -> Self::Output {
        self.state = state;
        let error = &self.state - &self.reference;
        -(&self.k * error)
    }

    fn reset(&mut self) {
        self.state = OVector::zeros_generic(R::name(), U1::name());
        self.reference = OVector::zeros_generic(R::name(), U1::name());
    }

    fn get_state(&self) -> Self::State {
        self.state.clone()
    }
}