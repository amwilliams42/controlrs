use nalgebra::{OMatrix, OVector, DefaultAllocator, Dim, DimMin, DimName, allocator::Allocator};
use crate::{solve_dare_sda, Number};

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
    pub k: OMatrix<T, C, R>
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
        max_iterations: Option<usize>,
    ) -> Result<Self, &'static str> {

        // Solve the DARE using the SDA algorithm.
        let p = solve_dare_sda(&a, &b, &q, &r, epsilon, max_iterations)?;

        // Compute the feedback gain matrix: K = (R + B^T P B)^-1 B^T P A
        let k = (r.clone() + b.transpose() * &p * &b)
            .try_inverse()
            .ok_or("Matrix inversion failed")?
            * b.transpose()
            * &p
            * &a;

        Ok(LQRController { a, b, q, r, k })
    }

    pub fn compute_control(
        &self,
        state: &OVector<T, R>,
        target: &OVector<T, R>
    ) -> OVector<T, C> {
        -&self.k * (state - target)
    }

    /// Recomputes the feedback gain matrix K if any system matrices are modified.
    ///
    /// # Returns
    /// - `Ok(())` if the gain matrix is successfully recomputed.
    /// - `Err(&str)` if the DARE solver fails to converge.
    pub fn recompute_gain(&mut self, epsilon: T, max_iterations: Option<usize>) -> Result<(), &'static str> {
        let p = solve_dare_sda(&self.a, &self.b, &self.q, &self.r, epsilon, max_iterations)?;
        self.k = (self.r.clone() + self.b.transpose() * &p * &self.b)
            .try_inverse()
            .ok_or("Matrix inversion failed")?
            * self.b.transpose()
            * &p
            * &self.a;
        Ok(())
    }
}