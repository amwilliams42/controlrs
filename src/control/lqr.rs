use std::marker::PhantomData;

use nalgebra::{DefaultAllocator, Dim, DimName, Dyn, OMatrix, OVector};
use nalgebra::allocator::Allocator;
use crate::Number;

pub enum LQRType {
    ContinuousTime,
    DiscreteTime,
}

pub enum Horizon {
    Infinite,
    Finite(usize),
}

pub struct LQRController<T, R, C>
where
    T: Number,
    R: Dim + DimName,
    C: Dim,
    DefaultAllocator: 
        Allocator<R, R> + 
        Allocator<R, C> + 
        Allocator<C, R> + 
        Allocator<C, C>, 
{
    k: Vec<OMatrix<T,C,R>>, // Feedback Gain Matrix
    a: OMatrix<T,R,R>, // State Transition Matrix
    b: OMatrix<T,R,C>, // Control Input Matrix
    lqr_type: LQRType,
    horizon: Horizon,
    _phantom: PhantomData<(R, C)>,
}

impl<T, R, C> LQRController<T, R, C>
where 
    T: Number,
    R: Dim + DimName,
    C: Dim,
    DefaultAllocator: 
        Allocator<R, R> + 
        Allocator<R, C> + 
        Allocator<C, R> + 
        Allocator<C, C> +
        Allocator<R>  + 
        Allocator<C>
{
    pub fn new(
        a: OMatrix<T, R, R>,
        b: OMatrix<T, R, C>,
        q: OMatrix<T, R, R>,
        r: OMatrix<T, C, C>,
        lqr_type: LQRType,
        horizon: Horizon,
        f: Option<OMatrix<T, R, R>>,
    ) -> Result<Self, String> {
        // Ensure all arms return Vec<OMatrix<T, C, R>> (gain matrices)
        let k = match (&lqr_type, &horizon) {
            (LQRType::ContinuousTime, Horizon::Infinite) => {
                // Compute the Riccati matrix P and then the gain matrix K
                let p = Self::solve_continuous_riccati(&a, &b, &q, &r)?;
                vec![Self::compute_gain(&b, &r, &p)]  // Return gain matrix K
            }
            (LQRType::DiscreteTime, Horizon::Infinite) => {
                // Compute the Riccati matrix P and then the gain matrix K
                let p = Self::solve_discrete_riccati(&a, &b, &q, &r)?;
                vec![Self::compute_gain(&b, &r, &p)]  // Return gain matrix K
            }
            (LQRType::ContinuousTime, Horizon::Finite(n)) => {
                let f = f.ok_or("Terminal cost matrix F is required for finite horizon")?;
                Self::solve_continuous_finite_horizon(&a, &b, &q, &r, f, *n)
            }
            (LQRType::DiscreteTime, Horizon::Finite(n)) => {
                let f = f.ok_or("Terminal cost matrix F is required for finite horizon")?;
                Self::solve_discrete_finite_horizon(&a, &b, &q, &r, f, *n)?
            }
        };

        Ok(Self {
            k,
            a,
            b,
            lqr_type,
            horizon,
            _phantom: PhantomData,
        })
    }

    pub fn integrate(&self, state: &OVector<T, R>, dt: T, time_step: usize) -> OVector<T, R> {
        match self.lqr_type {
            LQRType::ContinuousTime => self.integrate_rk4(state, dt, time_step),
            LQRType::DiscreteTime => {
                let control = self.compute_control(state, time_step);
                &self.a * state + &self.b * control // Discrete-time evolution
            }
        }
    }

    /// Runge-Kutta 4th-order (RK4) integration for continuous-time systems.
    fn integrate_rk4(&self, state: &OVector<T, R>, dt: T, time_step: usize) -> OVector<T, R> {
        let k1 = self.state_derivative(state, time_step);
        let k2 = self.state_derivative(&(state + &k1 * (dt / T::from_f64(2.0).unwrap())), time_step);
        let k3 = self.state_derivative(&(state + &k2 * (dt / T::from_f64(2.0).unwrap())), time_step);
        let k4 = self.state_derivative(&(state + &k3 * dt), time_step);

        state + (k1 + k2 * T::from_f64(2.0).unwrap() + k3 * T::from_f64(2.0).unwrap() + k4) * (dt / T::from_f64(6.0).unwrap())
    }

    /// Compute the state derivative for continuous-time systems.
    fn state_derivative(&self, state: &OVector<T, R>, time_step: usize) -> OVector<T, R> {
        let control = self.compute_control(state, time_step);
        &self.a * state + &self.b * control
    }


    /// Compute the control input for a given state and time step.
    pub fn compute_control(&self, state: &OVector<T, R>, time_step: usize) -> OVector<T, C> {
        let k = self.get_feedback_gain(time_step);
        -k * state
    }

    /// Get the feedback gain matrix for the given time step.
    pub fn get_feedback_gain(&self, time_step: usize) -> &OMatrix<T, C, R> {
        let k_index = match self.horizon {
            Horizon::Infinite => 0,
            Horizon::Finite(n) => n.saturating_sub(time_step + 1),
        };
        &self.k[k_index]
    }

    /// Compute the optimal feedback gain matrix K.
    fn compute_gain(
        b: &OMatrix<T, R, C>,
        r: &OMatrix<T, C, C>,
        p: &OMatrix<T, R, R>,
    ) -> OMatrix<T, C, R> {
        (r + b.transpose() * p * b)
            .try_inverse()
            .unwrap()
            * b.transpose()
            * p
    }
    fn solve_continuous_riccati(
        a: &OMatrix<T, R, R>,
        b: &OMatrix<T, R, C>,
        q: &OMatrix<T, R, R>,
        r: &OMatrix<T, C, C>,
    ) -> Result<OMatrix<T, R, R>, String> {
        let h = Self::hamiltonian_matrix(a, b, q, r);

        // Perform Schur decomposition on the Hamiltonian matrix.
        let schur = h.schur();
        let (q_matrix, _) = schur.unpack();

        // Extract the relevant blocks from the Q matrix.
        let n = R::try_to_usize().unwrap();
        let q21_slice = q_matrix.view((n, 0), (n, n));
        let q11_slice = q_matrix.view((0, 0), (n, n));

        // Manually copy the dynamic slices into statically-sized matrices.
        let mut q21_static = OMatrix::<T, R, R>::zeros();
        let mut q11_static = OMatrix::<T, R, R>::zeros();

        q21_static.copy_from(&q21_slice);
        q11_static.copy_from(&q11_slice);

        // Compute the inverse of Q11.
        let q11_inv = q11_static.try_inverse().ok_or("Failed to invert Q11 matrix")?;

        // Compute and return the Riccati solution P.
        Ok(q21_static * q11_inv)
    }

    /// Construct the Hamiltonian matrix for the continuous-time Riccati equation.
    fn hamiltonian_matrix(
        a: &OMatrix<T, R, R>,
        b: &OMatrix<T, R, C>,
        q: &OMatrix<T, R, R>,
        r: &OMatrix<T, C, C>,
    ) -> OMatrix<T, Dyn, Dyn> {

        let n = R::try_to_usize().unwrap();

        // Create a zero matrix of size (2n x 2n)
        let mut h = OMatrix::<T, Dyn, Dyn>::zeros(2 * n, 2 * n);

        // Fill the top-left block with A
        h.view_mut((0, 0), (n, n)).copy_from(a);

        // Fill the top-right block with B * R⁻¹ * Bᵀ
        let b_r_inv_bt = b * r.clone().try_inverse().unwrap() * b.transpose();
        h.view_mut((0, n), (n, n)).copy_from(&b_r_inv_bt);

        // Fill the bottom-left block with -Q
        h.view_mut((n, 0), (n, n)).copy_from(&-q);

        // Fill the bottom-right block with -Aᵀ
        h.view_mut((n, n), (n, n)).copy_from(&-a.transpose());

        h
    }

    /// Solve the discrete-time Riccati equation using an iterative approach.
    fn solve_discrete_riccati(
        a: &OMatrix<T, R, R>,
        b: &OMatrix<T, R, C>,
        q: &OMatrix<T, R, R>,
        r: &OMatrix<T, C, C>,
    ) -> Result<OMatrix<T, R, R>, String> {
        let mut p = q.clone();
        let tolerance = T::from_f64(1e-10).unwrap();
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let p_next = q + a.transpose() * &p * a
                - a.transpose() * &p * b
                * (r + b.transpose() * &p * b).try_inverse().unwrap()
                * b.transpose() * &p * a;

            if (&p_next - &p).norm() < tolerance {
                return Ok(p_next);
            }
            p = p_next;
        }

        Err("Riccati equation did not converge".into())
    }

    fn solve_continuous_finite_horizon(
        a: &OMatrix<T, R, R>,
        b: &OMatrix<T, R, C>,
        q: &OMatrix<T, R, R>,
        r: &OMatrix<T, C, C>,
        f: OMatrix<T, R, R>,
        n: usize,
    ) -> Vec<OMatrix<T, C, R>> {
        let mut p = f;
        let mut k_sequence = Vec::with_capacity(n);
        let dt = T::from_f64(1.0).unwrap() / T::from_usize(n).unwrap();
    
        for _ in (0..n).rev() {
            let k = (r + b.transpose() * &p * b)
                .try_inverse()
                .unwrap()
                * b.transpose()
                * &p;
            k_sequence.push(k);
    
            let dp = |p: &OMatrix<T, R, R>| {
                a.transpose() * p + p * a - p * b * r.clone().try_inverse().unwrap() * b.transpose() * p + q
            };
    
            let k1 = dp(&p);
            let k2 = dp(&(&p + &k1 * (dt / T::from_f64(2.0).unwrap())));
            let k3 = dp(&(&p + &k2 * (dt / T::from_f64(2.0).unwrap())));
            let k4 = dp(&(&p + &k3 * dt));
    
            p -= (k1 + k2 * T::from_f64(2.0).unwrap() + k3 * T::from_f64(2.0).unwrap() + k4)
                * (dt / T::from_f64(6.0).unwrap());
        }
    
        k_sequence.reverse();
        k_sequence
    }
    
    fn solve_discrete_finite_horizon(
        a: &OMatrix<T, R, R>,
        b: &OMatrix<T, R, C>,
        q: &OMatrix<T, R, R>,
        r: &OMatrix<T, C, C>,
        f: OMatrix<T, R, R>,  // Terminal cost matrix
        n: usize,             // Number of time steps
    ) -> Result<Vec<OMatrix<T, C, R>>, String> {
        let mut p = f; // Initialize with the terminal cost matrix F
        let mut k_sequence = Vec::with_capacity(n);
    
        for _ in (0..n).rev() {
            // Compute the gain matrix K for the current step
            let temp = r + b.transpose() * &p * b;
            let k = temp
                .try_inverse()
                .ok_or("Matrix inversion failed in discrete finite horizon solver")?
                * b.transpose()
                * &p
                * a;
            k_sequence.push(k.clone());
    
            // Update the Riccati matrix P for the next step
            p = a.transpose() * &p * a - a.transpose() * &p * b * k + q;
        }
    
        k_sequence.reverse();
        Ok(k_sequence)
    }
    
    

}