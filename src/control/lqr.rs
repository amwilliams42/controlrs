use nalgebra as na;
use std::f64;

pub enum LQRType {
    ContinuousTime,
    DiscreteTime,
}

pub enum Horizon {
    Infinite,
    Finite(usize),
}

pub struct LQRController {
    k: Vec<na::DMatrix<f64>>,
    a: na::DMatrix<f64>,
    b: na::DMatrix<f64>,
    lqr_type: LQRType,
    horizon: Horizon,
}

impl LQRController {
    pub fn new(
        a: na::DMatrix<f64>,
        b: na::DMatrix<f64>,
        q: na::DMatrix<f64>,
        r: na::DMatrix<f64>,
        lqr_type: LQRType,
        horizon: Horizon,
        f: Option<na::DMatrix<f64>>,
    ) -> Result<Self, String> {
        let k = match (&lqr_type, &horizon) {
            (LQRType::ContinuousTime, Horizon::Infinite) => {
                vec![Self::solve_continuous_riccati(&a, &b, &q, &r)?]
            }
            (LQRType::DiscreteTime, Horizon::Infinite) => {
                let p = Self::solve_discrete_riccati(&a, &b, &q, &r)?;
                println!("Riccati solution P:");
                println!("{}", p);
                
                let r_inv = r.try_inverse().ok_or("R matrix is not invertible")?;
                let k = &r_inv * &b.transpose() * &p * &a;
                println!("Computed K matrix:");
                println!("{}", k);
                
                vec![k]
            }
            (LQRType::ContinuousTime, Horizon::Finite(n)) => {
                Self::solve_continuous_finite_horizon(&a, &b, &q, &r, f.ok_or("Terminal cost matrix F is required for finite horizon")?, *n)
            }
            (LQRType::DiscreteTime, Horizon::Finite(n)) => {
                Self::solve_discrete_finite_horizon(&a, &b, &q, &r, f.ok_or("Terminal cost matrix F is required for finite horizon")?, *n)?
            }
        };
        
        Ok(LQRController { k, a, b, lqr_type, horizon })
    }

    pub fn compute_control(&self, state: &na::DVector<f64>, time_step: usize) -> na::DVector<f64> {
        let k_index = match self.horizon {
            Horizon::Infinite => 0,
            Horizon::Finite(n) => n.saturating_sub(time_step + 1),
        };
        -&self.k[k_index] * state
    }

    pub fn get_feedback_gain(&self, time_step: usize) -> &na::DMatrix<f64> {
        let k_index = match self.horizon {
            Horizon::Infinite => 0,
            Horizon::Finite(n) => n.saturating_sub(time_step + 1),
        };
        &self.k[k_index]
    }

    pub fn integrate(&self, state: &na::DVector<f64>, dt: f64, time_step: usize) -> na::DVector<f64> {
        match self.lqr_type {
            LQRType::ContinuousTime => self.integrate_rk4(state, dt, time_step),
            LQRType::DiscreteTime => {
                let control = self.compute_control(state, time_step);
                &self.a * state + &self.b * control
            }
        }
    }

    fn integrate_rk4(&self, state: &na::DVector<f64>, dt: f64, time_step: usize) -> na::DVector<f64> {
        let k1 = self.state_derivative(state, time_step);
        let k2 = self.state_derivative(&(state + &k1 * (dt / 2.0)), time_step);
        let k3 = self.state_derivative(&(state + &k2 * (dt / 2.0)), time_step);
        let k4 = self.state_derivative(&(state + &k3 * dt), time_step);
        state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
    }

    fn state_derivative(&self, state: &na::DVector<f64>, time_step: usize) -> na::DVector<f64> {
        let control = self.compute_control(state, time_step);
        &self.a * state + &self.b * control
    }

    fn solve_continuous_riccati(
        a: &na::DMatrix<f64>,
        b: &na::DMatrix<f64>,
        q: &na::DMatrix<f64>,
        r: &na::DMatrix<f64>,
    ) -> Result<na::DMatrix<f64>, String> {
        let n = a.nrows();
        let r_inv = r.clone().try_inverse().ok_or("R matrix is not invertible")?;
        let b_r_inv_bt = b * &r_inv * b.transpose();
    
        // Form the Hamiltonian matrix
        let mut h = na::DMatrix::zeros(2 * n, 2 * n);
        h.view_mut((0, 0), (n, n)).copy_from(a);
        h.view_mut((0, n), (n, n)).copy_from(&-b_r_inv_bt);
        h.view_mut((n, 0), (n, n)).copy_from(&-q);
        h.view_mut((n, n), (n, n)).copy_from(&-a.transpose());
    
        // Compute the Schur decomposition
        let schur = h.schur();
        let (q_matrix, _) = schur.unpack();
    
        // Extract the relevant blocks
        let q11 = q_matrix.view((0, 0), (n, n));
        let q21 = q_matrix.view((n, 0), (n, n));
    
        // Check if Q11 is invertible
        let q11_inv = q11.try_inverse().ok_or("Failed to invert Q11 matrix")?;
    
        // Solve for P
        Ok(q21 * q11_inv)
    }

    fn solve_discrete_riccati(
        a: &na::DMatrix<f64>,
        b: &na::DMatrix<f64>,
        q: &na::DMatrix<f64>,
        r: &na::DMatrix<f64>,
    ) -> Result<na::DMatrix<f64>, String> {
        let n = a.nrows();
        let r_inv = r.clone().try_inverse().ok_or("R matrix is not invertible")?;
        let mut g = b * &r_inv * b.transpose();
        let mut p = q.clone();
        let mut a_tilde = a.clone();
    
        let tolerance = 1e-10;
        let max_iterations = 1000;
        let divergence_threshold = 1e10;
    
        for iteration in 0..max_iterations {
            let a_tilde_t = a_tilde.transpose();
            let temp = na::DMatrix::identity(n, n) + &g * &p;
            let temp_inv = temp.try_inverse().ok_or("Matrix inversion failed in Riccati solver")?;
            
            let new_p = &a_tilde_t * &p * &temp_inv * &a_tilde + q;
            let new_g = &a_tilde * &g * &a_tilde_t + &g;
            let new_a_tilde = &a_tilde * &temp_inv * &a_tilde;
    
            let rel_error = (&new_p - &p).norm() / (1.0 + p.norm());
    
            if rel_error < tolerance {
                return Ok(new_p);
            }
    
            if new_p.norm() > divergence_threshold {
                return Err(format!("Discrete Riccati equation diverged at iteration {}", iteration));
            }
    
            p = new_p;
            g = new_g;
            a_tilde = new_a_tilde;
        }
    
        Err(format!("Discrete Riccati equation did not converge after {} iterations", max_iterations))
    }

    fn solve_continuous_finite_horizon(
        a: &na::DMatrix<f64>,
        b: &na::DMatrix<f64>,
        q: &na::DMatrix<f64>,
        r: &na::DMatrix<f64>,
        f: na::DMatrix<f64>,
        n: usize,
    ) -> Vec<na::DMatrix<f64>> {
        let mut p = f;
        let mut k_sequence = Vec::with_capacity(n);
        let r_inv = r.clone().try_inverse().expect("R matrix is not invertible");

        for _ in (0..n).rev() {
            let k = &r_inv * b.transpose() * &p;
            k_sequence.push(k);
            
            let temp = a.transpose() * &p * b * &r_inv * b.transpose() * &p;
            p = a.transpose() * &p * a - temp + q;
        }

        k_sequence.reverse();
        k_sequence
    }

    fn solve_discrete_finite_horizon(
        a: &na::DMatrix<f64>,
        b: &na::DMatrix<f64>,
        q: &na::DMatrix<f64>,
        r: &na::DMatrix<f64>,
        f: na::DMatrix<f64>,
        n: usize,
    ) -> Result<Vec<na::DMatrix<f64>>, String> {
        let mut p = f;
        let mut k_sequence = Vec::with_capacity(n);

        for _ in (0..n).rev() {
            let temp = r + b.transpose() * &p * b;
            let k = temp.try_inverse()
                .ok_or("Matrix inversion failed in discrete finite horizon solver")?
                * b.transpose() * &p * a;
            k_sequence.push(k);
            
            p = a.transpose() * &p * a - a.transpose() * &p * b * k_sequence.last().unwrap() + q;
        }

        k_sequence.reverse();
        Ok(k_sequence)
    }
}