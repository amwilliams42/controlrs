use nalgebra::{OMatrix, RealField, DefaultAllocator, Dim, DimName, allocator::Allocator};

/// Solves the Discrete Algebraic Riccati Equation (DARE) using the Structured Doubling Algorithm (SDA).
///
/// # Overview
/// The Discrete Algebraic Riccati Equation (DARE) solves for a stabilizing matrix `P`
/// by iteratively refining matrices using the **Structured Doubling Algorithm (SDA)**.
/// SDA provides **quadratic convergence** and stability.
///
/// The stopping criterion is:
/// \[ \frac{||H_{k+1} - H_k||}{||H_{k+1}||} < \epsilon \]
///
/// # Parameters
/// - `a`: State transition matrix \(A \in \mathbb{R}^{R \times R}\).
/// - `b`: Control matrix \(B \in \mathbb{R}^{R \times C}\).
/// - `q`: State cost matrix \(Q \in \mathbb{R}^{R \times R}\).
/// - `r`: Control cost matrix \(R \in \mathbb{R}^{C \times C}\).
/// - `epsilon`: Convergence tolerance.
/// - `max_iterations`: Optional maximum number of iterations. If `None`, the algorithm runs until convergence.
///
/// # Returns
/// - `Ok(P)` where `P` is the stabilizing solution to the DARE.
/// - `Err(&str)` if the matrix inversion fails or convergence cannot be achieved.
///
/// # Example
/// ```
/// use nalgebra::{OMatrix, U2, U4};
/// use your_crate::math::dare_sda::solve_dare_sda;
///
/// let a = OMatrix::<f64, U4, U4>::identity();
/// let b = OMatrix::<f64, U4, U2>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
/// let q = OMatrix::<f64, U4, U4>::identity() * 10.0;
/// let r = OMatrix::<f64, U2, U2>::identity();
///
/// let epsilon = 1e-9;
///
/// // Run the SDA algorithm without a max iteration limit.
/// match solve_dare_sda(&a, &b, &q, &r, epsilon, None) {
///     Ok(p) => println!("Solution matrix P: \n{}", p),
///     Err(e) => println!("Error: {}", e),
/// }
/// ```
pub fn solve_dare_sda<T, R, C>(
    a: &OMatrix<T, R, R>,
    b: &OMatrix<T, R, C>,
    q: &OMatrix<T, R, R>,
    r: &OMatrix<T, C, C>,
    epsilon: T,
    max_iterations: Option<usize>,
) -> Result<OMatrix<T, R, R>, &'static str>
where
    T: RealField,
    R: Dim + DimName,
    C: Dim + DimName,
    DefaultAllocator: Allocator<R, R>
        + Allocator<R, C>
        + Allocator<C, R>
        + Allocator<C, C>,
{
    // Initialize matrices A_0, G_0, and H_0
    let mut a_k = a.clone();
    let g_0 = b * r.clone().try_inverse().unwrap() * b.transpose();
    let mut g_k = g_0.clone();
    let mut h_k = q.clone();

    let mut iteration = 0;
    loop {
        // Compute (I + G_k * H_k)^-1
        let identity = OMatrix::<T, R, R>::identity();
        let inv_term = (identity.clone() + &g_k * &h_k)
            .try_inverse()
            .ok_or("Matrix inversion failed")?;

        // Update A_{k+1}
        let a_next = &a_k * &inv_term * &a_k;

        // Update G_{k+1}
        let g_next = &g_k + &a_k * &inv_term * &g_k * a_k.transpose();

        // Update H_{k+1}
        let h_next = &h_k + a_k.transpose() * &h_k * &inv_term * &a_k;

        // Check for convergence
        let diff = (&h_next - &h_k).norm();
        let norm = h_next.norm();
        if diff / norm < epsilon {
            return Ok(h_next);
        }

        // Prepare for the next iteration
        a_k = a_next;
        g_k = g_next;
        h_k = h_next;

        // Increment the iteration counter
        iteration += 1;

        // Check if we've hit the max iteration limit (if provided)
        if let Some(max_iter) = max_iterations {
            if iteration >= max_iter {
                return Err("SDA did not converge within the maximum number of iterations");
            }
        }
    }
}
