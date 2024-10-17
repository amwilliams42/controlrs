use controlrs::control::{LQRController, LQRType, Horizon};
use nalgebra::{self as na};


#[test]
fn test_discrete_infinite_horizon() {
    // System: Discrete-time double integrator
    // x[k+1] = [1  0.1] x[k] + [0.005] u[k]
    //          [0   1 ]        [0.1  ]
    let a = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
    let b = na::DMatrix::from_row_slice(2, 1, &[0.005, 0.1]);
    
    // Cost matrices: Equal weight on position and velocity states, less weight on control input
    let q = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let r = na::DMatrix::from_row_slice(1, 1, &[0.1]);

    let controller = LQRController::new(a.clone(), b.clone(), q, r, LQRType::DiscreteTime, Horizon::Infinite, None)
        .expect("Failed to create LQR controller");

    let k = controller.get_feedback_gain(0);
    
    println!("Computed K matrix: {:?}", k);
    println!("K matrix dimensions: {} x {}", k.nrows(), k.ncols());

    // Check that K has the correct dimensions
    assert_eq!(k.nrows(), 1, "K matrix should have 1 row");
    assert_eq!(k.ncols(), 2, "K matrix should have 2 columns");

    // Print the individual elements of K for inspection
    println!("K[0,0] = {}", k[(0, 0)]);
    println!("K[0,1] = {}", k[(0, 1)]);

    // Check if K has reasonable values
    assert!(k[(0, 0)].abs() > 1e-6, "K[0,0] should not be zero");
    assert!(k[(0, 1)].abs() > 1e-6, "K[0,1] should not be zero");

    // Manually compute closed-loop matrix A - BK
    let bk = &b * k;
    println!("B*K matrix:");
    println!("{}", bk);

    let closed_loop_a = &a - &bk;
    println!("Closed-loop A matrix (A - BK):");
    println!("{}", closed_loop_a);

    // Manually compute eigenvalues of closed-loop matrix
    let eigenvalues = closed_loop_a.complex_eigenvalues();
    println!("Closed-loop eigenvalues: {:?}", eigenvalues);

    for (i, eigenvalue) in eigenvalues.iter().enumerate() {
        let magnitude = eigenvalue.norm();
        println!("Eigenvalue {} magnitude: {}", i, magnitude);
        assert!(magnitude < 1.0, "Closed-loop system is not stable. Eigenvalue {} magnitude: {}", i, magnitude);
    }

    println!("Test completed successfully.");
}