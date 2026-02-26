"""
Verify LagPoly (Barycentric Lagrange Polynomial) implementation against scipy

This script generates reference values using scipy.interpolate.BarycentricInterpolator
to verify that our C++ LagPoly class produces correct results.

Usage:
    python scripts/verify_lagpoly_values.py
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator
import json


def chebyshev_nodes_second_kind(n, a, b):
    """
    Generate Chebyshev nodes of second kind on [a, b].

    Formula: x_k = (a+b)/2 + (b-a)/2 * cos(pi*k/(n-1)) for k=0..n-1

    NOTE: This generates nodes in DESCENDING order (cos(0)=1 first, cos(pi)=-1 last)
    to match the C++ implementation.
    """
    mid = (a + b) / 2.0
    half = (b - a) / 2.0
    return np.array([mid + half * np.cos(np.pi * k / (n - 1)) for k in range(n)])


def compute_barycentric_weights(nodes, a, b):
    """
    Compute barycentric weights, auto-detecting Chebyshev nodes.

    For Chebyshev nodes of second kind:
        w_k = (-1)^k * delta_k where delta = 0.5 at endpoints, 1 otherwise

    For general nodes:
        w_k = 1 / prod_{j!=k}(x_k - x_j)
    """
    n = len(nodes)

    # Check if nodes are Chebyshev second kind
    expected = chebyshev_nodes_second_kind(n, a, b)
    if np.allclose(nodes, expected, atol=1e-10):
        # Use optimized Chebyshev weights
        weights = np.array([
            ((-1)**k) * (0.5 if k == 0 or k == n-1 else 1.0)
            for k in range(n)
        ])
        return weights

    # General O(n^2) weight computation
    weights = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                weights[j] /= (nodes[j] - nodes[k])
    return weights


def build_diff_matrix(nodes, weights):
    """Build differentiation matrix for barycentric Lagrange interpolation."""
    n = len(nodes)
    D = np.zeros((n, n))

    for i in range(n):
        diag_sum = 0.0
        for j in range(n):
            if i != j:
                D[i, j] = (weights[j] / weights[i]) / (nodes[i] - nodes[j])
                diag_sum += D[i, j]
        D[i, i] = -diag_sum

    return D


def lagrange_derivative_at_nodes(nodes, values, nu, a, b):
    """
    Compute nu-th derivative values at nodes using differentiation matrix.

    Returns the derivative values at each node.
    """
    weights = compute_barycentric_weights(nodes, a, b)
    D = build_diff_matrix(nodes, weights)

    # Apply D^nu to values
    v = np.array(values)
    for _ in range(nu):
        v = D @ v

    return v


def generate_scipy_reference_data():
    """
    Generate reference data using scipy.interpolate.BarycentricInterpolator.
    This data will be used to verify the C++ implementation.
    """

    print("=== LagPoly (Barycentric Lagrange) Scipy Reference Values ===\n")

    test_data = {}

    # Test 1: Constant on [0, 1] with 5 Chebyshev nodes
    print("--- Test 1: Constant f(x) = 5 on [0, 1] ---")
    nodes = chebyshev_nodes_second_kind(5, 0.0, 1.0)
    values = np.full(5, 5.0)
    interp = BarycentricInterpolator(nodes, values)
    eval_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    results = [float(interp(x)) for x in eval_pts]
    print(f"Nodes: {nodes.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {results}")
    test_data["constant"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "expected_values": results
    }

    # Test 2: Linear f(x) = x on [0, 1]
    print("\n--- Test 2: Linear f(x) = x on [0, 1] ---")
    nodes = chebyshev_nodes_second_kind(5, 0.0, 1.0)
    values = nodes.copy()
    interp = BarycentricInterpolator(nodes, values)
    results = [float(interp(x)) for x in eval_pts]
    print(f"Nodes: {nodes.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {results}")
    test_data["linear"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "expected_values": results
    }

    # Test 3: Quadratic f(x) = x^2 on [0, 1]
    print("\n--- Test 3: Quadratic f(x) = x^2 on [0, 1] ---")
    nodes = chebyshev_nodes_second_kind(5, 0.0, 1.0)
    values = nodes**2
    interp = BarycentricInterpolator(nodes, values)
    results = [float(interp(x)) for x in eval_pts]
    expected = eval_pts**2
    print(f"Nodes: {nodes.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {results}")
    print(f"Expected (x^2): {expected.tolist()}")
    test_data["quadratic"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "expected_values": results
    }

    # Test 4 (Test 52): x^3 - x on [0, 2] with 5 Chebyshev nodes
    print("\n--- Test 4 (Test 52): f(x) = x^3 - x on [0, 2] ---")
    nodes = chebyshev_nodes_second_kind(5, 0.0, 2.0)
    values = nodes**3 - nodes
    interp = BarycentricInterpolator(nodes, values)
    eval_pts_52 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    results = [float(interp(x)) for x in eval_pts_52]
    expected = eval_pts_52**3 - eval_pts_52
    print(f"Nodes: {nodes.tolist()}")
    print(f"Values at nodes: {values.tolist()}")
    print(f"Interpolated at {eval_pts_52.tolist()}: {results}")
    print(f"Expected (x^3-x): {expected.tolist()}")
    test_data["test_52_x3_minus_x"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "breakpoints": [0.0, 2.0],
        "eval_points": eval_pts_52.tolist(),
        "expected_values": results
    }

    # Test 5 (Test 50): sin(x) on [0, pi] with 10 Chebyshev nodes
    print("\n--- Test 5 (Test 50): sin(x) on [0, pi] ---")
    nodes = chebyshev_nodes_second_kind(10, 0.0, np.pi)
    values = np.sin(nodes)
    interp = BarycentricInterpolator(nodes, values)
    eval_pts_50 = np.array([0.5, 1.0, 2.0])
    results = [float(interp(x)) for x in eval_pts_50]
    expected = np.sin(eval_pts_50)
    print(f"Nodes: {nodes.tolist()}")
    for i, x in enumerate(eval_pts_50):
        err = abs(results[i] - expected[i])
        print(f"  sin({x}): interp={results[i]:.15f}, exact={expected[i]:.15f}, err={err:.2e}")
    test_data["test_50_sin"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "breakpoints": [0.0, float(np.pi)],
        "eval_points": eval_pts_50.tolist(),
        "expected_values": results
    }

    # Test 6 (Test 51): Runge function 1/(1+25x^2) on [-1, 1] with 15 Chebyshev nodes
    print("\n--- Test 6 (Test 51): Runge 1/(1+25x^2) on [-1, 1] ---")
    nodes = chebyshev_nodes_second_kind(15, -1.0, 1.0)
    values = 1.0 / (1.0 + 25.0 * nodes**2)
    interp = BarycentricInterpolator(nodes, values)
    eval_pts_51 = np.array([0.0, 0.5])
    results = [float(interp(x)) for x in eval_pts_51]
    expected = 1.0 / (1.0 + 25.0 * eval_pts_51**2)
    print(f"Nodes: {nodes.tolist()}")
    for i, x in enumerate(eval_pts_51):
        err = abs(results[i] - expected[i])
        print(f"  Runge({x}): interp={results[i]:.15f}, exact={expected[i]:.15f}, err={err:.2e}")
    test_data["test_51_runge"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "breakpoints": [-1.0, 1.0],
        "eval_points": eval_pts_51.tolist(),
        "expected_values": results
    }

    # Test 7 (Tests 125-127): exp(x) on [0, 1] with 10 Chebyshev nodes
    print("\n--- Test 7 (Tests 125-127): exp(x) on [0, 1] ---")
    nodes = chebyshev_nodes_second_kind(10, 0.0, 1.0)
    exp_values = np.exp(nodes)
    interp = BarycentricInterpolator(nodes, exp_values)
    eval_pts_exp = np.array([0.0, 0.5, 1.0])
    results = [float(interp(x)) for x in eval_pts_exp]
    expected = np.exp(eval_pts_exp)
    print("exp(x) evaluation:")
    for i, x in enumerate(eval_pts_exp):
        err = abs(results[i] - expected[i])
        print(f"  exp({x}): interp={results[i]:.15f}, exact={expected[i]:.15f}, err={err:.2e}")

    # Derivative of exp(x) at 0.5 using differentiation matrix
    deriv_values = lagrange_derivative_at_nodes(nodes, exp_values, 1, 0.0, 1.0)
    deriv_interp = BarycentricInterpolator(nodes, deriv_values)
    deriv_at_05 = float(deriv_interp(0.5))
    exact_deriv = np.exp(0.5)
    print(f"  d/dx[exp](0.5): interp={deriv_at_05:.15f}, exact={exact_deriv:.15f}, err={abs(deriv_at_05-exact_deriv):.2e}")

    test_data["test_125_127_exp"] = {
        "nodes": nodes.tolist(),
        "values": exp_values.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts_exp.tolist(),
        "expected_values": results,
        "derivative_at_05": deriv_at_05
    }

    # Test 8 (Test 126): cos(x) on [0, 1] with 10 Chebyshev nodes
    print("\n--- Test 8 (Test 126): cos(x) on [0, 1] ---")
    cos_values = np.cos(nodes)
    cos_interp = BarycentricInterpolator(nodes, cos_values)
    cos_at_05 = float(cos_interp(0.5))
    exact_cos = np.cos(0.5)
    print(f"  cos(0.5): interp={cos_at_05:.15f}, exact={exact_cos:.15f}, err={abs(cos_at_05-exact_cos):.2e}")
    test_data["test_126_cos"] = {
        "nodes": nodes.tolist(),
        "values": cos_values.tolist(),
        "breakpoints": [0.0, 1.0],
        "expected_value_at_05": cos_at_05
    }

    # Test 9: Derivative verification - f(x) = x^3, f'(x) = 3x^2
    print("\n--- Test 9: Derivative of x^3 ---")
    nodes = chebyshev_nodes_second_kind(6, 0.0, 1.0)
    values = nodes**3
    interp = BarycentricInterpolator(nodes, values)

    # Compute derivatives using differentiation matrix
    deriv_values = lagrange_derivative_at_nodes(nodes, values, 1, 0.0, 1.0)
    deriv_interp = BarycentricInterpolator(nodes, deriv_values)

    deriv_pts = np.array([0.0, 0.5, 1.0])
    deriv_results = [float(deriv_interp(x)) for x in deriv_pts]
    expected_deriv = 3 * deriv_pts**2
    print(f"Derivative of x^3 at {deriv_pts.tolist()}:")
    for i, x in enumerate(deriv_pts):
        err = abs(deriv_results[i] - expected_deriv[i])
        print(f"  f'({x}): interp={deriv_results[i]:.10f}, exact={expected_deriv[i]:.10f}, err={err:.2e}")
    test_data["derivative_x3"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "breakpoints": [0.0, 1.0],
        "derivative_eval_points": deriv_pts.tolist(),
        "derivative_values": deriv_results
    }

    # Test 10: Chebyshev weights verification
    print("\n--- Test 10: Chebyshev weights ---")
    nodes = chebyshev_nodes_second_kind(5, 0.0, 1.0)
    weights = compute_barycentric_weights(nodes, 0.0, 1.0)
    print(f"5-node Chebyshev weights: {weights.tolist()}")
    print(f"Expected: [0.5, -1.0, 1.0, -1.0, 0.5]")
    test_data["chebyshev_weights_5"] = {
        "nodes": nodes.tolist(),
        "weights": weights.tolist()
    }

    nodes3 = chebyshev_nodes_second_kind(3, 0.0, 1.0)
    weights3 = compute_barycentric_weights(nodes3, 0.0, 1.0)
    print(f"3-node Chebyshev weights: {weights3.tolist()}")
    print(f"Expected: [0.5, -1.0, 0.5]")
    test_data["chebyshev_weights_3"] = {
        "nodes": nodes3.tolist(),
        "weights": weights3.tolist()
    }

    # Test 11: from_derivatives (Hermite) verification
    # f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1 (cubic)
    print("\n--- Test 11: from_derivatives Hermite cubic ---")
    # The Hermite cubic passing through these constraints
    # p(x) = 2x^3 - 3x^2 + x + 0 (can be verified)
    # Actually, let's compute it properly:
    # p(t) = (1+2t)(1-t)^2 * f0 + t(1-t)^2 * f'0 + t^2(3-2t) * f1 + t^2(t-1) * f'1
    # With f0=0, f'0=1, f1=1, f'1=-1:
    # p(t) = t(1-t)^2 + t^2(3-2t) - t^2(t-1)
    # = t(1-2t+t^2) + 3t^2 - 2t^3 - t^3 + t^2
    # = t - 2t^2 + t^3 + 3t^2 - 2t^3 - t^3 + t^2
    # = t + 2t^2 - 2t^3
    def hermite_cubic(x):
        return x + 2*x**2 - 2*x**3

    def hermite_cubic_deriv(x):
        return 1 + 4*x - 6*x**2

    test_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    hermite_values = [hermite_cubic(x) for x in test_pts]
    hermite_derivs = [hermite_cubic_deriv(x) for x in test_pts]
    print(f"Hermite cubic values at {test_pts.tolist()}: {hermite_values}")
    print(f"Hermite cubic derivatives at {test_pts.tolist()}: {hermite_derivs}")
    print(f"Check: f(0)={hermite_cubic(0)}, f'(0)={hermite_cubic_deriv(0)}, f(1)={hermite_cubic(1)}, f'(1)={hermite_cubic_deriv(1)}")
    test_data["hermite_cubic"] = {
        "xi": [0.0, 1.0],
        "yi": [[0.0, 1.0], [1.0, -1.0]],
        "test_points": test_pts.tolist(),
        "expected_values": hermite_values,
        "expected_derivatives": hermite_derivs
    }

    # Test 12: Extrapolation of linear
    print("\n--- Test 12: Linear extrapolation ---")
    nodes = chebyshev_nodes_second_kind(3, 0.0, 1.0)
    values = nodes.copy()
    interp = BarycentricInterpolator(nodes, values)
    extrap_pts = np.array([-0.5, 1.5])
    extrap_results = [float(interp(x)) for x in extrap_pts]
    print(f"Linear f(x)=x extrapolated: {extrap_results}")
    print(f"Expected: [-0.5, 1.5]")
    test_data["linear_extrapolation"] = {
        "nodes": nodes.tolist(),
        "values": values.tolist(),
        "extrap_points": extrap_pts.tolist(),
        "expected_values": extrap_results
    }

    # Save to JSON
    output_path = "scripts/lagpoly_reference_data.json"
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"\n=== Reference data saved to {output_path} ===")

    return test_data


def verify_barycentric_formula():
    """Verify that our understanding of barycentric formula matches scipy."""
    print("\n=== Barycentric Formula Verification ===\n")

    # Test with known polynomial: f(x) = x^2
    nodes = chebyshev_nodes_second_kind(5, 0.0, 1.0)
    values = nodes**2

    # Scipy interpolation
    interp = BarycentricInterpolator(nodes, values)
    scipy_result = interp(0.3)

    # Manual barycentric evaluation
    weights = compute_barycentric_weights(nodes, 0.0, 1.0)
    x = 0.3

    # Check if x is close to a node
    for i, node in enumerate(nodes):
        if abs(x - node) < 1e-14:
            manual_result = values[i]
            break
    else:
        numerator = sum(w * y / (x - xi) for xi, y, w in zip(nodes, values, weights))
        denominator = sum(w / (x - xi) for xi, w in zip(nodes, weights))
        manual_result = numerator / denominator

    print(f"Nodes: {nodes.tolist()}")
    print(f"Values (x^2): {values.tolist()}")
    print(f"Weights: {weights.tolist()}")
    print(f"Evaluation at x = {x}")
    print(f"Scipy result: {scipy_result}")
    print(f"Manual barycentric: {manual_result}")
    print(f"Expected (x^2): {x**2}")
    print(f"Match: {np.isclose(scipy_result, manual_result)}")


def verify_differentiation_matrix():
    """Verify differentiation matrix computation."""
    print("\n=== Differentiation Matrix Verification ===\n")

    # Use f(x) = x^3, f'(x) = 3x^2
    nodes = chebyshev_nodes_second_kind(6, 0.0, 1.0)
    values = nodes**3

    weights = compute_barycentric_weights(nodes, 0.0, 1.0)
    D = build_diff_matrix(nodes, weights)

    # Apply D to values
    deriv_values = D @ values
    expected_derivs = 3 * nodes**2

    print(f"Nodes: {nodes.tolist()}")
    print(f"Values (x^3): {values.tolist()}")
    print(f"D @ values: {deriv_values.tolist()}")
    print(f"Expected (3x^2): {expected_derivs.tolist()}")
    print(f"Match: {np.allclose(deriv_values, expected_derivs, atol=1e-8)}")


if __name__ == "__main__":
    verify_barycentric_formula()
    verify_differentiation_matrix()
    generate_scipy_reference_data()
