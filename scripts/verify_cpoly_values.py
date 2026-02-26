"""
Verify CPoly (Chebyshev Polynomial) implementation against numpy.polynomial.chebyshev

This script generates reference values using numpy's Chebyshev polynomial implementation
to verify that our C++ CPoly class produces correct results.

Usage:
    python scripts/verify_cpoly_values.py
"""

import numpy as np
from numpy.polynomial import chebyshev as C
import json


def chebyshev_from_values(x_points, y_values, degree=None):
    """
    Fit a Chebyshev polynomial to given points.

    Args:
        x_points: x coordinates
        y_values: y values at those points
        degree: polynomial degree (default: len(x_points) - 1)

    Returns:
        Chebyshev polynomial coefficients
    """
    if degree is None:
        degree = len(x_points) - 1
    return C.chebfit(x_points, y_values, degree)


def chebyshev_derivative_coeffs(coeffs):
    """Get derivative coefficients of a Chebyshev polynomial."""
    return C.chebder(coeffs)


def chebyshev_antiderivative_coeffs(coeffs):
    """Get antiderivative coefficients of a Chebyshev polynomial."""
    return C.chebint(coeffs)


def evaluate_chebyshev(coeffs, x):
    """Evaluate Chebyshev polynomial at point(s)."""
    return C.chebval(x, coeffs)


def power_to_chebyshev(power_coeffs):
    """Convert power basis coefficients to Chebyshev coefficients."""
    # Create polynomial in power basis and convert
    from numpy.polynomial import polynomial as P
    # poly2cheb converts from power basis to Chebyshev
    return C.poly2cheb(power_coeffs)


def chebyshev_to_power(cheb_coeffs):
    """Convert Chebyshev coefficients to power basis."""
    return C.cheb2poly(cheb_coeffs)


def generate_test_cases():
    """Generate comprehensive test cases for CPoly verification."""

    test_cases = []

    # Test 1: Constant polynomial on [0, 1]
    # f(x) = 5, mapped to [-1, 1]: f(s) = 5
    # Chebyshev coefficients: [5]
    test_cases.append({
        "name": "constant",
        "coefficients": [[5.0]],
        "breakpoints": [0.0, 1.0],
        "eval_points": [0.0, 0.25, 0.5, 0.75, 1.0],
        "expected_values": [5.0, 5.0, 5.0, 5.0, 5.0],
        "description": "Constant polynomial f(x) = 5"
    })

    # Test 2: Linear polynomial on [0, 1]
    # f(x) = x on [0,1]
    # Map to s in [-1,1]: x = (s+1)/2, so f(s) = (s+1)/2 = 0.5 + 0.5*s
    # Chebyshev: T_0(s) = 1, T_1(s) = s
    # f(s) = 0.5*T_0 + 0.5*T_1 = [0.5, 0.5]
    test_cases.append({
        "name": "linear_identity",
        "coefficients": [[0.5], [0.5]],
        "breakpoints": [0.0, 1.0],
        "eval_points": [0.0, 0.25, 0.5, 0.75, 1.0],
        "expected_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "description": "Linear polynomial f(x) = x"
    })

    # Test 3: Quadratic polynomial on [0, 1]
    # f(x) = x^2 on [0,1]
    # x = (s+1)/2, so x^2 = (s+1)^2/4 = (s^2 + 2s + 1)/4
    # = 0.25 + 0.5*s + 0.25*s^2
    # T_0 = 1, T_1 = s, T_2 = 2s^2 - 1, so s^2 = (T_2 + 1)/2
    # f(s) = 0.25 + 0.5*T_1 + 0.25*(T_2 + 1)/2
    #      = 0.25 + 0.125 + 0.5*T_1 + 0.125*T_2
    #      = 0.375 + 0.5*T_1 + 0.125*T_2
    # Chebyshev coeffs: [0.375, 0.5, 0.125]
    test_cases.append({
        "name": "quadratic_x_squared",
        "coefficients": [[0.375], [0.5], [0.125]],
        "breakpoints": [0.0, 1.0],
        "eval_points": [0.0, 0.25, 0.5, 0.75, 1.0],
        "expected_values": [0.0, 0.0625, 0.25, 0.5625, 1.0],
        "description": "Quadratic polynomial f(x) = x^2"
    })

    # Test 4: Derivative of linear
    # f(x) = x, f'(x) = 1
    # Chebyshev coeffs for x on [0,1]: [0.5, 0.5]
    # Derivative: need to scale by 2/h = 2/1 = 2
    # d/ds[0.5 + 0.5*s] = 0.5, then multiply by 2/h = 1
    test_cases.append({
        "name": "derivative_linear",
        "description": "Derivative of f(x) = x is f'(x) = 1",
        "original_coeffs": [[0.5], [0.5]],
        "breakpoints": [0.0, 1.0],
        "eval_points": [0.0, 0.5, 1.0],
        "expected_derivative_values": [1.0, 1.0, 1.0]
    })

    # Test 5: Derivative of quadratic
    # f(x) = x^2, f'(x) = 2x
    # Chebyshev coeffs: [0.375, 0.5, 0.125]
    test_cases.append({
        "name": "derivative_quadratic",
        "description": "Derivative of f(x) = x^2 is f'(x) = 2x",
        "original_coeffs": [[0.375], [0.5], [0.125]],
        "breakpoints": [0.0, 1.0],
        "eval_points": [0.0, 0.5, 1.0],
        "expected_derivative_values": [0.0, 1.0, 2.0]
    })

    # Test 6: Antiderivative of constant
    # f(x) = 2, integral from 0 to x is 2x
    # Chebyshev coeffs for 2: [2]
    test_cases.append({
        "name": "antiderivative_constant",
        "description": "Antiderivative of f(x) = 2 with F(0) = 0",
        "original_coeffs": [[2.0]],
        "breakpoints": [0.0, 1.0],
        "eval_points": [0.0, 0.5, 1.0],
        "expected_antiderivative_values": [0.0, 1.0, 2.0]
    })

    # Test 7: Integration
    # integral of x from 0 to 1 = 0.5
    test_cases.append({
        "name": "integration_linear",
        "description": "Integral of x from 0 to 1 = 0.5",
        "coefficients": [[0.5], [0.5]],
        "breakpoints": [0.0, 1.0],
        "integral_bounds": [0.0, 1.0],
        "expected_integral": 0.5
    })

    # Test 8: Power basis conversion
    # Power: 1 + 2x + 3x^2 on [0,1]
    # Need to map to [-1,1]: x = (s+1)/2
    # 1 + 2*(s+1)/2 + 3*((s+1)/2)^2
    # = 1 + (s+1) + 3*(s^2 + 2s + 1)/4
    # = 1 + s + 1 + 0.75*s^2 + 1.5*s + 0.75
    # = 2.75 + 2.5*s + 0.75*s^2
    # Convert to Chebyshev: s^2 = (T_2 + 1)/2
    # = 2.75 + 2.5*T_1 + 0.75*(T_2 + 1)/2
    # = 2.75 + 0.375 + 2.5*T_1 + 0.375*T_2
    # = 3.125 + 2.5*T_1 + 0.375*T_2
    # Chebyshev coeffs: [3.125, 2.5, 0.375]
    test_cases.append({
        "name": "power_to_chebyshev",
        "description": "Convert power basis [1, 2, 3] to Chebyshev",
        "power_coeffs": [1.0, 2.0, 3.0],
        "breakpoints": [0.0, 1.0],
        "eval_points": [0.0, 0.5, 1.0],
        "expected_values": [1.0, 2.75, 6.0]
    })

    return test_cases


def generate_scipy_reference_data():
    """
    Generate reference data using numpy.polynomial.chebyshev.
    This data will be used to verify the C++ implementation.
    """

    print("=== CPoly (Chebyshev Polynomial) Scipy Reference Values ===\n")

    # Standard domain mapping for interval [a, b]:
    # s = (2x - a - b) / (b - a)  maps x to [-1, 1]
    # x = ((b-a)*s + a + b) / 2   maps s back to x

    test_data = {}

    # Test 1: Constant on [0, 1]
    print("--- Test 1: Constant f(x) = 5 on [0, 1] ---")
    coeffs = np.array([5.0])  # Chebyshev coeffs for constant 5
    eval_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # Map to [-1, 1]: s = 2*x - 1
    s_pts = 2 * eval_pts - 1
    values = C.chebval(s_pts, coeffs)
    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    test_data["constant"] = {
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 2: Linear f(x) = x on [0, 1]
    print("\n--- Test 2: Linear f(x) = x on [0, 1] ---")
    # In Chebyshev basis on [-1, 1], x on [0,1] becomes:
    # f(s) = (s+1)/2 = 0.5 + 0.5*s = 0.5*T_0 + 0.5*T_1
    coeffs = np.array([0.5, 0.5])
    values = C.chebval(s_pts, coeffs)
    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    test_data["linear"] = {
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 3: Quadratic f(x) = x^2 on [0, 1]
    print("\n--- Test 3: Quadratic f(x) = x^2 on [0, 1] ---")
    # Power basis [0, 0, 1] for x^2 on standard domain
    # Map to [-1, 1]: x = (s+1)/2
    # x^2 = (s+1)^2/4 = (s^2 + 2s + 1)/4
    # In power basis on s: [0.25, 0.5, 0.25]
    power_coeffs = np.array([0.25, 0.5, 0.25])
    coeffs = C.poly2cheb(power_coeffs)
    values = C.chebval(s_pts, coeffs)
    print(f"Power coeffs (on s): {power_coeffs.tolist()}")
    print(f"Chebyshev coeffs: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    test_data["quadratic"] = {
        "power_coeffs_on_s": power_coeffs.tolist(),
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 4: Derivative of x^2
    print("\n--- Test 4: Derivative of f(x) = x^2 ---")
    deriv_coeffs = C.chebder(coeffs)
    # Scale by 2/(b-a) = 2/1 = 2 for derivative
    deriv_values = C.chebval(s_pts, deriv_coeffs) * 2
    print(f"Derivative coeffs (unscaled): {deriv_coeffs.tolist()}")
    print(f"Derivative values (scaled by 2/h): {deriv_values.tolist()}")
    print(f"Expected (2x): {(2*eval_pts).tolist()}")
    test_data["derivative_quadratic"] = {
        "original_coeffs": coeffs.tolist(),
        "derivative_coeffs": deriv_coeffs.tolist(),
        "scale_factor": 2.0,
        "eval_points": eval_pts.tolist(),
        "derivative_values": deriv_values.tolist()
    }

    # Test 5: Antiderivative of constant 2
    print("\n--- Test 5: Antiderivative of f(x) = 2 ---")
    const_coeffs = np.array([2.0])
    # Antiderivative in s-space
    anti_coeffs = C.chebint(const_coeffs)
    # Scale by (b-a)/2 = 0.5 for antiderivative
    # Also need to adjust constant for F(a) = 0
    anti_values_s = C.chebval(s_pts, anti_coeffs) * 0.5
    # Shift so F(0) = 0 (s = -1 when x = 0)
    anti_at_a = C.chebval(-1, anti_coeffs) * 0.5
    anti_values = anti_values_s - anti_at_a
    print(f"Antiderivative coeffs (unscaled): {anti_coeffs.tolist()}")
    print(f"Antiderivative values (F(0)=0): {anti_values.tolist()}")
    print(f"Expected (2x): {(2*eval_pts).tolist()}")
    test_data["antiderivative_constant"] = {
        "original_coeffs": const_coeffs.tolist(),
        "antiderivative_coeffs": anti_coeffs.tolist(),
        "scale_factor": 0.5,
        "eval_points": eval_pts.tolist(),
        "antiderivative_values": anti_values.tolist()
    }

    # Test 6: Integration of x from 0 to 1
    print("\n--- Test 6: Integration of x from 0 to 1 ---")
    linear_coeffs = np.array([0.5, 0.5])
    # Integral on [-1, 1] and scale
    anti = C.chebint(linear_coeffs)
    integral_s = C.chebval(1, anti) - C.chebval(-1, anti)
    integral = integral_s * 0.5  # Scale by h/2
    print(f"Integral of x from 0 to 1: {integral}")
    print(f"Expected: 0.5")
    test_data["integration_linear"] = {
        "coefficients": linear_coeffs.tolist(),
        "integral": float(integral)
    }

    # Test 7: Higher degree polynomial
    print("\n--- Test 7: Cubic f(x) = x^3 on [0, 1] ---")
    # x^3 = ((s+1)/2)^3 = (s+1)^3/8
    # = (s^3 + 3s^2 + 3s + 1)/8
    power_coeffs = np.array([1/8, 3/8, 3/8, 1/8])
    coeffs = C.poly2cheb(power_coeffs)
    values = C.chebval(s_pts, coeffs)
    print(f"Power coeffs (on s): {power_coeffs.tolist()}")
    print(f"Chebyshev coeffs: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    print(f"Expected (x^3): {(eval_pts**3).tolist()}")
    test_data["cubic"] = {
        "power_coeffs_on_s": power_coeffs.tolist(),
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 8: Second derivative
    print("\n--- Test 8: Second derivative of f(x) = x^3 ---")
    deriv1 = C.chebder(coeffs)
    deriv2 = C.chebder(deriv1)
    # Scale by (2/h)^2 = 4 for second derivative
    deriv2_values = C.chebval(s_pts, deriv2) * 4
    print(f"Second derivative values: {deriv2_values.tolist()}")
    print(f"Expected (6x): {(6*eval_pts).tolist()}")
    test_data["second_derivative_cubic"] = {
        "original_coeffs": coeffs.tolist(),
        "derivative2_coeffs": deriv2.tolist(),
        "scale_factor": 4.0,
        "derivative2_values": deriv2_values.tolist()
    }

    # Test 9: from_power_basis verification
    print("\n--- Test 9: Power to Chebyshev conversion ---")
    # Power basis: 1 + 2x + 3x^2 on [0, 1]
    # Need to express in terms of s = 2x - 1, so x = (s+1)/2
    # 1 + 2*(s+1)/2 + 3*((s+1)/2)^2
    # = 1 + (s+1) + 3*(s^2 + 2s + 1)/4
    # = 1 + s + 1 + 0.75*s^2 + 1.5*s + 0.75
    # = 2.75 + 2.5*s + 0.75*s^2
    power_on_s = np.array([2.75, 2.5, 0.75])
    cheb_coeffs = C.poly2cheb(power_on_s)
    values = C.chebval(s_pts, cheb_coeffs)
    print(f"Original power basis (on x): [1, 2, 3]")
    print(f"Mapped power basis (on s): {power_on_s.tolist()}")
    print(f"Chebyshev coeffs: {cheb_coeffs.tolist()}")
    print(f"Values at x={eval_pts.tolist()}: {values.tolist()}")
    expected = 1 + 2*eval_pts + 3*eval_pts**2
    print(f"Expected (1+2x+3x^2): {expected.tolist()}")
    test_data["power_to_chebyshev"] = {
        "power_coeffs_on_x": [1.0, 2.0, 3.0],
        "power_coeffs_on_s": power_on_s.tolist(),
        "chebyshev_coeffs": cheb_coeffs.tolist(),
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 10: to_power_basis verification
    print("\n--- Test 10: Chebyshev to Power conversion ---")
    # Start with Chebyshev coeffs and convert back
    cheb = np.array([1.0, 2.0, 3.0])  # arbitrary Chebyshev coeffs
    power = C.cheb2poly(cheb)
    # Verify round-trip
    cheb_back = C.poly2cheb(power)
    print(f"Chebyshev coeffs: {cheb.tolist()}")
    print(f"Power coeffs (on s): {power.tolist()}")
    print(f"Back to Chebyshev: {cheb_back.tolist()}")
    test_data["chebyshev_to_power"] = {
        "chebyshev_coeffs": cheb.tolist(),
        "power_coeffs": power.tolist(),
        "round_trip_coeffs": cheb_back.tolist()
    }

    # Test 11: Different interval [2, 5]
    print("\n--- Test 11: Polynomial on [2, 5] ---")
    a, b = 2.0, 5.0
    h = b - a  # 3
    # f(x) = x on [2, 5]
    # s = (2x - a - b) / (b - a) = (2x - 7) / 3
    # x = ((b-a)*s + a + b) / 2 = (3s + 7) / 2
    # f(s) = (3s + 7) / 2 = 3.5 + 1.5*s
    # Chebyshev: [3.5, 1.5]
    coeffs = np.array([3.5, 1.5])
    eval_x = np.array([2.0, 2.75, 3.5, 4.25, 5.0])
    s_pts_11 = (2 * eval_x - a - b) / (b - a)
    values = C.chebval(s_pts_11, coeffs)
    print(f"Interval: [{a}, {b}]")
    print(f"Chebyshev coeffs: {coeffs.tolist()}")
    print(f"s points: {s_pts_11.tolist()}")
    print(f"Values at x={eval_x.tolist()}: {values.tolist()}")
    print(f"Expected (x): {eval_x.tolist()}")
    test_data["different_interval"] = {
        "coefficients": coeffs.tolist(),
        "breakpoints": [a, b],
        "eval_points": eval_x.tolist(),
        "s_points": s_pts_11.tolist(),
        "values": values.tolist()
    }

    # Save to JSON
    output_path = "scripts/cpoly_reference_data.json"
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"\n=== Reference data saved to {output_path} ===")

    return test_data


def verify_clenshaw():
    """Verify that our understanding of Clenshaw algorithm matches numpy."""
    print("\n=== Clenshaw Algorithm Verification ===\n")

    # Test Clenshaw manually vs numpy
    coeffs = np.array([1.0, 2.0, 3.0, 4.0])  # Degree 3
    s_val = 0.5

    # Numpy evaluation
    np_result = C.chebval(s_val, coeffs)

    # Manual Clenshaw
    n = len(coeffs) - 1
    b_k1 = 0.0
    b_k2 = 0.0
    for k in range(n, 0, -1):
        b_k = 2 * s_val * b_k1 - b_k2 + coeffs[k]
        b_k2 = b_k1
        b_k1 = b_k
    manual_result = coeffs[0] + s_val * b_k1 - b_k2

    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Evaluation at s = {s_val}")
    print(f"Numpy result: {np_result}")
    print(f"Manual Clenshaw: {manual_result}")
    print(f"Match: {np.isclose(np_result, manual_result)}")


def verify_derivative_formula():
    """Verify Chebyshev derivative formula."""
    print("\n=== Derivative Formula Verification ===\n")

    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Degree 4
    np_deriv = C.chebder(coeffs)

    # Manual derivative using recurrence
    # c'_{n-1} = 2n * c_n
    # c'_k = c'_{k+2} + 2(k+1) * c_{k+1}  for k = n-2, ..., 1
    # c'_0 = c'_2/2 + c_1
    n = len(coeffs) - 1
    d = np.zeros(n)
    d[n-1] = 2 * n * coeffs[n]
    for k in range(n-2, 0, -1):
        d_k2 = d[k+2] if k+2 < n else 0.0
        d[k] = d_k2 + 2 * (k+1) * coeffs[k+1]
    d_2 = d[2] if 2 < n else 0.0
    d[0] = d_2 / 2 + coeffs[1]

    print(f"Original coeffs: {coeffs.tolist()}")
    print(f"Numpy derivative: {np_deriv.tolist()}")
    print(f"Manual derivative: {d.tolist()}")
    print(f"Match: {np.allclose(np_deriv, d)}")


def verify_antiderivative_formula():
    """Verify Chebyshev antiderivative formula."""
    print("\n=== Antiderivative Formula Verification ===\n")

    coeffs = np.array([1.0, 2.0, 3.0, 4.0])  # Degree 3
    np_anti = C.chebint(coeffs)

    # Manual antiderivative
    n = len(coeffs) - 1
    C_arr = np.zeros(n + 2)

    # C_1 = c_0 - c_2/2 (special case)
    c_0 = coeffs[0]
    c_2 = coeffs[2] if n >= 2 else 0.0
    C_arr[1] = c_0 - c_2 / 2

    # C_k = (c_{k-1} - c_{k+1}) / (2k) for k >= 2
    for k in range(2, n + 2):
        c_prev = coeffs[k-1] if k-1 <= n else 0.0
        c_next = coeffs[k+1] if k+1 <= n else 0.0
        C_arr[k] = (c_prev - c_next) / (2 * k)

    # C_0 is the integration constant (numpy sets it to 0)
    C_arr[0] = 0.0

    print(f"Original coeffs: {coeffs.tolist()}")
    print(f"Numpy antiderivative: {np_anti.tolist()}")
    print(f"Manual antiderivative: {C_arr.tolist()}")
    print(f"Match: {np.allclose(np_anti, C_arr)}")


if __name__ == "__main__":
    verify_clenshaw()
    verify_derivative_formula()
    verify_antiderivative_formula()
    generate_scipy_reference_data()
