"""
Verify LegPoly (Legendre Polynomial) implementation against numpy.polynomial.legendre

This script generates reference values using numpy's Legendre polynomial implementation
to verify that our C++ LegPoly class produces correct results.

Usage:
    python scripts/verify_legpoly_values.py
"""

import numpy as np
from numpy.polynomial import legendre as L
import json


def evaluate_legendre(coeffs, s):
    """Evaluate Legendre polynomial at point(s)."""
    return L.legval(s, coeffs)


def power_to_legendre(power_coeffs):
    """Convert power basis coefficients to Legendre coefficients."""
    return L.poly2leg(power_coeffs)


def legendre_to_power(leg_coeffs):
    """Convert Legendre coefficients to power basis."""
    return L.leg2poly(leg_coeffs)


def generate_scipy_reference_data():
    """
    Generate reference data using numpy.polynomial.legendre.
    This data will be used to verify the C++ implementation.
    """

    print("=== LegPoly (Legendre Polynomial) Numpy Reference Values ===\n")

    # Standard domain mapping for interval [a, b]:
    # s = (2x - a - b) / (b - a)  maps x to [-1, 1]
    # x = ((b-a)*s + a + b) / 2   maps s back to x

    test_data = {}

    # Test 1: Constant on [0, 1]
    print("--- Test 1: Constant f(x) = 5 on [0, 1] ---")
    coeffs = np.array([5.0])  # Legendre coeffs for constant 5
    eval_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    # Map to [-1, 1]: s = 2*x - 1
    s_pts = 2 * eval_pts - 1
    values = L.legval(s_pts, coeffs)
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
    # In Legendre basis on [-1, 1], x on [0,1] becomes:
    # f(s) = (s+1)/2 = 0.5 + 0.5*s = 0.5*P_0 + 0.5*P_1
    coeffs = np.array([0.5, 0.5])
    values = L.legval(s_pts, coeffs)
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
    # x = (s+1)/2, so x^2 = (s+1)^2/4 = (s^2 + 2s + 1)/4
    # In power basis on s: [0.25, 0.5, 0.25]
    power_coeffs = np.array([0.25, 0.5, 0.25])
    coeffs = L.poly2leg(power_coeffs)
    values = L.legval(s_pts, coeffs)
    print(f"Power coeffs (on s): {power_coeffs.tolist()}")
    print(f"Legendre coeffs: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    print(f"Expected (x^2): {(eval_pts**2).tolist()}")
    test_data["quadratic"] = {
        "power_coeffs_on_s": power_coeffs.tolist(),
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 4: Derivative of x^2
    print("\n--- Test 4: Derivative of f(x) = x^2 ---")
    deriv_coeffs = L.legder(coeffs)
    # Scale by 2/(b-a) = 2/1 = 2 for derivative
    deriv_values = L.legval(s_pts, deriv_coeffs) * 2
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
    anti_coeffs = L.legint(const_coeffs)
    # Scale by (b-a)/2 = 0.5 for antiderivative
    anti_values_s = L.legval(s_pts, anti_coeffs) * 0.5
    # Shift so F(0) = 0 (s = -1 when x = 0)
    anti_at_a = L.legval(-1, anti_coeffs) * 0.5
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
    anti = L.legint(linear_coeffs)
    integral_s = L.legval(1, anti) - L.legval(-1, anti)
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
    coeffs = L.poly2leg(power_coeffs)
    values = L.legval(s_pts, coeffs)
    print(f"Power coeffs (on s): {power_coeffs.tolist()}")
    print(f"Legendre coeffs: {coeffs.tolist()}")
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
    deriv1 = L.legder(coeffs)
    deriv2 = L.legder(deriv1)
    # Scale by (2/h)^2 = 4 for second derivative
    deriv2_values = L.legval(s_pts, deriv2) * 4
    print(f"Second derivative values: {deriv2_values.tolist()}")
    print(f"Expected (6x): {(6*eval_pts).tolist()}")
    test_data["second_derivative_cubic"] = {
        "original_coeffs": coeffs.tolist(),
        "derivative2_coeffs": deriv2.tolist(),
        "scale_factor": 4.0,
        "derivative2_values": deriv2_values.tolist()
    }

    # Test 9: from_power_basis verification
    print("\n--- Test 9: Power to Legendre conversion ---")
    # Power basis: 1 + 2x + 3x^2 on [0, 1]
    # Need to express in terms of s = 2x - 1, so x = (s+1)/2
    # 1 + 2*(s+1)/2 + 3*((s+1)/2)^2
    # = 1 + (s+1) + 3*(s^2 + 2s + 1)/4
    # = 1 + s + 1 + 0.75*s^2 + 1.5*s + 0.75
    # = 2.75 + 2.5*s + 0.75*s^2
    power_on_s = np.array([2.75, 2.5, 0.75])
    leg_coeffs = L.poly2leg(power_on_s)
    values = L.legval(s_pts, leg_coeffs)
    print(f"Original power basis (on x): [1, 2, 3]")
    print(f"Mapped power basis (on s): {power_on_s.tolist()}")
    print(f"Legendre coeffs: {leg_coeffs.tolist()}")
    print(f"Values at x={eval_pts.tolist()}: {values.tolist()}")
    expected = 1 + 2*eval_pts + 3*eval_pts**2
    print(f"Expected (1+2x+3x^2): {expected.tolist()}")
    test_data["power_to_legendre"] = {
        "power_coeffs_on_x": [1.0, 2.0, 3.0],
        "power_coeffs_on_s": power_on_s.tolist(),
        "legendre_coeffs": leg_coeffs.tolist(),
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 10: to_power_basis verification
    print("\n--- Test 10: Legendre to Power conversion ---")
    # Start with Legendre coeffs and convert back
    leg = np.array([1.0, 2.0, 3.0])  # arbitrary Legendre coeffs
    power = L.leg2poly(leg)
    # Verify round-trip
    leg_back = L.poly2leg(power)
    print(f"Legendre coeffs: {leg.tolist()}")
    print(f"Power coeffs (on s): {power.tolist()}")
    print(f"Back to Legendre: {leg_back.tolist()}")
    test_data["legendre_to_power"] = {
        "legendre_coeffs": leg.tolist(),
        "power_coeffs": power.tolist(),
        "round_trip_coeffs": leg_back.tolist()
    }

    # Test 11: Legendre polynomial values at specific points
    print("\n--- Test 11: Legendre polynomial values ---")
    # P_0(s) = 1
    # P_1(s) = s
    # P_2(s) = (3s^2 - 1)/2
    # P_3(s) = (5s^3 - 3s)/2
    s_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    P0_values = L.legval(s_test, [1])
    P1_values = L.legval(s_test, [0, 1])
    P2_values = L.legval(s_test, [0, 0, 1])
    P3_values = L.legval(s_test, [0, 0, 0, 1])

    print(f"P_0 at {s_test.tolist()}: {P0_values.tolist()}")
    print(f"P_1 at {s_test.tolist()}: {P1_values.tolist()}")
    print(f"P_2 at {s_test.tolist()}: {P2_values.tolist()}")
    print(f"P_3 at {s_test.tolist()}: {P3_values.tolist()}")

    # Manual verification
    print(f"Expected P_2 at 0.5: {(3*0.5**2 - 1)/2}")  # -0.125
    print(f"Expected P_3 at 0.5: {(5*0.5**3 - 3*0.5)/2}")  # -0.4375

    test_data["legendre_basis_values"] = {
        "s_points": s_test.tolist(),
        "P0": P0_values.tolist(),
        "P1": P1_values.tolist(),
        "P2": P2_values.tolist(),
        "P3": P3_values.tolist()
    }

    # Test 12: Legendre polynomial derivatives at endpoints
    print("\n--- Test 12: Legendre derivatives at endpoints ---")
    # At s = 1: P_n(1) = 1, P'_n(1) = n(n+1)/2
    # At s = -1: P_n(-1) = (-1)^n, P'_n(-1) = (-1)^{n+1} * n(n+1)/2
    for n in range(5):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0  # Pure P_n

        val_at_1 = L.legval(1, coeffs)
        val_at_m1 = L.legval(-1, coeffs)

        deriv = L.legder(coeffs)
        deriv_at_1 = L.legval(1, deriv) if len(deriv) > 0 else 0.0
        deriv_at_m1 = L.legval(-1, deriv) if len(deriv) > 0 else 0.0

        expected_val_1 = 1.0
        expected_val_m1 = (-1)**n
        expected_deriv_1 = n * (n + 1) / 2
        expected_deriv_m1 = (-1)**(n+1) * n * (n + 1) / 2

        print(f"P_{n}(1) = {val_at_1:.6f} (expected {expected_val_1})")
        print(f"P_{n}(-1) = {val_at_m1:.6f} (expected {expected_val_m1})")
        print(f"P'_{n}(1) = {deriv_at_1:.6f} (expected {expected_deriv_1})")
        print(f"P'_{n}(-1) = {deriv_at_m1:.6f} (expected {expected_deriv_m1})")
        print()

    # Save to JSON
    output_path = "scripts/legpoly_reference_data.json"
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"\n=== Reference data saved to {output_path} ===")

    return test_data


def verify_clenshaw_legendre():
    """Verify Clenshaw algorithm for Legendre polynomials."""
    print("\n=== Clenshaw Algorithm Verification for Legendre ===\n")

    # Test Clenshaw manually vs numpy
    coeffs = np.array([1.0, 2.0, 3.0, 4.0])  # Degree 3
    s_val = 0.5

    # Numpy evaluation
    np_result = L.legval(s_val, coeffs)

    # Manual Clenshaw for Legendre
    # The Legendre recurrence is: (k+1)*P_{k+1} = (2k+1)*s*P_k - k*P_{k-1}
    # Clenshaw backward iteration:
    # b_{n+1} = 0, b_{n+2} = 0
    # For k = n, n-1, ..., 1:
    #   b_k = c_k + alpha_k * s * b_{k+1} - beta_{k+1} * b_{k+2}
    # where alpha_k = (2k+1)/(k+1), beta_k = k/(k+1)
    # Result = c_0 + s * b_1 - 0.5 * b_2

    n = len(coeffs) - 1
    b_k1 = 0.0
    b_k2 = 0.0

    for k in range(n, 0, -1):
        alpha_k = (2*k + 1) / (k + 1)
        beta_k1 = (k + 1) / (k + 2)
        b_k = coeffs[k] + alpha_k * s_val * b_k1 - beta_k1 * b_k2
        b_k2 = b_k1
        b_k1 = b_k

    manual_result = coeffs[0] + s_val * b_k1 - 0.5 * b_k2

    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Evaluation at s = {s_val}")
    print(f"Numpy result: {np_result}")
    print(f"Manual Clenshaw: {manual_result}")
    print(f"Match: {np.isclose(np_result, manual_result)}")


def verify_derivative_formula():
    """Verify Legendre derivative formula."""
    print("\n=== Derivative Formula Verification ===\n")

    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Degree 4
    np_deriv = L.legder(coeffs)

    # Manual derivative using recurrence from numpy docs
    # c'_{n-1} = (2n-1) * c_n
    # c'_k = c'_{k+2} + (2k+3) * c_{k+1}  for k = n-2, ..., 0
    n = len(coeffs) - 1
    d = np.zeros(n)
    d[n-1] = (2*n - 1) * coeffs[n]
    for k in range(n-2, -1, -1):
        d_k2 = d[k+2] if k+2 < n else 0.0
        d[k] = d_k2 + (2*k + 3) * coeffs[k+1]

    print(f"Original coeffs: {coeffs.tolist()}")
    print(f"Numpy derivative: {np_deriv.tolist()}")
    print(f"Manual derivative: {d.tolist()}")
    print(f"Match: {np.allclose(np_deriv, d)}")


def verify_antiderivative_formula():
    """Verify Legendre antiderivative formula."""
    print("\n=== Antiderivative Formula Verification ===\n")

    coeffs = np.array([1.0, 2.0, 3.0, 4.0])  # Degree 3
    np_anti = L.legint(coeffs)

    # Manual antiderivative
    # integral P_k ds = (P_{k+1} - P_{k-1}) / (2k+1) for k >= 1
    # integral P_0 ds = P_1
    # So for f(s) = sum c_k P_k(s), the antiderivative F(s) = sum C_k P_k(s)
    # C_1 += c_0
    # C_{k+1} += c_k / (2k+1)
    # C_{k-1} -= c_k / (2k+1)

    n = len(coeffs) - 1
    C = np.zeros(n + 2)

    # From P_0
    C[1] += coeffs[0]

    # From higher terms
    for k in range(1, n + 1):
        factor = coeffs[k] / (2*k + 1)
        if k + 1 <= n + 1:
            C[k + 1] += factor
        if k - 1 >= 0:
            C[k - 1] -= factor

    # C[0] is the integration constant (numpy sets it to make F(-1) = 0 by default? Let's check)
    # Actually numpy sets C[0] based on the lower bound of integration

    print(f"Original coeffs: {coeffs.tolist()}")
    print(f"Numpy antiderivative: {np_anti.tolist()}")
    print(f"Manual antiderivative (without C_0 adjustment): {C.tolist()}")

    # Numpy's antiderivative sets C_0 such that the integral evaluates to 0 at -1
    # Let's verify by checking that sum C_k * P_k(-1) = 0
    val_at_m1 = L.legval(-1, np_anti)
    print(f"Numpy antiderivative at s=-1: {val_at_m1}")


def generate_legendre_property_tests():
    """
    Generate reference data for Legendre-specific mathematical properties.
    These properties are unique to Legendre polynomials and should be tested.
    """
    print("\n=== Legendre-Specific Property Tests ===\n")

    property_data = {}

    # ============================================================
    # Property 1: Orthogonality
    # Integral of P_n(s) * P_m(s) from -1 to 1 = 0 for n != m
    # ============================================================
    print("--- Property 1: Orthogonality ---")
    from scipy import integrate

    orthogonality_tests = []
    pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (0, 4), (2, 4)]

    for n, m in pairs:
        # Create coefficient vectors for P_n and P_m
        coeffs_n = np.zeros(n + 1)
        coeffs_n[n] = 1.0
        coeffs_m = np.zeros(m + 1)
        coeffs_m[m] = 1.0

        # Define the product P_n(s) * P_m(s)
        def integrand(s):
            return L.legval(s, coeffs_n) * L.legval(s, coeffs_m)

        integral, error = integrate.quad(integrand, -1, 1)
        print(f"  Integral of P_{n} * P_{m} from -1 to 1: {integral:.2e} (error: {error:.2e})")
        orthogonality_tests.append({
            "n": n, "m": m, "integral": float(integral), "expected": 0.0
        })

    property_data["orthogonality"] = orthogonality_tests

    # ============================================================
    # Property 2: Normalization
    # Integral of P_n(s)^2 from -1 to 1 = 2/(2n+1)
    # ============================================================
    print("\n--- Property 2: Normalization ---")

    normalization_tests = []
    for n in range(6):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0

        def integrand(s):
            return L.legval(s, coeffs) ** 2

        integral, error = integrate.quad(integrand, -1, 1)
        expected = 2.0 / (2 * n + 1)
        print(f"  Integral of P_{n}^2: {integral:.10f}, expected: {expected:.10f}")
        normalization_tests.append({
            "n": n, "integral": float(integral), "expected": expected
        })

    property_data["normalization"] = normalization_tests

    # ============================================================
    # Property 3: Endpoint Values
    # P_n(1) = 1 for all n
    # P_n(-1) = (-1)^n for all n
    # ============================================================
    print("\n--- Property 3: Endpoint Values ---")

    endpoint_tests = []
    for n in range(11):
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0

        val_at_1 = L.legval(1.0, coeffs)
        val_at_m1 = L.legval(-1.0, coeffs)
        expected_at_1 = 1.0
        expected_at_m1 = (-1.0) ** n

        print(f"  P_{n}(1) = {val_at_1:.10f} (expected 1), P_{n}(-1) = {val_at_m1:.10f} (expected {expected_at_m1})")
        endpoint_tests.append({
            "n": n,
            "val_at_1": float(val_at_1), "expected_at_1": expected_at_1,
            "val_at_m1": float(val_at_m1), "expected_at_m1": expected_at_m1
        })

    property_data["endpoint_values"] = endpoint_tests

    # ============================================================
    # Property 4: Recurrence Relation
    # (n+1)*P_{n+1}(s) = (2n+1)*s*P_n(s) - n*P_{n-1}(s)
    # ============================================================
    print("\n--- Property 4: Recurrence Relation ---")

    recurrence_tests = []
    s_values = [-0.5, 0.0, 0.5, 0.75]

    for n in range(1, 5):
        coeffs_n_minus_1 = np.zeros(n)
        coeffs_n_minus_1[n-1] = 1.0

        coeffs_n = np.zeros(n + 1)
        coeffs_n[n] = 1.0

        coeffs_n_plus_1 = np.zeros(n + 2)
        coeffs_n_plus_1[n+1] = 1.0

        for s in s_values:
            P_nm1 = L.legval(s, coeffs_n_minus_1)
            P_n = L.legval(s, coeffs_n)
            P_np1 = L.legval(s, coeffs_n_plus_1)

            lhs = (n + 1) * P_np1
            rhs = (2 * n + 1) * s * P_n - n * P_nm1

            print(f"  n={n}, s={s}: LHS={(n+1)*P_np1:.10f}, RHS={rhs:.10f}, diff={abs(lhs-rhs):.2e}")
            recurrence_tests.append({
                "n": n, "s": s, "lhs": float(lhs), "rhs": float(rhs),
                "P_n_minus_1": float(P_nm1), "P_n": float(P_n), "P_n_plus_1": float(P_np1)
            })

    property_data["recurrence_relation"] = recurrence_tests

    # ============================================================
    # Property 5: Derivative at Endpoints
    # P'_n(1) = n(n+1)/2
    # P'_n(-1) = (-1)^{n+1} * n(n+1)/2
    # ============================================================
    print("\n--- Property 5: Derivative at Endpoints ---")

    deriv_endpoint_tests = []
    for n in range(1, 8):  # Start from 1 since P_0 is constant
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0

        deriv_coeffs = L.legder(coeffs)
        deriv_at_1 = L.legval(1.0, deriv_coeffs)
        deriv_at_m1 = L.legval(-1.0, deriv_coeffs)

        expected_at_1 = n * (n + 1) / 2
        expected_at_m1 = ((-1) ** (n + 1)) * n * (n + 1) / 2

        print(f"  P'_{n}(1) = {deriv_at_1:.10f} (expected {expected_at_1})")
        print(f"  P'_{n}(-1) = {deriv_at_m1:.10f} (expected {expected_at_m1})")
        deriv_endpoint_tests.append({
            "n": n,
            "deriv_at_1": float(deriv_at_1), "expected_at_1": expected_at_1,
            "deriv_at_m1": float(deriv_at_m1), "expected_at_m1": expected_at_m1
        })

    property_data["derivative_endpoints"] = deriv_endpoint_tests

    # ============================================================
    # Property 6: High-Degree Polynomial Evaluation
    # Test numerical stability with degree 20, 50
    # ============================================================
    print("\n--- Property 6: High-Degree Stability ---")

    high_degree_tests = []
    for degree in [10, 20, 50]:
        # Polynomial that's 1.0 everywhere: just use c_0 = 1
        coeffs_constant = np.array([1.0])

        # Polynomial representing x (linear) in high-degree representation
        # Actually, just test pure P_n at various points
        coeffs_pure = np.zeros(degree + 1)
        coeffs_pure[degree] = 1.0

        eval_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        s_pts = 2 * eval_pts - 1  # Map [0,1] -> [-1,1]

        values = L.legval(s_pts, coeffs_pure)

        # Expected values at s points for P_n
        # At s=1: P_n(1) = 1
        # At s=-1: P_n(-1) = (-1)^n

        print(f"  P_{degree} at s={s_pts.tolist()}: {values.tolist()}")
        high_degree_tests.append({
            "degree": degree,
            "s_points": s_pts.tolist(),
            "values": values.tolist(),
            "expected_at_1": 1.0,
            "expected_at_m1": float((-1) ** degree)
        })

    property_data["high_degree"] = high_degree_tests

    # ============================================================
    # Property 7: from_derivatives Asymmetric Configurations
    # Test various (n_left, n_right) derivative counts
    # ============================================================
    print("\n--- Property 7: Asymmetric from_derivatives Configurations ---")

    from scipy.interpolate import BPoly

    asymmetric_tests = []
    configurations = [
        # (n_left, n_right) - number of derivatives specified at each endpoint
        (1, 2),  # f at left, f and f' at right
        (2, 1),  # f and f' at left, f at right
        (1, 3),  # f at left, f, f', f'' at right
        (3, 1),  # f, f', f'' at left, f at right
        (2, 3),  # f, f' at left, f, f', f'' at right
        (4, 2),  # f, f', f'', f''' at left, f, f' at right
    ]

    for n_left, n_right in configurations:
        # Create sample derivative values
        yi_left = [float(i) for i in range(n_left)]  # [0], [0, 1], [0, 1, 0], etc.
        yi_right = [float(i + 1) for i in range(n_right)]  # [1], [1, 2], [1, 2, 3], etc.

        xi = [0.0, 1.0]
        yi = [yi_left, yi_right]

        try:
            bp = BPoly.from_derivatives(xi, yi)
            eval_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            values = bp(eval_pts)
            derivatives = bp(eval_pts, 1)

            print(f"  Config ({n_left}, {n_right}): degree {bp.c.shape[0]-1}, values at 0.5: {values[2]:.6f}")
            asymmetric_tests.append({
                "n_left": n_left, "n_right": n_right,
                "xi": xi, "yi": yi,
                "degree": int(bp.c.shape[0] - 1),
                "eval_points": eval_pts.tolist(),
                "values": values.tolist(),
                "derivatives": derivatives.tolist(),
                "coefficients": bp.c.flatten().tolist()
            })
        except Exception as e:
            print(f"  Config ({n_left}, {n_right}): ERROR - {e}")

    property_data["asymmetric_from_derivatives"] = asymmetric_tests

    # Save property test data
    output_path = "scripts/legpoly_property_reference.json"
    with open(output_path, 'w') as f:
        json.dump(property_data, f, indent=2)
    print(f"\n=== Property test data saved to {output_path} ===")

    return property_data


if __name__ == "__main__":
    verify_clenshaw_legendre()
    verify_derivative_formula()
    verify_antiderivative_formula()
    generate_scipy_reference_data()
    generate_legendre_property_tests()
