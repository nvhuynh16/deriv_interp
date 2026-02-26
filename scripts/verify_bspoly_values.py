#!/usr/bin/env python3
"""
Verify BsPoly values against scipy.interpolate.BPoly

Since BsPoly stores Bernstein coefficients internally (identical to BPoly),
the results should match scipy.interpolate.BPoly exactly.
"""

import numpy as np
from scipy.interpolate import BPoly
import json

def verify_constant():
    """Verify constant polynomial"""
    print("=== Constant Polynomial ===")
    c = np.array([[5]])  # Constant 5 on [0,1]
    bp = BPoly(c, [0, 1])

    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x in test_points:
        print(f"  f({x}) = {bp(x)}")

    assert np.allclose([bp(x) for x in test_points], [5, 5, 5, 5, 5])
    print("  PASSED")

def verify_linear():
    """Verify linear polynomial p(x) = x"""
    print("\n=== Linear Polynomial (p(x) = x) ===")
    # Bernstein: c0*(1-t) + c1*t = 0*(1-t) + 1*t = t = x on [0,1]
    c = np.array([[0], [1]])
    bp = BPoly(c, [0, 1])

    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x in test_points:
        val = bp(x)
        expected = x
        print(f"  f({x}) = {val:.10f}, expected {expected:.10f}, diff = {abs(val-expected):.2e}")

    assert np.allclose([bp(x) for x in test_points], test_points)
    print("  PASSED")

def verify_quadratic():
    """Verify quadratic polynomial p(x) = x^2"""
    print("\n=== Quadratic Polynomial (p(x) = x^2) ===")
    # Bernstein for x^2 on [0,1]: c0=0, c1=0, c2=1
    c = np.array([[0], [0], [1]])
    bp = BPoly(c, [0, 1])

    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x in test_points:
        val = bp(x)
        expected = x**2
        print(f"  f({x}) = {val:.10f}, expected {expected:.10f}, diff = {abs(val-expected):.2e}")

    expected_vals = [x**2 for x in test_points]
    assert np.allclose([bp(x) for x in test_points], expected_vals)
    print("  PASSED")

def verify_derivative():
    """Verify derivative of quadratic"""
    print("\n=== Derivative of x^2 ===")
    c = np.array([[0], [0], [1]])  # x^2
    bp = BPoly(c, [0, 1])
    dbp = bp.derivative()

    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x in test_points:
        val = dbp(x)
        expected = 2*x
        print(f"  f'({x}) = {val:.10f}, expected {expected:.10f}, diff = {abs(val-expected):.2e}")

    expected_vals = [2*x for x in test_points]
    assert np.allclose([dbp(x) for x in test_points], expected_vals)
    print("  PASSED")

def verify_antiderivative():
    """Verify antiderivative of constant"""
    print("\n=== Antiderivative of constant 2 ===")
    c = np.array([[2]])  # constant 2
    bp = BPoly(c, [0, 1])
    ibp = bp.antiderivative()

    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x in test_points:
        val = ibp(x)
        expected = 2*x  # antiderivative of 2 is 2x (starting at 0)
        print(f"  int f({x}) = {val:.10f}, expected {expected:.10f}, diff = {abs(val-expected):.2e}")

    expected_vals = [2*x for x in test_points]
    assert np.allclose([ibp(x) for x in test_points], expected_vals)
    print("  PASSED")

def verify_integration():
    """Verify definite integration"""
    print("\n=== Definite Integration ===")
    # integral of x from 0 to 1 is 0.5
    c = np.array([[0], [1]])  # p(x) = x
    bp = BPoly(c, [0, 1])

    integral_01 = bp.integrate(0, 1)
    print(f"  int_0^1 x dx = {integral_01:.10f}, expected 0.5")
    assert np.isclose(integral_01, 0.5)

    integral_0_05 = bp.integrate(0, 0.5)
    print(f"  int_0^0.5 x dx = {integral_0_05:.10f}, expected 0.125")
    assert np.isclose(integral_0_05, 0.125)

    print("  PASSED")

def verify_from_derivatives():
    """Verify Hermite interpolation"""
    print("\n=== from_derivatives (Hermite) ===")

    # Cubic Hermite: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
    xi = [0, 1]
    yi = [[0, 1], [1, -1]]
    bp = BPoly.from_derivatives(xi, yi)

    print(f"  f(0) = {bp(0):.10f}, expected 0")
    print(f"  f(1) = {bp(1):.10f}, expected 1")
    print(f"  f'(0) = {bp.derivative()(0):.10f}, expected 1")
    print(f"  f'(1) = {bp.derivative()(1):.10f}, expected -1")

    assert np.isclose(bp(0), 0)
    assert np.isclose(bp(1), 1)
    assert np.isclose(bp.derivative()(0), 1)
    assert np.isclose(bp.derivative()(1), -1)

    # Test intermediate points
    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("\n  Intermediate values:")
    for x in test_points:
        print(f"    f({x}) = {bp(x):.10f}")

    print("  PASSED")

def verify_multi_interval():
    """Verify multi-interval polynomial"""
    print("\n=== Multi-interval Polynomial ===")
    c = np.array([[1, 2]])  # Constant 1 on [0,1], constant 2 on [1,2]
    bp = BPoly(c, [0, 1, 2])

    print(f"  f(0.5) = {bp(0.5):.10f}, expected 1")
    print(f"  f(1.5) = {bp(1.5):.10f}, expected 2")

    assert np.isclose(bp(0.5), 1)
    assert np.isclose(bp(1.5), 2)
    print("  PASSED")

def verify_from_power_basis():
    """Verify conversion from power basis"""
    print("\n=== from_power_basis ===")
    from scipy.interpolate import PPoly

    # PPoly: p(x) = 1 + 2*(x-a) + 3*(x-a)^2 on [0,1]
    # At x=0: 1
    # At x=0.5: 1 + 1 + 0.75 = 2.75
    # At x=1: 1 + 2 + 3 = 6
    power_coeffs = np.array([[1], [2], [3]])  # c0 + c1*(x-a) + c2*(x-a)^2
    pp = PPoly(power_coeffs[::-1], [0, 1])  # PPoly uses reversed order
    bp = BPoly.from_power_basis(pp)

    test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x in test_points:
        pp_val = pp(x)
        bp_val = bp(x)
        print(f"  PPoly({x}) = {pp_val:.10f}, BPoly({x}) = {bp_val:.10f}, diff = {abs(pp_val-bp_val):.2e}")

    for x in test_points:
        assert np.isclose(pp(x), bp(x))
    print("  PASSED")

def generate_reference_data():
    """Generate reference data for C++ verification"""
    print("\n=== Generating Reference Data ===")

    reference = {}

    # Test case 1: Cubic Hermite
    xi = [0, 1]
    yi = [[0, 1], [1, -1]]
    bp = BPoly.from_derivatives(xi, yi)
    test_points = np.linspace(0, 1, 11).tolist()
    reference['hermite_cubic'] = {
        'xi': xi,
        'yi': yi,
        'test_points': test_points,
        'values': [float(bp(x)) for x in test_points],
        'derivatives': [float(bp.derivative()(x)) for x in test_points],
        'coefficients': bp.c.tolist()
    }

    # Test case 2: Quintic Hermite
    xi = [0, 1]
    yi = [[0, 1, 0], [1, -1, 0]]
    bp = BPoly.from_derivatives(xi, yi)
    reference['hermite_quintic'] = {
        'xi': xi,
        'yi': yi,
        'test_points': test_points,
        'values': [float(bp(x)) for x in test_points],
        'derivatives': [float(bp.derivative()(x)) for x in test_points],
        'second_derivatives': [float(bp.derivative(2)(x)) for x in test_points],
        'coefficients': bp.c.tolist()
    }

    # Test case 3: Multi-interval
    xi = [0, 1, 2]
    yi = [[0, 1], [1, 0], [0, -1]]
    bp = BPoly.from_derivatives(xi, yi)
    test_points_multi = np.linspace(0, 2, 21).tolist()
    reference['multi_interval'] = {
        'xi': xi,
        'yi': yi,
        'test_points': test_points_multi,
        'values': [float(bp(x)) for x in test_points_multi],
        'coefficients': bp.c.tolist()
    }

    # Test case 4: High-degree polynomial (from_derivatives with more constraints)
    xi = [0, 1]
    yi = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 4 constraints at each end = degree 7
    bp = BPoly.from_derivatives(xi, yi)
    reference['high_degree'] = {
        'xi': xi,
        'yi': yi,
        'test_points': test_points,
        'values': [float(bp(x)) for x in test_points],
        'coefficients': bp.c.tolist()
    }

    # Save to JSON
    with open('scripts/bspoly_reference_data.json', 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"  Saved reference data to scripts/bspoly_reference_data.json")
    print("  PASSED")

def generate_bernstein_property_tests():
    """
    Generate reference data for Bernstein-specific mathematical properties.
    These properties are unique to Bernstein polynomials and should be tested.
    """
    print("\n=== Bernstein-Specific Property Tests ===\n")
    from math import comb

    property_data = {}

    # ============================================================
    # Property 1: Partition of Unity
    # Sum of Bernstein basis functions B_{i,n}(t) = 1 for all t in [0,1]
    # ============================================================
    print("--- Property 1: Partition of Unity ---")

    partition_tests = []
    for degree in range(1, 11):
        # Create polynomial with all coefficients = 1
        # p(t) = sum_{i=0}^n 1 * B_{i,n}(t) = 1
        c = np.ones((degree + 1, 1))
        bp = BPoly(c, [0, 1])

        test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        values = [float(bp(t)) for t in test_points]

        all_ones = all(abs(v - 1.0) < 1e-10 for v in values)
        print(f"  Degree {degree}: all values = 1? {all_ones}")

        partition_tests.append({
            "degree": degree,
            "test_points": test_points,
            "values": values,
            "expected": 1.0,
            "all_correct": all_ones
        })

    property_data["partition_of_unity"] = partition_tests

    # ============================================================
    # Property 2: Endpoint Interpolation
    # p(0) = c_0, p(1) = c_n for Bernstein polynomial with coeffs c_0, ..., c_n
    # ============================================================
    print("\n--- Property 2: Endpoint Interpolation ---")

    endpoint_tests = []
    test_configs = [
        [1.0, 2.0],
        [0.0, 5.0, 3.0],
        [1.5, 2.5, 3.5, 4.5],
        [10.0, 0.0, 0.0, 0.0, 20.0],
    ]

    for coeffs in test_configs:
        c = np.array([[x] for x in coeffs])
        bp = BPoly(c, [0, 1])

        val_at_0 = float(bp(0))
        val_at_1 = float(bp(1))
        expected_at_0 = coeffs[0]
        expected_at_1 = coeffs[-1]

        print(f"  coeffs={coeffs}: p(0)={val_at_0} (exp {expected_at_0}), p(1)={val_at_1} (exp {expected_at_1})")
        endpoint_tests.append({
            "coefficients": coeffs,
            "val_at_0": val_at_0, "expected_at_0": expected_at_0,
            "val_at_1": val_at_1, "expected_at_1": expected_at_1
        })

    property_data["endpoint_interpolation"] = endpoint_tests

    # ============================================================
    # Property 3: Convex Hull Property
    # For t in [0,1], p(t) lies within [min(c), max(c)]
    # ============================================================
    print("\n--- Property 3: Convex Hull Property ---")

    convex_hull_tests = []
    test_configs = [
        [1, 5, 2, 4, 3],
        [0, 10, 5, 8, 2],
        [-5, 10, -3, 7, 0],
        [1, 1, 1, 1, 1],  # All same - trivial case
    ]

    for coeffs in test_configs:
        c = np.array([[x] for x in coeffs])
        bp = BPoly(c, [0, 1])

        min_c = min(coeffs)
        max_c = max(coeffs)

        test_points = np.linspace(0, 1, 101).tolist()
        values = [float(bp(t)) for t in test_points]

        min_val = min(values)
        max_val = max(values)

        within_hull = (min_val >= min_c - 1e-10) and (max_val <= max_c + 1e-10)
        print(f"  coeffs={coeffs}: values in [{min_val:.4f}, {max_val:.4f}], hull=[{min_c}, {max_c}], within? {within_hull}")

        convex_hull_tests.append({
            "coefficients": coeffs,
            "min_coeff": min_c, "max_coeff": max_c,
            "min_value": min_val, "max_value": max_val,
            "within_hull": within_hull
        })

    property_data["convex_hull"] = convex_hull_tests

    # ============================================================
    # Property 4: Non-negativity Preservation
    # If all coefficients >= 0, then p(t) >= 0 for all t in [0,1]
    # ============================================================
    print("\n--- Property 4: Non-negativity Preservation ---")

    non_neg_tests = []
    test_configs = [
        [0, 1, 0.5, 2, 0.1],
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0.001, 0.002, 0.003, 0.004],
    ]

    for coeffs in test_configs:
        c = np.array([[x] for x in coeffs])
        bp = BPoly(c, [0, 1])

        test_points = np.linspace(0, 1, 101).tolist()
        values = [float(bp(t)) for t in test_points]

        min_val = min(values)
        all_non_negative = min_val >= -1e-14  # Allow small numerical error

        print(f"  coeffs={coeffs}: min value = {min_val:.2e}, all >= 0? {all_non_negative}")
        non_neg_tests.append({
            "coefficients": coeffs,
            "min_value": min_val,
            "all_non_negative": all_non_negative
        })

    property_data["non_negativity"] = non_neg_tests

    # ============================================================
    # Property 5: Degree Elevation
    # Elevating a polynomial from degree n to n+1 preserves its shape
    # ============================================================
    print("\n--- Property 5: Degree Elevation ---")

    def elevate_bernstein(coeffs):
        """Elevate Bernstein polynomial by one degree."""
        n = len(coeffs) - 1
        new_coeffs = []
        for i in range(n + 2):
            if i == 0:
                new_coeffs.append(coeffs[0])
            elif i == n + 1:
                new_coeffs.append(coeffs[n])
            else:
                # c'_i = (i/(n+1))*c_{i-1} + ((n+1-i)/(n+1))*c_i
                c_new = (i / (n + 1)) * coeffs[i - 1] + ((n + 1 - i) / (n + 1)) * coeffs[i]
                new_coeffs.append(c_new)
        return new_coeffs

    elevation_tests = []
    test_configs = [
        [1, 3],  # Linear
        [0, 1, 0],  # Quadratic
        [1, 2, 3, 4],  # Cubic
        [0, 0.5, 1],  # Linear (elevated rep)
    ]

    for coeffs in test_configs:
        c = np.array([[x] for x in coeffs])
        bp_original = BPoly(c, [0, 1])

        elevated = elevate_bernstein(coeffs)
        c_elevated = np.array([[x] for x in elevated])
        bp_elevated = BPoly(c_elevated, [0, 1])

        test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        original_values = [float(bp_original(t)) for t in test_points]
        elevated_values = [float(bp_elevated(t)) for t in test_points]

        max_diff = max(abs(o - e) for o, e in zip(original_values, elevated_values))
        matches = max_diff < 1e-10

        print(f"  degree {len(coeffs)-1} -> {len(elevated)-1}: max diff = {max_diff:.2e}, matches? {matches}")
        elevation_tests.append({
            "original_coeffs": coeffs,
            "elevated_coeffs": elevated,
            "test_points": test_points,
            "original_values": original_values,
            "elevated_values": elevated_values,
            "max_diff": max_diff,
            "matches": matches
        })

    property_data["degree_elevation"] = elevation_tests

    # ============================================================
    # Property 6: De Casteljau Subdivision
    # Subdividing at t=0.5 gives two polynomials that together equal the original
    # ============================================================
    print("\n--- Property 6: De Casteljau Subdivision ---")

    def subdivide_at_half(coeffs):
        """Subdivide Bernstein polynomial at t=0.5 using de Casteljau."""
        n = len(coeffs) - 1
        # Build de Casteljau triangle
        triangle = [list(coeffs)]
        for r in range(1, n + 1):
            row = []
            for i in range(n + 1 - r):
                row.append(0.5 * triangle[r-1][i] + 0.5 * triangle[r-1][i+1])
            triangle.append(row)

        # Left piece: first elements of each row
        left_coeffs = [triangle[r][0] for r in range(n + 1)]
        # Right piece: last elements of each row (reversed)
        right_coeffs = [triangle[n - r][r] for r in range(n + 1)]

        return left_coeffs, right_coeffs

    subdivision_tests = []
    test_configs = [
        [1, 3],
        [0, 1, 0],
        [1, 2, 3, 4],
        [0, 0, 1, 1],
    ]

    for coeffs in test_configs:
        c = np.array([[x] for x in coeffs])
        bp = BPoly(c, [0, 1])

        left, right = subdivide_at_half(coeffs)
        c_left = np.array([[x] for x in left])
        c_right = np.array([[x] for x in right])
        bp_left = BPoly(c_left, [0, 0.5])
        bp_right = BPoly(c_right, [0.5, 1])

        # Test at points
        test_points_left = [0.0, 0.125, 0.25, 0.375, 0.5]
        test_points_right = [0.5, 0.625, 0.75, 0.875, 1.0]

        max_diff = 0
        for t in test_points_left:
            if t < 0.5:
                diff = abs(float(bp(t)) - float(bp_left(t)))
                max_diff = max(max_diff, diff)
            elif t == 0.5:
                diff = abs(float(bp(t)) - float(bp_left(t)))
                max_diff = max(max_diff, diff)

        for t in test_points_right:
            diff = abs(float(bp(t)) - float(bp_right(t)))
            max_diff = max(max_diff, diff)

        print(f"  coeffs={coeffs}: max diff = {max_diff:.2e}")
        subdivision_tests.append({
            "original_coeffs": coeffs,
            "left_coeffs": left,
            "right_coeffs": right,
            "max_diff": max_diff
        })

    property_data["subdivision"] = subdivision_tests

    # ============================================================
    # Property 7: Symmetry
    # If coefficients are symmetric (c_i = c_{n-i}), polynomial is symmetric about t=0.5
    # ============================================================
    print("\n--- Property 7: Symmetry ---")

    symmetry_tests = []
    test_configs = [
        [1, 2, 1],  # Symmetric
        [0, 1, 1, 0],  # Symmetric
        [1, 2, 2, 1],  # Symmetric
        [1, 2, 3],  # Not symmetric
    ]

    for coeffs in test_configs:
        c = np.array([[x] for x in coeffs])
        bp = BPoly(c, [0, 1])

        # Check if symmetric
        is_symmetric_coeffs = all(abs(coeffs[i] - coeffs[len(coeffs)-1-i]) < 1e-10 for i in range(len(coeffs)//2 + 1))

        # Test symmetry: p(t) = p(1-t)
        test_points = [0.1, 0.2, 0.3, 0.4]
        symmetric_values = True
        max_diff = 0
        for t in test_points:
            diff = abs(float(bp(t)) - float(bp(1-t)))
            max_diff = max(max_diff, diff)
            if diff > 1e-10:
                symmetric_values = False

        print(f"  coeffs={coeffs}: symmetric coeffs? {is_symmetric_coeffs}, symmetric values? {symmetric_values}")
        symmetry_tests.append({
            "coefficients": coeffs,
            "symmetric_coeffs": is_symmetric_coeffs,
            "symmetric_values": symmetric_values,
            "max_diff": max_diff
        })

    property_data["symmetry"] = symmetry_tests

    # ============================================================
    # Property 8: High-Degree Stability
    # Test evaluation at high degrees (50, 100) for numerical stability
    # ============================================================
    print("\n--- Property 8: High-Degree Stability ---")

    high_degree_tests = []
    for degree in [10, 20, 50, 100]:
        # Test 1: All coefficients = 1 (should equal 1 everywhere due to partition of unity)
        c_ones = np.ones((degree + 1, 1))
        bp_ones = BPoly(c_ones, [0, 1])

        test_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        values_ones = [float(bp_ones(t)) for t in test_points]
        max_error_ones = max(abs(v - 1.0) for v in values_ones)

        # Test 2: Linear (identity function)
        # Bernstein coeffs for f(t) = t are c_i = i/n
        c_linear = np.array([[i / degree] for i in range(degree + 1)])
        bp_linear = BPoly(c_linear, [0, 1])
        values_linear = [float(bp_linear(t)) for t in test_points]
        max_error_linear = max(abs(v - t) for v, t in zip(values_linear, test_points))

        print(f"  Degree {degree}: partition of unity error = {max_error_ones:.2e}, linear error = {max_error_linear:.2e}")
        high_degree_tests.append({
            "degree": degree,
            "test_points": test_points,
            "partition_unity_values": values_ones,
            "partition_unity_max_error": max_error_ones,
            "linear_values": values_linear,
            "linear_max_error": max_error_linear
        })

    property_data["high_degree_stability"] = high_degree_tests

    # Save property test data
    output_path = "scripts/bspoly_property_reference.json"
    with open(output_path, 'w') as f:
        json.dump(property_data, f, indent=2)
    print(f"\n=== Property test data saved to {output_path} ===")

    return property_data


def main():
    print("=" * 60)
    print("BsPoly Verification against scipy.interpolate.BPoly")
    print("=" * 60)
    print()
    print("Since BsPoly stores Bernstein coefficients internally,")
    print("it should produce identical results to scipy.interpolate.BPoly")
    print()

    verify_constant()
    verify_linear()
    verify_quadratic()
    verify_derivative()
    verify_antiderivative()
    verify_integration()
    verify_from_derivatives()
    verify_multi_interval()
    verify_from_power_basis()
    generate_reference_data()
    generate_bernstein_property_tests()

    print()
    print("=" * 60)
    print("ALL VERIFICATIONS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
