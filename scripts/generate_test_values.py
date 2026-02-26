#!/usr/bin/env python3
"""
Generate expected values for C++ tests and perform live verification.

This script:
1. Generates expected values for Test 137 (multi-shape evaluation)
2. Verifies property-based mathematical identities
3. Can be used for live scipy verification of C++ results

Run: python scripts/generate_test_values.py
"""

import numpy as np
from scipy.interpolate import BPoly
import json
import sys


def generate_test137_values():
    """Generate expected values for Test 137 (multi-shape evaluation)."""
    print("=" * 70)
    print("Test 137: Multi-shape evaluation expected values")
    print("=" * 70)

    # Matches manual_test.cpp Test 137
    c = np.array([[3, 0], [0, 0], [0, 2]])
    x = np.array([0, 1, 3])
    bp = BPoly(c, x)

    xx = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = [float(bp(pt)) for pt in xx]

    print(f"Coefficients: {c.tolist()}")
    print(f"Breakpoints: {x.tolist()}")
    print(f"Eval points: {xx}")
    print(f"Expected results: {results}")
    print()
    print("C++ code:")
    print(f"    std::vector<double> expected_vals = {{{', '.join(f'{v:.10f}' for v in results)}}};")

    return xx, results


def verify_property_integrate_antiderivative():
    """
    Property: bp.integrate(a, b) == bp.antiderivative()(b) - bp.antiderivative()(a)
    """
    print("\n" + "=" * 70)
    print("Property: integrate(a,b) == antiderivative(b) - antiderivative(a)")
    print("=" * 70)

    test_cases = [
        # (name, coeffs, breaks)
        ("linear", [[0], [1]], [0, 1]),
        ("quadratic", [[0], [0], [1]], [0, 1]),
        ("cubic", [[1], [2], [3], [4]], [0, 1]),
        ("multi_interval", [[1, 2], [3, 4]], [0, 1, 2]),
        ("hermite", None, None),  # Special case: use from_derivatives
    ]

    all_pass = True
    for name, coeffs, breaks in test_cases:
        if name == "hermite":
            bp = BPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
        else:
            bp = BPoly(np.array(coeffs), np.array(breaks))

        anti = bp.antiderivative()

        # Test multiple intervals
        test_intervals = [(0.0, 0.5), (0.25, 0.75), (0.0, 1.0)]
        if name == "multi_interval":
            test_intervals = [(0.0, 0.5), (0.5, 1.5), (0.0, 2.0)]

        for a, b in test_intervals:
            via_integrate = bp.integrate(a, b)
            via_antideriv = anti(b) - anti(a)
            error = abs(via_integrate - via_antideriv)

            if error > 1e-10:
                print(f"  FAIL {name} [{a}, {b}]: integrate={via_integrate}, antideriv_diff={via_antideriv}, error={error:.2e}")
                all_pass = False
            else:
                print(f"  PASS {name} [{a}, {b}]: error={error:.2e}")

    return all_pass


def verify_property_derivative_antiderivative_inverse():
    """
    Property: derivative(antiderivative(f)) == f
    """
    print("\n" + "=" * 70)
    print("Property: derivative(antiderivative(f)) == f")
    print("=" * 70)

    test_cases = [
        ("constant", [[5]], [0, 1]),
        ("linear", [[1], [3]], [0, 1]),
        ("quadratic", [[1], [2], [3]], [0, 1]),
        ("multi", [[1, 2, 3], [4, 5, 6]], [0, 1, 2, 3]),
    ]

    all_pass = True
    for name, coeffs, breaks in test_cases:
        bp = BPoly(np.array(coeffs), np.array(breaks))
        recovered = bp.antiderivative().derivative()

        test_points = np.linspace(breaks[0], breaks[-1], 11)
        max_error = max(abs(bp(x) - recovered(x)) for x in test_points)

        if max_error > 1e-10:
            print(f"  FAIL {name}: max_error={max_error:.2e}")
            all_pass = False
        else:
            print(f"  PASS {name}: max_error={max_error:.2e}")

    return all_pass


def verify_property_bernstein_partition_of_unity():
    """
    Property: Sum of Bernstein basis functions == 1
    (All-ones coefficients should evaluate to 1 everywhere)
    """
    print("\n" + "=" * 70)
    print("Property: Bernstein partition of unity (all-1 coeffs -> 1)")
    print("=" * 70)

    all_pass = True
    for degree in [1, 2, 5, 10, 20, 50]:
        coeffs = np.ones((degree + 1, 1))
        bp = BPoly(coeffs, [0, 1])

        test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        max_error = max(abs(bp(x) - 1.0) for x in test_points)

        if max_error > 1e-10:
            print(f"  FAIL degree {degree}: max_error={max_error:.2e}")
            all_pass = False
        else:
            print(f"  PASS degree {degree}: max_error={max_error:.2e}")

    return all_pass


def verify_property_degree_elevation():
    """
    Property: Degree elevation preserves polynomial values
    Linear t elevated to degree n with c[i] = i/n should still equal t
    """
    print("\n" + "=" * 70)
    print("Property: Degree elevation preserves polynomial values")
    print("=" * 70)

    all_pass = True
    for target_degree in [2, 5, 10, 20, 50]:
        # Coefficients for linear t elevated to degree n: c[i] = i/n
        coeffs = np.array([[i / target_degree] for i in range(target_degree + 1)])
        bp = BPoly(coeffs, [0, 1])

        test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        max_error = max(abs(bp(x) - x) for x in test_points)

        if max_error > 1e-10:
            print(f"  FAIL elevated to degree {target_degree}: max_error={max_error:.2e}")
            all_pass = False
        else:
            print(f"  PASS elevated to degree {target_degree}: max_error={max_error:.2e}")

    return all_pass


def verify_property_from_derivatives_endpoints():
    """
    Property: from_derivatives matches specified values at endpoints
    """
    print("\n" + "=" * 70)
    print("Property: from_derivatives matches endpoint constraints")
    print("=" * 70)

    test_cases = [
        # (name, xi, yi)
        ("linear", [0, 1], [[0], [1]]),
        ("cubic_hermite", [0, 1], [[0, 1], [1, -1]]),
        ("quintic", [0, 1], [[0, 1, 0], [1, -1, 0]]),
        ("asymmetric", [0, 1], [[0, 1, 2], [1]]),
        ("three_point", [0, 1, 2], [[0, 1], [1], [0, -1]]),
    ]

    all_pass = True
    for name, xi, yi in test_cases:
        bp = BPoly.from_derivatives(xi, yi)

        # Check each breakpoint
        for i, (x, y_derivs) in enumerate(zip(xi, yi)):
            for order, expected_val in enumerate(y_derivs):
                # Evaluate order-th derivative at x
                if order == 0:
                    actual = bp(x)
                else:
                    actual = bp.derivative(order)(x)

                error = abs(actual - expected_val)
                if error > 1e-8:  # Looser tolerance for higher derivatives
                    print(f"  FAIL {name} at x={x}, order={order}: expected={expected_val}, got={actual}, error={error:.2e}")
                    all_pass = False

        if all_pass:
            print(f"  PASS {name}")

    return all_pass


def generate_property_test_data():
    """Generate test data for C++ property-based tests."""
    print("\n" + "=" * 70)
    print("Generating property test data for C++")
    print("=" * 70)

    data = {
        "integrate_antiderivative": [],
        "derivative_antiderivative": [],
    }

    # Test case: Hermite cubic
    bp = BPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
    anti = bp.antiderivative()

    intervals = [(0.0, 0.5), (0.25, 0.75), (0.0, 1.0)]
    for a, b in intervals:
        data["integrate_antiderivative"].append({
            "a": a,
            "b": b,
            "integrate_result": float(bp.integrate(a, b)),
            "antideriv_a": float(anti(a)),
            "antideriv_b": float(anti(b)),
        })

    print(f"Generated {len(data['integrate_antiderivative'])} integrate/antiderivative test points")

    return data


def main():
    print("BPoly Test Value Generator and Property Verifier")
    print("=" * 70)
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {__import__('scipy').__version__}")
    print()

    all_pass = True

    # Generate Test 137 values
    generate_test137_values()

    # Run property verifications
    all_pass &= verify_property_integrate_antiderivative()
    all_pass &= verify_property_derivative_antiderivative_inverse()
    all_pass &= verify_property_bernstein_partition_of_unity()
    all_pass &= verify_property_degree_elevation()
    all_pass &= verify_property_from_derivatives_endpoints()

    # Generate additional test data
    prop_data = generate_property_test_data()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"All property tests passed: {all_pass}")

    if not all_pass:
        print("\nWARNING: Some property tests failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
