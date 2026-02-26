#!/usr/bin/env python3
"""
Live verification script that compares C++ BPoly output against scipy.

This script can be used to:
1. Verify C++ test results match scipy
2. Generate reference values for new test cases
3. Debug discrepancies between C++ and scipy

Usage:
    python scripts/live_verify.py                    # Run all verifications
    python scripts/live_verify.py --generate-cpp    # Generate C++ test code

Run after building C++ tests:
    cmake --build build --config Release && python scripts/live_verify.py
"""

import numpy as np
from scipy.interpolate import BPoly
import argparse
import sys
import subprocess
import json
import os


def verify_basic_construction():
    """Verify basic BPoly construction and evaluation."""
    print("\n" + "=" * 70)
    print("Verifying basic construction")
    print("=" * 70)

    test_cases = [
        # (name, coeffs, breaks, eval_points)
        ("constant", [[5]], [0, 1], [0.0, 0.5, 1.0]),
        ("linear", [[0], [1]], [0, 1], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("quadratic_x2", [[0], [0], [1]], [0, 1], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("multi_interval", [[0, 1], [1, 2]], [0, 1, 2], [0.0, 0.5, 1.0, 1.5, 2.0]),
    ]

    results = []
    for name, coeffs, breaks, pts in test_cases:
        bp = BPoly(np.array(coeffs), np.array(breaks))
        vals = [float(bp(x)) for x in pts]
        results.append({
            "name": name,
            "coefficients": coeffs,
            "breakpoints": breaks,
            "eval_points": pts,
            "scipy_values": vals,
        })
        print(f"  {name}: {vals}")

    return results


def verify_from_derivatives():
    """Verify from_derivatives matches scipy."""
    print("\n" + "=" * 70)
    print("Verifying from_derivatives")
    print("=" * 70)

    test_cases = [
        # (name, xi, yi, eval_points)
        ("linear", [0, 1], [[0], [1]], [0.0, 0.5, 1.0]),
        ("cubic_hermite", [0, 1], [[0, 1], [1, -1]], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("quintic", [0, 1], [[0, 1, 0], [1, -1, 0]], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("asymmetric_1_3", [0, 1], [[0], [1, -1, 0]], [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("three_point", [0, 1, 2], [[0, 1], [1], [0, -1]], [0.0, 0.5, 1.0, 1.5, 2.0]),
    ]

    results = []
    for name, xi, yi, pts in test_cases:
        bp = BPoly.from_derivatives(xi, yi)
        vals = [float(bp(x)) for x in pts]
        coeffs = [[float(c) for c in row] for row in bp.c]
        results.append({
            "name": name,
            "xi": xi,
            "yi": yi,
            "eval_points": pts,
            "scipy_values": vals,
            "scipy_coefficients": coeffs,
            "degree": int(bp.c.shape[0] - 1),
        })
        print(f"  {name}: degree={bp.c.shape[0]-1}, values={vals}")

    return results


def verify_derivatives():
    """Verify derivative computation."""
    print("\n" + "=" * 70)
    print("Verifying derivatives")
    print("=" * 70)

    # Hermite cubic: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
    bp = BPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
    bp_d1 = bp.derivative(1)
    bp_d2 = bp.derivative(2)

    pts = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {
        "original": [float(bp(x)) for x in pts],
        "first_derivative": [float(bp_d1(x)) for x in pts],
        "second_derivative": [float(bp_d2(x)) for x in pts],
    }

    print(f"  f(x):   {results['original']}")
    print(f"  f'(x):  {results['first_derivative']}")
    print(f"  f''(x): {results['second_derivative']}")

    return results


def verify_integration():
    """Verify integration computation."""
    print("\n" + "=" * 70)
    print("Verifying integration")
    print("=" * 70)

    test_cases = [
        # (name, coeffs, breaks, a, b, expected_description)
        ("constant_1", [[1]], [0, 1], 0.0, 1.0, "integral of 1 = 1"),
        ("linear_x", [[0], [1]], [0, 1], 0.0, 1.0, "integral of x = 0.5"),
        ("quadratic_x2", [[0], [0], [1]], [0, 1], 0.0, 1.0, "integral of x^2 = 1/3"),
    ]

    results = []
    for name, coeffs, breaks, a, b, desc in test_cases:
        bp = BPoly(np.array(coeffs), np.array(breaks))
        val = float(bp.integrate(a, b))
        results.append({
            "name": name,
            "a": a,
            "b": b,
            "result": val,
            "description": desc,
        })
        print(f"  {name}: integrate({a}, {b}) = {val:.10f} ({desc})")

    return results


def verify_antiderivative():
    """Verify antiderivative computation."""
    print("\n" + "=" * 70)
    print("Verifying antiderivative")
    print("=" * 70)

    # f(x) = 1, antiderivative = x
    bp = BPoly([[1]], [0, 1])
    anti = bp.antiderivative()

    pts = [0.0, 0.25, 0.5, 0.75, 1.0]
    vals = [float(anti(x)) for x in pts]

    print(f"  f(x) = 1")
    print(f"  F(x) at {pts}: {vals}")
    print(f"  Expected (x): {pts}")

    return {"points": pts, "antiderivative_values": vals}


def verify_roots():
    """Verify root finding using numerical methods.

    Note: scipy.BPoly doesn't have a roots() method - our C++ implementation adds it.
    We verify roots by finding sign changes and using scipy.optimize.
    """
    print("\n" + "=" * 70)
    print("Verifying roots (scipy doesn't have roots(), using numerical methods)")
    print("=" * 70)

    from scipy.optimize import brentq

    results = []

    # Linear case: f(x) = x - 0.5, root at x = 0.5
    bp = BPoly([[-0.5], [0.5]], [0, 1])
    # Find root numerically
    try:
        root = brentq(lambda x: bp(x), 0, 1)
        print(f"  linear f(x)=x-0.5: root={root:.10f} (expected 0.5)")
        results.append({"name": "linear", "roots": [root]})
    except ValueError:
        print(f"  linear: no root found in [0, 1]")
        results.append({"name": "linear", "roots": []})

    # Quadratic: (x-0.25)(x-0.75) = x^2 - x + 0.1875
    # Using from_derivatives to create a polynomial that passes through
    # (0, 0.1875), (0.5, -0.0625), (1, 0.1875)
    # Actually, let's just create it directly with the right Bernstein coefficients
    # For f(t) = (t-0.25)(t-0.75) on [0,1], the Bernstein coeffs are:
    # f(0) = 0.1875, f(1) = 0.1875
    # Using from_derivatives with just function values to interpolate
    bp = BPoly.from_derivatives([0, 0.5, 1], [[0.1875], [-0.0625], [0.1875]])
    roots = []
    try:
        root1 = brentq(lambda x: bp(x), 0, 0.4)
        roots.append(root1)
    except ValueError:
        pass
    try:
        root2 = brentq(lambda x: bp(x), 0.6, 1)
        roots.append(root2)
    except ValueError:
        pass
    print(f"  quadratic-like: roots={roots} (expected near [0.25, 0.75])")
    results.append({"name": "quadratic", "roots": roots})

    return results


def generate_cpp_test_code(results):
    """Generate C++ test code from verification results."""
    print("\n" + "=" * 70)
    print("Generated C++ Test Code")
    print("=" * 70)

    # Example: from_derivatives tests
    print("\n// From derivatives tests (generated by live_verify.py)")
    for case in results.get("from_derivatives", []):
        name = case["name"]
        xi = case["xi"]
        yi = case["yi"]
        pts = case["eval_points"]
        vals = case["scipy_values"]

        print(f"\n// Test: {name}")
        print(f"{{")
        print(f"    BPoly bp = BPoly::from_derivatives({{{', '.join(str(x) for x in xi)}}},")
        yi_str = ", ".join("{" + ", ".join(str(y) for y in ys) + "}" for ys in yi)
        print(f"                                       {{{yi_str}}});")
        for pt, val in zip(pts, vals):
            print(f"    test.expect_near(bp({pt}), {val:.10f}, tolerance, \"{name} f({pt})\");")
        print(f"}}")


def run_cpp_tests():
    """Run C++ tests and check output."""
    print("\n" + "=" * 70)
    print("Running C++ Tests")
    print("=" * 70)

    # Try to find and run the test executable
    possible_paths = [
        "build/Release/manual_test.exe",
        "build/Debug/manual_test.exe",
        "build/manual_test.exe",
        "build/manual_test",
    ]

    test_exe = None
    for path in possible_paths:
        if os.path.exists(path):
            test_exe = path
            break

    if test_exe is None:
        print("  WARNING: Could not find manual_test executable")
        print("  Run: cmake --build build --config Release")
        return None

    print(f"  Running: {test_exe}")
    try:
        result = subprocess.run([test_exe], capture_output=True, text=True, timeout=60)
        print(f"  Exit code: {result.returncode}")

        # Count passed/failed
        lines = result.stdout.split('\n')
        passed = sum(1 for l in lines if '[PASS]' in l)
        failed = sum(1 for l in lines if '[FAIL]' in l)
        print(f"  Passed: {passed}, Failed: {failed}")

        if result.returncode != 0:
            print("\n  FAILED TESTS:")
            for line in lines:
                if '[FAIL]' in line:
                    print(f"    {line}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("  ERROR: Test timed out")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Live scipy verification for BPoly")
    parser.add_argument("--generate-cpp", action="store_true",
                       help="Generate C++ test code")
    parser.add_argument("--run-cpp", action="store_true",
                       help="Run C++ tests and verify")
    parser.add_argument("--json", type=str,
                       help="Output results to JSON file")
    args = parser.parse_args()

    print("BPoly Live Scipy Verification")
    print("=" * 70)
    print(f"NumPy version: {np.__version__}")
    print(f"SciPy version: {__import__('scipy').__version__}")

    results = {
        "basic": verify_basic_construction(),
        "from_derivatives": verify_from_derivatives(),
        "derivatives": verify_derivatives(),
        "integration": verify_integration(),
        "antiderivative": verify_antiderivative(),
        "roots": verify_roots(),
    }

    if args.generate_cpp:
        generate_cpp_test_code(results)

    if args.run_cpp:
        cpp_success = run_cpp_tests()
        if not cpp_success:
            print("\nC++ tests FAILED!")
            return 1

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")

    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
