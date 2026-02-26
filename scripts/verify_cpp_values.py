"""
Verify C++ hardcoded values against scipy at the same evaluation points.

This script generates scipy reference values for the 5-point evaluation grid
used in compare_from_derivatives.cpp and compares them against the hardcoded
values in that file.

Run: python scripts/verify_cpp_values.py
"""

import numpy as np
from scipy.interpolate import BPoly

# 5-point evaluation grid matching compare_from_derivatives.cpp
EVAL_PTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# All 11 test cases from compare_from_derivatives.cpp
TEST_CASES = [
    {
        "name": "linear_2pt",
        "xi": [0, 1],
        "yi": [[0], [1]],
        "expected_degree": 1,
        # Hardcoded in C++: {0.0, 0.25, 0.5, 0.75, 1.0}
        "cpp_values": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    {
        "name": "hermite_cubic",
        "xi": [0, 1],
        "yi": [[0, 1], [1, -1]],
        "expected_degree": 3,
        # Hardcoded in C++: {0.0, 0.34375, 0.75, 1.03125, 1.0}
        "cpp_values": [0.0, 0.34375, 0.75, 1.03125, 1.0],
    },
    {
        "name": "hermite_quintic",
        "xi": [0, 1],
        "yi": [[0, 1, 0], [1, -1, 0]],
        "expected_degree": 5,
        # Hardcoded in C++: {0.0, 0.326171875, 0.8125, 1.119140625, 1.0}
        "cpp_values": [0.0, 0.326171875, 0.8125, 1.119140625, 1.0],
    },
    {
        "name": "asymmetric_1_2",
        "xi": [0, 1],
        "yi": [[0], [1, 0]],
        "expected_degree": 2,
        # Hardcoded in C++: {0.0, 0.4375, 0.75, 0.9375, 1.0}
        "cpp_values": [0.0, 0.4375, 0.75, 0.9375, 1.0],
    },
    {
        "name": "asymmetric_2_1",
        "xi": [0, 1],
        "yi": [[0, 1], [1]],
        "expected_degree": 2,
        # Hardcoded in C++: {0.0, 0.25, 0.5, 0.75, 1.0}
        "cpp_values": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    {
        "name": "asymmetric_3_1",
        "xi": [0, 1],
        "yi": [[0, 1, 2], [1]],
        "expected_degree": 3,
        # Hardcoded in C++: {0.0, 0.296875, 0.625, 0.890625, 1.0}
        "cpp_values": [0.0, 0.296875, 0.625, 0.890625, 1.0],
    },
    {
        "name": "asymmetric_1_3",
        "xi": [0, 1],
        "yi": [[0], [1, -1, 0]],
        "expected_degree": 3,
        # Hardcoded in C++: {0.0, 0.90625, 1.25, 1.21875, 1.0}
        "cpp_values": [0.0, 0.90625, 1.25, 1.21875, 1.0],
    },
    {
        "name": "higher_order_4",
        "xi": [0, 1],
        "yi": [[0, 1, 0, 0], [1, -1, 0, 0]],
        "expected_degree": 7,
        # Hardcoded in C++: {0.0, 0.3063964844, 0.84375, 1.1652832031, 1.0}
        "cpp_values": [0.0, 0.3063964844, 0.84375, 1.1652832031, 1.0],
    },
    {
        "name": "non_unit_interval",
        "xi": [0, 2],
        "yi": [[0, 0.5], [4, 2]],
        "expected_degree": 3,
        "eval_pts": [0.0, 0.5, 1.0, 1.5, 2.0],  # Different grid for this case
        # Hardcoded in C++: {0.0, 0.578125, 1.625, 2.859375, 4.0}
        "cpp_values": [0.0, 0.578125, 1.625, 2.859375, 4.0],
    },
    {
        "name": "negative_interval",
        "xi": [-1, 1],
        "yi": [[1, 0], [1, 0]],
        "expected_degree": 3,
        "eval_pts": [-1.0, -0.5, 0.0, 0.5, 1.0],  # Different grid
        # Hardcoded in C++: {1.0, 1.0, 1.0, 1.0, 1.0}
        "cpp_values": [1.0, 1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "three_points_mixed",
        "xi": [0, 1, 2],
        "yi": [[0, 1], [1], [0, -1]],
        "expected_degree": 2,
        "eval_pts": [0.0, 0.5, 1.0, 1.5, 2.0],  # Different grid
        # Hardcoded in C++: {0.0, 0.5, 1.0, 0.5, 0.0}
        "cpp_values": [0.0, 0.5, 1.0, 0.5, 0.0],
    },
]


def verify_case(case):
    """Verify a single test case against scipy."""
    name = case["name"]
    xi = case["xi"]
    yi = case["yi"]
    cpp_values = case["cpp_values"]
    eval_pts = case.get("eval_pts", EVAL_PTS)
    expected_degree = case["expected_degree"]

    try:
        bp = BPoly.from_derivatives(xi, yi)
    except Exception as e:
        return {
            "name": name,
            "status": "ERROR",
            "message": f"Failed to create BPoly: {e}",
        }

    # Check degree
    actual_degree = bp.c.shape[0] - 1
    if actual_degree != expected_degree:
        print(f"  WARNING: degree mismatch: expected {expected_degree}, got {actual_degree}")

    # Evaluate at test points
    scipy_values = [float(bp(x)) for x in eval_pts]

    # Compare
    max_error = 0.0
    errors = []
    for i, (cpp_val, scipy_val, x) in enumerate(zip(cpp_values, scipy_values, eval_pts)):
        error = abs(cpp_val - scipy_val)
        max_error = max(max_error, error)
        if error > 1e-10:
            errors.append(f"  x={x}: cpp={cpp_val}, scipy={scipy_val:.15f}, error={error:.2e}")

    return {
        "name": name,
        "status": "PASS" if max_error < 1e-10 else "FAIL",
        "max_error": max_error,
        "scipy_values": scipy_values,
        "errors": errors,
    }


def print_cpp_code():
    """Print C++ code with correct scipy values for copy-paste."""
    print("\n" + "=" * 70)
    print("C++ CODE WITH CORRECT SCIPY VALUES")
    print("=" * 70 + "\n")

    for case in TEST_CASES:
        name = case["name"]
        xi = case["xi"]
        yi = case["yi"]
        eval_pts = case.get("eval_pts", EVAL_PTS)

        bp = BPoly.from_derivatives(xi, yi)
        scipy_values = [float(bp(x)) for x in eval_pts]
        coeffs = bp.c.flatten().tolist()

        print(f"// {name}")
        print(f"// xi: {xi}")
        print(f"// yi: {yi}")
        print(f"// scipy coefficients: {coeffs}")
        print(f"// eval_points: {list(eval_pts)}")
        print(f"// scipy_results:")
        print(f"  {{{', '.join(f'{v:.10f}' for v in scipy_values)}}},")
        print()


def main():
    print("=" * 70)
    print("BPoly C++ Hardcoded Values Verification")
    print("=" * 70)
    print(f"\nComparing C++ hardcoded values against scipy.interpolate.BPoly")
    print(f"Scipy version: {__import__('scipy').__version__}")
    print(f"NumPy version: {np.__version__}")
    print()

    total_pass = 0
    total_fail = 0

    for case in TEST_CASES:
        result = verify_case(case)
        status_symbol = "[PASS]" if result["status"] == "PASS" else "[FAIL]"
        print(f"{status_symbol} {result['name']}: max_error = {result.get('max_error', 'N/A'):.2e}")

        if result["status"] == "PASS":
            total_pass += 1
        else:
            total_fail += 1
            for err in result.get("errors", []):
                print(err)

    print()
    print("=" * 70)
    print(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL")
    print("=" * 70)

    if total_fail > 0:
        print("\nDiscrepancies found! Generating correct C++ values...")
        print_cpp_code()
        return 1
    else:
        print("\nAll hardcoded C++ values match scipy!")
        return 0


if __name__ == "__main__":
    exit(main())
