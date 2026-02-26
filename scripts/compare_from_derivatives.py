#!/usr/bin/env python3
"""
Compare scipy.interpolate.BPoly.from_derivatives with C++ implementation.
Identifies accuracy gaps in various derivative configurations.
"""

import numpy as np
from scipy.interpolate import BPoly
import json

def compare_case(name, xi, yi, eval_points=None):
    """Compare a single from_derivatives case."""
    try:
        bp = BPoly.from_derivatives(xi, yi)

        if eval_points is None:
            # Generate evaluation points across all intervals
            eval_points = []
            for i in range(len(xi) - 1):
                pts = np.linspace(xi[i], xi[i+1], 11)
                eval_points.extend(pts.tolist())
            eval_points = sorted(set(eval_points))

        results = [float(bp(x)) for x in eval_points]

        # Get derivative results too
        bp_d1 = bp.derivative(1)
        deriv_results = [float(bp_d1(x)) for x in eval_points]

        # Get coefficients
        coeffs = bp.c.tolist()

        return {
            'name': name,
            'xi': xi,
            'yi': yi,
            'eval_points': eval_points,
            'scipy_results': results,
            'scipy_derivative': deriv_results,
            'scipy_coefficients': coeffs,
            'degree': bp.c.shape[0] - 1,
            'success': True
        }
    except Exception as e:
        return {
            'name': name,
            'xi': xi,
            'yi': yi,
            'error': str(e),
            'success': False
        }

def main():
    print("=" * 70)
    print("SciPy BPoly.from_derivatives Comparison Data Generator")
    print("=" * 70)

    test_cases = []

    # Case 1: Simple linear (n0=1, n1=1)
    print("\n1. Linear interpolation (function values only)")
    result = compare_case(
        "linear_2pt",
        xi=[0, 1],
        yi=[[0], [1]]
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 2: Cubic Hermite (n0=2, n1=2) - the common case
    print("\n2. Cubic Hermite (function + first derivative)")
    result = compare_case(
        "hermite_cubic",
        xi=[0, 1],
        yi=[[0, 1], [1, -1]]  # f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 3: Quintic Hermite (n0=3, n1=3)
    print("\n3. Quintic Hermite (function + first + second derivative)")
    result = compare_case(
        "hermite_quintic",
        xi=[0, 1],
        yi=[[0, 1, 0], [1, -1, 0]]  # f, f', f'' at each endpoint
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 4: Asymmetric derivatives (n0=1, n1=2) - just function at left, function+derivative at right
    print("\n4. Asymmetric (n0=1, n1=2)")
    result = compare_case(
        "asymmetric_1_2",
        xi=[0, 1],
        yi=[[0], [1, 0]]  # f(0)=0; f(1)=1, f'(1)=0
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 5: Asymmetric (n0=2, n1=1)
    print("\n5. Asymmetric (n0=2, n1=1)")
    result = compare_case(
        "asymmetric_2_1",
        xi=[0, 1],
        yi=[[0, 1], [1]]  # f(0)=0, f'(0)=1; f(1)=1
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 6: Asymmetric (n0=3, n1=1)
    print("\n6. Asymmetric (n0=3, n1=1)")
    result = compare_case(
        "asymmetric_3_1",
        xi=[0, 1],
        yi=[[0, 1, 2], [1]]  # f(0)=0, f'(0)=1, f''(0)=2; f(1)=1
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 7: Asymmetric (n0=1, n1=3)
    print("\n7. Asymmetric (n0=1, n1=3)")
    result = compare_case(
        "asymmetric_1_3",
        xi=[0, 1],
        yi=[[0], [1, -1, 0]]  # f(0)=0; f(1)=1, f'(1)=-1, f''(1)=0
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 8: Three points with mixed derivatives
    print("\n8. Three points with mixed derivatives")
    result = compare_case(
        "three_points_mixed",
        xi=[0, 1, 2],
        yi=[[0, 1], [1], [0, -1]]  # Different derivative counts
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 9: Higher order - 4th derivative matching
    print("\n9. Higher order (4 derivatives at each end)")
    result = compare_case(
        "higher_order_4",
        xi=[0, 1],
        yi=[[0, 1, 0, 0], [1, -1, 0, 0]]
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 10: Non-unit interval
    print("\n10. Non-unit interval [0, 2]")
    result = compare_case(
        "non_unit_interval",
        xi=[0, 2],
        yi=[[0, 0.5], [4, 2]]  # f(0)=0, f'(0)=0.5; f(2)=4, f'(2)=2
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Case 11: Negative interval
    print("\n11. Negative interval [-1, 1]")
    result = compare_case(
        "negative_interval",
        xi=[-1, 1],
        yi=[[1, 0], [1, 0]]  # Parabola-like
    )
    test_cases.append(result)
    print(f"   Degree: {result.get('degree', 'N/A')}")
    print(f"   Coefficients: {result.get('scipy_coefficients', 'N/A')}")

    # Save results
    output_file = 'from_derivatives_comparison.json'
    with open(output_file, 'w') as f:
        json.dump({
            'description': 'SciPy from_derivatives reference data for C++ comparison',
            'cases': test_cases
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Saved {len(test_cases)} test cases to {output_file}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Derivative Configurations")
    print(f"{'=' * 70}")
    print(f"{'Case':<30} {'n0':<5} {'n1':<5} {'Degree':<8} {'Status'}")
    print("-" * 70)

    for case in test_cases:
        name = case['name']
        yi = case['yi']
        n0 = len(yi[0]) if yi else 0
        n1 = len(yi[-1]) if len(yi) > 1 else 0
        degree = case.get('degree', 'N/A')
        status = 'OK' if case['success'] else 'FAIL'
        print(f"{name:<30} {n0:<5} {n1:<5} {degree:<8} {status}")

    return test_cases

if __name__ == '__main__':
    main()
