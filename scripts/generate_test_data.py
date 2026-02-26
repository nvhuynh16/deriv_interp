#!/usr/bin/env python3
"""
Comprehensive corner case verification for BPoly implementation.
Generates test data and explores scipy edge cases to ensure C++ compatibility.
"""

import numpy as np
from scipy.interpolate import BPoly
import json
import sys
import warnings
import traceback

def test_basic_edge_cases():
    """Test basic edge cases and boundary conditions"""
    print("=== BASIC EDGE CASES ===")
    
    cases = []
    
    # 1. Single interval, constant polynomial
    try:
        c = np.array([[5.0]])  # degree 0, constant 5 - shape (1, 1) for 1 interval
        x = [0, 1]
        bp = BPoly(c, x)
        eval_points = [0.0, 0.5, 1.0, -0.5, 1.5]  # Include extrapolation
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'single_interval_constant',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Constant polynomial on single interval'
        })
        print(f"OK Single interval constant: {results}")
    except Exception as e:
        print(f"FAIL Single interval constant: {e}")
    
    # 2. Single interval, high degree
    try:
        c = np.array([[1], [0], [0], [0], [0], [2]])  # degree 5: shape (6, 1)
        x = [0, 1]
        bp = BPoly(c, x)
        eval_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'single_interval_degree5',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'High degree polynomial on single interval'
        })
        print(f"OK Single interval degree 5: {results}")
    except Exception as e:
        print(f"FAIL Single interval degree 5: {e}")
    
    # 3. Two intervals, different degrees (scipy handles this)
    try:
        c = np.array([[0, 1], [1, 0], [0, 0]])  # Two intervals, degree 2, shape (3, 2)
        x = [0, 0.5, 1]
        bp = BPoly(c, x)
        eval_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'two_intervals_degree2',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Two intervals with degree 2'
        })
        print(f"OK Two intervals degree 2: {results}")
    except Exception as e:
        print(f"FAIL Two intervals degree 2: {e}")
    
    # 4. Many small intervals
    try:
        num_intervals = 10
        c = np.array([[i for i in range(num_intervals)], [i+1 for i in range(num_intervals)]])  # shape (2, 10)
        x = list(range(num_intervals + 1))  # [0, 1, 2, ..., 10]
        bp = BPoly(c, x)
        eval_points = [0.5, 5.5, 9.5]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'many_intervals',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Many intervals with linear pieces'
        })
        print(f"OK Many intervals: {results}")
    except Exception as e:
        print(f"FAIL Many intervals: {e}")
    
    return cases

def test_degenerate_cases():
    """Test degenerate and boundary cases that should fail"""
    print("\n=== DEGENERATE CASES ===")
    
    cases = []
    
    # 1. Identical breakpoints (should fail)
    try:
        c = np.array([[1, 2]])
        x = [0, 0]  # Identical breakpoints
        bp = BPoly(c, x)
        print("FAIL Identical breakpoints should have failed but didn't!")
        cases.append({
            'name': 'identical_breakpoints',
            'should_fail': False,  # Unexpectedly succeeded
            'description': 'Identical breakpoints should raise error'
        })
    except Exception as e:
        print(f"OK Identical breakpoints correctly failed: {type(e).__name__}")
        cases.append({
            'name': 'identical_breakpoints',
            'should_fail': True,
            'error_type': type(e).__name__,
            'description': 'Identical breakpoints should raise error'
        })
    
    # 2. Unsorted breakpoints (should fail)
    try:
        c = np.array([[1, 2]])
        x = [1, 0]  # Reversed order
        bp = BPoly(c, x)
        print("FAIL Unsorted breakpoints should have failed but didn't!")
    except Exception as e:
        print(f"OK Unsorted breakpoints correctly failed: {type(e).__name__}")
        cases.append({
            'name': 'unsorted_breakpoints',
            'should_fail': True,
            'error_type': type(e).__name__,
            'description': 'Unsorted breakpoints should raise error'
        })
    
    # 3. Empty coefficient arrays (should fail)
    try:
        c = np.array([])
        x = [0, 1]
        bp = BPoly(c, x)
        print("FAIL Empty coefficients should have failed but didn't!")
    except Exception as e:
        print(f"OK Empty coefficients correctly failed: {type(e).__name__}")
        cases.append({
            'name': 'empty_coefficients',
            'should_fail': True,
            'error_type': type(e).__name__,
            'description': 'Empty coefficients should raise error'
        })
    
    # 4. Wrong number of intervals
    try:
        c = np.array([[1, 2]])  # 1 interval worth of coefficients
        x = [0, 1, 2]  # 2 intervals
        bp = BPoly(c, x)
        print("FAIL Mismatched intervals should have failed but didn't!")
    except Exception as e:
        print(f"OK Mismatched intervals correctly failed: {type(e).__name__}")
        cases.append({
            'name': 'mismatched_intervals',
            'should_fail': True,
            'error_type': type(e).__name__,
            'description': 'Wrong number of intervals should raise error'
        })
    
    return cases

def test_from_derivatives_edge_cases():
    """Test from_derivatives with various edge cases"""
    print("\n=== FROM_DERIVATIVES EDGE CASES ===")
    
    cases = []
    
    # 1. Single point (should fail - need at least 2 points)
    try:
        xi = [0]
        yi = [[1]]
        bp = BPoly.from_derivatives(xi, yi)
        print("FAIL Single point should have failed but didn't!")
    except Exception as e:
        print(f"OK Single point correctly failed: {type(e).__name__}")
        cases.append({
            'name': 'from_derivatives_single_point',
            'should_fail': True,
            'error_type': type(e).__name__,
            'description': 'from_derivatives with single point should fail'
        })
    
    # 2. Mismatched dimensions
    try:
        xi = [0, 1]
        yi = [[1, 2]]  # Only one point but two breakpoints
        bp = BPoly.from_derivatives(xi, yi)
        print("FAIL Mismatched from_derivatives should have failed!")
    except Exception as e:
        print(f"OK Mismatched from_derivatives correctly failed: {type(e).__name__}")
        cases.append({
            'name': 'from_derivatives_mismatched',
            'should_fail': True,
            'error_type': type(e).__name__,
            'description': 'Mismatched xi/yi dimensions should fail'
        })
    
    # 3. Basic two point interpolation
    try:
        xi = [0, 1]
        yi = [[0], [1]]  # f(0)=0, f(1)=1 (linear)
        bp = BPoly.from_derivatives(xi, yi)
        eval_points = [0.0, 0.5, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'from_derivatives_linear',
            'xi': xi,
            'yi': yi,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Simple linear interpolation'
        })
        print(f"OK Linear from_derivatives: {results}")
    except Exception as e:
        print(f"FAIL Linear from_derivatives: {e}")
    
    # 4. Hermite cubic with first derivatives
    try:
        xi = [0, 1]
        yi = [[0, 1], [1, -1]]  # f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        bp = BPoly.from_derivatives(xi, yi)
        eval_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        # Test derivatives
        bp_d1 = bp.derivative()
        deriv_results = [float(bp_d1(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'hermite_cubic',
            'xi': xi,
            'yi': yi,
            'eval_points': eval_points,
            'expected_results': results,
            'derivative_1': deriv_results,
            'description': 'Hermite cubic interpolation'
        })
        print(f"OK Hermite cubic: f={results}")
        print(f"   derivatives: {deriv_results}")
    except Exception as e:
        print(f"FAIL Hermite cubic: {e}")
    
    # 5. Three points with mixed derivative orders
    try:
        xi = [0, 0.5, 1]
        yi = [[0], [0.25, 1], [1]]  # Different derivative orders
        bp = BPoly.from_derivatives(xi, yi)
        eval_points = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'mixed_derivative_orders',
            'xi': xi,
            'yi': yi,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Three points with mixed derivative data'
        })
        print(f"OK Mixed derivative orders: {results}")
    except Exception as e:
        print(f"FAIL Mixed derivative orders: {e}")
    
    return cases

def test_numerical_extremes():
    """Test with very large and very small numbers"""
    print("\n=== NUMERICAL EXTREMES ===")
    
    cases = []
    
    # 1. Very large coefficients
    try:
        c = np.array([[1e10], [2e10]])  # Linear, shape (2, 1)
        x = [0, 1]
        bp = BPoly(c, x)
        eval_points = [0.0, 0.5, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'large_coefficients',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Very large coefficients'
        })
        print(f"OK Large coefficients: {results}")
    except Exception as e:
        print(f"FAIL Large coefficients: {e}")
    
    # 2. Very small coefficients
    try:
        c = np.array([[1e-15], [2e-15]])
        x = [0, 1]
        bp = BPoly(c, x)
        eval_points = [0.0, 0.5, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'small_coefficients',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Very small coefficients'
        })
        print(f"OK Small coefficients: {results}")
    except Exception as e:
        print(f"FAIL Small coefficients: {e}")
    
    # 3. Zero coefficients
    try:
        c = np.array([[0], [0], [0]])  # All zeros, degree 2, shape (3, 1)
        x = [0, 1]
        bp = BPoly(c, x)
        eval_points = [0.0, 0.5, 1.0]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'all_zero_coefficients',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'All zero coefficients'
        })
        print(f"OK Zero coefficients: {results}")
    except Exception as e:
        print(f"FAIL Zero coefficients: {e}")
    
    return cases

def test_boundary_conditions():
    """Test boundary and extrapolation behavior"""
    print("\n=== BOUNDARY CONDITIONS ===")
    
    cases = []
    
    # 1. Extrapolation behavior
    try:
        c = np.array([[0], [1]])  # f(x) = x on [0,1], linear, shape (2, 1)
        x = [0, 1]
        bp = BPoly(c, x)
        
        # Test various extrapolation points
        extrap_points = [-2, -0.1, 1.1, 5]
        results = [float(bp(xi)) for xi in extrap_points]
        
        cases.append({
            'name': 'linear_extrapolation',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': extrap_points,
            'expected_results': results,
            'description': 'Linear extrapolation far from domain'
        })
        print(f"OK Linear extrapolation: {results}")
    except Exception as e:
        print(f"FAIL Linear extrapolation: {e}")
    
    # 2. Points very close to boundaries
    try:
        c = np.array([[1, 4], [2, 5], [3, 6]])  # Two intervals, quadratic, shape (3, 2)
        x = [0, 0.5, 1]
        bp = BPoly(c, x)
        
        # Points very close to breakpoint 0.5
        epsilon = 1e-14
        close_points = [0.5 - epsilon, 0.5, 0.5 + epsilon]
        results = [float(bp(xi)) for xi in close_points]
        
        cases.append({
            'name': 'near_boundary_evaluation',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': close_points,
            'expected_results': results,
            'description': 'Evaluation very close to interval boundaries'
        })
        print(f"OK Near boundary: {results}")
    except Exception as e:
        print(f"FAIL Near boundary: {e}")
    
    # 3. Tiny intervals
    try:
        c = np.array([[1], [2]])  # Linear, shape (2, 1)
        x = [0, 1e-12]  # Extremely small interval
        bp = BPoly(c, x)
        eval_points = [0.0, 0.5e-12, 1e-12]
        results = [float(bp(xi)) for xi in eval_points]
        
        cases.append({
            'name': 'tiny_interval',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': eval_points,
            'expected_results': results,
            'description': 'Extremely small interval'
        })
        print(f"OK Tiny interval: {results}")
    except Exception as e:
        print(f"FAIL Tiny interval: {e}")
    
    return cases

def test_derivative_edge_cases():
    """Test derivative computation edge cases"""
    print("\n=== DERIVATIVE EDGE CASES ===")
    
    cases = []
    
    # 1. Derivative of constant (should be zero)
    try:
        c = np.array([[5]])  # Constant 5
        x = [0, 1]
        bp = BPoly(c, x)
        bp_deriv = bp.derivative()
        
        deriv_points = [0.0, 0.5, 1.0]
        deriv_results = [float(bp_deriv(xi)) for xi in deriv_points]
        
        cases.append({
            'name': 'derivative_of_constant',
            'original_coefficients': c.tolist(),
            'original_breakpoints': x,
            'eval_points': deriv_points,
            'derivative_results': deriv_results,
            'description': 'Derivative of constant should be zero'
        })
        print(f"OK Derivative of constant: {deriv_results}")
    except Exception as e:
        print(f"FAIL Derivative of constant: {e}")
    
    # 2. Multiple derivatives until zero
    try:
        c = np.array([[0], [0], [0], [1]])  # Cubic: t^3, shape (4, 1)
        x = [0, 1]
        bp = BPoly(c, x)
        
        derivative_results = {}
        for order in range(1, 6):
            try:
                bp_dn = bp.derivative(order)
                result = float(bp_dn(0.5))
                derivative_results[f'order_{order}'] = result
                print(f"   d^{order}/dx^{order} at 0.5: {result}")
            except Exception as de:
                print(f"   d^{order}/dx^{order} failed: {de}")
                derivative_results[f'order_{order}'] = None
        
        cases.append({
            'name': 'multiple_derivatives_cubic',
            'original_coefficients': c.tolist(),
            'original_breakpoints': x,
            'derivative_results': derivative_results,
            'description': 'Multiple derivatives of cubic until zero'
        })
        print(f"OK Multiple derivatives")
    except Exception as e:
        print(f"FAIL Multiple derivatives: {e}")
    
    # 3. Derivative order higher than polynomial degree
    try:
        c = np.array([[1], [2]])  # Linear (degree 1), shape (2, 1)
        x = [0, 1]
        bp = BPoly(c, x)
        
        # Try derivatives of order 2, 3, 4 (should all be zero)
        high_order_results = {}
        for order in [2, 3, 4]:
            try:
                bp_dn = bp.derivative(order)
                result = float(bp_dn(0.5))
                high_order_results[f'order_{order}'] = result
            except Exception as de:
                print(f"   derivative order {order} failed: {de}")
                high_order_results[f'order_{order}'] = None
        
        cases.append({
            'name': 'high_order_derivative_linear',
            'original_coefficients': c.tolist(),
            'derivative_orders': high_order_results,
            'description': 'High order derivatives of linear polynomial'
        })
        print(f"OK High order derivatives: {high_order_results}")
    except Exception as e:
        print(f"FAIL High order derivatives: {e}")
    
    return cases

def test_special_floating_point():
    """Test with special floating point values"""
    print("\n=== SPECIAL FLOATING POINT VALUES ===")
    
    cases = []
    
    # 1. NaN coefficients
    try:
        c = np.array([[np.nan], [1]])
        x = [0, 1]
        bp = BPoly(c, x)
        result = bp(0.5)
        
        cases.append({
            'name': 'nan_coefficients',
            'coefficients': 'contains_nan',  # Special marker
            'breakpoints': x,
            'eval_points': [0.5],
            'expected_results': [float(result)],
            'description': 'NaN in coefficients',
            'contains_nan': True
        })
        print(f"OK NaN coefficients: {result} (is_nan: {np.isnan(result)})")
    except Exception as e:
        print(f"OK NaN coefficients failed: {e}")
        cases.append({
            'name': 'nan_coefficients',
            'should_fail': True,
            'error_type': type(e).__name__,
            'description': 'NaN coefficients should fail or propagate NaN'
        })
    
    # 2. Infinite coefficients
    try:
        c = np.array([[np.inf], [1]])
        x = [0, 1]
        bp = BPoly(c, x)
        result = bp(0.5)
        
        cases.append({
            'name': 'inf_coefficients',
            'coefficients': 'contains_inf',  # Special marker
            'breakpoints': x,
            'eval_points': [0.5],
            'expected_results': [float(result)],
            'description': 'Infinity in coefficients',
            'contains_inf': True
        })
        print(f"OK Inf coefficients: {result} (is_inf: {np.isinf(result)})")
    except Exception as e:
        print(f"OK Inf coefficients failed: {e}")
    
    # 3. Evaluation at NaN
    try:
        c = np.array([[1], [2]])  # Normal coefficients, shape (2, 1)
        x = [0, 1]
        bp = BPoly(c, x)
        result = bp(np.nan)
        
        cases.append({
            'name': 'evaluate_at_nan',
            'coefficients': c.tolist(),
            'breakpoints': x,
            'eval_points': ['nan'],  # Special marker
            'expected_results': [float(result)],
            'description': 'Evaluation at NaN input'
        })
        print(f"OK Evaluate at NaN: {result}")
    except Exception as e:
        print(f"FAIL Evaluate at NaN: {e}")
    
    return cases

def test_comprehensive_from_derivatives():
    """Comprehensive from_derivatives testing"""
    print("\n=== COMPREHENSIVE FROM_DERIVATIVES ===")
    
    cases = []
    
    # Test various configurations that work
    test_configs = [
        # Two points, function values only
        {
            'xi': [0, 1],
            'yi': [[1], [3]],
            'name': 'two_points_values_only'
        },
        # Two points with first derivatives
        {
            'xi': [0, 1],
            'yi': [[1, 2], [3, -1]],
            'name': 'two_points_with_derivatives'
        },
        # Three points, values only
        {
            'xi': [0, 0.5, 1],
            'yi': [[0], [0.25], [1]],
            'name': 'three_points_values_only'
        },
        # Three points with some derivatives
        {
            'xi': [0, 0.5, 1],
            'yi': [[0, 0], [0.25], [1, 0]],
            'name': 'three_points_mixed_derivatives'
        },
        # Four points for higher order
        {
            'xi': [0, 1, 2, 3],
            'yi': [[0], [1, 2], [4], [9, 6]],
            'name': 'four_points_mixed'
        }
    ]
    
    for config in test_configs:
        try:
            bp = BPoly.from_derivatives(config['xi'], config['yi'])
            
            # Evaluate at boundaries and midpoints
            eval_points = []
            for i in range(len(config['xi']) - 1):
                eval_points.extend([
                    config['xi'][i],
                    (config['xi'][i] + config['xi'][i+1]) / 2,
                    config['xi'][i+1]
                ])
            
            # Remove duplicates and sort
            eval_points = sorted(list(set(eval_points)))
            results = [float(bp(xi)) for xi in eval_points]
            
            cases.append({
                'name': config['name'],
                'xi': config['xi'],
                'yi': config['yi'],
                'eval_points': eval_points,
                'expected_results': results,
                'description': f"from_derivatives: {config['name']}"
            })
            print(f"OK {config['name']}: {len(results)} points evaluated")
            
        except Exception as e:
            print(f"FAIL {config['name']}: {e}")
            cases.append({
                'name': config['name'],
                'should_fail': True,
                'error_type': type(e).__name__,
                'description': f"from_derivatives {config['name']} failed"
            })
    
    return cases

def run_comprehensive_tests():
    """Run all corner case tests and save results"""
    
    print("Scipy BPoly Corner Case Analysis")
    print("=" * 50)
    
    all_cases = []
    
    # Run all test categories
    test_functions = [
        test_basic_edge_cases,
        test_degenerate_cases,
        test_from_derivatives_edge_cases,
        test_numerical_extremes,
        test_boundary_conditions,
        test_derivative_edge_cases,
        test_special_floating_point,
        test_comprehensive_from_derivatives
    ]
    
    for test_func in test_functions:
        try:
            cases = test_func()
            all_cases.extend(cases)
        except Exception as e:
            print(f"Test function {test_func.__name__} failed: {e}")
            traceback.print_exc()
    
    # Save results to JSON file
    output_file = 'corner_case_test_data.json'
    
    try:
        # Convert special values for JSON serialization
        for case in all_cases:
            if 'expected_results' in case:
                for i, result in enumerate(case['expected_results']):
                    if isinstance(result, float):
                        if np.isnan(result):
                            case['expected_results'][i] = "NaN"
                        elif np.isinf(result):
                            case['expected_results'][i] = "Infinity" if result > 0 else "-Infinity"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'description': 'Comprehensive corner case test data for BPoly',
                    'generated_by': 'verify_corner_cases.py',
                    'total_cases': len(all_cases),
                    'success_cases': len([c for c in all_cases if not c.get('should_fail', False)]),
                    'failure_cases': len([c for c in all_cases if c.get('should_fail', False)])
                },
                'test_cases': all_cases
            }, f, indent=2)
        
        print(f"\nOK Saved {len(all_cases)} test cases to {output_file}")
        
        # Print summary
        success_cases = [c for c in all_cases if not c.get('should_fail', False)]
        failure_cases = [c for c in all_cases if c.get('should_fail', False)]
        
        print(f"\nSUMMARY:")
        print(f"- {len(success_cases)} cases that should succeed")
        print(f"- {len(failure_cases)} cases that should fail")
        print(f"- Total: {len(all_cases)} test cases")
        
        return True
        
    except Exception as e:
        print(f"FAIL Failed to save test data: {e}")
        return False

if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)