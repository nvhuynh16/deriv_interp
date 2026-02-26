"""
Verify HPoly (Hermite Polynomial) implementation against numpy.polynomial.hermite
and numpy.polynomial.hermite_e.

This script generates reference values using numpy's Hermite polynomial implementations
to verify that our C++ HPoly class produces correct results for both:
- Physicist's Hermite (H_n) - numpy.polynomial.hermite
- Probabilist's Hermite (He_n) - numpy.polynomial.hermite_e

Usage:
    python scripts/verify_hpoly_values.py
"""

import numpy as np
from numpy.polynomial import hermite as H   # Physicist's
from numpy.polynomial import hermite_e as He  # Probabilist's
import json


def generate_reference_data():
    """
    Generate reference data using numpy.polynomial.hermite and hermite_e.
    This data will be used to verify the C++ implementation.
    """

    print("=== HPoly (Hermite Polynomial) Numpy Reference Values ===\n")

    # Standard domain mapping for interval [a, b]:
    # s = (2x - a - b) / (b - a)  maps x to [-1, 1]
    # x = ((b-a)*s + a + b) / 2   maps s back to x

    test_data = {"physicist": {}, "probabilist": {}}

    # ===========================================================
    # PHYSICIST'S HERMITE (H_n)
    # ===========================================================
    print("=" * 60)
    print("PHYSICIST'S HERMITE (H_n)")
    print("=" * 60)

    # Test 1: Constant on [0, 1]
    print("\n--- Physicist Test 1: Constant f(x) = 5 on [0, 1] ---")
    coeffs = np.array([5.0])  # Hermite coeffs for constant 5
    eval_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    s_pts = 2 * eval_pts - 1
    values = H.hermval(s_pts, coeffs)
    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    test_data["physicist"]["constant"] = {
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 2: Linear f(x) = x on [0, 1]
    print("\n--- Physicist Test 2: Linear f(x) = x on [0, 1] ---")
    # H_0 = 1, H_1 = 2s
    # f(s) = c_0*H_0 + c_1*H_1 = c_0 + 2*c_1*s
    # We want f(x) = x = (s+1)/2
    # So: c_0 + 2*c_1*s = (s+1)/2 = 0.5 + 0.5*s
    # => c_0 = 0.5, 2*c_1 = 0.5 => c_1 = 0.25
    coeffs = np.array([0.5, 0.25])
    values = H.hermval(s_pts, coeffs)
    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    print(f"Expected (x): {eval_pts.tolist()}")
    test_data["physicist"]["linear"] = {
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 3: Derivative formula verification
    print("\n--- Physicist Test 3: Derivative of linear ---")
    deriv_coeffs = H.hermder(coeffs)
    # Scale by 2/(b-a) = 2 for derivative
    deriv_values = H.hermval(s_pts, deriv_coeffs) * 2
    print(f"Original coeffs: {coeffs.tolist()}")
    print(f"Derivative coeffs (unscaled): {deriv_coeffs.tolist()}")
    print(f"Derivative values (scaled by 2): {deriv_values.tolist()}")
    print(f"Expected (1): {[1.0]*len(eval_pts)}")
    test_data["physicist"]["derivative_linear"] = {
        "original_coeffs": coeffs.tolist(),
        "derivative_coeffs": deriv_coeffs.tolist(),
        "scale_factor": 2.0,
        "eval_points": eval_pts.tolist(),
        "derivative_values": deriv_values.tolist()
    }

    # Test 4: Antiderivative of constant
    print("\n--- Physicist Test 4: Antiderivative of f(x) = 2 ---")
    const_coeffs = np.array([2.0])
    anti_coeffs = H.hermint(const_coeffs)
    # Scale by (b-a)/2 = 0.5 for antiderivative
    anti_values_s = H.hermval(s_pts, anti_coeffs) * 0.5
    # Shift so F(0) = 0 (s = -1 when x = 0)
    anti_at_a = H.hermval(-1, anti_coeffs) * 0.5
    anti_values = anti_values_s - anti_at_a
    print(f"Antiderivative coeffs (unscaled): {anti_coeffs.tolist()}")
    print(f"Antiderivative values (F(0)=0): {anti_values.tolist()}")
    print(f"Expected (2x): {(2*eval_pts).tolist()}")
    test_data["physicist"]["antiderivative_constant"] = {
        "original_coeffs": const_coeffs.tolist(),
        "antiderivative_coeffs": anti_coeffs.tolist(),
        "scale_factor": 0.5,
        "eval_points": eval_pts.tolist(),
        "antiderivative_values": anti_values.tolist()
    }

    # Test 5: Integration
    print("\n--- Physicist Test 5: Integration of x from 0 to 1 ---")
    linear_coeffs = np.array([0.5, 0.25])
    anti = H.hermint(linear_coeffs)
    integral_s = H.hermval(1, anti) - H.hermval(-1, anti)
    integral = integral_s * 0.5  # Scale by h/2
    print(f"Integral of x from 0 to 1: {integral}")
    print(f"Expected: 0.5")
    test_data["physicist"]["integration_linear"] = {
        "coefficients": linear_coeffs.tolist(),
        "integral": float(integral)
    }

    # Test 6: Power basis conversion
    print("\n--- Physicist Test 6: Power to Hermite conversion ---")
    # Power basis: 1 + 2x + 3x^2 on [0, 1]
    power_on_x = np.array([1.0, 2.0, 3.0])
    # x = (s+1)/2 on [0,1] -> [-1,1]
    # 1 + 2*(s+1)/2 + 3*((s+1)/2)^2
    # = 1 + (s+1) + 3*(s^2 + 2s + 1)/4
    # = 1 + s + 1 + 0.75*s^2 + 1.5*s + 0.75
    # = 2.75 + 2.5*s + 0.75*s^2
    power_on_s = np.array([2.75, 2.5, 0.75])
    herm_coeffs = H.poly2herm(power_on_s)
    values = H.hermval(s_pts, herm_coeffs)
    expected = 1 + 2*eval_pts + 3*eval_pts**2
    print(f"Power coeffs (on x): {power_on_x.tolist()}")
    print(f"Power coeffs (on s): {power_on_s.tolist()}")
    print(f"Hermite coeffs: {herm_coeffs.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Expected (1+2x+3x^2): {expected.tolist()}")
    test_data["physicist"]["power_to_hermite"] = {
        "power_coeffs_on_x": power_on_x.tolist(),
        "power_coeffs_on_s": power_on_s.tolist(),
        "hermite_coeffs": herm_coeffs.tolist(),
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 7: Hermite to power conversion
    print("\n--- Physicist Test 7: Hermite to Power conversion ---")
    herm_in = np.array([1.0, 0.5, 0.25])
    power_out = H.herm2poly(herm_in)
    print(f"Hermite coeffs: {herm_in.tolist()}")
    print(f"Power coeffs (on s): {power_out.tolist()}")
    test_data["physicist"]["hermite_to_power"] = {
        "hermite_coeffs": herm_in.tolist(),
        "power_coeffs_on_s": power_out.tolist()
    }

    # ===========================================================
    # PROBABILIST'S HERMITE (He_n)
    # ===========================================================
    print("\n" + "=" * 60)
    print("PROBABILIST'S HERMITE (He_n)")
    print("=" * 60)

    # Test 1: Constant on [0, 1]
    print("\n--- Probabilist Test 1: Constant f(x) = 5 on [0, 1] ---")
    coeffs = np.array([5.0])
    values = He.hermeval(s_pts, coeffs)
    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    test_data["probabilist"]["constant"] = {
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 2: Linear f(x) = x on [0, 1]
    print("\n--- Probabilist Test 2: Linear f(x) = x on [0, 1] ---")
    # He_0 = 1, He_1 = s
    # f(s) = c_0*He_0 + c_1*He_1 = c_0 + c_1*s
    # We want f(x) = x = (s+1)/2
    # So: c_0 + c_1*s = (s+1)/2 = 0.5 + 0.5*s
    # => c_0 = 0.5, c_1 = 0.5
    coeffs = np.array([0.5, 0.5])
    values = He.hermeval(s_pts, coeffs)
    print(f"Coefficients: {coeffs.tolist()}")
    print(f"Values at {eval_pts.tolist()}: {values.tolist()}")
    print(f"Expected (x): {eval_pts.tolist()}")
    test_data["probabilist"]["linear"] = {
        "coefficients": coeffs.tolist(),
        "breakpoints": [0.0, 1.0],
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 3: Derivative formula verification
    print("\n--- Probabilist Test 3: Derivative of linear ---")
    deriv_coeffs = He.hermeder(coeffs)
    deriv_values = He.hermeval(s_pts, deriv_coeffs) * 2
    print(f"Original coeffs: {coeffs.tolist()}")
    print(f"Derivative coeffs (unscaled): {deriv_coeffs.tolist()}")
    print(f"Derivative values (scaled by 2): {deriv_values.tolist()}")
    print(f"Expected (1): {[1.0]*len(eval_pts)}")
    test_data["probabilist"]["derivative_linear"] = {
        "original_coeffs": coeffs.tolist(),
        "derivative_coeffs": deriv_coeffs.tolist(),
        "scale_factor": 2.0,
        "eval_points": eval_pts.tolist(),
        "derivative_values": deriv_values.tolist()
    }

    # Test 4: Antiderivative of constant
    print("\n--- Probabilist Test 4: Antiderivative of f(x) = 2 ---")
    const_coeffs = np.array([2.0])
    anti_coeffs = He.hermeint(const_coeffs)
    anti_values_s = He.hermeval(s_pts, anti_coeffs) * 0.5
    anti_at_a = He.hermeval(-1, anti_coeffs) * 0.5
    anti_values = anti_values_s - anti_at_a
    print(f"Antiderivative coeffs (unscaled): {anti_coeffs.tolist()}")
    print(f"Antiderivative values (F(0)=0): {anti_values.tolist()}")
    print(f"Expected (2x): {(2*eval_pts).tolist()}")
    test_data["probabilist"]["antiderivative_constant"] = {
        "original_coeffs": const_coeffs.tolist(),
        "antiderivative_coeffs": anti_coeffs.tolist(),
        "scale_factor": 0.5,
        "eval_points": eval_pts.tolist(),
        "antiderivative_values": anti_values.tolist()
    }

    # Test 5: Integration
    print("\n--- Probabilist Test 5: Integration of x from 0 to 1 ---")
    linear_coeffs = np.array([0.5, 0.5])
    anti = He.hermeint(linear_coeffs)
    integral_s = He.hermeval(1, anti) - He.hermeval(-1, anti)
    integral = integral_s * 0.5
    print(f"Integral of x from 0 to 1: {integral}")
    print(f"Expected: 0.5")
    test_data["probabilist"]["integration_linear"] = {
        "coefficients": linear_coeffs.tolist(),
        "integral": float(integral)
    }

    # Test 6: Power basis conversion
    print("\n--- Probabilist Test 6: Power to Hermite conversion ---")
    power_on_s = np.array([2.75, 2.5, 0.75])
    herme_coeffs = He.poly2herme(power_on_s)
    values = He.hermeval(s_pts, herme_coeffs)
    expected = 1 + 2*eval_pts + 3*eval_pts**2
    print(f"Power coeffs (on s): {power_on_s.tolist()}")
    print(f"Hermite_e coeffs: {herme_coeffs.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Expected (1+2x+3x^2): {expected.tolist()}")
    test_data["probabilist"]["power_to_hermite"] = {
        "power_coeffs_on_s": power_on_s.tolist(),
        "hermite_coeffs": herme_coeffs.tolist(),
        "eval_points": eval_pts.tolist(),
        "values": values.tolist()
    }

    # Test 7: Hermite to power conversion
    print("\n--- Probabilist Test 7: Hermite to Power conversion ---")
    herme_in = np.array([1.0, 0.5, 0.25])
    power_out = He.herme2poly(herme_in)
    print(f"Hermite_e coeffs: {herme_in.tolist()}")
    print(f"Power coeffs (on s): {power_out.tolist()}")
    test_data["probabilist"]["hermite_to_power"] = {
        "hermite_coeffs": herme_in.tolist(),
        "power_coeffs_on_s": power_out.tolist()
    }

    # ===========================================================
    # HERMITE POLYNOMIAL VALUES (for verification)
    # ===========================================================
    print("\n" + "=" * 60)
    print("HERMITE POLYNOMIAL BASE VALUES")
    print("=" * 60)

    print("\n--- Physicist's H_n(s) values ---")
    s_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for n in range(5):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        vals = [H.hermval(s, coeffs) for s in s_vals]
        print(f"H_{n}({s_vals}) = {vals}")
    test_data["physicist"]["base_polynomials"] = {
        "s_values": s_vals,
        "H_0": [H.hermval(s, [1.0]) for s in s_vals],
        "H_1": [H.hermval(s, [0.0, 1.0]) for s in s_vals],
        "H_2": [H.hermval(s, [0.0, 0.0, 1.0]) for s in s_vals],
        "H_3": [H.hermval(s, [0.0, 0.0, 0.0, 1.0]) for s in s_vals],
        "H_4": [H.hermval(s, [0.0, 0.0, 0.0, 0.0, 1.0]) for s in s_vals],
    }

    print("\n--- Probabilist's He_n(s) values ---")
    for n in range(5):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        vals = [He.hermeval(s, coeffs) for s in s_vals]
        print(f"He_{n}({s_vals}) = {vals}")
    test_data["probabilist"]["base_polynomials"] = {
        "s_values": s_vals,
        "He_0": [He.hermeval(s, [1.0]) for s in s_vals],
        "He_1": [He.hermeval(s, [0.0, 1.0]) for s in s_vals],
        "He_2": [He.hermeval(s, [0.0, 0.0, 1.0]) for s in s_vals],
        "He_3": [He.hermeval(s, [0.0, 0.0, 0.0, 1.0]) for s in s_vals],
        "He_4": [He.hermeval(s, [0.0, 0.0, 0.0, 0.0, 1.0]) for s in s_vals],
    }

    # ===========================================================
    # DERIVATIVE FORMULAS
    # ===========================================================
    print("\n" + "=" * 60)
    print("DERIVATIVE FORMULAS VERIFICATION")
    print("=" * 60)

    print("\n--- Physicist's d/ds H_n = 2n * H_{n-1} ---")
    for n in range(1, 5):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        deriv_coeffs = H.hermder(coeffs)
        print(f"d/ds H_{n}: coeffs = {deriv_coeffs.tolist()}")
    test_data["physicist"]["derivative_formula"] = {
        "d_H_1": H.hermder([0.0, 1.0]).tolist(),
        "d_H_2": H.hermder([0.0, 0.0, 1.0]).tolist(),
        "d_H_3": H.hermder([0.0, 0.0, 0.0, 1.0]).tolist(),
        "d_H_4": H.hermder([0.0, 0.0, 0.0, 0.0, 1.0]).tolist(),
    }

    print("\n--- Probabilist's d/ds He_n = n * He_{n-1} ---")
    for n in range(1, 5):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        deriv_coeffs = He.hermeder(coeffs)
        print(f"d/ds He_{n}: coeffs = {deriv_coeffs.tolist()}")
    test_data["probabilist"]["derivative_formula"] = {
        "d_He_1": He.hermeder([0.0, 1.0]).tolist(),
        "d_He_2": He.hermeder([0.0, 0.0, 1.0]).tolist(),
        "d_He_3": He.hermeder([0.0, 0.0, 0.0, 1.0]).tolist(),
        "d_He_4": He.hermeder([0.0, 0.0, 0.0, 0.0, 1.0]).tolist(),
    }

    # ===========================================================
    # ANTIDERIVATIVE FORMULAS
    # ===========================================================
    print("\n" + "=" * 60)
    print("ANTIDERIVATIVE FORMULAS VERIFICATION")
    print("=" * 60)

    print("\n--- Physicist's integral H_n ---")
    for n in range(5):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        anti_coeffs = H.hermint(coeffs)
        print(f"integral H_{n}: coeffs = {anti_coeffs.tolist()}")
    test_data["physicist"]["antiderivative_formula"] = {
        "int_H_0": H.hermint([1.0]).tolist(),
        "int_H_1": H.hermint([0.0, 1.0]).tolist(),
        "int_H_2": H.hermint([0.0, 0.0, 1.0]).tolist(),
        "int_H_3": H.hermint([0.0, 0.0, 0.0, 1.0]).tolist(),
    }

    print("\n--- Probabilist's integral He_n ---")
    for n in range(5):
        coeffs = [0.0] * (n + 1)
        coeffs[n] = 1.0
        anti_coeffs = He.hermeint(coeffs)
        print(f"integral He_{n}: coeffs = {anti_coeffs.tolist()}")
    test_data["probabilist"]["antiderivative_formula"] = {
        "int_He_0": He.hermeint([1.0]).tolist(),
        "int_He_1": He.hermeint([0.0, 1.0]).tolist(),
        "int_He_2": He.hermeint([0.0, 0.0, 1.0]).tolist(),
        "int_He_3": He.hermeint([0.0, 0.0, 0.0, 1.0]).tolist(),
    }

    # Save test data to JSON
    with open('scripts/hpoly_reference_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"\nReference data saved to scripts/hpoly_reference_data.json")

    return test_data


def verify_with_cpp_bindings():
    """
    If deriv_poly bindings are available, verify HPoly against numpy.
    """
    try:
        import deriv_poly as dp
        print("\n" + "=" * 60)
        print("VERIFICATION AGAINST C++ BINDINGS")
        print("=" * 60)

        tolerance = 1e-10

        # Test Physicist's constant
        print("\n--- Verifying Physicist's constant ---")
        coeffs = [[5.0]]
        breaks = [0.0, 1.0]
        hp_cpp = dp.HPoly(coeffs, breaks, kind=dp.HermiteKind.Physicist)
        hp_np = lambda x: 5.0
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cpp_val = hp_cpp(x)
            np_val = hp_np(x)
            match = abs(cpp_val - np_val) < tolerance
            print(f"  x={x}: C++={cpp_val:.10f}, numpy={np_val:.10f}, match={match}")

        # Test Physicist's linear
        print("\n--- Verifying Physicist's linear f(x) = x ---")
        coeffs = [[0.5], [0.25]]
        hp_cpp = dp.HPoly(coeffs, breaks, kind=dp.HermiteKind.Physicist)
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cpp_val = hp_cpp(x)
            np_val = x  # f(x) = x
            match = abs(cpp_val - np_val) < tolerance
            print(f"  x={x}: C++={cpp_val:.10f}, numpy={np_val:.10f}, match={match}")

        # Test Probabilist's linear
        print("\n--- Verifying Probabilist's linear f(x) = x ---")
        coeffs = [[0.5], [0.5]]
        hp_cpp = dp.HPoly(coeffs, breaks, kind=dp.HermiteKind.Probabilist)
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cpp_val = hp_cpp(x)
            np_val = x
            match = abs(cpp_val - np_val) < tolerance
            print(f"  x={x}: C++={cpp_val:.10f}, numpy={np_val:.10f}, match={match}")

        # Test from_derivatives
        print("\n--- Verifying from_derivatives (both kinds) ---")
        xi = [0.0, 1.0]
        yi = [[0.0, 1.0], [1.0, -1.0]]

        hp_phys = dp.HPoly.from_derivatives(xi, yi, kind=dp.HermiteKind.Physicist)
        hp_prob = dp.HPoly.from_derivatives(xi, yi, kind=dp.HermiteKind.Probabilist)

        print("  Endpoint values:")
        print(f"    Physicist:   f(0)={hp_phys(0):.10f}, f(1)={hp_phys(1):.10f}")
        print(f"    Probabilist: f(0)={hp_prob(0):.10f}, f(1)={hp_prob(1):.10f}")

        print("  Endpoint derivatives:")
        print(f"    Physicist:   f'(0)={hp_phys(0, nu=1):.10f}, f'(1)={hp_phys(1, nu=1):.10f}")
        print(f"    Probabilist: f'(0)={hp_prob(0, nu=1):.10f}, f'(1)={hp_prob(1, nu=1):.10f}")

        # Test integration
        print("\n--- Verifying integration ---")
        coeffs = [[0.5], [0.25]]  # f(x) = x
        hp_cpp = dp.HPoly(coeffs, breaks, kind=dp.HermiteKind.Physicist)
        integral = hp_cpp.integrate(0, 1)
        expected = 0.5
        match = abs(integral - expected) < tolerance
        print(f"  Integral of x from 0 to 1: C++={integral:.10f}, expected={expected:.10f}, match={match}")

        print("\n=== Verification complete ===")

    except ImportError:
        print("\nderiv_poly not installed. Run 'pip install -e .' to build bindings.")
        print("Skipping C++ verification.")


if __name__ == "__main__":
    generate_reference_data()
    verify_with_cpp_bindings()
