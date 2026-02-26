#include "include/cpoly.h"
#include "test_utils.h"
#include <cassert>
#include <limits>
#include <thread>
#include <atomic>
#include <random>

int main() {
    TestRunner test;
    const double tolerance = 1e-10;

    std::cout << "=== CPoly (Chebyshev Polynomial) Test Suite ===" << std::endl;

    // ============================================================
    // Group 1: Basic Construction and Evaluation
    // ============================================================
    std::cout << "\n--- Group 1: Basic Construction and Evaluation ---\n" << std::endl;

    // Test 1: Basic construction
    test.expect_no_throw([]() {
        // For Chebyshev: T_0 = 1 (constant), so c = {1} gives p(x) = 1
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
    }, "Test 1: Basic construction");

    // Test 2: Constant polynomial T_0 = 1
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 5.0, tolerance, "Test 2a: Constant f(0)");
        test.expect_near(cp(0.5), 5.0, tolerance, "Test 2b: Constant f(0.5)");
        test.expect_near(cp(1.0), 5.0, tolerance, "Test 2c: Constant f(1)");
    }

    // Test 3: Linear polynomial T_1(s) = s
    // On interval [0,1], s = 2x - 1
    // So T_1(s) = 2x - 1 maps [0,1] -> [-1,1]
    // p(x) = c_0 * T_0(s) + c_1 * T_1(s) = c_0 + c_1 * (2x - 1)
    // For p(x) = x, we need c_0 + c_1*(2x-1) = x
    // => c_0 - c_1 + 2*c_1*x = x
    // => c_0 - c_1 = 0 and 2*c_1 = 1
    // => c_1 = 0.5, c_0 = 0.5
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 0.0, tolerance, "Test 3a: Linear f(0)");
        test.expect_near(cp(0.25), 0.25, tolerance, "Test 3b: Linear f(0.25)");
        test.expect_near(cp(0.5), 0.5, tolerance, "Test 3c: Linear f(0.5)");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 3d: Linear f(1)");
    }

    // Test 4: Quadratic polynomial using T_0, T_1, T_2
    // T_0 = 1, T_1(s) = s, T_2(s) = 2s^2 - 1
    // On [0,1]: s = 2x - 1
    // T_2(s) = 2(2x-1)^2 - 1 = 2(4x^2 - 4x + 1) - 1 = 8x^2 - 8x + 1
    // For p(x) = x^2, we need: c_0 + c_1*(2x-1) + c_2*(8x^2 - 8x + 1) = x^2
    // => c_0 - c_1 + c_2 + (2c_1 - 8c_2)*x + 8c_2*x^2 = x^2
    // => 8*c_2 = 1 => c_2 = 1/8
    // => 2*c_1 - 8*c_2 = 0 => c_1 = 4*c_2 = 0.5
    // => c_0 - c_1 + c_2 = 0 => c_0 = c_1 - c_2 = 0.5 - 0.125 = 0.375
    {
        std::vector<std::vector<double>> coeffs = {{0.375}, {0.5}, {0.125}};  // p(x) = x^2 on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 0.0, tolerance, "Test 4a: Quadratic f(0)");
        test.expect_near(cp(0.5), 0.25, tolerance, "Test 4b: Quadratic f(0.5)");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 4c: Quadratic f(1)");
        test.expect_near(cp(0.25), 0.0625, tolerance, "Test 4d: Quadratic f(0.25)");
    }

    // Test 5: Multiple intervals
    {
        // Two constant pieces: [0,1] has value 1, [1,2] has value 2
        std::vector<std::vector<double>> coeffs = {{1, 2}};
        std::vector<double> breaks = {0, 1, 2};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.5), 1.0, tolerance, "Test 5a: Multi-interval f(0.5)");
        test.expect_near(cp(1.5), 2.0, tolerance, "Test 5b: Multi-interval f(1.5)");
    }

    // ============================================================
    // Group 2: Error Handling
    // ============================================================
    std::cout << "\n--- Group 2: Error Handling ---\n" << std::endl;

    // Test 6: Empty coefficients error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs;  // Empty
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
    }, "Test 6: Empty coefficients error");

    // Test 7: Too few breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0};  // Only 1 breakpoint
        CPoly cp(coeffs, breaks);
    }, "Test 7: Too few breakpoints error");

    // Test 8: Non-monotonic breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 0.5, 0.3};  // Not monotonic
        CPoly cp(coeffs, breaks);
    }, "Test 8: Non-monotonic breakpoints error");

    // ============================================================
    // Group 3: Vector Evaluation
    // ============================================================
    std::cout << "\n--- Group 3: Vector Evaluation ---\n" << std::endl;

    // Test 9: Evaluate at multiple points
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        std::vector<double> xs = {0.0, 0.25, 0.5, 0.75, 1.0};
        std::vector<double> results = cp(xs);

        test.expect_eq(results.size(), 5ul, "Test 9a: Vector result size");
        test.expect_near(results[0], 0.0, tolerance, "Test 9b: Vector result[0]");
        test.expect_near(results[2], 0.5, tolerance, "Test 9c: Vector result[2]");
        test.expect_near(results[4], 1.0, tolerance, "Test 9d: Vector result[4]");
    }

    // ============================================================
    // Group 4: Extrapolation Modes
    // ============================================================
    std::cout << "\n--- Group 4: Extrapolation Modes ---\n" << std::endl;

    // Test 10: Extrapolate mode (default)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Extrapolate);

        // Extrapolate beyond [0,1]
        test.expect_near(cp(-0.5), -0.5, tolerance, "Test 10a: Extrapolate f(-0.5)");
        test.expect_near(cp(1.5), 1.5, tolerance, "Test 10b: Extrapolate f(1.5)");
    }

    // Test 11: NoExtrapolate mode
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::NoExtrapolate);

        test.expect_true(std::isnan(cp(-0.5)), "Test 11a: NoExtrapolate f(-0.5) is NaN");
        test.expect_true(std::isnan(cp(1.5)), "Test 11b: NoExtrapolate f(1.5) is NaN");
    }

    // Test 12: Periodic mode
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Periodic);

        // Test periodic wrapping
        test.expect_near(cp(0.5), 0.5, tolerance, "Test 12a: Periodic f(0.5)");
        test.expect_near(cp(1.5), cp(0.5), tolerance, "Test 12b: Periodic f(1.5) = f(0.5)");
        test.expect_near(cp(2.5), cp(0.5), tolerance, "Test 12c: Periodic f(2.5) = f(0.5)");
    }

    // ============================================================
    // Group 5: Derivative Operations
    // ============================================================
    std::cout << "\n--- Group 5: Derivative Operations ---\n" << std::endl;

    // Test 13: Derivative of linear polynomial
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly dcp = cp.derivative();

        // Derivative of x is 1
        test.expect_near(dcp(0.0), 1.0, tolerance, "Test 13a: d/dx[x] at 0");
        test.expect_near(dcp(0.5), 1.0, tolerance, "Test 13b: d/dx[x] at 0.5");
        test.expect_near(dcp(1.0), 1.0, tolerance, "Test 13c: d/dx[x] at 1");
    }

    // Test 14: Derivative of quadratic polynomial
    {
        std::vector<std::vector<double>> coeffs = {{0.375}, {0.5}, {0.125}};  // p(x) = x^2
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly dcp = cp.derivative();

        // Derivative of x^2 is 2x
        test.expect_near(dcp(0.0), 0.0, tolerance, "Test 14a: d/dx[x^2] at 0");
        test.expect_near(dcp(0.5), 1.0, tolerance, "Test 14b: d/dx[x^2] at 0.5");
        test.expect_near(dcp(1.0), 2.0, tolerance, "Test 14c: d/dx[x^2] at 1");
    }

    // Test 15: operator()(x, nu) syntax for derivatives
    {
        std::vector<std::vector<double>> coeffs = {{0.375}, {0.5}, {0.125}};  // p(x) = x^2
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.5, 0), 0.25, tolerance, "Test 15a: x^2(0.5, 0)");
        test.expect_near(cp(0.5, 1), 1.0, tolerance, "Test 15b: x^2(0.5, 1) = 2*0.5");
        test.expect_near(cp(0.5, 2), 2.0, tolerance, "Test 15c: x^2(0.5, 2) = 2");
    }

    // ============================================================
    // Group 6: Antiderivative and Integration
    // ============================================================
    std::cout << "\n--- Group 6: Antiderivative and Integration ---\n" << std::endl;

    // Test 16: Antiderivative of constant
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly icp = cp.antiderivative();

        // Antiderivative of 2 is 2x (starting at 0)
        test.expect_near(icp(0), 0.0, tolerance, "Test 16a: int[2] at 0");
        test.expect_near(icp(1), 2.0, tolerance, "Test 16b: int[2] at 1");
        test.expect_near(icp(0.5), 1.0, tolerance, "Test 16c: int[2] at 0.5");
    }

    // Test 17: Integration (definite integral)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        // Integral of x from 0 to 1 is 0.5
        double integral = cp.integrate(0, 1);
        test.expect_near(integral, 0.5, tolerance, "Test 17a: int_0^1 x dx");

        // Integral of x from 0 to 0.5 is 0.125
        test.expect_near(cp.integrate(0, 0.5), 0.125, tolerance, "Test 17b: int_0^0.5 x dx");
    }

    // Test 18: Negative derivative order = antiderivative
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        CPoly icp = cp.derivative(-1);  // -1 means antiderivative
        test.expect_near(icp(1), 2.0, tolerance, "Test 18: derivative(-1) = antiderivative");
    }

    // ============================================================
    // Group 7: from_derivatives (Hermite Interpolation)
    // ============================================================
    std::cout << "\n--- Group 7: from_derivatives ---\n" << std::endl;

    // Test 19: Simple Hermite cubic from_derivatives
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};  // f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1

        CPoly cp = CPoly::from_derivatives(xi, yi);

        // Verify endpoint values
        test.expect_near(cp(0), 0.0, tolerance, "Test 19a: Hermite f(0)");
        test.expect_near(cp(1), 1.0, tolerance, "Test 19b: Hermite f(1)");

        // Verify derivatives at endpoints
        test.expect_near(cp(0, 1), 1.0, tolerance, "Test 19c: Hermite f'(0)");
        test.expect_near(cp(1, 1), -1.0, tolerance, "Test 19d: Hermite f'(1)");
    }

    // Test 20: from_derivatives with second derivatives
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, -1, 0}};  // f(0)=0, f'(0)=1, f''(0)=0, f(1)=1, f'(1)=-1, f''(1)=0

        CPoly cp = CPoly::from_derivatives(xi, yi);

        test.expect_near(cp(0), 0.0, tolerance, "Test 20a: Quintic f(0)");
        test.expect_near(cp(1), 1.0, tolerance, "Test 20b: Quintic f(1)");
        test.expect_near(cp(0, 1), 1.0, tolerance, "Test 20c: Quintic f'(0)");
        test.expect_near(cp(1, 1), -1.0, tolerance, "Test 20d: Quintic f'(1)");
        test.expect_near(cp(0, 2), 0.0, tolerance, "Test 20e: Quintic f''(0)");
        test.expect_near(cp(1, 2), 0.0, tolerance, "Test 20f: Quintic f''(1)");
    }

    // Test 21: from_derivatives multi-interval
    {
        std::vector<double> xi = {0, 1, 2};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, 0}, {0, -1}};

        CPoly cp = CPoly::from_derivatives(xi, yi);

        test.expect_near(cp(0), 0.0, tolerance, "Test 21a: Multi-interval f(0)");
        test.expect_near(cp(1), 1.0, tolerance, "Test 21b: Multi-interval f(1)");
        test.expect_near(cp(2), 0.0, tolerance, "Test 21c: Multi-interval f(2)");
    }

    // ============================================================
    // Group 8: Basis Conversions
    // ============================================================
    std::cout << "\n--- Group 8: Basis Conversions ---\n" << std::endl;

    // Test 22: from_power_basis (constant)
    {
        std::vector<std::vector<double>> power_coeffs = {{3}};  // p(x) = 3
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(cp(0.5), 3.0, tolerance, "Test 22: from_power_basis constant");
    }

    // Test 23: from_power_basis (linear)
    {
        // Power basis: p(x) = 1 + 2*(x-0) = 1 + 2x on [0,1]
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}};
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(cp(0), 1.0, tolerance, "Test 23a: from_power_basis linear f(0)");
        test.expect_near(cp(0.5), 2.0, tolerance, "Test 23b: from_power_basis linear f(0.5)");
        test.expect_near(cp(1), 3.0, tolerance, "Test 23c: from_power_basis linear f(1)");
    }

    // Test 24: from_power_basis (quadratic)
    {
        // Power basis: p(x) = 1 + 0*x + 1*x^2 = 1 + x^2 on [0,1]
        std::vector<std::vector<double>> power_coeffs = {{1}, {0}, {1}};
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(cp(0), 1.0, tolerance, "Test 24a: from_power_basis quadratic f(0)");
        test.expect_near(cp(0.5), 1.25, tolerance, "Test 24b: from_power_basis quadratic f(0.5)");
        test.expect_near(cp(1), 2.0, tolerance, "Test 24c: from_power_basis quadratic f(1)");
    }

    // Test 25: to_power_basis round-trip
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        auto recovered = cp.to_power_basis();

        test.expect_near(recovered[0][0], 1.0, tolerance, "Test 25a: Round-trip c0");
        test.expect_near(recovered[1][0], 2.0, tolerance, "Test 25b: Round-trip c1");
        test.expect_near(recovered[2][0], 3.0, tolerance, "Test 25c: Round-trip c2");
    }

    // ============================================================
    // Group 9: Extend Operation
    // ============================================================
    std::cout << "\n--- Group 9: Extend Operation ---\n" << std::endl;

    // Test 26: Extend to the right
    {
        std::vector<std::vector<double>> coeffs1 = {{1}};  // p(x) = 1 on [0,1]
        std::vector<double> breaks1 = {0, 1};
        CPoly cp1(coeffs1, breaks1);

        std::vector<std::vector<double>> coeffs2 = {{2}};  // p(x) = 2 on [1,2]
        std::vector<double> breaks2 = {1, 2};

        CPoly extended = cp1.extend(coeffs2, breaks2, true);

        test.expect_near(extended(0.5), 1.0, tolerance, "Test 26a: Extended f(0.5)");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 26b: Extended f(1.5)");
        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 26c: Extended num_intervals");
    }

    // Test 27: Extend to the left
    {
        std::vector<std::vector<double>> coeffs1 = {{2}};  // p(x) = 2 on [1,2]
        std::vector<double> breaks1 = {1, 2};
        CPoly cp1(coeffs1, breaks1);

        std::vector<std::vector<double>> coeffs2 = {{1}};  // p(x) = 1 on [0,1]
        std::vector<double> breaks2 = {0, 1};

        CPoly extended = cp1.extend(coeffs2, breaks2, false);

        test.expect_near(extended(0.5), 1.0, tolerance, "Test 27a: Left-extended f(0.5)");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 27b: Left-extended f(1.5)");
    }

    // ============================================================
    // Group 10: Root Finding
    // ============================================================
    std::cout << "\n--- Group 10: Root Finding ---\n" << std::endl;

    // Test 28: Linear root finding
    {
        // p(x) = x - 0.5 on [0,1] has root at x = 0.5
        // In Chebyshev: need c_0 + c_1*(2x-1) = x - 0.5
        // c_0 - c_1 + 2*c_1*x = x - 0.5
        // 2*c_1 = 1 => c_1 = 0.5
        // c_0 - c_1 = -0.5 => c_0 = 0
        std::vector<std::vector<double>> coeffs = {{0}, {0.5}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        auto r = cp.roots();
        test.expect_eq(r.size(), 1ul, "Test 28a: Linear root count");
        if (r.size() >= 1) {
            test.expect_near(r[0], 0.5, tolerance, "Test 28b: Linear root value");
        }
    }

    // Test 29: Quadratic with two roots
    {
        // p(x) = (x - 0.25)(x - 0.75) = x^2 - x + 0.1875 on [0,1]
        // Convert to Chebyshev using from_power_basis
        std::vector<std::vector<double>> power_coeffs = {{0.1875}, {-1.0}, {1.0}};
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        auto r = cp.roots();
        test.expect_eq(r.size(), 2ul, "Test 29a: Quadratic root count");
        if (r.size() >= 2) {
            // Note: Bisection root-finding has slightly looser tolerance than evaluation
            // due to numerical precision limits in the iterative refinement process.
            const double root_tolerance = 1e-10;
            test.expect_near(r[0], 0.25, root_tolerance, "Test 29b: Quadratic root 1");
            test.expect_near(r[1], 0.75, root_tolerance, "Test 29c: Quadratic root 2");
        }
    }

    // Test 30: No roots
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        auto r = cp.roots(true, false);  // Don't extrapolate
        test.expect_eq(r.size(), 0ul, "Test 30: No roots for positive constant");
    }

    // ============================================================
    // Group 11: Accessors
    // ============================================================
    std::cout << "\n--- Group 11: Accessors ---\n" << std::endl;

    // Test 31: Property accessors
    {
        std::vector<std::vector<double>> coeffs = {{1, 3}, {2, 4}};  // degree 1, 2 intervals
        std::vector<double> breaks = {0, 1, 2};  // 2 intervals
        CPoly cp(coeffs, breaks);

        test.expect_eq(static_cast<size_t>(cp.degree()), 1ul, "Test 31a: degree()");
        test.expect_eq(static_cast<size_t>(cp.num_intervals()), 2ul, "Test 31b: num_intervals()");
        test.expect_eq(cp.c().size(), 2ul, "Test 31c: c() alias");
        test.expect_eq(cp.x().size(), 3ul, "Test 31d: x() alias");
        test.expect_true(cp.is_ascending(), "Test 31e: is_ascending()");
    }

    // ============================================================
    // Group 12: Descending Breakpoints
    // ============================================================
    std::cout << "\n--- Group 12: Descending Breakpoints ---\n" << std::endl;

    // Test 32: Descending breakpoints construction
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5, 1 interval
        std::vector<double> breaks = {1, 0};  // Descending, 1 interval
        CPoly cp(coeffs, breaks);

        test.expect_true(!cp.is_ascending(), "Test 32a: Descending detected");
        test.expect_near(cp(0.5), 5.0, tolerance, "Test 32b: Descending evaluation");
        test.expect_near(cp(0.25), 5.0, tolerance, "Test 32c: Descending evaluation 2");
    }

    // ============================================================
    // Group 13: Independent Verification
    // ============================================================
    std::cout << "\n--- Group 13: Independent Verification ---\n" << std::endl;

    // Test 33: Verify derivative with finite differences
    {
        // Use from_power_basis for known polynomial: p(x) = x^3
        std::vector<std::vector<double>> power_coeffs = {{0}, {0}, {0}, {1}};
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        double x = 0.5;
        double analytical = cp(x, 1);  // Should be 3*x^2 = 0.75
        double numerical = finite_diff_derivative(cp, x);

        test.expect_near(analytical, 0.75, tolerance, "Test 33a: Analytical derivative at 0.5");
        test.expect_near(numerical, 0.75, 1e-5, "Test 33b: Numerical derivative at 0.5");
    }

    // Test 34: Verify antiderivative with numerical integration
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        CPoly antideriv = cp.antiderivative();

        double analytical = antideriv(0.5) - antideriv(0);  // Should be 0.125
        double numerical = numerical_integrate(cp, 0, 0.5);

        test.expect_near(analytical, 0.125, tolerance, "Test 34a: Analytical integral");
        test.expect_near(numerical, 0.125, 1e-6, "Test 34b: Numerical integral");
    }

    // ============================================================
    // Group 14: Property-Based Tests
    // ============================================================
    std::cout << "\n--- Group 14: Property-Based Tests ---\n" << std::endl;

    // Test 35: Integral additivity
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {-1}};  // 1 + 2x - x^2
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        double int_0_1 = cp.integrate(0, 1);
        double int_0_half = cp.integrate(0, 0.5);
        double int_half_1 = cp.integrate(0.5, 1);

        test.expect_near(int_0_1, int_0_half + int_half_1, tolerance,
                        "Test 35: Integral additivity");
    }

    // Test 36: Derivative-integral relationship
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        CPoly antideriv = cp.antiderivative();
        CPoly recovered = antideriv.derivative();

        // Evaluate at several points
        bool all_close = true;
        for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            if (std::abs(cp(x) - recovered(x)) > tolerance) {
                all_close = false;
                break;
            }
        }
        test.expect_true(all_close, "Test 36: d/dx[antiderivative] = original");
    }

    // Test 37: Integral reversal
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        double int_ab = cp.integrate(0.2, 0.8);
        double int_ba = cp.integrate(0.8, 0.2);

        test.expect_near(int_ab, -int_ba, tolerance, "Test 37: Integral reversal");
    }

    // ============================================================
    // Group 15: High-Degree Polynomials
    // ============================================================
    std::cout << "\n--- Group 15: High-Degree Polynomials ---\n" << std::endl;

    // Test 38: High-degree polynomial (degree 10)
    {
        // Create polynomial x^10 using from_power_basis
        std::vector<std::vector<double>> power_coeffs(11, std::vector<double>(1, 0.0));
        power_coeffs[10][0] = 1.0;  // x^10
        std::vector<double> breaks = {0, 1};

        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(cp(0), 0.0, tolerance, "Test 38a: x^10 at 0");
        test.expect_near(cp(1), 1.0, tolerance, "Test 38b: x^10 at 1");
        test.expect_near(cp(0.5), std::pow(0.5, 10), tolerance, "Test 38c: x^10 at 0.5");
    }

    // ============================================================
    // Group 16: NaN and Infinity Handling
    // ============================================================
    std::cout << "\n--- Group 16: NaN and Infinity Handling ---\n" << std::endl;

    // Test 39: NaN input
    {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_true(std::isnan(cp(std::numeric_limits<double>::quiet_NaN())),
                        "Test 39: NaN input gives NaN output");
    }

    // Test 40: Infinity input with NoExtrapolate
    {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::NoExtrapolate);

        test.expect_true(std::isnan(cp(std::numeric_limits<double>::infinity())),
                        "Test 40: Inf input with NoExtrapolate gives NaN");
    }

    // ============================================================
    // Group 17: Orders Parameter in from_derivatives
    // ============================================================
    std::cout << "\n--- Group 17: Orders Parameter ---\n" << std::endl;

    // Test 41: from_derivatives with limited orders
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 5}, {1, -1, 3}};  // More derivatives than needed
        std::vector<int> orders = {0};  // Only use function values

        CPoly cp = CPoly::from_derivatives(xi, yi, orders);

        // With only function values, should be linear
        test.expect_near(cp(0), 0.0, tolerance, "Test 41a: Limited orders f(0)");
        test.expect_near(cp(1), 1.0, tolerance, "Test 41b: Limited orders f(1)");
        test.expect_eq(static_cast<size_t>(cp.degree()), 1ul, "Test 41c: Limited orders degree");
    }

    // ============================================================
    // Group 18: Move Semantics
    // ============================================================
    std::cout << "\n--- Group 18: Move Semantics ---\n" << std::endl;

    // Test 42: Move constructor
    {
        std::vector<std::vector<double>> coeffs = {{5}};
        std::vector<double> breaks = {0, 1};
        CPoly cp1(coeffs, breaks);
        CPoly cp2(std::move(cp1));

        test.expect_near(cp2(0.5), 5.0, tolerance, "Test 42a: Move constructor preserves data");

        // Verify moved-from object is in a valid (empty) state
        test.expect_true(cp1.coefficients().empty(), "Test 42b: Moved-from coefficients empty");
        test.expect_true(cp1.breakpoints().empty(), "Test 42c: Moved-from breakpoints empty");
    }

    // ============================================================
    // Group 19: Controlled Extrapolation
    // ============================================================
    std::cout << "\n--- Group 19: Controlled Extrapolation ---\n" << std::endl;

    // Test 43: Constant extrapolation (order 0)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Extrapolate, 0, 0);  // Constant extrapolation

        // At x < 0, should use value at x=0, which is 0
        test.expect_near(cp(-0.5), 0.0, tolerance, "Test 43a: Constant extrapolation left");
        // At x > 1, should use value at x=1, which is 1
        test.expect_near(cp(1.5), 1.0, tolerance, "Test 43b: Constant extrapolation right");
    }

    // Test 44: Linear extrapolation (order 1)
    {
        std::vector<std::vector<double>> coeffs = {{0.375}, {0.5}, {0.125}};  // p(x) = x^2 on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Extrapolate, 1, 1);  // Linear extrapolation

        // At x=0: f(0)=0, f'(0)=0, so linear is just 0
        test.expect_near(cp(-0.5), 0.0, tolerance, "Test 44a: Linear extrapolation left");
        // At x=1: f(1)=1, f'(1)=2, so linear is 1 + 2*(x-1)
        test.expect_near(cp(1.5), 2.0, tolerance, "Test 44b: Linear extrapolation right");
    }

    // ============================================================
    // Group 20: Scipy/Numpy Reference Verification
    // ============================================================
    // Values verified against numpy.polynomial.chebyshev
    std::cout << "\n--- Group 20: Scipy/Numpy Reference Verification ---\n" << std::endl;

    // Test 45: Constant verification (scipy reference)
    // numpy: C.chebval(s_pts, [5.0]) where s = 2*x - 1
    {
        std::vector<std::vector<double>> coeffs = {{5.0}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        // scipy values: [5.0, 5.0, 5.0, 5.0, 5.0] at x=[0, 0.25, 0.5, 0.75, 1.0]
        test.expect_near(cp(0.0), 5.0, tolerance, "Test 45a: scipy constant at 0");
        test.expect_near(cp(0.25), 5.0, tolerance, "Test 45b: scipy constant at 0.25");
        test.expect_near(cp(0.5), 5.0, tolerance, "Test 45c: scipy constant at 0.5");
        test.expect_near(cp(0.75), 5.0, tolerance, "Test 45d: scipy constant at 0.75");
        test.expect_near(cp(1.0), 5.0, tolerance, "Test 45e: scipy constant at 1.0");
    }

    // Test 46: Linear verification (scipy reference)
    // numpy: coeffs=[0.5, 0.5] gives values [0.0, 0.25, 0.5, 0.75, 1.0]
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 0.0, tolerance, "Test 46a: scipy linear at 0");
        test.expect_near(cp(0.25), 0.25, tolerance, "Test 46b: scipy linear at 0.25");
        test.expect_near(cp(0.5), 0.5, tolerance, "Test 46c: scipy linear at 0.5");
        test.expect_near(cp(0.75), 0.75, tolerance, "Test 46d: scipy linear at 0.75");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 46e: scipy linear at 1.0");
    }

    // Test 47: Quadratic verification (scipy reference)
    // numpy: poly2cheb([0.25, 0.5, 0.25]) = [0.375, 0.5, 0.125]
    // values at x=[0, 0.25, 0.5, 0.75, 1.0]: [0.0, 0.0625, 0.25, 0.5625, 1.0]
    {
        std::vector<std::vector<double>> coeffs = {{0.375}, {0.5}, {0.125}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 0.0, tolerance, "Test 47a: scipy quadratic at 0");
        test.expect_near(cp(0.25), 0.0625, tolerance, "Test 47b: scipy quadratic at 0.25");
        test.expect_near(cp(0.5), 0.25, tolerance, "Test 47c: scipy quadratic at 0.5");
        test.expect_near(cp(0.75), 0.5625, tolerance, "Test 47d: scipy quadratic at 0.75");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 47e: scipy quadratic at 1.0");
    }

    // Test 48: Cubic verification (scipy reference)
    // numpy: poly2cheb([0.125, 0.375, 0.375, 0.125]) = [0.3125, 0.46875, 0.1875, 0.03125]
    // values at x=[0, 0.25, 0.5, 0.75, 1.0]: [0.0, 0.015625, 0.125, 0.421875, 1.0]
    {
        std::vector<std::vector<double>> coeffs = {{0.3125}, {0.46875}, {0.1875}, {0.03125}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 0.0, tolerance, "Test 48a: scipy cubic at 0");
        test.expect_near(cp(0.25), 0.015625, tolerance, "Test 48b: scipy cubic at 0.25");
        test.expect_near(cp(0.5), 0.125, tolerance, "Test 48c: scipy cubic at 0.5");
        test.expect_near(cp(0.75), 0.421875, tolerance, "Test 48d: scipy cubic at 0.75");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 48e: scipy cubic at 1.0");
    }

    // Test 49: Derivative verification (scipy reference)
    // d/dx[x^2] = 2x, scipy derivative coeffs (unscaled): [0.5, 0.5]
    // derivative values (scaled by 2/h=2): [0.0, 0.5, 1.0, 1.5, 2.0]
    {
        std::vector<std::vector<double>> coeffs = {{0.375}, {0.5}, {0.125}};  // x^2
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly dcp = cp.derivative();

        test.expect_near(dcp(0.0), 0.0, tolerance, "Test 49a: scipy derivative at 0");
        test.expect_near(dcp(0.25), 0.5, tolerance, "Test 49b: scipy derivative at 0.25");
        test.expect_near(dcp(0.5), 1.0, tolerance, "Test 49c: scipy derivative at 0.5");
        test.expect_near(dcp(0.75), 1.5, tolerance, "Test 49d: scipy derivative at 0.75");
        test.expect_near(dcp(1.0), 2.0, tolerance, "Test 49e: scipy derivative at 1.0");
    }

    // Test 50: Second derivative verification (scipy reference)
    // d^2/dx^2[x^3] = 6x
    {
        std::vector<std::vector<double>> coeffs = {{0.3125}, {0.46875}, {0.1875}, {0.03125}};  // x^3
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly d2cp = cp.derivative(2);

        test.expect_near(d2cp(0.0), 0.0, tolerance, "Test 50a: scipy 2nd derivative at 0");
        test.expect_near(d2cp(0.25), 1.5, tolerance, "Test 50b: scipy 2nd derivative at 0.25");
        test.expect_near(d2cp(0.5), 3.0, tolerance, "Test 50c: scipy 2nd derivative at 0.5");
        test.expect_near(d2cp(0.75), 4.5, tolerance, "Test 50d: scipy 2nd derivative at 0.75");
        test.expect_near(d2cp(1.0), 6.0, tolerance, "Test 50e: scipy 2nd derivative at 1.0");
    }

    // Test 51: Antiderivative verification (scipy reference)
    // antiderivative of 2 is 2x with F(0)=0
    // scipy values: [0.0, 0.5, 1.0, 1.5, 2.0]
    {
        std::vector<std::vector<double>> coeffs = {{2.0}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly icp = cp.antiderivative();

        test.expect_near(icp(0.0), 0.0, tolerance, "Test 51a: scipy antiderivative at 0");
        test.expect_near(icp(0.25), 0.5, tolerance, "Test 51b: scipy antiderivative at 0.25");
        test.expect_near(icp(0.5), 1.0, tolerance, "Test 51c: scipy antiderivative at 0.5");
        test.expect_near(icp(0.75), 1.5, tolerance, "Test 51d: scipy antiderivative at 0.75");
        test.expect_near(icp(1.0), 2.0, tolerance, "Test 51e: scipy antiderivative at 1.0");
    }

    // Test 52: Integration verification (scipy reference)
    // integral of x from 0 to 1 = 0.5
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp.integrate(0, 1), 0.5, tolerance, "Test 52: scipy integration of x");
    }

    // Test 53: Power to Chebyshev verification (scipy reference)
    // Power basis: [1, 2, 3] for 1+2x+3x^2 on [0,1]
    // Mapped power (on s): [2.75, 2.5, 0.75]
    // Chebyshev coeffs: [3.125, 2.5, 0.375]
    // Values: [1.0, 1.6875, 2.75, 4.1875, 6.0]
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(cp(0.0), 1.0, tolerance, "Test 53a: scipy power_basis at 0");
        test.expect_near(cp(0.25), 1.6875, tolerance, "Test 53b: scipy power_basis at 0.25");
        test.expect_near(cp(0.5), 2.75, tolerance, "Test 53c: scipy power_basis at 0.5");
        test.expect_near(cp(0.75), 4.1875, tolerance, "Test 53d: scipy power_basis at 0.75");
        test.expect_near(cp(1.0), 6.0, tolerance, "Test 53e: scipy power_basis at 1.0");
    }

    // Test 54: Chebyshev to Power verification (scipy reference)
    // Chebyshev: [1, 2, 3] -> Power: [-2, 2, 6]
    {
        std::vector<std::vector<double>> coeffs = {{1}, {2}, {3}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        auto power = cp.to_power_basis();
        // Note: power coeffs are in terms of (x-a) on each interval
        // For interval [0,1]: should match scipy's cheb2poly for s-domain
        // Then transformed back to x-domain
        // We verify via evaluation instead of exact coefficients
        double x = 0.5;
        double s = 2*x - 1;
        // scipy: cheb2poly([1,2,3]) = [-2, 2, 6] in s-domain
        double expected_s = -2 + 2*s + 6*s*s;
        test.expect_near(cp(x), expected_s, tolerance, "Test 54: cheb2poly evaluation match");
    }

    // Test 55: Different interval [2, 5] verification (scipy reference)
    // f(x) = x on [2,5], Chebyshev coeffs: [3.5, 1.5]
    // Values at [2.0, 2.75, 3.5, 4.25, 5.0]: [2.0, 2.75, 3.5, 4.25, 5.0]
    {
        std::vector<std::vector<double>> coeffs = {{3.5}, {1.5}};
        std::vector<double> breaks = {2.0, 5.0};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(2.0), 2.0, tolerance, "Test 55a: scipy [2,5] at 2.0");
        test.expect_near(cp(2.75), 2.75, tolerance, "Test 55b: scipy [2,5] at 2.75");
        test.expect_near(cp(3.5), 3.5, tolerance, "Test 55c: scipy [2,5] at 3.5");
        test.expect_near(cp(4.25), 4.25, tolerance, "Test 55d: scipy [2,5] at 4.25");
        test.expect_near(cp(5.0), 5.0, tolerance, "Test 55e: scipy [2,5] at 5.0");
    }

    // Test 56: Clenshaw algorithm verification
    // Verify with degree 3 poly: coeffs = [1, 2, 3, 4]
    // At s=0.5, numpy gives: C.chebval(0.5, [1, 2, 3, 4]) = -3.5
    {
        std::vector<std::vector<double>> coeffs = {{1}, {2}, {3}, {4}};
        std::vector<double> breaks = {0, 1};  // s = 2x - 1, so x=0.75 gives s=0.5
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.75), -3.5, tolerance, "Test 56: Clenshaw at s=0.5");
    }

    // Test 57: Derivative formula verification
    // coeffs = [1, 2, 3, 4, 5], derivative = [14, 52, 24, 40]
    // Verify by evaluation
    {
        std::vector<std::vector<double>> coeffs = {{1}, {2}, {3}, {4}, {5}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly dcp = cp.derivative();

        // Verify derivative degree is n-1
        test.expect_eq(static_cast<size_t>(dcp.degree()), 3ul, "Test 57a: derivative degree");

        // Verify with finite differences
        double x = 0.5;
        double fd = finite_diff_derivative(cp, x);
        test.expect_near(dcp(x), fd, 1e-5, "Test 57b: derivative matches finite diff");
    }

    // Test 58: Antiderivative formula verification
    // coeffs = [1, 2, 3, 4], antiderivative should increase degree by 1
    {
        std::vector<std::vector<double>> coeffs = {{1}, {2}, {3}, {4}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly icp = cp.antiderivative();

        test.expect_eq(static_cast<size_t>(icp.degree()), 4ul, "Test 58a: antiderivative degree");

        // Verify d/dx[antiderivative] = original
        CPoly recovered = icp.derivative();
        for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            test.expect_near(recovered(x), cp(x), tolerance,
                           "Test 58b: d/dx[antiderivative] at " + std::to_string(x));
        }
    }

    // Test 59: Multi-interval scipy verification
    // Two intervals with different constants
    {
        std::vector<std::vector<double>> coeffs = {{1.0, 3.0}};  // 1 on [0,1], 3 on [1,2]
        std::vector<double> breaks = {0, 1, 2};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.5), 1.0, tolerance, "Test 59a: multi-interval first");
        test.expect_near(cp(1.5), 3.0, tolerance, "Test 59b: multi-interval second");
    }

    // Test 60: Multi-interval linear verification
    {
        // f(x) = x on [0,1], f(x) = 2-x on [1,2]
        // Chebyshev for x on [0,1]: [0.5, 0.5]
        // Chebyshev for 2-x on [1,2]: first need to express in s-domain
        // x on [1,2]: s = 2x - 3, x = (s+3)/2
        // 2 - x = 2 - (s+3)/2 = 0.5 - 0.5*s = [0.5, -0.5]
        std::vector<std::vector<double>> coeffs = {{0.5, 0.5}, {0.5, -0.5}};
        std::vector<double> breaks = {0, 1, 2};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.5), 0.5, tolerance, "Test 60a: multi-linear at 0.5");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 60b: multi-linear at 1.0");
        test.expect_near(cp(1.5), 0.5, tolerance, "Test 60c: multi-linear at 1.5");
    }

    // ============================================================
    // Group 21: Comprehensive Coefficient Tests
    // ============================================================
    std::cout << "\n--- Group 21: Comprehensive Coefficient Tests ---\n" << std::endl;

    // Test 61: Verify from_power_basis coefficients match scipy
    // Power: [1, 2, 3] on [0,1]
    // Expected Chebyshev: [3.125, 2.5, 0.375]
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        auto cheb_coeffs = cp.coefficients();
        test.expect_near(cheb_coeffs[0][0], 3.125, tolerance, "Test 61a: from_power c0");
        test.expect_near(cheb_coeffs[1][0], 2.5, tolerance, "Test 61b: from_power c1");
        test.expect_near(cheb_coeffs[2][0], 0.375, tolerance, "Test 61c: from_power c2");
    }

    // Test 62: Derivative coefficient verification
    // Original: [0.375, 0.5, 0.125] (x^2)
    // Derivative coeffs (unscaled in s-domain): [0.5, 0.5]
    // After scaling by 2/h = 2, we get [1.0, 1.0] which represents 2x
    {
        std::vector<std::vector<double>> coeffs = {{0.375}, {0.5}, {0.125}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly dcp = cp.derivative();

        auto deriv_coeffs = dcp.coefficients();
        // Verify by evaluation that it matches 2x
        test.expect_near(dcp(0.0), 0.0, tolerance, "Test 62a: deriv eval at 0");
        test.expect_near(dcp(0.5), 1.0, tolerance, "Test 62b: deriv eval at 0.5");
        test.expect_near(dcp(1.0), 2.0, tolerance, "Test 62c: deriv eval at 1");
    }

    // Test 63: Antiderivative coefficient verification
    // Original: [2] (constant 2)
    // Antiderivative should give 2x
    {
        std::vector<std::vector<double>> coeffs = {{2}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);
        CPoly icp = cp.antiderivative();

        // Verify by evaluation
        test.expect_near(icp(0), 0.0, tolerance, "Test 63a: antideriv eval at 0");
        test.expect_near(icp(0.5), 1.0, tolerance, "Test 63b: antideriv eval at 0.5");
        test.expect_near(icp(1), 2.0, tolerance, "Test 63c: antideriv eval at 1");
    }

    // Test 64: Round-trip coefficient verification
    // Power -> Chebyshev -> Power
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {-2}, {3}, {-4}};
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);
        auto recovered = cp.to_power_basis();

        test.expect_near(recovered[0][0], 1.0, tolerance, "Test 64a: round-trip c0");
        test.expect_near(recovered[1][0], -2.0, tolerance, "Test 64b: round-trip c1");
        test.expect_near(recovered[2][0], 3.0, tolerance, "Test 64c: round-trip c2");
        test.expect_near(recovered[3][0], -4.0, tolerance, "Test 64d: round-trip c3");
    }

    // ============================================================
    // Group 22: Copy Constructor and Accessor Tests
    // ============================================================
    std::cout << "\n--- Group 22: Copy Constructor and Accessor Tests ---\n" << std::endl;

    // Test 65: Copy constructor
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        CPoly cp1(coeffs, breaks);
        CPoly cp2(cp1);  // Copy

        test.expect_near(cp2(0.5), cp1(0.5), tolerance, "Test 65a: Copy constructor preserves data");
        test.expect_near(cp2(0.25), 0.25, tolerance, "Test 65b: Copy constructor evaluation");

        // Verify they are independent (modifying cp2 via derivative doesn't affect cp1)
        CPoly cp2_deriv = cp2.derivative();
        test.expect_near(cp1(0.5), 0.5, tolerance, "Test 65c: Original unchanged after copy derivative");
    }

    // Test 66: Accessor methods
    {
        CPoly cp({{5}}, {0, 1}, ExtrapolateMode::Periodic, 1, 2);

        test.expect_true(cp.extrapolate() == ExtrapolateMode::Periodic, "Test 66a: extrapolate() getter");
        test.expect_eq(static_cast<size_t>(cp.extrapolate_order_left()), 1ul, "Test 66b: extrapolate_order_left() getter");
        test.expect_eq(static_cast<size_t>(cp.extrapolate_order_right()), 2ul, "Test 66c: extrapolate_order_right() getter");
    }

    // Test 67: Default extrapolation order accessors
    {
        CPoly cp({{5}}, {0, 1});  // Default extrapolation orders

        test.expect_true(cp.extrapolate() == ExtrapolateMode::Extrapolate, "Test 67a: default extrapolate mode");
        test.expect_true(cp.extrapolate_order_left() == -1, "Test 67b: default left order is -1");
        test.expect_true(cp.extrapolate_order_right() == -1, "Test 67c: default right order is -1");
    }

    // ============================================================
    // Group 23: Extended Root Finding Tests
    // ============================================================
    std::cout << "\n--- Group 23: Extended Root Finding Tests ---\n" << std::endl;

    // Test 68: Roots with discontinuity=false
    {
        // Two intervals with different constants, discontinuity at x=1
        std::vector<std::vector<double>> coeffs = {{-1, 1}};  // -1 on [0,1], +1 on [1,2]
        std::vector<double> breaks = {0, 1, 2};
        CPoly cp(coeffs, breaks);

        auto r_with = cp.roots(true, false);  // Include discontinuity
        auto r_without = cp.roots(false, false);  // Exclude discontinuity

        test.expect_eq(r_with.size(), 1ul, "Test 68a: Roots with discontinuity finds boundary root");
        test.expect_eq(r_without.size(), 0ul, "Test 68b: Roots without discontinuity excludes boundary");
    }

    // Test 69: Roots with extrapolate=false
    {
        // p(x) = x - 2 on [0,1] has root at x=2, outside domain
        std::vector<std::vector<double>> coeffs = {{-1.5}, {0.5}};  // -1.5 + 0.5*(2x-1) = x - 2
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        auto r_extrap = cp.roots(true, true);  // With extrapolation
        auto r_no_extrap = cp.roots(true, false);  // No extrapolation

        test.expect_eq(r_extrap.size(), 1ul, "Test 69a: Roots with extrapolation finds root");
        if (r_extrap.size() >= 1) {
            test.expect_near(r_extrap[0], 2.0, 1e-10, "Test 69b: Extrapolated root value");
        }
        test.expect_eq(r_no_extrap.size(), 0ul, "Test 69c: Roots without extrapolation excludes");
    }

    // Test 70: Roots at exact breakpoint
    {
        // p(x) = x on [0,1] has root at x=0 (breakpoint)
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        auto r = cp.roots(true, false);
        test.expect_eq(r.size(), 1ul, "Test 70a: Root at breakpoint count");
        if (!r.empty()) {
            test.expect_near(r[0], 0.0, 1e-10, "Test 70b: Root at left breakpoint");
        }
    }

    // Test 71: Multi-interval with multiple roots
    {
        // Two linear functions crossing zero in each interval
        // Interval [0,1]: p(x) = x - 0.5 (root at 0.5)
        // Interval [1,2]: p(x) = x - 1.5 (root at 1.5)
        // In Chebyshev: c_0 + c_1*(2x-a-b)/(b-a) where [a,b] is interval
        // For x-0.5 on [0,1]: x = 0.5 + 0.5*s, so x - 0.5 = 0.5*s = 0.5*T_1
        // coeffs = [0, 0.5]
        // For x-1.5 on [1,2]: x = 1.5 + 0.5*s, so x - 1.5 = 0.5*s = 0.5*T_1
        // coeffs = [0, 0.5]
        std::vector<std::vector<double>> coeffs = {{0, 0}, {0.5, 0.5}};
        std::vector<double> breaks = {0, 1, 2};
        CPoly cp(coeffs, breaks);

        // Note: With discontinuity=true, it also finds a root at boundary x=1
        // where the polynomial is continuous (both sides equal 0.5 there, not zero)
        // Actually p(1) = 0.5 from left, p(1) = -0.5 from right - sign change at boundary!
        // So roots are: 0.5, 1.0 (boundary with sign change), 1.5
        auto r = cp.roots(true, false);
        test.expect_eq(r.size(), 3ul, "Test 71a: Multi-interval root count");
        if (r.size() >= 3) {
            test.expect_near(r[0], 0.5, 1e-10, "Test 71b: First interval root");
            test.expect_near(r[1], 1.0, 1e-10, "Test 71c: Boundary root");
            test.expect_near(r[2], 1.5, 1e-10, "Test 71d: Second interval root");
        }
    }

    // Test 72: High-degree polynomial roots
    {
        // x^3 - x = x(x-1)(x+1) has roots at -1, 0, 1
        // On [0, 1], only roots 0 and 1 are in domain
        std::vector<std::vector<double>> power_coeffs = {{0}, {-1}, {0}, {1}};  // -x + x^3
        std::vector<double> breaks = {0, 1};
        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        auto r = cp.roots(true, false);
        test.expect_eq(r.size(), 2ul, "Test 72a: Cubic root count");
        if (r.size() >= 2) {
            test.expect_near(r[0], 0.0, 1e-10, "Test 72b: Cubic root at 0");
            test.expect_near(r[1], 1.0, 1e-10, "Test 72c: Cubic root at 1");
        }
    }

    // ============================================================
    // Group 24: Periodic Mode Calculus Tests
    // ============================================================
    std::cout << "\n--- Group 24: Periodic Mode Calculus Tests ---\n" << std::endl;

    // Test 73: Periodic mode derivative
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Periodic);
        CPoly dcp = cp.derivative();

        // Derivative of x is 1, should be 1 everywhere (periodic)
        test.expect_near(dcp(0.5), 1.0, tolerance, "Test 73a: Periodic derivative at 0.5");
        test.expect_near(dcp(1.5), 1.0, tolerance, "Test 73b: Periodic derivative at 1.5 (wrapped)");
        test.expect_near(dcp(2.5), 1.0, tolerance, "Test 73c: Periodic derivative at 2.5 (wrapped)");
    }

    // Test 74: Periodic mode antiderivative
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2 (constant)
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Periodic);
        CPoly icp = cp.antiderivative();

        // Antiderivative of 2 is 2x, but with periodic wrapping
        test.expect_near(icp(0), 0.0, tolerance, "Test 74a: Periodic antiderivative at 0");
        test.expect_near(icp(0.5), 1.0, tolerance, "Test 74b: Periodic antiderivative at 0.5");
    }

    // Test 75: Periodic mode integration across boundary
    {
        std::vector<std::vector<double>> coeffs = {{1}};  // p(x) = 1 (constant)
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Periodic);

        // Integral of 1 over any interval of length L should be L
        double int_0_1 = cp.integrate(0, 1);
        double int_0_half = cp.integrate(0, 0.5);

        test.expect_near(int_0_1, 1.0, tolerance, "Test 75a: Periodic integral [0,1]");
        test.expect_near(int_0_half, 0.5, tolerance, "Test 75b: Periodic integral [0,0.5]");
    }

    // Test 75c: Periodic mode integration truly crossing the period boundary
    {
        // Use linear function f(x) = x on [0,1] with periodic mode
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Periodic);

        // Integration from 0.5 to 1.5 should cross the boundary
        // From 0.5 to 1.0: integral of x = [x^2/2] = 0.5 - 0.125 = 0.375
        // From 1.0 to 1.5: wraps to 0.0 to 0.5, integral of x = 0.125
        // Total should be 0.375 + 0.125 = 0.5
        double int_across = cp.integrate(0.5, 1.5);
        test.expect_near(int_across, 0.5, tolerance, "Test 75c: Periodic integral [0.5,1.5] crossing boundary");

        // Also verify integration spanning multiple periods
        // From 0 to 2: two full periods of integral 0.5 each = 1.0
        double int_two_periods = cp.integrate(0, 2);
        test.expect_near(int_two_periods, 1.0, tolerance, "Test 75d: Periodic integral spanning two periods");
    }

    // ============================================================
    // Group 25: Corner Cases (Extreme Values)
    // ============================================================
    std::cout << "\n--- Group 25: Corner Cases (Extreme Values) ---\n" << std::endl;

    // Test 76: High-degree polynomial (degree 20)
    {
        std::vector<std::vector<double>> power_coeffs(21, std::vector<double>(1, 0.0));
        power_coeffs[20][0] = 1.0;  // x^20
        std::vector<double> breaks = {0, 1};

        CPoly cp = CPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(cp(0), 0.0, tolerance, "Test 76a: x^20 at 0");
        test.expect_near(cp(1), 1.0, tolerance, "Test 76b: x^20 at 1");
        test.expect_near(cp(0.5), std::pow(0.5, 20), tolerance, "Test 76c: x^20 at 0.5");
    }

    // Test 77: Large coefficients
    {
        std::vector<std::vector<double>> coeffs = {{1e10}, {2e10}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        // Looser tolerance for large values
        test.expect_near(cp(0.0), 1e10 - 2e10, 1e5, "Test 77a: Large coeffs at 0");
        test.expect_near(cp(1.0), 1e10 + 2e10, 1e5, "Test 77b: Large coeffs at 1");
    }

    // Test 78: Small coefficients
    {
        std::vector<std::vector<double>> coeffs = {{1e-15}, {2e-15}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 1e-15 - 2e-15, 1e-25, "Test 78a: Small coeffs at 0");
        test.expect_near(cp(1.0), 1e-15 + 2e-15, 1e-25, "Test 78b: Small coeffs at 1");
    }

    // Test 79: Very small interval
    {
        // For x on [0, h], Chebyshev representation is x = (s+1)*h/2
        // where s in [-1, 1]. So coefficients are [h/2, h/2]
        double h = 1e-10;
        std::vector<std::vector<double>> coeffs = {{h/2}, {h/2}};  // x on [0, h]
        std::vector<double> breaks = {0, h};
        CPoly cp(coeffs, breaks);

        double mid = h / 2;  // midpoint
        test.expect_near(cp(mid), mid, 1e-20, "Test 79: Small interval evaluation");
    }

    // Test 80: Far extrapolation
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks, ExtrapolateMode::Extrapolate);

        // Extrapolate 100x the interval width
        test.expect_near(cp(100.0), 100.0, tolerance, "Test 80a: Far extrapolation right");
        test.expect_near(cp(-100.0), -100.0, tolerance, "Test 80b: Far extrapolation left");
    }

    // Test 81: Near-boundary evaluation
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        double eps = std::numeric_limits<double>::epsilon();
        test.expect_near(cp(eps), eps, 1e-15, "Test 81a: Evaluation at epsilon");
        test.expect_near(cp(1.0 - eps), 1.0 - eps, tolerance, "Test 81b: Evaluation at 1-epsilon");
    }

    // ============================================================
    // Group 26: Property-Based Tests
    // ============================================================
    std::cout << "\n--- Group 26: Property-Based Tests ---\n" << std::endl;

    // Test 82: Property - integrate(a,b) == antiderivative(b) - antiderivative(a)
    {
        std::vector<std::pair<std::string, CPoly>> test_polys;
        test_polys.push_back({"constant", CPoly({{3}}, {0, 1})});
        test_polys.push_back({"linear", CPoly({{0.5}, {0.5}}, {0, 1})});
        test_polys.push_back({"quadratic", CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1})});

        bool all_pass = true;
        for (const auto& [name, cp] : test_polys) {
            CPoly anti = cp.antiderivative();
            for (double a : {0.0, 0.2, 0.5}) {
                for (double b : {0.5, 0.8, 1.0}) {
                    if (a >= b) continue;
                    double integral = cp.integrate(a, b);
                    double anti_diff = anti(b) - anti(a);
                    if (std::abs(integral - anti_diff) > tolerance) {
                        test.fail("property: integrate==antideriv_diff for " + name);
                        all_pass = false;
                    }
                }
            }
        }
        if (all_pass) {
            test.pass("Test 82: integrate(a,b) == antiderivative(b) - antiderivative(a)");
        }
    }

    // Test 83: Property - derivative(antiderivative(f)) == f
    {
        std::vector<std::pair<std::string, CPoly>> test_polys;
        test_polys.push_back({"constant", CPoly({{5}}, {0, 1})});
        test_polys.push_back({"linear", CPoly({{0.5}, {0.5}}, {0, 1})});
        test_polys.push_back({"quadratic", CPoly::from_power_basis({{1}, {-2}, {3}}, {0, 1})});

        bool all_pass = true;
        for (const auto& [name, cp] : test_polys) {
            CPoly anti = cp.antiderivative();
            CPoly recovered = anti.derivative();
            for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                if (std::abs(cp(x) - recovered(x)) > tolerance) {
                    test.fail("property: deriv(antideriv)==original for " + name + " at " + std::to_string(x));
                    all_pass = false;
                }
            }
        }
        if (all_pass) {
            test.pass("Test 83: derivative(antiderivative(f)) == f");
        }
    }

    // Test 84: Property - derivative composition d^m(d^n(f)) == d^(m+n)(f)
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}, {4}, {5}}, {0, 1});  // degree 4

        bool all_pass = true;
        for (int m = 1; m <= 2; ++m) {
            for (int n = 1; n <= 2; ++n) {
                CPoly d_m_n = cp.derivative(m).derivative(n);
                CPoly d_mn = cp.derivative(m + n);
                for (double x : {0.2, 0.5, 0.8}) {
                    if (std::abs(d_m_n(x) - d_mn(x)) > tolerance) {
                        test.fail("property: d^m(d^n)==d^(m+n) for m=" + std::to_string(m) + " n=" + std::to_string(n));
                        all_pass = false;
                    }
                }
            }
        }
        if (all_pass) {
            test.pass("Test 84: derivative^m(derivative^n(f)) == derivative^(m+n)(f)");
        }
    }

    // Test 85: Property - antiderivative composition A^m(A^n(f)) == A^(m+n)(f)
    {
        CPoly cp({{3}}, {0, 1});  // constant

        bool all_pass = true;
        for (int m = 1; m <= 2; ++m) {
            for (int n = 1; n <= 2; ++n) {
                CPoly a_m_n = cp.antiderivative(m).antiderivative(n);
                CPoly a_mn = cp.antiderivative(m + n);
                for (double x : {0.2, 0.5, 0.8}) {
                    if (std::abs(a_m_n(x) - a_mn(x)) > tolerance) {
                        test.fail("property: A^m(A^n)==A^(m+n) for m=" + std::to_string(m) + " n=" + std::to_string(n));
                        all_pass = false;
                    }
                }
            }
        }
        if (all_pass) {
            test.pass("Test 85: antiderivative^m(antiderivative^n(f)) == antiderivative^(m+n)(f)");
        }
    }

    // Test 86: Property - integral additivity integrate(a,c) == integrate(a,b) + integrate(b,c)
    {
        std::vector<std::pair<std::string, CPoly>> test_polys;
        test_polys.push_back({"linear", CPoly({{0.5}, {0.5}}, {0, 1})});
        test_polys.push_back({"quadratic", CPoly::from_power_basis({{1}, {2}, {-1}}, {0, 1})});

        bool all_pass = true;
        for (const auto& [name, cp] : test_polys) {
            double a = 0.1, b = 0.5, c = 0.9;
            double int_ac = cp.integrate(a, c);
            double int_ab = cp.integrate(a, b);
            double int_bc = cp.integrate(b, c);
            if (std::abs(int_ac - (int_ab + int_bc)) > tolerance) {
                test.fail("property: integrate additivity for " + name);
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 86: integrate(a,c) == integrate(a,b) + integrate(b,c)");
        }
    }

    // Test 87: Property - integral reversal integrate(a,b) == -integrate(b,a)
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});

        bool all_pass = true;
        for (double a : {0.1, 0.3}) {
            for (double b : {0.6, 0.9}) {
                double int_ab = cp.integrate(a, b);
                double int_ba = cp.integrate(b, a);
                if (std::abs(int_ab + int_ba) > tolerance) {
                    test.fail("property: integrate reversal at a=" + std::to_string(a) + " b=" + std::to_string(b));
                    all_pass = false;
                }
            }
        }
        if (all_pass) {
            test.pass("Test 87: integrate(a,b) == -integrate(b,a)");
        }
    }

    // Test 88: Property - from_derivatives matches endpoint constraints
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{1, 2, 3}, {4, 5, 6}};  // f, f', f'' at each endpoint

        CPoly cp = CPoly::from_derivatives(xi, yi);

        bool all_pass = true;
        // Check left endpoint
        if (std::abs(cp(0) - 1) > tolerance) { test.fail("from_deriv f(0)"); all_pass = false; }
        if (std::abs(cp(0, 1) - 2) > tolerance) { test.fail("from_deriv f'(0)"); all_pass = false; }
        if (std::abs(cp(0, 2) - 3) > tolerance) { test.fail("from_deriv f''(0)"); all_pass = false; }
        // Check right endpoint
        if (std::abs(cp(1) - 4) > tolerance) { test.fail("from_deriv f(1)"); all_pass = false; }
        if (std::abs(cp(1, 1) - 5) > tolerance) { test.fail("from_deriv f'(1)"); all_pass = false; }
        if (std::abs(cp(1, 2) - 6) > tolerance) { test.fail("from_deriv f''(1)"); all_pass = false; }

        if (all_pass) {
            test.pass("Test 88: from_derivatives matches all endpoint constraints");
        }
    }

    // Test 89: Property - Chebyshev evaluation consistency
    {
        // Verify that direct Chebyshev evaluation matches from_power_basis evaluation
        std::vector<std::vector<double>> power = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        CPoly cp = CPoly::from_power_basis(power, {0, 1});

        bool all_pass = true;
        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            double expected = 1 + 2*x + 3*x*x;
            if (std::abs(cp(x) - expected) > tolerance) {
                test.fail("Chebyshev eval consistency at x=" + std::to_string(x));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 89: Chebyshev evaluation matches power basis");
        }
    }

    // ============================================================
    // Group 27: Extended Independent Verification
    // ============================================================
    std::cout << "\n--- Group 27: Extended Independent Verification ---\n" << std::endl;

    // Test 90: Multi-point finite difference derivative verification
    {
        CPoly cp = CPoly::from_power_basis({{1}, {-2}, {3}, {-1}}, {0, 1});  // 1 - 2x + 3x^2 - x^3

        bool all_pass = true;
        for (double x : {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}) {
            double analytical = cp(x, 1);
            double numerical = finite_diff_derivative(cp, x);
            if (std::abs(analytical - numerical) > 1e-5) {
                test.fail("finite diff derivative at x=" + std::to_string(x));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 90: Multi-point derivative matches finite differences");
        }
    }

    // Test 91: Multi-point second derivative verification
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});  // 1 + 2x + 3x^2 + 4x^3

        bool all_pass = true;
        for (double x : {0.2, 0.4, 0.5, 0.6, 0.8}) {
            double analytical = cp(x, 2);
            double numerical = finite_diff_second_derivative(cp, x);
            if (std::abs(analytical - numerical) > 1e-4) {
                test.fail("finite diff 2nd derivative at x=" + std::to_string(x));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 91: Multi-point 2nd derivative matches finite differences");
        }
    }

    // Test 92: Multi-interval numerical integration verification
    {
        // Use a polynomial where we know the exact integral
        CPoly cp = CPoly::from_power_basis({{0}, {1}}, {0, 1});  // f(x) = x

        bool all_pass = true;
        std::vector<std::pair<double, double>> intervals = {{0, 0.5}, {0.25, 0.75}, {0, 1}};
        for (const auto& [a, b] : intervals) {
            double analytical = cp.integrate(a, b);
            double numerical = numerical_integrate(cp, a, b);
            double expected = (b*b - a*a) / 2.0;  // exact integral of x

            if (std::abs(analytical - expected) > tolerance) {
                test.fail("analytical integral [" + std::to_string(a) + "," + std::to_string(b) + "]");
                all_pass = false;
            }
            if (std::abs(numerical - expected) > 1e-6) {
                test.fail("numerical integral [" + std::to_string(a) + "," + std::to_string(b) + "]");
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 92: Multi-interval integration verification");
        }
    }

    // ============================================================
    // Group 28: Thread Safety Verification
    // ============================================================
    std::cout << "\n--- Group 28: Thread Safety Verification ---\n" << std::endl;

    // Test 93: Repeated evaluation produces consistent results
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}, {4}, {5}}, {0, 1});

        bool all_pass = true;
        std::vector<double> points = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

        // Evaluate multiple times and verify consistency
        std::vector<double> results1 = cp(points);
        std::vector<double> results2 = cp(points);
        std::vector<double> results3 = cp(points);

        for (size_t i = 0; i < points.size(); ++i) {
            if (results1[i] != results2[i] || results2[i] != results3[i]) {
                test.fail("consistency at x=" + std::to_string(points[i]));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 93: Repeated evaluation produces identical results");
        }
    }

    // Test 93b: Actual multi-threaded concurrent access
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}, {4}, {5}}, {0, 1});
        const int num_threads = 10;
        std::atomic<bool> all_correct{true};
        std::vector<std::thread> threads;

        auto thread_func = [&cp, &all_correct](unsigned int seed) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            for (int i = 0; i < 100; ++i) {
                double x = dist(rng);
                double result = cp(x);
                // Expected: 1 + 2x + 3x^2 + 4x^3 + 5x^4
                double expected = 1 + 2*x + 3*x*x + 4*x*x*x + 5*x*x*x*x;
                if (std::abs(result - expected) > 1e-10) {
                    all_correct.store(false);
                }
            }
        };

        // Spawn threads
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(thread_func, static_cast<unsigned int>(t * 12345));
        }

        // Wait for all threads
        for (auto& th : threads) {
            th.join();
        }

        if (all_correct.load()) {
            test.pass("Test 93b: Multi-threaded concurrent evaluation");
        } else {
            test.fail("Test 93b: Multi-threaded concurrent evaluation");
        }
    }

    // Test 94: Const-correctness verification (compile-time check, runtime validation)
    {
        const CPoly cp({{0.5}, {0.5}}, {0, 1});

        // All these operations should work on a const object
        double v1 = cp(0.5);
        double v2 = cp(0.5, 1);
        std::vector<double> v3 = cp({0.1, 0.2, 0.3});
        double v4 = cp.integrate(0, 1);
        int d = cp.degree();
        int n = cp.num_intervals();
        bool asc = cp.is_ascending();
        const auto& c = cp.coefficients();
        const auto& x = cp.breakpoints();
        ExtrapolateMode mode = cp.extrapolate();

        // Verify values are reasonable
        test.expect_near(v1, 0.5, tolerance, "Test 94a: Const eval");
        test.expect_near(v2, 1.0, tolerance, "Test 94b: Const derivative eval");
        test.expect_eq(v3.size(), 3ul, "Test 94c: Const vector eval");
        test.expect_near(v4, 0.5, tolerance, "Test 94d: Const integrate");
        test.expect_eq(static_cast<size_t>(d), 1ul, "Test 94e: Const degree");
        test.expect_eq(static_cast<size_t>(n), 1ul, "Test 94f: Const num_intervals");
        test.expect_true(asc, "Test 94g: Const is_ascending");
        test.expect_true(!c.empty(), "Test 94h: Const coefficients");
        test.expect_true(!x.empty(), "Test 94i: Const breakpoints");
        test.expect_true(mode == ExtrapolateMode::Extrapolate, "Test 94j: Const extrapolate");
    }

    // ============================================================
    // Group 29: Periodic Derivative Evaluation
    // ============================================================
    std::cout << "\n--- Group 29: Periodic Derivative Evaluation ---\n" << std::endl;

    // Test 95: Periodic mode with derivative evaluation
    {
        CPoly cp = CPoly::from_power_basis({{0, 1}, {1, -0.5}}, {0, 1, 3}, ExtrapolateMode::Periodic);

        // Period = 3 - 0 = 3
        test.expect_near(cp(3.5), cp(0.5), tolerance, "Test 95a: Periodic f(3.5) = f(0.5)");
        test.expect_near(cp(-0.5), cp(2.5), tolerance, "Test 95b: Periodic f(-0.5) = f(2.5)");
        test.expect_near(cp(3.5, 1), cp(0.5, 1), tolerance, "Test 95c: Periodic f'(3.5) = f'(0.5)");
        test.expect_near(cp(-0.5, 1), cp(2.5, 1), tolerance, "Test 95d: Periodic f'(-0.5) = f'(2.5)");
    }

    // ============================================================
    // Group 30: Corner Case Tests
    // ============================================================
    std::cout << "\n--- Group 30: Corner Case Tests ---\n" << std::endl;

    // Test 96: Two intervals with from_power_basis
    {
        CPoly cp = CPoly::from_power_basis({{0, 0.5}, {1, 0}}, {0, 0.5, 1});
        test.expect_near(cp(0.0), 0.0, tolerance, "Test 96a: Two intervals at 0");
        test.expect_near(cp(0.25), 0.25, tolerance, "Test 96b: Two intervals at 0.25");
        test.expect_near(cp(0.5), 0.5, tolerance, "Test 96c: Two intervals at 0.5");
        test.expect_near(cp(0.75), 0.5, tolerance, "Test 96d: Two intervals at 0.75");
        test.expect_near(cp(1.0), 0.5, tolerance, "Test 96e: Two intervals at 1");
    }

    // Test 97: Many intervals (10 linear pieces)
    {
        std::vector<std::vector<double>> c(2, std::vector<double>(10));
        std::vector<double> breaks(11);
        for (int i = 0; i <= 10; ++i) {
            breaks[i] = static_cast<double>(i);
        }
        // For Chebyshev on each [a,b]: f(x) = x means s = (2x-a-b)/(b-a)
        // f(s) = c_0*T_0(s) + c_1*T_1(s) = c_0 + c_1*s
        // At unit intervals: mid = (a+b)/2, half_width = 0.5
        // f(x) = mid + half_width * s = c_0 + c_1*s => c_0 = mid, c_1 = 0.5
        for (int i = 0; i < 10; ++i) {
            double a = breaks[i];
            double b = breaks[i + 1];
            double mid = (a + b) / 2.0;
            c[0][i] = mid;
            c[1][i] = 0.5;
        }
        CPoly cp(c, breaks);
        test.expect_near(cp(0.5), 0.5, tolerance, "Test 97a: Many intervals at 0.5");
        test.expect_near(cp(5.5), 5.5, tolerance, "Test 97b: Many intervals at 5.5");
        test.expect_near(cp(9.5), 9.5, tolerance, "Test 97c: Many intervals at 9.5");
    }

    // Test 98: Large coefficients (1e10 scale)
    {
        CPoly cp = CPoly::from_power_basis({{1e10}, {1e10}}, {0, 1});
        test.expect_near(cp(0.0), 1e10, 1.0, "Test 98a: Large coeffs at 0");
        test.expect_near(cp(0.5), 1.5e10, 1e5, "Test 98b: Large coeffs at 0.5");
        test.expect_near(cp(1.0), 2e10, 1.0, "Test 98c: Large coeffs at 1");
    }

    // Test 99: Small coefficients (1e-15 scale)
    {
        CPoly cp = CPoly::from_power_basis({{1e-15}, {1e-15}}, {0, 1});
        test.expect_near(cp(0.0), 1e-15, 1e-25, "Test 99a: Small coeffs at 0");
        test.expect_near(cp(0.5), 1.5e-15, 1e-25, "Test 99b: Small coeffs at 0.5");
        test.expect_near(cp(1.0), 2e-15, 1e-25, "Test 99c: Small coeffs at 1");
    }

    // Test 100: Far extrapolation
    {
        CPoly cp = CPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);
        test.expect_near(cp(10.0), 10.0, tolerance, "Test 100a: Far extrapolation at 10");
        test.expect_near(cp(-10.0), -10.0, tolerance, "Test 100b: Far extrapolation at -10");
        test.expect_near(cp(100.0), 100.0, 1e-8, "Test 100c: Far extrapolation at 100");
    }

    // Test 101: Near-boundary evaluation
    {
        CPoly cp = CPoly::from_power_basis({{0, 1}, {1, 1}}, {0, 1, 2});

        double eps = 1e-14;
        test.expect_true(std::isfinite(cp(1.0 - eps)), "Test 101a: Near boundary 1-eps finite");
        test.expect_true(std::isfinite(cp(1.0 + eps)), "Test 101b: Near boundary 1+eps finite");
        double val_at_1 = cp(1.0);
        test.expect_near(cp(1.0 - eps), val_at_1, 1e-10, "Test 101c: Near boundary 1-eps value");
        test.expect_near(cp(1.0 + eps), val_at_1, 1e-10, "Test 101d: Near boundary 1+eps value");
    }

    // Test 102: from_derivatives with values only
    {
        CPoly cp = CPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}});
        test.expect_near(cp(0.0), 0.0, tolerance, "Test 102a: Values-only f(0)");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 102b: Values-only f(1)");
        test.expect_near(cp(2.0), 0.0, tolerance, "Test 102c: Values-only f(2)");
        test.expect_near(cp(0.5), 0.5, tolerance, "Test 102d: Values-only f(0.5)");
    }

    // Test 103: from_derivatives with asymmetric orders
    {
        CPoly cp = CPoly::from_derivatives({0, 1}, {{0, 1, 0, 0}, {1}});
        test.expect_near(cp(0.0), 0.0, tolerance, "Test 103a: Asymmetric f(0)");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 103b: Asymmetric f(1)");
        double h = 1e-8;
        double deriv_at_0 = (cp(h) - cp(0)) / h;
        test.expect_near(deriv_at_0, 1.0, 1e-5, "Test 103c: Asymmetric f'(0) approx");
    }

    // ============================================================
    // Group 31: C0 Continuity at Breakpoints
    // ============================================================
    std::cout << "\n--- Group 31: C0 Continuity at Breakpoints ---\n" << std::endl;

    // Test 104: from_derivatives produces C0 continuous polynomial
    {
        CPoly cp = CPoly::from_derivatives({0, 1, 2, 3}, {{0, 1}, {1, 0}, {0.5, -0.5}, {0, 0}});

        double eps = 1e-12;
        bool continuous = true;

        for (double bp : {1.0, 2.0}) {
            double left = cp(bp - eps);
            double right = cp(bp + eps);
            double at_bp = cp(bp);
            if (std::abs(left - at_bp) > 1e-8 || std::abs(right - at_bp) > 1e-8) {
                test.fail("Test 104: C0 continuity at x=" + std::to_string(bp));
                continuous = false;
            }
        }
        if (continuous) {
            test.pass("Test 104: from_derivatives produces C0 continuous polynomial");
        }
    }

    // Test 105: from_derivatives with matching derivatives produces C1 continuity
    {
        CPoly cp = CPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1, 1}, {3, 1}});

        double eps = 1e-10;
        bool c1_continuous = true;

        double f_left = cp(1.0 - eps);
        double f_right = cp(1.0 + eps);
        double df_left = cp(1.0 - eps, 1);
        double df_right = cp(1.0 + eps, 1);

        if (std::abs(f_left - f_right) > 1e-8) {
            test.fail("Test 105a: C0 continuity at x=1");
            c1_continuous = false;
        }
        if (std::abs(df_left - df_right) > 1e-6) {
            test.fail("Test 105b: C1 continuity (derivative) at x=1");
            c1_continuous = false;
        }
        if (c1_continuous) {
            test.pass("Test 105: from_derivatives with matching f' produces C1 continuity");
        }
    }

    // ============================================================
    // Group 32: extend() with Mixed Degrees
    // ============================================================
    std::cout << "\n--- Group 32: extend() with Mixed Degrees ---\n" << std::endl;

    // Test 106: Extend cubic polynomial with linear polynomial
    {
        CPoly cp_cubic = CPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1});

        // Linear on [1,2]: g(x) = 2x - 1
        // Chebyshev on [1,2]: s = 2*(x-1.5)/1 = 2x - 3
        // g(s) = c_0 + c_1*s, g(-1) = 1, g(1) = 3 => c_0 = 2, c_1 = 1
        CPoly cp_linear({{2}, {1}}, {1, 2});

        CPoly extended = cp_cubic.extend(cp_linear.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 106a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.125, tolerance, "Test 106b: Cubic part at 0.5");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 106c: Linear part at 1.5");
        test.expect_near(extended(1.0), 1.0, tolerance, "Test 106d: Continuity at boundary");
    }

    // Test 107: Extend quadratic with quintic
    {
        CPoly cp_quad = CPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});
        CPoly cp_quint = CPoly::from_derivatives({1, 2}, {{1, 2, 2}, {4, 4, 2}});

        CPoly extended = cp_quad.extend(cp_quint.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 107a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.25, tolerance, "Test 107b: Quadratic part");
        test.expect_near(extended(1.5), cp_quint(1.5), tolerance, "Test 107c: Quintic part");
    }

    // ============================================================
    // Group 33: Edge Case Coverage
    // ============================================================
    std::cout << "\n--- Group 33: Edge Case Coverage ---\n" << std::endl;

    // Test 108: Empty vector evaluation
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> empty_input;
        std::vector<double> result = cp(empty_input);
        test.expect_eq(result.size(), 0ul, "Test 108: Empty vector evaluation returns empty");
    }

    // Test 109: Single-point vector evaluation
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> single_point = {0.5};
        std::vector<double> result = cp(single_point);
        test.expect_eq(result.size(), 1ul, "Test 109a: Single-point vector size");
        test.expect_near(result[0], 2.0, tolerance, "Test 109b: Single-point value correct");
    }

    // Test 110: Evaluation exactly at all breakpoints
    {
        CPoly cp = CPoly::from_power_basis({{0, 1, 2}, {1, 1, 1}}, {0, 1, 2, 3});
        std::vector<double> breakpts = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> results = cp(breakpts);

        test.expect_near(results[0], 0.0, tolerance, "Test 110a: Eval at bp[0]");
        test.expect_near(results[1], 1.0, tolerance, "Test 110b: Eval at bp[1]");
        test.expect_near(results[2], 2.0, tolerance, "Test 110c: Eval at bp[2]");
        test.expect_near(results[3], 3.0, tolerance, "Test 110d: Eval at bp[3]");
    }

    // ============================================================
    // Group 34: Additional Edge Case Coverage
    // ============================================================
    std::cout << "\n--- Group 34: Additional Edge Case Coverage ---\n" << std::endl;

    // Test 111: Zero polynomial
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {0}};
        std::vector<double> breaks = {0, 1};
        CPoly cp(coeffs, breaks);

        test.expect_near(cp(0.0), 0.0, tolerance, "Test 111a: Zero polynomial at 0");
        test.expect_near(cp(0.5), 0.0, tolerance, "Test 111b: Zero polynomial at 0.5");
        test.expect_near(cp(1.0), 0.0, tolerance, "Test 111c: Zero polynomial at 1");
        test.expect_near(cp(0.5, 1), 0.0, tolerance, "Test 111d: Zero polynomial derivative");
        test.expect_near(cp.integrate(0, 1), 0.0, tolerance, "Test 111e: Zero polynomial integral");
    }

    // Test 112: Repeated roots - (x-0.5)^2
    {
        CPoly cp = CPoly::from_power_basis({{0.25}, {-1}, {1}}, {0, 1});

        auto roots = cp.roots();
        test.expect_true(roots.size() >= 1, "Test 112a: Repeated root found");
        if (!roots.empty()) {
            test.expect_near(roots[0], 0.5, 1e-6, "Test 112b: Repeated root at 0.5");
        }
        test.expect_near(cp(0.5), 0.0, tolerance, "Test 112c: f(0.5) = 0");
        test.expect_near(cp(0.5, 1), 0.0, tolerance, "Test 112d: f'(0.5) = 0");
    }

    // Test 113: Integration beyond domain
    {
        CPoly cp = CPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);

        double integral = cp.integrate(-1, 2);
        test.expect_near(integral, 1.5, tolerance, "Test 113a: Integration beyond domain");
        test.expect_near(cp.integrate(-1, 0), -0.5, tolerance, "Test 113b: Integration in left extrapolation");
        test.expect_near(cp.integrate(1, 2), 1.5, tolerance, "Test 113c: Integration in right extrapolation");
    }

    // ============================================================
    // Group 35: Chebyshev Basis-Specific Properties
    // ============================================================
    std::cout << "\n--- Group 35: Chebyshev Basis-Specific Properties ---\n" << std::endl;

    // Test 114: T_n(1) = 1 for all n
    {
        for (int n = 0; n <= 10; ++n) {
            std::vector<std::vector<double>> coeffs(n + 1, {0.0});
            coeffs[n][0] = 1.0;
            CPoly cp(coeffs, {-1, 1});

            double val = cp(1.0);
            std::string test_name = "Test 114: T_" + std::to_string(n) + "(1) = 1";
            test.expect_near(val, 1.0, 1e-8, test_name);
        }
    }

    // Test 115: T_n(-1) = (-1)^n
    {
        for (int n = 0; n <= 10; ++n) {
            std::vector<std::vector<double>> coeffs(n + 1, {0.0});
            coeffs[n][0] = 1.0;
            CPoly cp(coeffs, {-1, 1});

            double val = cp(-1.0);
            double expected = (n % 2 == 0) ? 1.0 : -1.0;
            std::string test_name = "Test 115: T_" + std::to_string(n) + "(-1) = " + std::to_string(static_cast<int>(expected));
            test.expect_near(val, expected, 1e-8, test_name);
        }
    }

    // Test 116: T_n(cos(theta)) = cos(n*theta)
    {
        std::vector<double> thetas = {0.3, 0.7, 1.0, 1.5, 2.0};
        for (int n = 0; n <= 6; ++n) {
            std::vector<std::vector<double>> coeffs(n + 1, {0.0});
            coeffs[n][0] = 1.0;
            CPoly cp(coeffs, {-1, 1});

            for (double theta : thetas) {
                double s = std::cos(theta);
                double val = cp(s);
                double expected = std::cos(n * theta);
                std::string test_name = "Test 116: T_" + std::to_string(n) + "(cos(" + std::to_string(theta) + "))";
                test.expect_near(val, expected, 1e-8, test_name);
            }
        }
    }

    // Test 117: Recurrence T_{n+1}(s) = 2s*T_n(s) - T_{n-1}(s)
    {
        std::vector<double> test_points = {-0.5, 0.0, 0.5, 0.75};
        for (int n = 1; n <= 4; ++n) {
            auto make_T = [](int deg) {
                std::vector<std::vector<double>> coeffs(deg + 1, {0.0});
                coeffs[deg][0] = 1.0;
                return CPoly(coeffs, {-1, 1});
            };
            CPoly T_nm1 = make_T(n - 1);
            CPoly T_n = make_T(n);
            CPoly T_np1 = make_T(n + 1);

            for (double s : test_points) {
                double lhs = T_np1(s);
                double rhs = 2.0 * s * T_n(s) - T_nm1(s);

                std::string test_name = "Test 117: Recurrence n=" + std::to_string(n) + " s=" + std::to_string(s);
                test.expect_near(lhs, rhs, 1e-10, test_name);
            }
        }
    }

    // ============================================================
    // Group 36: Derivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 36: Derivative Edge Cases ---\n" << std::endl;

    // Test 118: Derivative of constant is 0
    {
        CPoly cp({{5}}, {0, 1});
        CPoly deriv = cp.derivative();
        test.expect_near(deriv(0.5), 0.0, tolerance, "Test 118a: Derivative of constant is 0");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 0ul, "Test 118b: Derivative degree is 0");
    }

    // Test 119: Over-differentiate
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        CPoly d3 = cp.derivative(3);
        test.expect_near(d3(0.5), 0.0, tolerance, "Test 119a: 3rd derivative of quadratic is 0");

        CPoly d10 = cp.derivative(10);
        test.expect_near(d10(0.5), 0.0, tolerance, "Test 119b: 10th derivative of quadratic is 0");
    }

    // Test 120: Chained derivatives vs single derivative(n)
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});
        CPoly d3_chained = cp.derivative().derivative().derivative();
        CPoly d3_single = cp.derivative(3);

        test.expect_near(d3_chained(0.5), d3_single(0.5), tolerance, "Test 120: Chained vs single derivative(3)");
    }

    // ============================================================
    // Group 37: Antiderivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 37: Antiderivative Edge Cases ---\n" << std::endl;

    // Test 121: Antiderivative of zero polynomial
    {
        CPoly cp({{0}, {0}}, {0, 1});
        CPoly anti = cp.antiderivative();
        test.expect_near(anti(0.0), 0.0, tolerance, "Test 121a: Antiderivative of zero at 0");
        test.expect_near(anti(0.5), 0.0, tolerance, "Test 121b: Antiderivative of zero at 0.5");
        test.expect_near(anti(1.0), 0.0, tolerance, "Test 121c: Antiderivative of zero at 1");
    }

    // Test 122: Antiderivative(n).derivative(n) = original
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        CPoly round_trip = cp.antiderivative(2).derivative(2);

        test.expect_near(cp(0.25), round_trip(0.25), tolerance, "Test 122a: antiderivative(2).derivative(2) at 0.25");
        test.expect_near(cp(0.5), round_trip(0.5), tolerance, "Test 122b: antiderivative(2).derivative(2) at 0.5");
        test.expect_near(cp(0.75), round_trip(0.75), tolerance, "Test 122c: antiderivative(2).derivative(2) at 0.75");
    }

    // Test 123: Chained antiderivatives vs single antiderivative(n)
    {
        CPoly cp = CPoly::from_power_basis({{2}}, {0, 1});
        CPoly a2_chained = cp.antiderivative().antiderivative();
        CPoly a2_single = cp.antiderivative(2);

        test.expect_near(a2_chained(0.5), a2_single(0.5), tolerance, "Test 123: Chained vs single antiderivative(2)");
    }

    // ============================================================
    // Group 37b: Derivative/Antiderivative Structural Properties
    // ============================================================
    std::cout << "\n--- Group 37b: Derivative/Antiderivative Structural Properties ---\n" << std::endl;

    // Test 123b: Derivative reduces degree by 1
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});
        CPoly deriv = cp.derivative();
        test.expect_eq(static_cast<size_t>(cp.degree()), 3ul, "Test 123b-1: Original degree is 3");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 2ul, "Test 123b-2: Derivative degree is 2");
    }

    // Test 123c: Antiderivative increases degree by 1
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        CPoly anti = cp.antiderivative();
        test.expect_eq(static_cast<size_t>(cp.degree()), 2ul, "Test 123c-1: Original degree is 2");
        test.expect_eq(static_cast<size_t>(anti.degree()), 3ul, "Test 123c-2: Antiderivative degree is 3");
    }

    // Test 123d: Derivative preserves num_intervals
    {
        CPoly cp({{1, 2, 3}}, {0, 1, 2, 3});
        CPoly deriv = cp.derivative();
        test.expect_eq(static_cast<size_t>(deriv.num_intervals()), 3ul,
            "Test 123d: Derivative preserves num_intervals");
    }

    // Test 123e: Derivative preserves breakpoints
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        CPoly deriv = cp.derivative();
        test.expect_near(deriv.breakpoints()[0], 0.0, tolerance, "Test 123e-1: Derivative preserves left breakpoint");
        test.expect_near(deriv.breakpoints()[1], 1.0, tolerance, "Test 123e-2: Derivative preserves right breakpoint");
    }

    // Test 123f: Antiderivative starts at 0 at left boundary
    {
        CPoly cp = CPoly::from_power_basis({{5}}, {0, 1});
        CPoly anti = cp.antiderivative();
        test.expect_near(anti(0), 0.0, tolerance, "Test 123f: Antiderivative(left_boundary) = 0");
    }

    // ============================================================
    // Group 38: Integration Edge Cases
    // ============================================================
    std::cout << "\n--- Group 38: Integration Edge Cases ---\n" << std::endl;

    // Test 124: integrate(a, a) = 0
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        test.expect_near(cp.integrate(0.5, 0.5), 0.0, tolerance, "Test 124: integrate(a, a) = 0");
    }

    // Test 125: Integration crossing multiple intervals
    {
        CPoly cp({{1, 2}}, {0, 1, 2});
        test.expect_near(cp.integrate(0, 2), 3.0, tolerance, "Test 125: Integration across multiple intervals");
    }

    // Test 126: NoExtrapolate mode returns NaN beyond bounds
    {
        CPoly temp = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        CPoly cp(temp.c(), {0, 1}, ExtrapolateMode::NoExtrapolate);
        double result = cp(-1);
        test.expect_true(std::isnan(result), "Test 126: NoExtrapolate evaluation beyond bounds returns NaN");
    }

    // ============================================================
    // Group 39: Root Finding Edge Cases
    // ============================================================
    std::cout << "\n--- Group 39: Root Finding Edge Cases ---\n" << std::endl;

    // Test 127: No roots (always positive polynomial)
    {
        CPoly cp = CPoly::from_power_basis({{1}, {0}, {1}}, {0, 1});
        auto roots = cp.roots();
        test.expect_eq(roots.size(), 0ul, "Test 127: No roots for always-positive polynomial");
    }

    // Test 128: Root at domain boundary
    {
        CPoly cp = CPoly::from_power_basis({{0}, {1}}, {0, 1});
        auto roots = cp.roots();
        test.expect_true(roots.size() >= 1, "Test 128a: Root at boundary found");
        if (!roots.empty()) {
            double min_root = *std::min_element(roots.begin(), roots.end());
            test.expect_near(min_root, 0.0, 1e-6, "Test 128b: Root at x=0");
        }
    }

    // Test 129: Many roots
    {
        CPoly cp = CPoly::from_power_basis({{-0.09375}, {0.6875}, {-1.5}, {1}}, {0, 1});
        auto roots = cp.roots();
        test.expect_eq(roots.size(), 3ul, "Test 129a: Three roots found");

        if (roots.size() == 3) {
            std::sort(roots.begin(), roots.end());
            test.expect_near(roots[0], 0.25, 1e-6, "Test 129b: Root at 0.25");
            test.expect_near(roots[1], 0.5, 1e-6, "Test 129c: Root at 0.5");
            test.expect_near(roots[2], 0.75, 1e-6, "Test 129d: Root at 0.75");
        }
    }

    // ============================================================
    // Group 40: Extrapolation Order Edge Cases
    // ============================================================
    std::cout << "\n--- Group 40: Extrapolation Order Edge Cases ---\n" << std::endl;

    // Test 130: extrapolation_order > polynomial degree
    {
        CPoly temp = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        CPoly cp_full(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, -1, -1);
        CPoly cp_high(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 10, 10);

        test.expect_near(cp_full(-0.5), cp_high(-0.5), tolerance, "Test 130a: High extrapolation order matches full");
        test.expect_near(cp_full(1.5), cp_high(1.5), tolerance, "Test 130b: High extrapolation order matches full");
    }

    // Test 131: extrapolation_order = 0 gives constant extrapolation
    {
        CPoly temp = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        CPoly cp(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 0, 0);
        test.expect_near(cp(-0.5), 1.0, tolerance, "Test 131a: Order 0 extrapolation at left");
        test.expect_near(cp(1.5), 3.0, tolerance, "Test 131b: Order 0 extrapolation at right");
    }

    // Test 132: Asymmetric extrapolation orders
    {
        CPoly temp = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        CPoly cp(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 0, -1);
        test.expect_near(cp(-0.5), 1.0, tolerance, "Test 132a: Order 0 on left");
        test.expect_near(cp(1.5), 1.0 + 2*1.5, tolerance, "Test 132b: Full order on right");
    }

    // ============================================================
    // Group 41: Move/Copy Semantics
    // ============================================================
    std::cout << "\n--- Group 41: Move/Copy Semantics ---\n" << std::endl;

    // Test 133: Copy constructor
    {
        CPoly cp1 = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        CPoly cp2(cp1);

        test.expect_near(cp1(0.5), cp2(0.5), tolerance, "Test 133a: Copy constructor preserves value");
        test.expect_eq(static_cast<size_t>(cp1.degree()), static_cast<size_t>(cp2.degree()), "Test 133b: Copy constructor preserves degree");
    }

    // Test 134: Move constructor
    {
        CPoly cp1 = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double original_val = cp1(0.5);
        CPoly cp2(std::move(cp1));

        test.expect_near(cp2(0.5), original_val, tolerance, "Test 134: Move constructor preserves value");
    }

    // ============================================================
    // Group 42: Const-Correctness and Thread Safety
    // ============================================================
    std::cout << "\n--- Group 42: Const-Correctness and Thread Safety ---\n" << std::endl;

    // Test 135: All const methods work on const object
    {
        const CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double v1 = cp(0.5);
        double v2 = cp(0.5, 1);
        std::vector<double> v3 = cp({0.25, 0.5, 0.75});
        int d = cp.degree();
        int n = cp.num_intervals();
        const auto& c = cp.c();
        const auto& x = cp.x();
        double integ = cp.integrate(0, 1);
        auto roots = cp.roots();

        test.expect_near(v1, 1 + 2*0.5 + 3*0.25, tolerance, "Test 135a: Const evaluation");
        test.expect_near(v2, 2 + 6*0.5, tolerance, "Test 135b: Const derivative evaluation");
        test.expect_eq(v3.size(), 3ul, "Test 135c: Const vector evaluation");
        test.expect_eq(static_cast<size_t>(d), 2ul, "Test 135d: Const degree()");
        test.expect_eq(static_cast<size_t>(n), 1ul, "Test 135e: Const num_intervals()");
        test.expect_true(!c.empty(), "Test 135f: Const c() returns coefficients");
        test.expect_true(!x.empty(), "Test 135g: Const x() returns breakpoints");
        test.expect_near(integ, 3.0, tolerance, "Test 135h: Const integrate()");
        (void)roots;
        test.pass("Test 135: All const methods work on const object");
    }

    // Test 136: Thread safety - multiple threads evaluating same polynomial
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        std::atomic<int> correct_count{0};
        const int num_threads = 10;
        const int evals_per_thread = 1000;

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&cp, &correct_count, evals_per_thread]() {
                std::mt19937 gen(std::random_device{}());
                std::uniform_real_distribution<> dis(0.0, 1.0);

                for (int i = 0; i < evals_per_thread; ++i) {
                    double x = dis(gen);
                    double result = cp(x);
                    double expected = 1.0 + 2.0 * x + 3.0 * x * x;
                    if (std::abs(result - expected) < 1e-10) {
                        correct_count++;
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        test.expect_eq(static_cast<size_t>(correct_count.load()),
                      static_cast<size_t>(num_threads * evals_per_thread),
                      "Test 136: Thread safety - all concurrent evaluations correct");
    }

    // ============================================================
    // Group 43: Property-Based Tests
    // ============================================================
    std::cout << "\n--- Group 43: Property-Based Tests ---\n" << std::endl;

    // Test 137: Linearity of integration
    {
        CPoly cp1 = CPoly::from_power_basis({{1}, {2}}, {0, 1});
        CPoly cp2 = CPoly::from_power_basis({{3}, {4}}, {0, 1});
        CPoly cp_sum = CPoly::from_power_basis({{4}, {6}}, {0, 1});

        double int_sum = cp_sum.integrate(0, 1);
        double sum_int = cp1.integrate(0, 1) + cp2.integrate(0, 1);

        test.expect_near(int_sum, sum_int, tolerance, "Test 137: Linearity of integration");
    }

    // Test 138: FTC d/dx(antiderivative(f)) = f
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        CPoly anti = cp.antiderivative();
        CPoly anti_deriv = anti.derivative();

        for (double x : {0.25, 0.5, 0.75}) {
            test.expect_near(cp(x), anti_deriv(x), tolerance,
                           "Test 138: FTC d/dx(antiderivative(f)) = f at x=" + std::to_string(x));
        }
    }

    // Test 139: Integration reversal
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double int_ab = cp.integrate(0.2, 0.8);
        double int_ba = cp.integrate(0.8, 0.2);

        test.expect_near(int_ab, -int_ba, tolerance, "Test 139: Integration reversal");
    }

    // Test 140: Integration additivity
    {
        CPoly cp = CPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double int_ac = cp.integrate(0.2, 0.8);
        double int_ab = cp.integrate(0.2, 0.5);
        double int_bc = cp.integrate(0.5, 0.8);

        test.expect_near(int_ac, int_ab + int_bc, tolerance, "Test 140: Integration additivity");
    }

    // ============================================================
    // Group 44: Symmetry Tests
    // ============================================================
    std::cout << "\n--- Group 44: Symmetry Tests ---\n" << std::endl;

    // Test 141: Even polynomial p(-x) = p(x) using even T_n (T_0, T_2)
    {
        // T_0(s) = 1, T_2(s) = 2s^2 - 1
        // p(s) = 1*T_0 + 2*T_2 = 1 + 2*(2s^2 - 1) = 4s^2 - 1
        CPoly cp({{1}, {0}, {2}}, {-1, 1});

        test.expect_near(cp(-0.5), cp(0.5), tolerance, "Test 141a: Even polynomial at +/-0.5");
        test.expect_near(cp(-0.8), cp(0.8), tolerance, "Test 141b: Even polynomial at +/-0.8");
    }

    // Test 142: Odd polynomial p(-x) = -p(x) using odd T_n (T_1, T_3)
    {
        // T_1(s) = s, T_3(s) = 4s^3 - 3s
        CPoly cp({{0}, {1}, {0}, {1}}, {-1, 1});

        test.expect_near(cp(-0.5), -cp(0.5), tolerance, "Test 142a: Odd polynomial at +/-0.5");
        test.expect_near(cp(-0.8), -cp(0.8), tolerance, "Test 142b: Odd polynomial at +/-0.8");
    }

    // ============================================================
    // Group 45: Boundary Edge Cases
    // ============================================================
    std::cout << "\n--- Group 45: Boundary Edge Cases ---\n" << std::endl;

    // Test 143: Evaluation at breakpoint + epsilon
    {
        CPoly cp1 = CPoly::from_power_basis({{0}, {1}}, {0, 1});
        CPoly cp2 = CPoly::from_power_basis({{1}, {2}}, {1, 2});
        CPoly cp = cp1.extend(cp2.c(), {1, 2}, true);

        double eps = std::numeric_limits<double>::epsilon();
        double at_bp = cp(1.0);
        double after_bp = cp(1.0 + eps);
        double before_bp = cp(1.0 - eps);

        test.expect_near(at_bp, 1.0, tolerance, "Test 143a: At breakpoint");
        test.expect_near(before_bp, 1.0, tolerance, "Test 143b: Just before breakpoint");
        test.expect_near(after_bp, 1.0, tolerance, "Test 143c: Just after breakpoint");
    }

    // Test 144: Very wide interval
    {
        CPoly cp = CPoly::from_power_basis({{0}, {1}}, {0, 1e10});
        test.expect_near(cp(0), 0.0, tolerance, "Test 144a: Wide interval at left");
        test.expect_near(cp(5e9), 5e9, 1e-3, "Test 144b: Wide interval at midpoint");
        test.expect_near(cp(1e10), 1e10, 1e-3, "Test 144c: Wide interval at right");
    }

    // Test 145: Very narrow interval
    {
        CPoly cp = CPoly::from_power_basis({{0}, {1}}, {0, 1e-10});
        test.expect_near(cp(0), 0.0, tolerance, "Test 145a: Narrow interval at left");
        test.expect_near(cp(5e-11), 5e-11, 1e-20, "Test 145b: Narrow interval at midpoint");
        test.expect_near(cp(1e-10), 1e-10, 1e-20, "Test 145c: Narrow interval at right");
    }

    // ============================================================
    // Group 46: Reference Data Verification
    // ============================================================
    std::cout << "\n--- Group 46: Reference Data Verification ---\n" << std::endl;

    // Test 146: T_2(s) at reference points
    // T_2(s) = 2s^2 - 1 at s = [-1, -0.5, 0, 0.5, 1] = [1, -0.5, -1, -0.5, 1]
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};
        CPoly cp(coeffs, {-1, 1});

        test.expect_near(cp(-1.0), 1.0, tolerance, "Test 146a: T_2(-1) = 1");
        test.expect_near(cp(-0.5), -0.5, tolerance, "Test 146b: T_2(-0.5) = -0.5");
        test.expect_near(cp(0.0), -1.0, tolerance, "Test 146c: T_2(0) = -1");
        test.expect_near(cp(0.5), -0.5, tolerance, "Test 146d: T_2(0.5) = -0.5");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 146e: T_2(1) = 1");
    }

    // Test 147: T_3(s) at reference points
    // T_3(s) = 4s^3 - 3s at s = [-1, -0.5, 0, 0.5, 1] = [-1, 1, 0, -1, 1]
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {0}, {1}};
        CPoly cp(coeffs, {-1, 1});

        test.expect_near(cp(-1.0), -1.0, tolerance, "Test 147a: T_3(-1) = -1");
        test.expect_near(cp(-0.5), 1.0, tolerance, "Test 147b: T_3(-0.5) = 1");
        test.expect_near(cp(0.0), 0.0, tolerance, "Test 147c: T_3(0) = 0");
        test.expect_near(cp(0.5), -1.0, tolerance, "Test 147d: T_3(0.5) = -1");
        test.expect_near(cp(1.0), 1.0, tolerance, "Test 147e: T_3(1) = 1");
    }

    // Test 148: T_10(0.5) reference value
    // T_10(0.5) = cos(10 * arccos(0.5)) = cos(10 * pi/3) = cos(10*pi/3)
    {
        std::vector<std::vector<double>> coeffs(11, {0.0});
        coeffs[10][0] = 1.0;
        CPoly cp(coeffs, {-1, 1});

        double expected = std::cos(10.0 * std::acos(0.5));
        test.expect_near(cp(0.5), expected, 1e-8, "Test 148: T_10(0.5) matches reference");
    }

    // Test 149: from_power_basis Chebyshev coefficients for x^2 on [0,1]
    {
        CPoly cp = CPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});
        auto& c = cp.c();

        // x^2 on [0,1]: map to s = 2x-1, x = (s+1)/2, x^2 = (s+1)^2/4 = (s^2+2s+1)/4
        // Chebyshev expansion: s^2 = (T_2(s)+1)/2, s = T_1(s)
        // x^2 = ((T_2+1)/2 + 2*T_1 + 1)/4 = (T_2/2 + 1/2 + 2*T_1 + 1)/4
        //      = T_2/8 + 1/8 + T_1/2 + 1/4 = 3/8 + T_1/2 + T_2/8
        // So c_0 = 3/8, c_1 = 1/2, c_2 = 1/8
        test.expect_near(c[0][0], 3.0/8.0, tolerance, "Test 149a: x^2 Chebyshev c_0 = 3/8");
        test.expect_near(c[1][0], 0.5, tolerance, "Test 149b: x^2 Chebyshev c_1 = 1/2");
        test.expect_near(c[2][0], 1.0/8.0, tolerance, "Test 149c: x^2 Chebyshev c_2 = 1/8");
    }

    // ============================================================
    // Summary
    // ============================================================
    test.summary();

    return test.all_passed() ? 0 : 1;
}
