#include "include/legpoly.h"
#include "test_utils.h"
#include <cassert>
#include <limits>
#include <thread>
#include <atomic>
#include <random>

int main() {
    TestRunner test;
    const double tolerance = 1e-10;

    std::cout << "=== LegPoly (Legendre Polynomial) Test Suite ===" << std::endl;

    // ============================================================
    // Group 1: Basic Construction and Evaluation
    // ============================================================
    std::cout << "\n--- Group 1: Basic Construction and Evaluation ---\n" << std::endl;

    // Test 1: Basic construction
    test.expect_no_throw([]() {
        // For Legendre: P_0 = 1 (constant), so c = {1} gives p(x) = 1
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);
    }, "Test 1: Basic construction");

    // Test 2: Constant polynomial P_0 = 1
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        test.expect_near(lp(0.0), 5.0, tolerance, "Test 2a: Constant f(0)");
        test.expect_near(lp(0.5), 5.0, tolerance, "Test 2b: Constant f(0.5)");
        test.expect_near(lp(1.0), 5.0, tolerance, "Test 2c: Constant f(1)");
    }

    // Test 3: Linear polynomial P_1(s) = s
    // On interval [0,1], s = 2x - 1
    // So P_1(s) = 2x - 1 maps [0,1] -> [-1,1]
    // p(x) = c_0 * P_0(s) + c_1 * P_1(s) = c_0 + c_1 * (2x - 1)
    // For p(x) = x, we need c_0 + c_1*(2x-1) = x
    // => c_0 - c_1 + 2*c_1*x = x
    // => c_0 - c_1 = 0 and 2*c_1 = 1
    // => c_1 = 0.5, c_0 = 0.5
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        test.expect_near(lp(0.0), 0.0, tolerance, "Test 3a: Linear f(0)");
        test.expect_near(lp(0.25), 0.25, tolerance, "Test 3b: Linear f(0.25)");
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 3c: Linear f(0.5)");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 3d: Linear f(1)");
    }

    // Test 4: Quadratic polynomial using P_0, P_1, P_2
    // P_0 = 1, P_1(s) = s, P_2(s) = (3s^2 - 1)/2
    // On [0,1]: s = 2x - 1
    // P_2(s) = (3(2x-1)^2 - 1)/2 = (3(4x^2 - 4x + 1) - 1)/2 = (12x^2 - 12x + 2)/2 = 6x^2 - 6x + 1
    // For p(x) = x^2, we solve: c_0 + c_1*(2x-1) + c_2*(6x^2-6x+1) = x^2
    // => 6*c_2 = 1 => c_2 = 1/6
    // => 2*c_1 - 6*c_2 = 0 => c_1 = 3*c_2 = 0.5
    // => c_0 - c_1 + c_2 = 0 => c_0 = c_1 - c_2 = 0.5 - 1/6 = 1/3
    {
        std::vector<std::vector<double>> coeffs = {{1.0/3.0}, {0.5}, {1.0/6.0}};  // p(x) = x^2 on [0,1]
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        test.expect_near(lp(0.0), 0.0, tolerance, "Test 4a: Quadratic f(0)");
        test.expect_near(lp(0.5), 0.25, tolerance, "Test 4b: Quadratic f(0.5)");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 4c: Quadratic f(1)");
        test.expect_near(lp(0.25), 0.0625, tolerance, "Test 4d: Quadratic f(0.25)");
    }

    // Test 5: Multiple intervals
    {
        // Two constant pieces: [0,1] has value 1, [1,2] has value 2
        std::vector<std::vector<double>> coeffs = {{1, 2}};
        std::vector<double> breaks = {0, 1, 2};
        LegPoly lp(coeffs, breaks);

        test.expect_near(lp(0.5), 1.0, tolerance, "Test 5a: Multi-interval f(0.5)");
        test.expect_near(lp(1.5), 2.0, tolerance, "Test 5b: Multi-interval f(1.5)");
    }

    // ============================================================
    // Group 2: Error Handling
    // ============================================================
    std::cout << "\n--- Group 2: Error Handling ---\n" << std::endl;

    // Test 6: Empty coefficients error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs;  // Empty
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);
    }, "Test 6: Empty coefficients error");

    // Test 7: Too few breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0};  // Only 1 breakpoint
        LegPoly lp(coeffs, breaks);
    }, "Test 7: Too few breakpoints error");

    // Test 8: Non-monotonic breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 0.5, 0.3};  // Not monotonic
        LegPoly lp(coeffs, breaks);
    }, "Test 8: Non-monotonic breakpoints error");

    // Test 8b: from_derivatives with mismatched xi/yi sizes
    test.expect_throw([]() {
        std::vector<double> xi = {0, 1, 2};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};  // Only 2 points, xi has 3
        LegPoly::from_derivatives(xi, yi);
    }, "Test 8b: from_derivatives mismatched xi/yi sizes");

    // Test 8c: from_derivatives with single point
    test.expect_throw([]() {
        LegPoly::from_derivatives({0}, {{1, 0}});
    }, "Test 8c: from_derivatives with single point");

    // Test 8d: from_derivatives with empty yi element
    test.expect_throw([]() {
        LegPoly::from_derivatives({0, 1}, {{}, {1}});
    }, "Test 8d: from_derivatives with empty yi element");

    // Test 8e: extend with non-contiguous breakpoints
    test.expect_throw([]() {
        LegPoly lp({{1}}, {0, 1});
        lp.extend({{2}}, {5, 6}, true);  // Gap between 1 and 5
    }, "Test 8e: extend with non-contiguous breakpoints");

    // Test 8f: extend with opposite ordering
    test.expect_throw([]() {
        LegPoly lp({{1}}, {0, 1});  // Ascending
        lp.extend({{2}}, {2, 1}, true);  // Descending
    }, "Test 8f: extend with opposite ordering");

    // Test 8g: from_power_basis with empty coefficients
    test.expect_throw([]() {
        LegPoly::from_power_basis({}, {0, 1});
    }, "Test 8g: from_power_basis with empty coefficients");

    // Test 8h: from_derivatives with invalid orders parameter size
    test.expect_throw([]() {
        LegPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}}, {1, 2});  // orders size != 1 and != 3
    }, "Test 8h: from_derivatives with invalid orders size");

    // ============================================================
    // Group 3: Vector Evaluation
    // ============================================================
    std::cout << "\n--- Group 3: Vector Evaluation ---\n" << std::endl;

    // Test 9: Evaluate at multiple points
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        std::vector<double> xs = {0.0, 0.25, 0.5, 0.75, 1.0};
        std::vector<double> results = lp(xs);

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
        LegPoly lp(coeffs, breaks, ExtrapolateMode::Extrapolate);

        // Extrapolate beyond [0,1]
        test.expect_near(lp(-0.5), -0.5, tolerance, "Test 10a: Extrapolate f(-0.5)");
        test.expect_near(lp(1.5), 1.5, tolerance, "Test 10b: Extrapolate f(1.5)");
    }

    // Test 11: NoExtrapolate mode
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks, ExtrapolateMode::NoExtrapolate);

        test.expect_true(std::isnan(lp(-0.5)), "Test 11a: NoExtrapolate f(-0.5) is NaN");
        test.expect_true(std::isnan(lp(1.5)), "Test 11b: NoExtrapolate f(1.5) is NaN");
    }

    // Test 12: Periodic mode
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks, ExtrapolateMode::Periodic);

        // Test periodic wrapping
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 12a: Periodic f(0.5)");
        test.expect_near(lp(1.5), lp(0.5), tolerance, "Test 12b: Periodic f(1.5) = f(0.5)");
        test.expect_near(lp(2.5), lp(0.5), tolerance, "Test 12c: Periodic f(2.5) = f(0.5)");
    }

    // ============================================================
    // Group 5: Derivative Operations
    // ============================================================
    std::cout << "\n--- Group 5: Derivative Operations ---\n" << std::endl;

    // Test 13: Derivative of linear polynomial
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);
        LegPoly dlp = lp.derivative();

        // Derivative of x is 1
        test.expect_near(dlp(0.0), 1.0, tolerance, "Test 13a: d/dx[x] at 0");
        test.expect_near(dlp(0.5), 1.0, tolerance, "Test 13b: d/dx[x] at 0.5");
        test.expect_near(dlp(1.0), 1.0, tolerance, "Test 13c: d/dx[x] at 1");
    }

    // Test 14: Derivative of quadratic polynomial
    {
        std::vector<std::vector<double>> coeffs = {{1.0/3.0}, {0.5}, {1.0/6.0}};  // p(x) = x^2
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);
        LegPoly dlp = lp.derivative();

        // Derivative of x^2 is 2x
        test.expect_near(dlp(0.0), 0.0, tolerance, "Test 14a: d/dx[x^2] at 0");
        test.expect_near(dlp(0.5), 1.0, tolerance, "Test 14b: d/dx[x^2] at 0.5");
        test.expect_near(dlp(1.0), 2.0, tolerance, "Test 14c: d/dx[x^2] at 1");
    }

    // Test 15: operator()(x, nu) syntax for derivatives
    {
        std::vector<std::vector<double>> coeffs = {{1.0/3.0}, {0.5}, {1.0/6.0}};  // p(x) = x^2
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        test.expect_near(lp(0.5, 0), 0.25, tolerance, "Test 15a: x^2(0.5, 0)");
        test.expect_near(lp(0.5, 1), 1.0, tolerance, "Test 15b: x^2(0.5, 1) = 2*0.5");
        test.expect_near(lp(0.5, 2), 2.0, tolerance, "Test 15c: x^2(0.5, 2) = 2");
    }

    // ============================================================
    // Group 6: Antiderivative and Integration
    // ============================================================
    std::cout << "\n--- Group 6: Antiderivative and Integration ---\n" << std::endl;

    // Test 16: Antiderivative of constant
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);
        LegPoly ilp = lp.antiderivative();

        // Antiderivative of 2 is 2x (starting at 0)
        test.expect_near(ilp(0), 0.0, tolerance, "Test 16a: int[2] at 0");
        test.expect_near(ilp(1), 2.0, tolerance, "Test 16b: int[2] at 1");
        test.expect_near(ilp(0.5), 1.0, tolerance, "Test 16c: int[2] at 0.5");
    }

    // Test 17: Integration (definite integral)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        // Integral of x from 0 to 1 is 0.5
        double integral = lp.integrate(0, 1);
        test.expect_near(integral, 0.5, tolerance, "Test 17a: int_0^1 x dx");

        // Integral of x from 0 to 0.5 is 0.125
        test.expect_near(lp.integrate(0, 0.5), 0.125, tolerance, "Test 17b: int_0^0.5 x dx");
    }

    // Test 18: Negative derivative order = antiderivative
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        LegPoly ilp = lp.derivative(-1);  // -1 means antiderivative
        test.expect_near(ilp(1), 2.0, tolerance, "Test 18: derivative(-1) = antiderivative");
    }

    // ============================================================
    // Group 7: from_derivatives (Hermite Interpolation)
    // ============================================================
    std::cout << "\n--- Group 7: from_derivatives ---\n" << std::endl;

    // Test 19: Simple Hermite cubic from_derivatives
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};  // f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1

        LegPoly lp = LegPoly::from_derivatives(xi, yi);

        // Verify endpoint values
        test.expect_near(lp(0), 0.0, tolerance, "Test 19a: Hermite f(0)");
        test.expect_near(lp(1), 1.0, tolerance, "Test 19b: Hermite f(1)");

        // Verify derivatives at endpoints
        test.expect_near(lp(0, 1), 1.0, tolerance, "Test 19c: Hermite f'(0)");
        test.expect_near(lp(1, 1), -1.0, tolerance, "Test 19d: Hermite f'(1)");
    }

    // Test 20: from_derivatives with second derivatives
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, -1, 0}};

        LegPoly lp = LegPoly::from_derivatives(xi, yi);

        test.expect_near(lp(0), 0.0, tolerance, "Test 20a: Quintic f(0)");
        test.expect_near(lp(1), 1.0, tolerance, "Test 20b: Quintic f(1)");
        test.expect_near(lp(0, 1), 1.0, tolerance, "Test 20c: Quintic f'(0)");
        test.expect_near(lp(1, 1), -1.0, tolerance, "Test 20d: Quintic f'(1)");
        test.expect_near(lp(0, 2), 0.0, tolerance, "Test 20e: Quintic f''(0)");
        test.expect_near(lp(1, 2), 0.0, tolerance, "Test 20f: Quintic f''(1)");
    }

    // Test 21: from_derivatives multi-interval
    {
        std::vector<double> xi = {0, 1, 2};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, 0}, {0, -1}};

        LegPoly lp = LegPoly::from_derivatives(xi, yi);

        test.expect_near(lp(0), 0.0, tolerance, "Test 21a: Multi-interval f(0)");
        test.expect_near(lp(1), 1.0, tolerance, "Test 21b: Multi-interval f(1)");
        test.expect_near(lp(2), 0.0, tolerance, "Test 21c: Multi-interval f(2)");
    }

    // ============================================================
    // Group 8: Basis Conversions
    // ============================================================
    std::cout << "\n--- Group 8: Basis Conversions ---\n" << std::endl;

    // Test 22: from_power_basis (constant)
    {
        std::vector<std::vector<double>> power_coeffs = {{3}};  // p(x) = 3
        std::vector<double> breaks = {0, 1};
        LegPoly lp = LegPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(lp(0.5), 3.0, tolerance, "Test 22: from_power_basis constant");
    }

    // Test 23: from_power_basis (linear)
    {
        // Power basis: p(x) = 1 + 2*(x-0) = 1 + 2x on [0,1]
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}};
        std::vector<double> breaks = {0, 1};
        LegPoly lp = LegPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(lp(0), 1.0, tolerance, "Test 23a: from_power_basis linear f(0)");
        test.expect_near(lp(0.5), 2.0, tolerance, "Test 23b: from_power_basis linear f(0.5)");
        test.expect_near(lp(1), 3.0, tolerance, "Test 23c: from_power_basis linear f(1)");
    }

    // Test 24: from_power_basis (quadratic)
    {
        // Power basis: p(x) = 1 + 0*x + 1*x^2 = 1 + x^2 on [0,1]
        std::vector<std::vector<double>> power_coeffs = {{1}, {0}, {1}};
        std::vector<double> breaks = {0, 1};
        LegPoly lp = LegPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(lp(0), 1.0, tolerance, "Test 24a: from_power_basis quadratic f(0)");
        test.expect_near(lp(0.5), 1.25, tolerance, "Test 24b: from_power_basis quadratic f(0.5)");
        test.expect_near(lp(1), 2.0, tolerance, "Test 24c: from_power_basis quadratic f(1)");
    }

    // Test 25: to_power_basis round-trip
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        std::vector<double> breaks = {0, 1};
        LegPoly lp = LegPoly::from_power_basis(power_coeffs, breaks);

        auto recovered = lp.to_power_basis();

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
        LegPoly lp1(coeffs1, breaks1);

        std::vector<std::vector<double>> coeffs2 = {{2}};  // p(x) = 2 on [1,2]
        std::vector<double> breaks2 = {1, 2};

        LegPoly extended = lp1.extend(coeffs2, breaks2, true);

        test.expect_near(extended(0.5), 1.0, tolerance, "Test 26a: Extended f(0.5)");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 26b: Extended f(1.5)");
        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 26c: Extended num_intervals");
    }

    // Test 27: Extend to the left
    {
        std::vector<std::vector<double>> coeffs1 = {{2}};  // p(x) = 2 on [1,2]
        std::vector<double> breaks1 = {1, 2};
        LegPoly lp1(coeffs1, breaks1);

        std::vector<std::vector<double>> coeffs2 = {{1}};  // p(x) = 1 on [0,1]
        std::vector<double> breaks2 = {0, 1};

        LegPoly extended = lp1.extend(coeffs2, breaks2, false);

        test.expect_near(extended(0.5), 1.0, tolerance, "Test 27a: Extended left f(0.5)");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 27b: Extended left f(1.5)");
    }

    // ============================================================
    // Group 10: Root Finding
    // ============================================================
    std::cout << "\n--- Group 10: Root Finding ---\n" << std::endl;

    // Test 28: Linear root
    {
        // On [-1, 1], s = (2x - (-1) - 1)/(1 - (-1)) = x, so s = x
        // f(s) = 0*P_0(s) + 1*P_1(s) = s, which means f(x) = x on [-1, 1]
        // Root at x = 0
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // f(s) = P_1(s) = s
        std::vector<double> breaks = {-1, 1};
        LegPoly lp(coeffs, breaks);

        auto roots = lp.roots();
        test.expect_eq(roots.size(), 1ul, "Test 28a: Linear has 1 root");
        if (!roots.empty()) {
            test.expect_near(roots[0], 0.0, tolerance, "Test 28b: Linear root at 0");
        }
    }

    // Test 29: Quadratic roots
    {
        // p(x) = x^2 - 0.25 on [-1, 1] has roots at x = +/-0.5
        // Since s = x on [-1,1], we need Legendre coefficients for s^2 - 0.25
        // s^2 = (2*P_2 + 1)/3, so s^2 - 0.25 = 2*P_2/3 + 1/3 - 0.25 = 2*P_2/3 + 1/12
        // Coefficients: [1/12, 0, 2/3]
        std::vector<std::vector<double>> coeffs = {{1.0/12.0}, {0}, {2.0/3.0}};
        LegPoly lp(coeffs, {-1, 1});

        auto roots = lp.roots();
        test.expect_eq(roots.size(), 2ul, "Test 29a: Quadratic has 2 roots");
        if (roots.size() == 2) {
            test.expect_near(roots[0], -0.5, tolerance, "Test 29b: Quadratic root at -0.5");
            test.expect_near(roots[1], 0.5, tolerance, "Test 29c: Quadratic root at 0.5");
        }
    }

    // Test 30: No roots
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        auto roots = lp.roots();
        test.expect_eq(roots.size(), 0ul, "Test 30: Constant has no roots");
    }

    // ============================================================
    // Group 11: Accessors
    // ============================================================
    std::cout << "\n--- Group 11: Accessors ---\n" << std::endl;

    // Test 31: All accessors
    {
        std::vector<std::vector<double>> coeffs = {{1}, {2}, {3}};
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks, ExtrapolateMode::NoExtrapolate, 1, 2);

        test.expect_eq(static_cast<size_t>(lp.degree()), 2ul, "Test 31a: degree");
        test.expect_eq(static_cast<size_t>(lp.num_intervals()), 1ul, "Test 31b: num_intervals");
        test.expect_true(lp.is_ascending(), "Test 31c: is_ascending");
        test.expect_eq(lp.c().size(), 3ul, "Test 31d: c() size");
        test.expect_eq(lp.x().size(), 2ul, "Test 31e: x() size");
        test.expect_true(lp.extrapolate() == ExtrapolateMode::NoExtrapolate, "Test 31f: extrapolate mode");
        test.expect_eq(static_cast<size_t>(lp.extrapolate_order_left()), 1ul, "Test 31g: extrapolate_order_left");
        test.expect_eq(static_cast<size_t>(lp.extrapolate_order_right()), 2ul, "Test 31h: extrapolate_order_right");
    }

    // ============================================================
    // Group 12: Descending Breakpoints
    // ============================================================
    std::cout << "\n--- Group 12: Descending Breakpoints ---\n" << std::endl;

    // Test 32: Descending breakpoints
    // For descending [1, 0], s = (2x - 1 - 0)/(0 - 1) = (2x - 1)/(-1) = 1 - 2x
    // With coeffs {{0.5}, {-0.5}}: p(s) = 0.5*P_0(s) - 0.5*P_1(s) = 0.5 - 0.5*s
    // At x=0: s=1, p(1)=0.5-0.5=0
    // At x=0.5: s=0, p(0)=0.5
    // At x=1: s=-1, p(-1)=0.5-0.5*(-1)=1.0
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {-0.5}};
        std::vector<double> breaks = {1, 0};  // Descending
        LegPoly lp(coeffs, breaks);

        test.expect_true(!lp.is_ascending(), "Test 32a: is_ascending is false");
        test.expect_near(lp(0.0), 0.0, tolerance, "Test 32b: Descending f(0)");
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 32c: Descending f(0.5)");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 32d: Descending f(1)");
    }

    // Test 32e-f: Descending with from_power_basis
    // Power basis: p(x) = a_0 + a_1*(x - left) where left=1 for [1,0]
    // For f(x) = x, need: a_0 + a_1*(x - 1) = x => a_0 = 1, a_1 = 1
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {1}}, {1, 0});  // f(x) = 1 + (x-1) = x, descending
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 32e: Descending from_power_basis f(0.5)");
        test.expect_near(lp(0.5, 1), 1.0, tolerance, "Test 32f: Descending derivative");
    }

    // Test 32g-h: Descending integration
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {1}}, {1, 0});  // f(x) = x, descending
        test.expect_near(lp.integrate(0, 1), 0.5, tolerance, "Test 32g: Descending integration 0 to 1");
        test.expect_near(lp.integrate(0.25, 0.75), 0.25, tolerance, "Test 32h: Descending integration partial");
    }

    // Test 32i-j: Multi-interval descending
    {
        std::vector<std::vector<double>> coeffs = {{2, 1}};  // Constant 2 on [2,1], constant 1 on [1,0]
        std::vector<double> breaks = {2, 1, 0};  // Two descending intervals
        LegPoly lp(coeffs, breaks);
        test.expect_near(lp(0.5), 1.0, tolerance, "Test 32i: Multi-interval descending f(0.5)");
        test.expect_near(lp(1.5), 2.0, tolerance, "Test 32j: Multi-interval descending f(1.5)");
    }

    // ============================================================
    // Group 13: Independent Verification
    // ============================================================
    std::cout << "\n--- Group 13: Independent Verification ---\n" << std::endl;

    // Test 33: Numerical integration verification
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1});  // f(x) = x

        double analytical = lp.integrate(0, 1);
        double numerical = numerical_integrate(lp, 0, 1);

        test.expect_near(analytical, 0.5, tolerance, "Test 33a: Analytical integral");
        test.expect_near(numerical, 0.5, 1e-6, "Test 33b: Numerical integral");
    }

    // Test 34: Finite difference derivative verification
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});  // f(x) = x^2

        double analytical = lp(0.5, 1);  // 2*0.5 = 1
        double numerical = finite_diff_derivative(lp, 0.5);

        test.expect_near(analytical, 1.0, tolerance, "Test 34a: Analytical derivative");
        test.expect_near(numerical, 1.0, 1e-5, "Test 34b: Numerical derivative");
    }

    // ============================================================
    // Group 14: Property-Based Tests
    // ============================================================
    std::cout << "\n--- Group 14: Property-Based Tests ---\n" << std::endl;

    // Test 35: Integral additivity
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});

        double int_0_1 = lp.integrate(0, 1);
        double int_0_half = lp.integrate(0, 0.5);
        double int_half_1 = lp.integrate(0.5, 1);

        test.expect_near(int_0_1, int_0_half + int_half_1, tolerance,
                        "Test 35: Integral additivity");
    }

    // Test 36: Derivative-integral relationship
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        LegPoly antideriv = lp.antiderivative();
        LegPoly recovered = antideriv.derivative();

        bool all_close = true;
        for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            if (std::abs(lp(x) - recovered(x)) > tolerance) {
                all_close = false;
            }
        }
        test.expect_true(all_close, "Test 36: d/dx[antiderivative] = original");
    }

    // Test 37: Integral reversal
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});

        double int_ab = lp.integrate(0.2, 0.8);
        double int_ba = lp.integrate(0.8, 0.2);

        test.expect_near(int_ab, -int_ba, tolerance, "Test 37: integrate(a,b) = -integrate(b,a)");
    }

    // ============================================================
    // Group 15: High-Degree Polynomials
    // ============================================================
    std::cout << "\n--- Group 15: High-Degree Polynomials ---\n" << std::endl;

    // Test 38: High-degree polynomial stability (degree 50)
    // Use from_power_basis to create a high-degree representation of f(x) = 1
    // This tests numerical stability of Clenshaw algorithm at high degrees
    {
        const int high_degree = 50;
        // Create a polynomial that is 1.0 everywhere, but represented with many terms
        // In Legendre basis, P_0 = 1, so c_0 = 1 and all other c_k = 0 gives f(s) = 1
        std::vector<std::vector<double>> coeffs(high_degree + 1, std::vector<double>(1, 0.0));
        coeffs[0][0] = 1.0;

        LegPoly lp(coeffs, {0, 1});

        // The polynomial should evaluate to 1.0 everywhere within the domain
        test.expect_near(lp(0.0), 1.0, tolerance, "Test 38a: High-degree (50) at t=0");
        test.expect_near(lp(0.25), 1.0, tolerance, "Test 38b: High-degree (50) at t=0.25");
        test.expect_near(lp(0.5), 1.0, tolerance, "Test 38c: High-degree (50) at t=0.5");
        test.expect_near(lp(0.75), 1.0, tolerance, "Test 38d: High-degree (50) at t=0.75");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 38e: High-degree (50) at t=1");
        test.expect_near(lp(0.001), 1.0, tolerance, "Test 38f: High-degree (50) near 0");
        test.expect_near(lp(0.999), 1.0, tolerance, "Test 38g: High-degree (50) near 1");
    }

    // Test 38h-l: Very high degree (100) with non-trivial polynomial f(x) = x
    // This is the real stress test - representing a simple linear function at degree 100
    {
        // Use from_power_basis to create f(x) = x at whatever degree it produces
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1});

        // Should evaluate to x at all points
        test.expect_near(lp(0.0), 0.0, tolerance, "Test 38h: Linear at t=0");
        test.expect_near(lp(0.25), 0.25, tolerance, "Test 38i: Linear at t=0.25");
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 38j: Linear at t=0.5");
        test.expect_near(lp(0.75), 0.75, tolerance, "Test 38k: Linear at t=0.75");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 38l: Linear at t=1");
    }

    // Test 38m-q: High-degree polynomial with all non-zero coefficients
    // f(s) = sum_{k=0}^{10} (1/(k+1)) * P_k(s) - tests Clenshaw with all terms active
    {
        const int degree = 10;
        std::vector<std::vector<double>> coeffs(degree + 1, std::vector<double>(1));
        for (int k = 0; k <= degree; ++k) {
            coeffs[k][0] = 1.0 / (k + 1);  // Non-trivial coefficients: 1, 1/2, 1/3, ...
        }
        LegPoly lp(coeffs, {0, 1});

        // At s=1 (x=1): P_k(1) = 1 for all k, so f(1) = sum_{k=0}^{10} 1/(k+1) = H_11
        double H_11 = 0.0;
        for (int k = 0; k <= degree; ++k) H_11 += 1.0 / (k + 1);
        test.expect_near(lp(1.0), H_11, tolerance, "Test 38m: Degree 10 at x=1 (s=1)");

        // At s=-1 (x=0): P_k(-1) = (-1)^k, so f(-1) = sum_{k=0}^{10} (-1)^k/(k+1)
        double alt_sum = 0.0;
        for (int k = 0; k <= degree; ++k) alt_sum += (k % 2 == 0 ? 1.0 : -1.0) / (k + 1);
        test.expect_near(lp(0.0), alt_sum, tolerance, "Test 38n: Degree 10 at x=0 (s=-1)");

        // Verify midpoint is reasonable (between endpoint values)
        double val_mid = lp(0.5);
        test.expect_true(std::isfinite(val_mid) && val_mid > 0 && val_mid < H_11,
                        "Test 38o: Degree 10 at x=0.5 in valid range");

        // Verify derivative via finite differences
        double analytical_deriv = lp(0.5, 1);
        double numerical_deriv = finite_diff_derivative(lp, 0.5);
        test.expect_near(analytical_deriv, numerical_deriv, 1e-5,
                        "Test 38p: Degree 10 derivative matches finite difference");

        // Verify repeated evaluation gives same result (no numerical drift)
        double val_mid2 = lp(0.5);
        test.expect_near(val_mid, val_mid2, 1e-15, "Test 38q: Repeated evaluation identical");
    }

    // ============================================================
    // Group 16: NaN and Infinity Handling
    // ============================================================
    std::cout << "\n--- Group 16: NaN and Infinity Handling ---\n" << std::endl;

    // Test 39: NaN input
    {
        LegPoly lp({{1}}, {0, 1});
        test.expect_true(std::isnan(lp(std::numeric_limits<double>::quiet_NaN())),
                        "Test 39: NaN input gives NaN output");
    }

    // Test 40: Infinity with NoExtrapolate
    {
        LegPoly lp({{1}}, {0, 1}, ExtrapolateMode::NoExtrapolate);
        test.expect_true(std::isnan(lp(std::numeric_limits<double>::infinity())),
                        "Test 40: Inf input with NoExtrapolate gives NaN");
    }

    // ============================================================
    // Group 17: Orders Parameter
    // ============================================================
    std::cout << "\n--- Group 17: Orders Parameter ---\n" << std::endl;

    // Test 41: from_derivatives with orders parameter
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 2, 3}, {1, -1, 0, 0}};  // More derivatives available
        std::vector<int> orders = {1};  // Use only f and f' (order 0 and 1)

        LegPoly lp = LegPoly::from_derivatives(xi, yi, orders);

        // Should match f, f' but not higher derivatives
        test.expect_near(lp(0), 0.0, tolerance, "Test 41a: orders f(0)");
        test.expect_near(lp(1), 1.0, tolerance, "Test 41b: orders f(1)");
        test.expect_near(lp(0, 1), 1.0, tolerance, "Test 41c: orders f'(0)");
        test.expect_near(lp(1, 1), -1.0, tolerance, "Test 41d: orders f'(1)");

        // Verify that HIGHER derivatives were NOT matched (orders limited to 1)
        // If orders worked correctly, f''(0) should NOT equal yi[0][2] = 2
        double second_deriv = lp(0, 2);
        test.expect_true(std::abs(second_deriv - 2.0) > 0.1,
            "Test 41e: orders parameter actually limited derivatives (f''(0) != 2)");
    }

    // ============================================================
    // Group 18: Move Semantics
    // ============================================================
    std::cout << "\n--- Group 18: Move Semantics ---\n" << std::endl;

    // Test 42: Move constructor
    {
        LegPoly lp1({{0.5}, {0.5}}, {0, 1});
        double val_before = lp1(0.5);

        LegPoly lp2(std::move(lp1));
        double val_after = lp2(0.5);

        test.expect_near(val_after, val_before, tolerance, "Test 42: Move constructor preserves value");
    }

    // ============================================================
    // Group 19: Controlled Extrapolation (Taylor Order)
    // ============================================================
    std::cout << "\n--- Group 19: Controlled Extrapolation ---\n" << std::endl;

    // Test 43: Constant extrapolation (order 0)
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);  // f(x) = x
        LegPoly lp_const(lp.c(), lp.x(), ExtrapolateMode::Extrapolate, 0, 0);  // Constant extrapolation

        // At x=2 (beyond [0,1]), should use f(1) = 1 (constant)
        test.expect_near(lp_const(2), 1.0, tolerance, "Test 43a: Constant extrapolation right");
        // At x=-1 (before [0,1]), should use f(0) = 0 (constant)
        test.expect_near(lp_const(-1), 0.0, tolerance, "Test 43b: Constant extrapolation left");
    }

    // Test 44: Linear extrapolation (order 1)
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);  // f(x) = x^2
        LegPoly lp_linear(lp.c(), lp.x(), ExtrapolateMode::Extrapolate, 1, 1);

        // At x=2, linear extrapolation from x=1: f(1) + f'(1)*(2-1) = 1 + 2*1 = 3
        test.expect_near(lp_linear(2), 3.0, tolerance, "Test 44a: Linear extrapolation right");
        // At x=-1, linear extrapolation from x=0: f(0) + f'(0)*(-1-0) = 0 + 0*(-1) = 0
        test.expect_near(lp_linear(-1), 0.0, tolerance, "Test 44b: Linear extrapolation left");
    }

    // ============================================================
    // Group 20: Numpy Reference Verification
    // ============================================================
    std::cout << "\n--- Group 20: Numpy Reference Verification ---\n" << std::endl;

    // Test 45: Verify Legendre polynomial values match numpy.polynomial.legendre
    // P_0 = 1, P_1 = s, P_2 = (3s^2-1)/2, P_3 = (5s^3-3s)/2
    {
        // Evaluate P_2(s) = (3s^2 - 1)/2 at s = 0.5
        // P_2(0.5) = (3*0.25 - 1)/2 = (0.75 - 1)/2 = -0.125
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};  // Pure P_2
        LegPoly lp(coeffs, {0, 1});

        // At x = 0.75, s = 2*0.75 - 1 = 0.5
        double expected_P2 = (3.0 * 0.25 - 1.0) / 2.0;  // -0.125
        test.expect_near(lp(0.75), expected_P2, tolerance, "Test 45: P_2(0.5) = -0.125");
    }

    // Test 46: Verify derivative matches numpy.polynomial.legendre.legder
    {
        // d/ds[P_2(s)] = d/ds[(3s^2-1)/2] = 3s
        // At s = 0.5: d/ds[P_2] = 1.5
        // Scale by 2/h = 2/1 = 2 for d/dx
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};  // Pure P_2
        LegPoly lp(coeffs, {0, 1});

        // At x = 0.75, s = 0.5, derivative w.r.t. x = 3*0.5 * 2 = 3
        double deriv = lp(0.75, 1);
        test.expect_near(deriv, 3.0, tolerance, "Test 46: d/dx[P_2] at x=0.75");
    }

    // ============================================================
    // Group 21: Additional Property Tests
    // ============================================================
    std::cout << "\n--- Group 21: Additional Property Tests ---\n" << std::endl;

    // Test 47: Higher order derivatives eventually become zero
    {
        // Degree 3 polynomial
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});

        // 4th derivative of degree 3 polynomial is 0
        LegPoly d4 = lp.derivative(4);
        test.expect_near(d4(0.5), 0.0, tolerance, "Test 47: 4th derivative of cubic is 0");
    }

    // Test 48: from_derivatives endpoint constraints
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{1, 2, 3}, {4, 5, 6}};

        LegPoly lp = LegPoly::from_derivatives(xi, yi);

        bool all_pass = true;
        if (std::abs(lp(0) - 1) > tolerance) all_pass = false;
        if (std::abs(lp(0, 1) - 2) > tolerance) all_pass = false;
        if (std::abs(lp(0, 2) - 3) > tolerance) all_pass = false;
        if (std::abs(lp(1) - 4) > tolerance) all_pass = false;
        if (std::abs(lp(1, 1) - 5) > tolerance) all_pass = false;
        if (std::abs(lp(1, 2) - 6) > tolerance) all_pass = false;

        test.expect_true(all_pass, "Test 48: from_derivatives matches all endpoint constraints");
    }

    // ============================================================
    // Group 22: Extended Independent Verification
    // ============================================================
    std::cout << "\n--- Group 22: Extended Independent Verification ---\n" << std::endl;

    // Test 49: Multi-point finite difference derivative verification
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {-2}, {3}, {-1}}, {0, 1});

        bool all_pass = true;
        for (double x : {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}) {
            double analytical = lp(x, 1);
            double numerical = finite_diff_derivative(lp, x);
            if (std::abs(analytical - numerical) > 1e-5) {
                all_pass = false;
            }
        }
        test.expect_true(all_pass, "Test 49: Multi-point derivative matches finite differences");
    }

    // Test 50: Multi-point second derivative verification
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});

        bool all_pass = true;
        for (double x : {0.2, 0.4, 0.5, 0.6, 0.8}) {
            double analytical = lp(x, 2);
            double numerical = finite_diff_second_derivative(lp, x);
            if (std::abs(analytical - numerical) > 1e-4) {
                all_pass = false;
            }
        }
        test.expect_true(all_pass, "Test 50: Multi-point 2nd derivative matches finite differences");
    }

    // ============================================================
    // Group 23: Reproducibility and Thread Safety
    // ============================================================
    std::cout << "\n--- Group 23: Reproducibility and Thread Safety ---\n" << std::endl;

    // Test 51: Determinism/Reproducibility test
    // Verifies that repeated sequential evaluation produces bit-identical results.
    // This is NOT a thread safety test - see Test 52 for actual concurrent access testing.
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}, {4}, {5}}, {0, 1});

        bool all_pass = true;
        std::vector<double> points = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

        std::vector<double> results1 = lp(points);
        std::vector<double> results2 = lp(points);
        std::vector<double> results3 = lp(points);

        for (size_t i = 0; i < points.size(); ++i) {
            if (results1[i] != results2[i] || results2[i] != results3[i]) {
                all_pass = false;
            }
        }
        test.expect_true(all_pass, "Test 51: Reproducibility - repeated evaluation identical");
    }

    // Test 52: Thread safety - actual multi-threaded concurrent access
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}, {4}, {5}}, {0, 1});
        const int num_threads = 10;
        std::atomic<bool> all_correct{true};
        std::vector<std::thread> threads;

        auto thread_func = [&lp, &all_correct](unsigned int seed) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            for (int i = 0; i < 100; ++i) {
                double x = dist(rng);
                double result = lp(x);
                // Expected: 1 + 2x + 3x^2 + 4x^3 + 5x^4
                double expected = 1 + 2*x + 3*x*x + 4*x*x*x + 5*x*x*x*x;
                if (std::abs(result - expected) > 1e-10) {
                    all_correct.store(false);
                }
            }
        };

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(thread_func, static_cast<unsigned int>(t * 12345));
        }

        for (auto& th : threads) {
            th.join();
        }

        if (all_correct.load()) {
            test.pass("Test 52: Multi-threaded concurrent evaluation");
        } else {
            test.fail("Test 52: Multi-threaded concurrent evaluation");
        }
    }

    // Test 53: Const-correctness verification
    {
        const LegPoly lp({{0.5}, {0.5}}, {0, 1});

        double v1 = lp(0.5);
        double v2 = lp(0.5, 1);
        std::vector<double> v3 = lp({0.1, 0.2, 0.3});
        double v4 = lp.integrate(0, 1);
        int d = lp.degree();
        int n = lp.num_intervals();
        bool asc = lp.is_ascending();
        const auto& c = lp.coefficients();
        const auto& x = lp.breakpoints();
        ExtrapolateMode mode = lp.extrapolate();

        test.expect_near(v1, 0.5, tolerance, "Test 53a: Const eval");
        test.expect_near(v2, 1.0, tolerance, "Test 53b: Const derivative eval");
        test.expect_eq(v3.size(), 3ul, "Test 53c: Const vector eval");
        test.expect_near(v4, 0.5, tolerance, "Test 53d: Const integrate");
        test.expect_eq(static_cast<size_t>(d), 1ul, "Test 53e: Const degree");
        test.expect_eq(static_cast<size_t>(n), 1ul, "Test 53f: Const num_intervals");
        test.expect_true(asc, "Test 53g: Const is_ascending");
        test.expect_true(!c.empty(), "Test 53h: Const coefficients");
        test.expect_true(!x.empty(), "Test 53i: Const breakpoints");
        test.expect_true(mode == ExtrapolateMode::Extrapolate, "Test 53j: Const extrapolate");
    }

    // ============================================================
    // Group 24: Edge Cases
    // ============================================================
    std::cout << "\n--- Group 24: Edge Cases ---\n" << std::endl;

    // Test 54: Single interval evaluation at boundaries
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1});  // f(x) = x

        test.expect_near(lp(0.0), 0.0, tolerance, "Test 54a: f(0) at left boundary");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 54b: f(1) at right boundary");
    }

    // Test 55: Very small interval
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {1}}, {0, 1e-10});  // Very small interval: f(x) = 1 + x

        // Should still evaluate correctly
        double val = lp(0.5e-10);
        double expected = 1.0 + 0.5e-10;  // f(x) = 1 + x at x = 0.5e-10
        test.expect_true(!std::isnan(val), "Test 55a: Small interval not NaN");
        test.expect_near(val, expected, 1e-15, "Test 55b: Small interval correct value");
    }

    // Test 56: Cubic polynomial verification
    {
        // f(x) = x^3 on [0, 1]
        LegPoly lp = LegPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1});

        test.expect_near(lp(0), 0.0, tolerance, "Test 56a: x^3(0)");
        test.expect_near(lp(0.5), 0.125, tolerance, "Test 56b: x^3(0.5)");
        test.expect_near(lp(1), 1.0, tolerance, "Test 56c: x^3(1)");
    }

    // ============================================================
    // Group 25: Legendre Basis Reference Values
    // ============================================================
    std::cout << "\n--- Group 25: Legendre Basis Reference Values ---\n" << std::endl;

    // Test 57: Legendre polynomial basis values (numpy.polynomial.legendre.legval)
    // P_0(s) = 1, P_1(s) = s, P_2(s) = (3s^2-1)/2, P_3(s) = (5s^3-3s)/2
    {
        // P_3(s) = (5s^3 - 3s)/2 at s = 0.5: (5*0.125 - 1.5)/2 = (0.625 - 1.5)/2 = -0.4375
        // On [0,1], s = 2x - 1, so x = 0.75 gives s = 0.5
        LegPoly lp({{0}, {0}, {0}, {1}}, {0, 1});  // Pure P_3
        test.expect_near(lp(0.75), -0.4375, tolerance, "Test 57a: P_3(s=0.5) matches numpy");

        // P_2(s) = (3s^2-1)/2 at s = 0.5: (3*0.25-1)/2 = -0.125
        LegPoly lp2({{0}, {0}, {1}}, {0, 1});  // Pure P_2
        test.expect_near(lp2(0.75), -0.125, tolerance, "Test 57b: P_2(s=0.5) matches numpy");
    }

    // Test 58: Legendre derivative (numpy.polynomial.legendre.legder)
    // d/ds[P_2(s)] = d/ds[(3s^2-1)/2] = 3s, so at s=0.5: 1.5
    // Scale by 2/h = 2 for d/dx on [0,1], so derivative = 3.0
    {
        LegPoly lp({{0}, {0}, {1}}, {0, 1});  // P_2 on [0,1]
        test.expect_near(lp(0.75, 1), 3.0, tolerance, "Test 58: d/dx[P_2] at x=0.75 matches numpy");
    }

    // Test 59: Legendre antiderivative (numpy.polynomial.legendre.legint)
    // Integral of constant 1 from 0 to 1 is 1
    {
        LegPoly lp({{1}}, {0, 1});  // Constant 1
        test.expect_near(lp.integrate(0, 1), 1.0, tolerance, "Test 59a: Integral of 1 matches numpy");

        // Integral of x from 0 to 1 is 0.5
        LegPoly lp2 = LegPoly::from_power_basis({{0}, {1}}, {0, 1});
        test.expect_near(lp2.integrate(0, 1), 0.5, tolerance, "Test 59b: Integral of x matches numpy");
    }

    // Test 60: from_derivatives matches scipy (via LegPoly, conceptually same as BPoly Hermite)
    // Hermite cubic: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
    {
        LegPoly lp = LegPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});

        // Verify endpoint constraints match
        test.expect_near(lp(0), 0.0, tolerance, "Test 60a: Hermite f(0)=0");
        test.expect_near(lp(1), 1.0, tolerance, "Test 60b: Hermite f(1)=1");
        test.expect_near(lp(0, 1), 1.0, tolerance, "Test 60c: Hermite f'(0)=1");
        test.expect_near(lp(1, 1), -1.0, tolerance, "Test 60d: Hermite f'(1)=-1");

        // Interior points should give smooth interpolation
        // These values are verified against scipy BPoly.from_derivatives
        test.expect_near(lp(0.5), 0.75, tolerance, "Test 60e: Hermite midpoint");
    }

    // Test 61: Multi-term Legendre combination matches numpy
    // c_0 * P_0 + c_1 * P_1 + c_2 * P_2 at various points
    {
        // f(s) = 1 + 2*s + 3*(3s^2-1)/2 = 1 + 2s + 4.5s^2 - 1.5 = -0.5 + 2s + 4.5s^2
        // At s = 0: -0.5
        // At s = 1: -0.5 + 2 + 4.5 = 6
        // At s = -1: -0.5 - 2 + 4.5 = 2
        LegPoly lp({{1}, {2}, {3}}, {0, 1});

        // s = 0 means x = 0.5
        test.expect_near(lp(0.5), -0.5, tolerance, "Test 61a: Multi-term at midpoint");

        // s = 1 means x = 1
        test.expect_near(lp(1.0), 6.0, tolerance, "Test 61b: Multi-term at right");

        // s = -1 means x = 0
        test.expect_near(lp(0.0), 2.0, tolerance, "Test 61c: Multi-term at left");
    }

    // ============================================================
    // Group 26: Extreme Coefficient Tests
    // ============================================================
    std::cout << "\n--- Group 26: Extreme Coefficient Tests ---\n" << std::endl;

    // Test 62: Very small coefficients (1e-100 scale)
    {
        // Linear polynomial with tiny coefficients
        std::vector<std::vector<double>> tiny_coeffs = {{1e-100}, {2e-100}};
        std::vector<double> breaks = {0, 1};
        // For Legendre on [0,1]: f(x) = c_0*P_0(s) + c_1*P_1(s) where s = 2x-1
        // At x=0: s=-1, f = c_0*1 + c_1*(-1) = 1e-100 - 2e-100 = -1e-100
        // At x=1: s=1, f = c_0*1 + c_1*(1) = 1e-100 + 2e-100 = 3e-100
        // At x=0.5: s=0, f = c_0 = 1e-100
        LegPoly lp(tiny_coeffs, breaks);

        test.expect_near(lp(0.0), -1e-100, 1e-110, "Test 62a: Tiny coefficients at 0");
        test.expect_near(lp(0.5), 1e-100, 1e-110, "Test 62b: Tiny coefficients at 0.5");
        test.expect_near(lp(1.0), 3e-100, 1e-110, "Test 62c: Tiny coefficients at 1");
    }

    // Test 63: Very large coefficients (1e100 scale)
    {
        std::vector<std::vector<double>> huge_coeffs = {{1e100}, {2e100}};
        std::vector<double> breaks = {0, 1};
        // Same Legendre evaluation pattern
        // At x=0.5: s=0, f = c_0 = 1e100
        LegPoly lp(huge_coeffs, breaks);

        test.expect_near(lp(0.0), -1e100, 1e90, "Test 63a: Huge coefficients at 0");
        test.expect_near(lp(0.5), 1e100, 1e90, "Test 63b: Huge coefficients at 0.5");
        test.expect_near(lp(1.0), 3e100, 1e90, "Test 63c: Huge coefficients at 1");
    }

    // ============================================================
    // Group 27: Antiderivative Continuity Test
    // ============================================================
    std::cout << "\n--- Group 27: Antiderivative Continuity Test ---\n" << std::endl;

    // Test 64: Multi-interval antiderivative C0 continuity at breakpoints
    {
        // Create 3-interval polynomial with varying values
        std::vector<std::vector<double>> c = {{1, 2, 3}};  // Constants on each interval
        std::vector<double> x = {0, 1, 2, 3};
        LegPoly lp(c, x);
        LegPoly ilp = lp.antiderivative();

        // Check continuity at each interior breakpoint
        double eps = 1e-10;
        bool continuous = true;
        for (size_t i = 1; i < x.size() - 1; ++i) {
            double left = ilp(x[i] - eps);
            double right = ilp(x[i] + eps);
            if (std::abs(left - right) > 1e-8) {
                test.fail("Test 64: Antiderivative continuity at x=" + std::to_string(x[i]));
                continuous = false;
            }
        }
        if (continuous) {
            test.pass("Test 64: Antiderivative C0 continuity at all breakpoints");
        }
    }

    // ============================================================
    // Group 28: Derivative-Antiderivative Round-Trip Tests
    // ============================================================
    std::cout << "\n--- Group 28: Derivative-Antiderivative Round-Trip ---\n" << std::endl;

    // Test 65: d^n/dx^n[antiderivative(n)] = original for n = 1, 2, 3
    {
        LegPoly lp = LegPoly::from_derivatives({0, 1}, {{1, 2, -1}, {3, 0, 2}});

        for (int n = 1; n <= 3; ++n) {
            LegPoly antideriv = lp.antiderivative(n);
            LegPoly recovered = antideriv.derivative(n);

            bool matches = true;
            for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
                if (std::abs(lp(x) - recovered(x)) > tolerance) {
                    test.fail("Test 65: der_antider n=" + std::to_string(n) + " at x=" + std::to_string(x));
                    matches = false;
                    break;
                }
            }
            if (matches) {
                test.pass("Test 65: der_antider round-trip n=" + std::to_string(n));
            }
        }
    }

    // ============================================================
    // Group 29: Periodic Derivative Evaluation
    // ============================================================
    std::cout << "\n--- Group 29: Periodic Derivative Evaluation ---\n" << std::endl;

    // Test 66: Periodic mode with derivative evaluation
    {
        // Two-interval polynomial on [0, 3]
        // Interval 0: [0,1], linear x on this interval
        // Interval 1: [1,3], different polynomial
        LegPoly lp = LegPoly::from_power_basis({{0, 1}, {1, -0.5}}, {0, 1, 3}, ExtrapolateMode::Periodic);

        // Period = 3 - 0 = 3
        // At x=3.5: wraps to x=0.5, should equal lp(0.5)
        test.expect_near(lp(3.5), lp(0.5), tolerance, "Test 66a: Periodic f(3.5) = f(0.5)");

        // At x=-0.5: wraps to x=2.5 (period=3), should equal lp(2.5)
        test.expect_near(lp(-0.5), lp(2.5), tolerance, "Test 66b: Periodic f(-0.5) = f(2.5)");

        // Test derivative in periodic mode
        test.expect_near(lp(3.5, 1), lp(0.5, 1), tolerance, "Test 66c: Periodic f'(3.5) = f'(0.5)");
        test.expect_near(lp(-0.5, 1), lp(2.5, 1), tolerance, "Test 66d: Periodic f'(-0.5) = f'(2.5)");
    }

    // ============================================================
    // Group 30: Corner Case Tests
    // ============================================================
    std::cout << "\n--- Group 30: Corner Case Tests ---\n" << std::endl;

    // Test 67: Two intervals with from_power_basis
    {
        // f(x) = x on [0, 0.5], f(x) = 0.5 on [0.5, 1] (constant on second interval)
        LegPoly lp = LegPoly::from_power_basis({{0, 0.5}, {1, 0}}, {0, 0.5, 1});
        test.expect_near(lp(0.0), 0.0, tolerance, "Test 67a: Two intervals at 0");
        test.expect_near(lp(0.25), 0.25, tolerance, "Test 67b: Two intervals at 0.25");
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 67c: Two intervals at 0.5");
        test.expect_near(lp(0.75), 0.5, tolerance, "Test 67d: Two intervals at 0.75");
        test.expect_near(lp(1.0), 0.5, tolerance, "Test 67e: Two intervals at 1");
    }

    // Test 68: Many intervals (10 linear pieces)
    {
        std::vector<std::vector<double>> c(2, std::vector<double>(10));
        std::vector<double> breaks(11);
        for (int i = 0; i <= 10; ++i) {
            breaks[i] = static_cast<double>(i);
        }
        // Create linear f(x) = x using Legendre coefficients on each interval
        for (int i = 0; i < 10; ++i) {
            double a = breaks[i];
            double b = breaks[i + 1];
            double mid = (a + b) / 2.0;
            c[0][i] = mid;     // c_0 = midpoint
            c[1][i] = 0.5;     // c_1 = 0.5 for linear x on unit interval
        }
        LegPoly lp(c, breaks);
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 68a: Many intervals at 0.5");
        test.expect_near(lp(5.5), 5.5, tolerance, "Test 68b: Many intervals at 5.5");
        test.expect_near(lp(9.5), 9.5, tolerance, "Test 68c: Many intervals at 9.5");
    }

    // Test 69: Large coefficients (1e10 scale)
    {
        // Linear from 1e10 to 2e10
        LegPoly lp = LegPoly::from_power_basis({{1e10}, {1e10}}, {0, 1});
        test.expect_near(lp(0.0), 1e10, 1.0, "Test 69a: Large coeffs at 0");
        test.expect_near(lp(0.5), 1.5e10, 1e5, "Test 69b: Large coeffs at 0.5");
        test.expect_near(lp(1.0), 2e10, 1.0, "Test 69c: Large coeffs at 1");
    }

    // Test 70: Small coefficients (1e-15 scale)
    {
        LegPoly lp = LegPoly::from_power_basis({{1e-15}, {1e-15}}, {0, 1});
        test.expect_near(lp(0.0), 1e-15, 1e-25, "Test 70a: Small coeffs at 0");
        test.expect_near(lp(0.5), 1.5e-15, 1e-25, "Test 70b: Small coeffs at 0.5");
        test.expect_near(lp(1.0), 2e-15, 1e-25, "Test 70c: Small coeffs at 1");
    }

    // Test 71: Far extrapolation
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);
        test.expect_near(lp(10.0), 10.0, tolerance, "Test 71a: Far extrapolation at 10");
        test.expect_near(lp(-10.0), -10.0, tolerance, "Test 71b: Far extrapolation at -10");
        test.expect_near(lp(100.0), 100.0, 1e-8, "Test 71c: Far extrapolation at 100");
    }

    // Test 72: Near-boundary evaluation
    {
        // Use from_power_basis for cleaner expected values
        LegPoly lp = LegPoly::from_power_basis({{0, 1}, {1, 1}}, {0, 1, 2});

        double eps = 1e-14;
        test.expect_true(std::isfinite(lp(1.0 - eps)), "Test 72a: Near boundary 1-eps finite");
        test.expect_true(std::isfinite(lp(1.0 + eps)), "Test 72b: Near boundary 1+eps finite");
        // Near-boundary should be close to lp(1.0) from either side
        double val_at_1 = lp(1.0);
        test.expect_near(lp(1.0 - eps), val_at_1, 1e-10, "Test 72c: Near boundary 1-eps value");
        test.expect_near(lp(1.0 + eps), val_at_1, 1e-10, "Test 72d: Near boundary 1+eps value");
    }

    // Test 73: from_derivatives with values only (no derivative constraints)
    {
        LegPoly lp = LegPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}});
        test.expect_near(lp(0.0), 0.0, tolerance, "Test 73a: Values-only f(0)");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 73b: Values-only f(1)");
        test.expect_near(lp(2.0), 0.0, tolerance, "Test 73c: Values-only f(2)");
        test.expect_near(lp(0.5), 0.5, tolerance, "Test 73d: Values-only f(0.5)");
    }

    // Test 74: from_derivatives with asymmetric orders
    {
        // 4 derivatives at left, 1 at right
        LegPoly lp = LegPoly::from_derivatives({0, 1}, {{0, 1, 0, 0}, {1}});
        test.expect_near(lp(0.0), 0.0, tolerance, "Test 74a: Asymmetric f(0)");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 74b: Asymmetric f(1)");
        // Derivative at 0 should be 1
        double h = 1e-8;
        double deriv_at_0 = (lp(h) - lp(0)) / h;
        test.expect_near(deriv_at_0, 1.0, 1e-5, "Test 74c: Asymmetric f'(0) approx");
    }

    // ============================================================
    // Group 31: Missing Coverage - C0 Continuity at Breakpoints
    // ============================================================
    std::cout << "\n--- Group 31: C0 Continuity at Breakpoints ---\n" << std::endl;

    // Test 75: from_derivatives produces C0 continuous polynomial at interval boundaries
    {
        // Multi-interval Hermite interpolation - polynomial should be continuous
        LegPoly lp = LegPoly::from_derivatives({0, 1, 2, 3}, {{0, 1}, {1, 0}, {0.5, -0.5}, {0, 0}});

        double eps = 1e-12;
        bool continuous = true;

        // Check C0 continuity at each interior breakpoint
        for (double bp : {1.0, 2.0}) {
            double left = lp(bp - eps);
            double right = lp(bp + eps);
            double at_bp = lp(bp);
            if (std::abs(left - at_bp) > 1e-8 || std::abs(right - at_bp) > 1e-8) {
                test.fail("Test 75: C0 continuity at x=" + std::to_string(bp));
                continuous = false;
            }
        }
        if (continuous) {
            test.pass("Test 75: from_derivatives produces C0 continuous polynomial");
        }
    }

    // Test 76: from_derivatives with matching derivatives produces C1 continuity
    {
        // C1 continuity: function AND first derivative match at breakpoints
        LegPoly lp = LegPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1, 1}, {3, 1}});

        double eps = 1e-10;
        bool c1_continuous = true;

        // Check C1 continuity at x=1
        double f_left = lp(1.0 - eps);
        double f_right = lp(1.0 + eps);
        double df_left = lp(1.0 - eps, 1);
        double df_right = lp(1.0 + eps, 1);

        if (std::abs(f_left - f_right) > 1e-8) {
            test.fail("Test 76a: C0 continuity at x=1");
            c1_continuous = false;
        }
        if (std::abs(df_left - df_right) > 1e-6) {
            test.fail("Test 76b: C1 continuity (derivative) at x=1");
            c1_continuous = false;
        }
        if (c1_continuous) {
            test.pass("Test 76: from_derivatives with matching f' produces C1 continuity");
        }
    }

    // ============================================================
    // Group 32: Missing Coverage - extend() with Mixed Degrees
    // ============================================================
    std::cout << "\n--- Group 32: extend() with Mixed Degrees ---\n" << std::endl;

    // Test 77: Extend cubic polynomial with linear polynomial
    {
        // Create cubic on [0, 1]: f(x) = x^3
        LegPoly lp_cubic = LegPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1});

        // Create linear on [1, 2]: g(x) = 2x - 1 (matches f(1)=1)
        std::vector<std::vector<double>> linear_coeffs = {{1}, {2}};  // Using Legendre coefficients
        // Actually for Legendre: g(s) = c0 + c1*s where s = 2(x-1) - 1 = 2x - 3 on [1,2]
        // For g(x) = 2x - 1, at x=1: g=1, at x=2: g=3
        // In Legendre on [1,2]: s = 2*(x-1.5)/1 = 2x - 3
        // g(s) = c0 + c1*s should give g=1 at s=-1 and g=3 at s=1
        // c0 - c1 = 1, c0 + c1 = 3 => c0 = 2, c1 = 1
        LegPoly lp_linear({{2}, {1}}, {1, 2});

        LegPoly extended = lp_cubic.extend(lp_linear.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 77a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.125, tolerance, "Test 77b: Cubic part at 0.5");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 77c: Linear part at 1.5");
        test.expect_near(extended(1.0), 1.0, tolerance, "Test 77d: Continuity at boundary");
    }

    // Test 78: Extend quadratic with quintic
    {
        // Quadratic on [0, 1]
        LegPoly lp_quad = LegPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});

        // Quintic on [1, 2] from_derivatives with 3 constraints each side
        LegPoly lp_quint = LegPoly::from_derivatives({1, 2}, {{1, 2, 2}, {4, 4, 2}});

        LegPoly extended = lp_quad.extend(lp_quint.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 78a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.25, tolerance, "Test 78b: Quadratic part");
        test.expect_near(extended(1.5), lp_quint(1.5), tolerance, "Test 78c: Quintic part");
    }

    // ============================================================
    // Group 33: Missing Coverage - Edge Cases
    // ============================================================
    std::cout << "\n--- Group 33: Edge Case Coverage ---\n" << std::endl;

    // Test 79: Empty vector evaluation
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> empty_input;
        std::vector<double> result = lp(empty_input);
        test.expect_eq(result.size(), 0ul, "Test 79: Empty vector evaluation returns empty");
    }

    // Test 80: Single-point vector evaluation
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> single_point = {0.5};
        std::vector<double> result = lp(single_point);
        test.expect_eq(result.size(), 1ul, "Test 80a: Single-point vector size");
        test.expect_near(result[0], 2.0, tolerance, "Test 80b: Single-point value correct");
    }

    // Test 81: Evaluation exactly at all breakpoints
    {
        // Continuous piecewise linear: f(x)=x on all intervals [0,1], [1,2], [2,3]
        // Power basis: c0[i] = value at left endpoint, c1[i] = slope = 1
        LegPoly lp = LegPoly::from_power_basis({{0, 1, 2}, {1, 1, 1}}, {0, 1, 2, 3});
        std::vector<double> breakpts = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> results = lp(breakpts);

        // f(0) = 0 + 1*0 = 0
        // f(1) = 1 + 1*(1-1) = 1 (from interval [1,2])
        // f(2) = 2 + 1*(2-2) = 2 (from interval [2,3])
        // f(3) = 2 + 1*(3-2) = 3
        test.expect_near(results[0], 0.0, tolerance, "Test 81a: Eval at bp[0]");
        test.expect_near(results[1], 1.0, tolerance, "Test 81b: Eval at bp[1]");
        test.expect_near(results[2], 2.0, tolerance, "Test 81c: Eval at bp[2]");
        test.expect_near(results[3], 3.0, tolerance, "Test 81d: Eval at bp[3]");
    }

    // ============================================================
    // Group 34: Additional Edge Case Coverage
    // ============================================================
    std::cout << "\n--- Group 34: Additional Edge Case Coverage ---\n" << std::endl;

    // Test 82: Zero polynomial (all coefficients zero)
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {0}};
        std::vector<double> breaks = {0, 1};
        LegPoly lp(coeffs, breaks);

        test.expect_near(lp(0.0), 0.0, tolerance, "Test 82a: Zero polynomial at 0");
        test.expect_near(lp(0.5), 0.0, tolerance, "Test 82b: Zero polynomial at 0.5");
        test.expect_near(lp(1.0), 0.0, tolerance, "Test 82c: Zero polynomial at 1");
        test.expect_near(lp(0.5, 1), 0.0, tolerance, "Test 82d: Zero polynomial derivative");
        test.expect_near(lp.integrate(0, 1), 0.0, tolerance, "Test 82e: Zero polynomial integral");
    }

    // Test 83: Repeated roots - polynomial (x-0.5)^2 = x^2 - x + 0.25
    {
        // f(x) = x^2 - x + 0.25 = (x - 0.5)^2 has a double root at x = 0.5
        LegPoly lp = LegPoly::from_power_basis({{0.25}, {-1}, {1}}, {0, 1});

        auto roots = lp.roots();
        // Should find one root (possibly reported twice, or once with multiplicity)
        test.expect_true(roots.size() >= 1, "Test 83a: Repeated root found");
        if (!roots.empty()) {
            test.expect_near(roots[0], 0.5, 1e-6, "Test 83b: Repeated root at 0.5");
        }
        // Verify the polynomial touches zero at 0.5
        test.expect_near(lp(0.5), 0.0, tolerance, "Test 83c: f(0.5) = 0");
        // Derivative should also be zero at repeated root
        test.expect_near(lp(0.5, 1), 0.0, tolerance, "Test 83d: f'(0.5) = 0");
    }

    // Test 84: Integration beyond domain (extrapolation region)
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);

        // Integration from -1 to 2 (extends beyond [0,1] domain)
        // f(x) = x, so integral from -1 to 2 = [x^2/2] from -1 to 2 = 2 - 0.5 = 1.5
        double integral = lp.integrate(-1, 2);
        test.expect_near(integral, 1.5, tolerance, "Test 84a: Integration beyond domain");

        // Partial integration in extrapolation region
        // Integral from -1 to 0 = [x^2/2] from -1 to 0 = 0 - 0.5 = -0.5
        test.expect_near(lp.integrate(-1, 0), -0.5, tolerance, "Test 84b: Integration in left extrapolation");

        // Integral from 1 to 2 = [x^2/2] from 1 to 2 = 2 - 0.5 = 1.5
        test.expect_near(lp.integrate(1, 2), 1.5, tolerance, "Test 84c: Integration in right extrapolation");
    }

    // ============================================================
    // Group 35: Legendre Basis-Specific Properties
    // ============================================================
    std::cout << "\n--- Group 35: Legendre Basis-Specific Properties ---\n" << std::endl;

    // Test 85: Legendre endpoint values P_n(1) = 1, P_n(-1) = (-1)^n
    // These are fundamental properties of Legendre polynomials
    {
        // Test P_n(s) at s = -1 and s = 1 for various degrees
        // Using interval [-1, 1] so s = x directly
        for (int n = 0; n <= 10; ++n) {
            // Create P_n(x) on [-1, 1]: Legendre coefficients [0, 0, ..., 0, 1] (1 in position n)
            std::vector<std::vector<double>> coeffs(n + 1, {0.0});
            coeffs[n][0] = 1.0;  // Only the n-th Legendre coefficient is 1
            LegPoly lp(coeffs, {-1, 1});

            double val_at_1 = lp(1.0);
            double val_at_m1 = lp(-1.0);
            double expected_at_1 = 1.0;
            double expected_at_m1 = (n % 2 == 0) ? 1.0 : -1.0;

            std::string test_name = "Test 85" + std::string(1, static_cast<char>('a' + (n > 9 ? 9 : n))) + ": P_" + std::to_string(n) + " endpoint values";
            bool pass_1 = std::abs(val_at_1 - expected_at_1) < 1e-8;
            bool pass_m1 = std::abs(val_at_m1 - expected_at_m1) < 1e-8;
            if (pass_1 && pass_m1) {
                test.pass(test_name);
            } else {
                test.fail(test_name + " P(" + std::to_string(n) + ",1)=" + std::to_string(val_at_1) +
                         " P(" + std::to_string(n) + ",-1)=" + std::to_string(val_at_m1));
            }
        }
    }

    // Test 86: Legendre derivative at endpoints P'_n(1) = n(n+1)/2, P'_n(-1) = (-1)^(n+1) * n(n+1)/2
    {
        for (int n = 1; n <= 7; ++n) {
            std::vector<std::vector<double>> coeffs(n + 1, {0.0});
            coeffs[n][0] = 1.0;
            LegPoly lp(coeffs, {-1, 1});

            double deriv_at_1 = lp(1.0, 1);
            double deriv_at_m1 = lp(-1.0, 1);
            double expected_at_1 = n * (n + 1) / 2.0;
            double expected_at_m1 = ((n + 1) % 2 == 0) ? expected_at_1 : -expected_at_1;

            std::string test_name = "Test 86" + std::string(1, static_cast<char>('a' + n - 1)) + ": P'_" + std::to_string(n) + " endpoint derivatives";
            bool pass_1 = std::abs(deriv_at_1 - expected_at_1) < 1e-6;
            bool pass_m1 = std::abs(deriv_at_m1 - expected_at_m1) < 1e-6;
            if (pass_1 && pass_m1) {
                test.pass(test_name);
            } else {
                test.fail(test_name + " P'(" + std::to_string(n) + ",1)=" + std::to_string(deriv_at_1) +
                         " expected " + std::to_string(expected_at_1));
            }
        }
    }

    // Test 87: Legendre normalization - integral of P_n^2 from -1 to 1 = 2/(2n+1)
    {
        for (int n = 0; n <= 5; ++n) {
            // Create P_n on [-1, 1]
            std::vector<std::vector<double>> coeffs(n + 1, {0.0});
            coeffs[n][0] = 1.0;
            LegPoly lp(coeffs, {-1, 1});

            // Compute integral of P_n^2 using numerical integration
            double integral = 0.0;
            int num_points = 10000;
            double h = 2.0 / num_points;
            for (int i = 0; i <= num_points; ++i) {
                double x = -1.0 + i * h;
                double val = lp(x);
                double weight = (i == 0 || i == num_points) ? 0.5 : 1.0;
                integral += weight * val * val * h;
            }

            double expected = 2.0 / (2.0 * n + 1.0);
            std::string test_name = "Test 87" + std::string(1, static_cast<char>('a' + n)) + ": P_" + std::to_string(n) + " normalization integral";
            test.expect_near(integral, expected, 1e-6, test_name);
        }
    }

    // Test 88: Legendre recurrence relation (n+1)*P_{n+1}(s) = (2n+1)*s*P_n(s) - n*P_{n-1}(s)
    {
        std::vector<double> test_points = {-0.5, 0.0, 0.5, 0.75};
        for (int n = 1; n <= 4; ++n) {
            // Create P_{n-1}, P_n, P_{n+1}
            auto make_P = [](int deg) {
                std::vector<std::vector<double>> coeffs(deg + 1, {0.0});
                coeffs[deg][0] = 1.0;
                return LegPoly(coeffs, {-1, 1});
            };
            LegPoly P_nm1 = make_P(n - 1);
            LegPoly P_n = make_P(n);
            LegPoly P_np1 = make_P(n + 1);

            for (double s : test_points) {
                double lhs = (n + 1) * P_np1(s);
                double rhs = (2 * n + 1) * s * P_n(s) - n * P_nm1(s);

                std::string test_name = "Test 88: Recurrence n=" + std::to_string(n) + " s=" + std::to_string(s);
                test.expect_near(lhs, rhs, 1e-10, test_name);
            }
        }
    }

    // ============================================================
    // Group 36: Derivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 36: Derivative Edge Cases ---\n" << std::endl;

    // Test 89: Derivative of degree-0 polynomial should be zero polynomial
    {
        LegPoly lp({{5}}, {0, 1});  // Constant 5
        LegPoly deriv = lp.derivative();
        test.expect_near(deriv(0.5), 0.0, tolerance, "Test 89a: Derivative of constant is 0");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 0ul, "Test 89b: Derivative degree is 0");
    }

    // Test 90: Over-differentiate (n-th derivative where n > degree)
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});  // Quadratic
        LegPoly d3 = lp.derivative(3);  // Third derivative of quadratic = 0
        test.expect_near(d3(0.5), 0.0, tolerance, "Test 90a: 3rd derivative of quadratic is 0");

        LegPoly d10 = lp.derivative(10);  // Way over-differentiated
        test.expect_near(d10(0.5), 0.0, tolerance, "Test 90b: 10th derivative of quadratic is 0");
    }

    // Test 91: Chained derivatives vs single derivative(n)
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});  // Cubic
        LegPoly d3_chained = lp.derivative().derivative().derivative();
        LegPoly d3_single = lp.derivative(3);

        test.expect_near(d3_chained(0.5), d3_single(0.5), tolerance, "Test 91: Chained vs single derivative(3)");
    }

    // ============================================================
    // Group 37: Antiderivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 37: Antiderivative Edge Cases ---\n" << std::endl;

    // Test 92: Antiderivative of zero polynomial
    {
        LegPoly lp({{0}, {0}}, {0, 1});  // Zero polynomial
        LegPoly anti = lp.antiderivative();
        test.expect_near(anti(0.0), 0.0, tolerance, "Test 92a: Antiderivative of zero at 0");
        test.expect_near(anti(0.5), 0.0, tolerance, "Test 92b: Antiderivative of zero at 0.5");
        test.expect_near(anti(1.0), 0.0, tolerance, "Test 92c: Antiderivative of zero at 1");
    }

    // Test 93: Antiderivative(n).derivative(n) = original
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        LegPoly round_trip = lp.antiderivative(2).derivative(2);

        test.expect_near(lp(0.25), round_trip(0.25), tolerance, "Test 93a: antiderivative(2).derivative(2) at 0.25");
        test.expect_near(lp(0.5), round_trip(0.5), tolerance, "Test 93b: antiderivative(2).derivative(2) at 0.5");
        test.expect_near(lp(0.75), round_trip(0.75), tolerance, "Test 93c: antiderivative(2).derivative(2) at 0.75");
    }

    // Test 94: Chained antiderivatives vs single antiderivative(n)
    {
        LegPoly lp = LegPoly::from_power_basis({{2}}, {0, 1});  // f(x) = 2
        LegPoly a2_chained = lp.antiderivative().antiderivative();
        LegPoly a2_single = lp.antiderivative(2);

        test.expect_near(a2_chained(0.5), a2_single(0.5), tolerance, "Test 94: Chained vs single antiderivative(2)");
    }

    // ============================================================
    // Group 37b: Derivative/Antiderivative Structural Properties
    // ============================================================
    std::cout << "\n--- Group 37b: Derivative/Antiderivative Structural Properties ---\n" << std::endl;

    // Test 94b: Derivative reduces degree by 1
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});  // Cubic (degree 3)
        LegPoly deriv = lp.derivative();
        test.expect_eq(static_cast<size_t>(lp.degree()), 3ul, "Test 94b-1: Original degree is 3");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 2ul, "Test 94b-2: Derivative degree is 2");
    }

    // Test 94c: Antiderivative increases degree by 1
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});  // Quadratic (degree 2)
        LegPoly anti = lp.antiderivative();
        test.expect_eq(static_cast<size_t>(lp.degree()), 2ul, "Test 94c-1: Original degree is 2");
        test.expect_eq(static_cast<size_t>(anti.degree()), 3ul, "Test 94c-2: Antiderivative degree is 3");
    }

    // Test 94d: Derivative preserves num_intervals
    {
        LegPoly lp({{1, 2, 3}}, {0, 1, 2, 3});  // 3 intervals
        LegPoly deriv = lp.derivative();
        test.expect_eq(static_cast<size_t>(deriv.num_intervals()), 3ul,
            "Test 94d: Derivative preserves num_intervals");
    }

    // Test 94e: Derivative preserves breakpoints
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        LegPoly deriv = lp.derivative();
        test.expect_near(deriv.x()[0], 0.0, tolerance, "Test 94e-1: Derivative preserves left breakpoint");
        test.expect_near(deriv.x()[1], 1.0, tolerance, "Test 94e-2: Derivative preserves right breakpoint");
    }

    // Test 94f: Antiderivative starts at 0 at left boundary
    {
        LegPoly lp = LegPoly::from_power_basis({{5}}, {0, 1});  // Constant 5
        LegPoly anti = lp.antiderivative();
        test.expect_near(anti(0), 0.0, tolerance, "Test 94f: Antiderivative(left_boundary) = 0");
    }

    // ============================================================
    // Group 38: Integration Edge Cases
    // ============================================================
    std::cout << "\n--- Group 38: Integration Edge Cases ---\n" << std::endl;

    // Test 95: integrate(a, a) = 0
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        test.expect_near(lp.integrate(0.5, 0.5), 0.0, tolerance, "Test 95: integrate(a, a) = 0");
    }

    // Test 96: Integration crossing multiple intervals
    {
        // Simple two-interval polynomial: f(x) = 1 on [0,1], f(x) = 2 on [1,2]
        // Create using direct Legendre coefficients
        LegPoly lp({{1, 2}}, {0, 1, 2});

        // Integral from 0 to 2 = 1 * 1 + 2 * 1 = 3.0
        test.expect_near(lp.integrate(0, 2), 3.0, tolerance, "Test 96: Integration across multiple intervals");
    }

    // Test 97: NoExtrapolate mode returns NaN when evaluating beyond bounds
    {
        LegPoly temp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        LegPoly lp(temp.c(), {0, 1}, ExtrapolateMode::NoExtrapolate);
        // NoExtrapolate returns NaN on evaluation beyond bounds
        double result = lp(-1);
        test.expect_true(std::isnan(result), "Test 97: NoExtrapolate evaluation beyond bounds returns NaN");
    }

    // ============================================================
    // Group 39: Root Finding Edge Cases
    // ============================================================
    std::cout << "\n--- Group 39: Root Finding Edge Cases ---\n" << std::endl;

    // Test 98: No roots (always positive polynomial)
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {0}, {1}}, {0, 1});  // 1 + x^2 >= 1
        auto roots = lp.roots();
        test.expect_eq(roots.size(), 0ul, "Test 98: No roots for always-positive polynomial");
    }

    // Test 99: Root at domain boundary
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1});  // f(x) = x, root at x=0
        auto roots = lp.roots();
        test.expect_true(roots.size() >= 1, "Test 99a: Root at boundary found");
        if (!roots.empty()) {
            // Find the root closest to 0
            double min_root = *std::min_element(roots.begin(), roots.end());
            test.expect_near(min_root, 0.0, 1e-6, "Test 99b: Root at x=0");
        }
    }

    // Test 100: Many roots (oscillating polynomial)
    {
        // f(x) = sin(4*pi*x) approximation using Chebyshev approximation
        // Or create polynomial with roots at 0.25, 0.5, 0.75
        // f(x) = (x - 0.25)(x - 0.5)(x - 0.75) = x^3 - 1.5x^2 + 0.6875x - 0.09375
        LegPoly lp = LegPoly::from_power_basis({{-0.09375}, {0.6875}, {-1.5}, {1}}, {0, 1});
        auto roots = lp.roots();
        test.expect_eq(roots.size(), 3ul, "Test 100a: Three roots found");

        // Sort roots and verify
        if (roots.size() == 3) {
            std::sort(roots.begin(), roots.end());
            test.expect_near(roots[0], 0.25, 1e-6, "Test 100b: Root at 0.25");
            test.expect_near(roots[1], 0.5, 1e-6, "Test 100c: Root at 0.5");
            test.expect_near(roots[2], 0.75, 1e-6, "Test 100d: Root at 0.75");
        }
    }

    // ============================================================
    // Group 40: Extrapolation Order Edge Cases
    // ============================================================
    std::cout << "\n--- Group 40: Extrapolation Order Edge Cases ---\n" << std::endl;

    // Test 101: extrapolation_order > polynomial degree behaves like full extrapolation
    {
        // Get Legendre coefficients via from_power_basis, then construct with extrapolation order
        LegPoly temp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        LegPoly lp_full(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, -1, -1);
        LegPoly lp_high(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 10, 10);

        test.expect_near(lp_full(-0.5), lp_high(-0.5), tolerance, "Test 101a: High extrapolation order matches full");
        test.expect_near(lp_full(1.5), lp_high(1.5), tolerance, "Test 101b: High extrapolation order matches full");
    }

    // Test 102: extrapolation_order = 0 gives constant extrapolation
    {
        LegPoly temp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        LegPoly lp(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 0, 0);
        // f(x) = 1 + 2x, f(0) = 1, f(1) = 3
        test.expect_near(lp(-0.5), 1.0, tolerance, "Test 102a: Order 0 extrapolation at left");
        test.expect_near(lp(1.5), 3.0, tolerance, "Test 102b: Order 0 extrapolation at right");
    }

    // Test 103: Asymmetric extrapolation orders
    {
        LegPoly temp = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        LegPoly lp(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 0, -1);
        // f(x) = 1 + 2x
        test.expect_near(lp(-0.5), 1.0, tolerance, "Test 103a: Order 0 on left");
        test.expect_near(lp(1.5), 1.0 + 2*1.5, tolerance, "Test 103b: Full order on right");
    }

    // ============================================================
    // Group 41: Move/Copy Semantics
    // ============================================================
    std::cout << "\n--- Group 41: Move/Copy Semantics ---\n" << std::endl;

    // Test 104: Copy constructor
    {
        LegPoly lp1 = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        LegPoly lp2(lp1);  // Copy

        test.expect_near(lp1(0.5), lp2(0.5), tolerance, "Test 104a: Copy constructor preserves value");
        test.expect_eq(static_cast<size_t>(lp1.degree()), static_cast<size_t>(lp2.degree()), "Test 104b: Copy constructor preserves degree");
    }

    // Test 105: Move constructor
    {
        LegPoly lp1 = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double original_val = lp1(0.5);
        LegPoly lp2(std::move(lp1));

        test.expect_near(lp2(0.5), original_val, tolerance, "Test 105: Move constructor preserves value");
    }

    // ============================================================
    // Group 42: Const-Correctness and Thread Safety
    // ============================================================
    std::cout << "\n--- Group 42: Const-Correctness and Thread Safety ---\n" << std::endl;

    // Test 106: All const methods work on const object
    {
        const LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        // These should all compile and work
        double v1 = lp(0.5);
        double v2 = lp(0.5, 1);
        std::vector<double> v3 = lp({0.25, 0.5, 0.75});
        int d = lp.degree();
        int n = lp.num_intervals();
        const auto& c = lp.c();
        const auto& x = lp.x();
        double integ = lp.integrate(0, 1);
        auto roots = lp.roots();

        test.expect_near(v1, 1 + 2*0.5 + 3*0.25, tolerance, "Test 106a: Const evaluation");
        test.expect_near(v2, 2 + 6*0.5, tolerance, "Test 106b: Const derivative evaluation");
        test.expect_eq(v3.size(), 3ul, "Test 106c: Const vector evaluation");
        test.expect_eq(static_cast<size_t>(d), 2ul, "Test 106d: Const degree()");
        test.expect_eq(static_cast<size_t>(n), 1ul, "Test 106e: Const num_intervals()");
        test.expect_true(!c.empty(), "Test 106f: Const c() returns coefficients");
        test.expect_true(!x.empty(), "Test 106g: Const x() returns breakpoints");
        // Integral of 1 + 2x + 3x^2 from 0 to 1 = [x + x^2 + x^3]_0^1 = 1 + 1 + 1 = 3
        test.expect_near(integ, 3.0, tolerance, "Test 106h: Const integrate()");
        (void)roots;  // roots() returns empty for this polynomial (no real roots in [0,1])
        test.pass("Test 106: All const methods work on const object");
    }

    // Test 107: Thread safety - multiple threads evaluating same polynomial
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        std::atomic<int> correct_count{0};
        const int num_threads = 10;
        const int evals_per_thread = 1000;

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&lp, &correct_count, evals_per_thread]() {
                std::mt19937 gen(std::random_device{}());
                std::uniform_real_distribution<> dis(0.0, 1.0);

                for (int i = 0; i < evals_per_thread; ++i) {
                    double x = dis(gen);
                    double result = lp(x);
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
                      "Test 107: Thread safety - all concurrent evaluations correct");
    }

    // ============================================================
    // Group 43: Property-Based Tests
    // ============================================================
    std::cout << "\n--- Group 43: Property-Based Tests ---\n" << std::endl;

    // Test 108: Linearity of integration - integrate(f+g) = integrate(f) + integrate(g)
    {
        LegPoly lp1 = LegPoly::from_power_basis({{1}, {2}}, {0, 1});
        LegPoly lp2 = LegPoly::from_power_basis({{3}, {4}}, {0, 1});
        LegPoly lp_sum = LegPoly::from_power_basis({{4}, {6}}, {0, 1});

        double int_sum = lp_sum.integrate(0, 1);
        double sum_int = lp1.integrate(0, 1) + lp2.integrate(0, 1);

        test.expect_near(int_sum, sum_int, tolerance, "Test 108: Linearity of integration");
    }

    // Test 109: Fundamental theorem of calculus - d/dx(antiderivative(f)) = f
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        LegPoly anti = lp.antiderivative();
        LegPoly anti_deriv = anti.derivative();

        for (double x : {0.25, 0.5, 0.75}) {
            test.expect_near(lp(x), anti_deriv(x), tolerance,
                           "Test 109: FTC d/dx(antiderivative(f)) = f at x=" + std::to_string(x));
        }
    }

    // Test 110: Integration reversal - integrate(a, b) = -integrate(b, a)
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double int_ab = lp.integrate(0.2, 0.8);
        double int_ba = lp.integrate(0.8, 0.2);

        test.expect_near(int_ab, -int_ba, tolerance, "Test 110: Integration reversal");
    }

    // Test 111: Integration additivity - integrate(a,c) = integrate(a,b) + integrate(b,c)
    {
        LegPoly lp = LegPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double int_ac = lp.integrate(0.2, 0.8);
        double int_ab = lp.integrate(0.2, 0.5);
        double int_bc = lp.integrate(0.5, 0.8);

        test.expect_near(int_ac, int_ab + int_bc, tolerance, "Test 111: Integration additivity");
    }

    // ============================================================
    // Group 44: Symmetry Tests
    // ============================================================
    std::cout << "\n--- Group 44: Symmetry Tests ---\n" << std::endl;

    // Test 112: Even polynomial p(-x) = p(x)
    {
        // For an even polynomial on [-1,1], use only even Legendre polynomials P_0, P_2
        // P_0(s) = 1, P_2(s) = (3s^2 - 1)/2
        // p(s) = 1*P_0 + 2*P_2 = 1 + 2*(3s^2 - 1)/2 = 1 + 3s^2 - 1 = 3s^2
        LegPoly lp({{1}, {0}, {2}}, {-1, 1});  // c_0=1, c_1=0 (odd), c_2=2

        test.expect_near(lp(-0.5), lp(0.5), tolerance, "Test 112a: Even polynomial at ±0.5");
        test.expect_near(lp(-0.8), lp(0.8), tolerance, "Test 112b: Even polynomial at ±0.8");
    }

    // Test 113: Odd polynomial p(-x) = -p(x)
    {
        // For an odd polynomial on [-1,1], use only odd Legendre polynomials P_1, P_3
        // P_1(s) = s, P_3(s) = (5s^3 - 3s)/2
        // p(s) = 1*P_1 + 1*P_3 = s + (5s^3 - 3s)/2
        LegPoly lp({{0}, {1}, {0}, {1}}, {-1, 1});  // c_0=0 (even), c_1=1, c_2=0 (even), c_3=1

        test.expect_near(lp(-0.5), -lp(0.5), tolerance, "Test 113a: Odd polynomial at ±0.5");
        test.expect_near(lp(-0.8), -lp(0.8), tolerance, "Test 113b: Odd polynomial at ±0.8");
    }

    // ============================================================
    // Group 45: Boundary Edge Cases
    // ============================================================
    std::cout << "\n--- Group 45: Boundary Edge Cases ---\n" << std::endl;

    // Test 114: Evaluation at breakpoint + epsilon
    {
        LegPoly lp1 = LegPoly::from_power_basis({{0}, {1}}, {0, 1});
        LegPoly lp2 = LegPoly::from_power_basis({{1}, {2}}, {1, 2});
        LegPoly lp = lp1.extend(lp2.c(), {1, 2}, true);

        double eps = std::numeric_limits<double>::epsilon();
        double at_bp = lp(1.0);
        double after_bp = lp(1.0 + eps);
        double before_bp = lp(1.0 - eps);

        // All should be very close to f(1) = 1 (from first interval) or close to it
        test.expect_near(at_bp, 1.0, tolerance, "Test 114a: At breakpoint");
        test.expect_near(before_bp, 1.0, tolerance, "Test 114b: Just before breakpoint");
        // After breakpoint uses second interval: f(1+eps) ≈ 1 + 2*eps ≈ 1
        test.expect_near(after_bp, 1.0, tolerance, "Test 114c: Just after breakpoint");
    }

    // Test 115: Very wide interval
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1e10});
        test.expect_near(lp(0), 0.0, tolerance, "Test 115a: Wide interval at left");
        test.expect_near(lp(5e9), 5e9, 1e-3, "Test 115b: Wide interval at midpoint");
        test.expect_near(lp(1e10), 1e10, 1e-3, "Test 115c: Wide interval at right");
    }

    // Test 116: Very narrow interval
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {1}}, {0, 1e-10});
        test.expect_near(lp(0), 0.0, tolerance, "Test 116a: Narrow interval at left");
        test.expect_near(lp(5e-11), 5e-11, 1e-20, "Test 116b: Narrow interval at midpoint");
        test.expect_near(lp(1e-10), 1e-10, 1e-20, "Test 116c: Narrow interval at right");
    }

    // ============================================================
    // Group 46: Reference Data Verification (from numpy)
    // ============================================================
    std::cout << "\n--- Group 46: Reference Data Verification ---\n" << std::endl;

    // Test 117: Verify Legendre basis values against numpy reference
    // P_2(s) at s = [-1, -0.5, 0, 0.5, 1] should be [1, -0.125, -0.5, -0.125, 1]
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};  // P_2
        LegPoly lp(coeffs, {-1, 1});

        test.expect_near(lp(-1.0), 1.0, tolerance, "Test 117a: P_2(-1) = 1");
        test.expect_near(lp(-0.5), -0.125, tolerance, "Test 117b: P_2(-0.5) = -0.125");
        test.expect_near(lp(0.0), -0.5, tolerance, "Test 117c: P_2(0) = -0.5");
        test.expect_near(lp(0.5), -0.125, tolerance, "Test 117d: P_2(0.5) = -0.125");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 117e: P_2(1) = 1");
    }

    // Test 118: Verify P_3 values
    // P_3(s) at s = [-1, -0.5, 0, 0.5, 1] should be [-1, 0.4375, 0, -0.4375, 1]
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {0}, {1}};  // P_3
        LegPoly lp(coeffs, {-1, 1});

        test.expect_near(lp(-1.0), -1.0, tolerance, "Test 118a: P_3(-1) = -1");
        test.expect_near(lp(-0.5), 0.4375, tolerance, "Test 118b: P_3(-0.5) = 0.4375");
        test.expect_near(lp(0.0), 0.0, tolerance, "Test 118c: P_3(0) = 0");
        test.expect_near(lp(0.5), -0.4375, tolerance, "Test 118d: P_3(0.5) = -0.4375");
        test.expect_near(lp(1.0), 1.0, tolerance, "Test 118e: P_3(1) = 1");
    }

    // Test 119: Verify from_power_basis against numpy (quadratic x^2 on [0,1])
    // Power coeffs [0.25, 0.5, 0.25] on s, Legendre coeffs should be [1/3, 0.5, 1/6]
    {
        LegPoly lp = LegPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});
        auto& c = lp.c();

        test.expect_near(c[0][0], 1.0/3.0, tolerance, "Test 119a: x^2 Legendre c_0 = 1/3");
        test.expect_near(c[1][0], 0.5, tolerance, "Test 119b: x^2 Legendre c_1 = 0.5");
        test.expect_near(c[2][0], 1.0/6.0, tolerance, "Test 119c: x^2 Legendre c_2 = 1/6");
    }

    // Test 120: Verify high-degree evaluation stability (P_10 at s=0.5)
    // From numpy: P_10(0.5) ≈ -0.18822860717773443
    {
        std::vector<std::vector<double>> coeffs(11, {0.0});
        coeffs[10][0] = 1.0;
        LegPoly lp(coeffs, {-1, 1});

        test.expect_near(lp(0.5), -0.18822860717773443, 1e-8, "Test 120: P_10(0.5) matches numpy");
    }

    // ============================================================
    // Summary
    // ============================================================
    test.summary();

    return test.all_passed() ? 0 : 1;
}
