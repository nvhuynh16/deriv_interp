#include "include/bspoly.h"
#include "test_utils.h"
#include <cassert>
#include <limits>
#include <thread>
#include <atomic>
#include <random>

int main() {
    TestRunner test;
    const double tolerance = 1e-10;

    std::cout << "=== BsPoly (B-Spline Polynomial with Bernstein Storage) Test Suite ===" << std::endl;

    // ============================================================
    // Group 1: Basic Construction and Evaluation
    // ============================================================
    std::cout << "\n--- Group 1: Basic Construction and Evaluation ---\n" << std::endl;

    // Test 1: Basic construction
    test.expect_no_throw([]() {
        // Constant polynomial: Bernstein coefficient {c0} gives p(x) = c0
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);
    }, "Test 1: Basic construction");

    // Test 2: Constant polynomial
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        test.expect_near(bp(0.0), 5.0, tolerance, "Test 2a: Constant f(0)");
        test.expect_near(bp(0.5), 5.0, tolerance, "Test 2b: Constant f(0.5)");
        test.expect_near(bp(1.0), 5.0, tolerance, "Test 2c: Constant f(1)");
    }

    // Test 3: Linear polynomial
    // Bernstein basis: B_0(t) = 1-t, B_1(t) = t
    // For p(x) = x on [0,1], t = x, so p(t) = t = 0*(1-t) + 1*t
    // Coefficients: c0 = 0, c1 = 1
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        test.expect_near(bp(0.0), 0.0, tolerance, "Test 3a: Linear f(0)");
        test.expect_near(bp(0.25), 0.25, tolerance, "Test 3b: Linear f(0.25)");
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 3c: Linear f(0.5)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 3d: Linear f(1)");
    }

    // Test 4: Quadratic polynomial
    // Bernstein basis for degree 2: B_0(t) = (1-t)^2, B_1(t) = 2t(1-t), B_2(t) = t^2
    // For p(x) = x^2 on [0,1], coefficients are c0=0, c1=0, c2=1
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};  // p(x) = x^2 on [0,1]
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        test.expect_near(bp(0.0), 0.0, tolerance, "Test 4a: Quadratic f(0)");
        test.expect_near(bp(0.5), 0.25, tolerance, "Test 4b: Quadratic f(0.5)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 4c: Quadratic f(1)");
        test.expect_near(bp(0.25), 0.0625, tolerance, "Test 4d: Quadratic f(0.25)");
    }

    // Test 5: Multiple intervals
    {
        // Two constant pieces: [0,1] has value 1, [1,2] has value 2
        std::vector<std::vector<double>> coeffs = {{1, 2}};
        std::vector<double> breaks = {0, 1, 2};
        BsPoly bp(coeffs, breaks);

        test.expect_near(bp(0.5), 1.0, tolerance, "Test 5a: Multi-interval f(0.5)");
        test.expect_near(bp(1.5), 2.0, tolerance, "Test 5b: Multi-interval f(1.5)");
    }

    // ============================================================
    // Group 2: Error Handling
    // ============================================================
    std::cout << "\n--- Group 2: Error Handling ---\n" << std::endl;

    // Test 6: Empty coefficients error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs;  // Empty
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);
    }, "Test 6: Empty coefficients error");

    // Test 7: Too few breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0};  // Only 1 breakpoint
        BsPoly bp(coeffs, breaks);
    }, "Test 7: Too few breakpoints error");

    // Test 8: Non-monotonic breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 0.5, 0.3};  // Not monotonic
        BsPoly bp(coeffs, breaks);
    }, "Test 8: Non-monotonic breakpoints error");

    // Test 8b: from_derivatives with mismatched xi/yi sizes
    test.expect_throw([]() {
        std::vector<double> xi = {0, 1, 2};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};  // Only 2 points, xi has 3
        BsPoly::from_derivatives(xi, yi);
    }, "Test 8b: from_derivatives mismatched xi/yi sizes");

    // Test 8c: from_derivatives with single point
    test.expect_throw([]() {
        BsPoly::from_derivatives({0}, {{1, 0}});
    }, "Test 8c: from_derivatives with single point");

    // Test 8d: from_derivatives with empty yi element
    test.expect_throw([]() {
        BsPoly::from_derivatives({0, 1}, {{}, {1}});
    }, "Test 8d: from_derivatives with empty yi element");

    // Test 8e: extend with non-contiguous breakpoints
    test.expect_throw([]() {
        BsPoly bp({{1}}, {0, 1});
        bp.extend({{2}}, {5, 6}, true);  // Gap between 1 and 5
    }, "Test 8e: extend with non-contiguous breakpoints");

    // Test 8f: extend with opposite ordering
    test.expect_throw([]() {
        BsPoly bp({{1}}, {0, 1});  // Ascending
        bp.extend({{2}}, {2, 1}, true);  // Descending
    }, "Test 8f: extend with opposite ordering");

    // Test 8g: from_power_basis with empty coefficients
    test.expect_throw([]() {
        BsPoly::from_power_basis({}, {0, 1});
    }, "Test 8g: from_power_basis with empty coefficients");

    // Test 8h: from_derivatives with invalid orders parameter size
    test.expect_throw([]() {
        BsPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}}, {1, 2});  // orders size != 1 and != 3
    }, "Test 8h: from_derivatives with invalid orders size");

    // ============================================================
    // Group 3: Vector Evaluation
    // ============================================================
    std::cout << "\n--- Group 3: Vector Evaluation ---\n" << std::endl;

    // Test 9: Evaluate at multiple points
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        std::vector<double> xs = {0.0, 0.25, 0.5, 0.75, 1.0};
        std::vector<double> results = bp(xs);

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
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks, ExtrapolateMode::Extrapolate);

        // Extrapolate beyond [0,1]
        test.expect_near(bp(-0.5), -0.5, tolerance, "Test 10a: Extrapolate f(-0.5)");
        test.expect_near(bp(1.5), 1.5, tolerance, "Test 10b: Extrapolate f(1.5)");
    }

    // Test 11: NoExtrapolate mode
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks, ExtrapolateMode::NoExtrapolate);

        test.expect_true(std::isnan(bp(-0.5)), "Test 11a: NoExtrapolate f(-0.5) is NaN");
        test.expect_true(std::isnan(bp(1.5)), "Test 11b: NoExtrapolate f(1.5) is NaN");
    }

    // Test 12: Periodic mode
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks, ExtrapolateMode::Periodic);

        // Test periodic wrapping
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 12a: Periodic f(0.5)");
        test.expect_near(bp(1.5), bp(0.5), tolerance, "Test 12b: Periodic f(1.5) = f(0.5)");
        test.expect_near(bp(2.5), bp(0.5), tolerance, "Test 12c: Periodic f(2.5) = f(0.5)");
    }

    // ============================================================
    // Group 5: Derivative Operations
    // ============================================================
    std::cout << "\n--- Group 5: Derivative Operations ---\n" << std::endl;

    // Test 13: Derivative of linear polynomial
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);
        BsPoly dbp = bp.derivative();

        // Derivative of x is 1
        test.expect_near(dbp(0.0), 1.0, tolerance, "Test 13a: d/dx[x] at 0");
        test.expect_near(dbp(0.5), 1.0, tolerance, "Test 13b: d/dx[x] at 0.5");
        test.expect_near(dbp(1.0), 1.0, tolerance, "Test 13c: d/dx[x] at 1");
    }

    // Test 14: Derivative of quadratic polynomial
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};  // p(x) = x^2
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);
        BsPoly dbp = bp.derivative();

        // Derivative of x^2 is 2x
        test.expect_near(dbp(0.0), 0.0, tolerance, "Test 14a: d/dx[x^2] at 0");
        test.expect_near(dbp(0.5), 1.0, tolerance, "Test 14b: d/dx[x^2] at 0.5");
        test.expect_near(dbp(1.0), 2.0, tolerance, "Test 14c: d/dx[x^2] at 1");
    }

    // Test 15: operator()(x, nu) syntax for derivatives
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};  // p(x) = x^2
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        test.expect_near(bp(0.5, 0), 0.25, tolerance, "Test 15a: x^2(0.5, 0)");
        test.expect_near(bp(0.5, 1), 1.0, tolerance, "Test 15b: x^2(0.5, 1) = 2*0.5");
        test.expect_near(bp(0.5, 2), 2.0, tolerance, "Test 15c: x^2(0.5, 2) = 2");
    }

    // ============================================================
    // Group 6: Antiderivative and Integration
    // ============================================================
    std::cout << "\n--- Group 6: Antiderivative and Integration ---\n" << std::endl;

    // Test 16: Antiderivative of constant
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);
        BsPoly ibp = bp.antiderivative();

        // Antiderivative of 2 is 2x (starting at 0)
        test.expect_near(ibp(0), 0.0, tolerance, "Test 16a: int[2] at 0");
        test.expect_near(ibp(1), 2.0, tolerance, "Test 16b: int[2] at 1");
        test.expect_near(ibp(0.5), 1.0, tolerance, "Test 16c: int[2] at 0.5");
    }

    // Test 17: Integration (definite integral)
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        // Integral of x from 0 to 1 is 0.5
        double integral = bp.integrate(0, 1);
        test.expect_near(integral, 0.5, tolerance, "Test 17a: int_0^1 x dx");

        // Integral of x from 0 to 0.5 is 0.125
        test.expect_near(bp.integrate(0, 0.5), 0.125, tolerance, "Test 17b: int_0^0.5 x dx");
    }

    // Test 18: Negative derivative order = antiderivative
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        BsPoly ibp = bp.derivative(-1);  // -1 means antiderivative
        test.expect_near(ibp(1), 2.0, tolerance, "Test 18: derivative(-1) = antiderivative");
    }

    // ============================================================
    // Group 7: from_derivatives (Hermite Interpolation)
    // ============================================================
    std::cout << "\n--- Group 7: from_derivatives ---\n" << std::endl;

    // Test 19: Simple Hermite cubic from_derivatives
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};  // f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1

        BsPoly bp = BsPoly::from_derivatives(xi, yi);

        // Verify endpoint values
        test.expect_near(bp(0), 0.0, tolerance, "Test 19a: Hermite f(0)");
        test.expect_near(bp(1), 1.0, tolerance, "Test 19b: Hermite f(1)");

        // Verify derivatives at endpoints
        test.expect_near(bp(0, 1), 1.0, tolerance, "Test 19c: Hermite f'(0)");
        test.expect_near(bp(1, 1), -1.0, tolerance, "Test 19d: Hermite f'(1)");
    }

    // Test 20: from_derivatives with second derivatives
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, -1, 0}};

        BsPoly bp = BsPoly::from_derivatives(xi, yi);

        test.expect_near(bp(0), 0.0, tolerance, "Test 20a: Quintic f(0)");
        test.expect_near(bp(1), 1.0, tolerance, "Test 20b: Quintic f(1)");
        test.expect_near(bp(0, 1), 1.0, tolerance, "Test 20c: Quintic f'(0)");
        test.expect_near(bp(1, 1), -1.0, tolerance, "Test 20d: Quintic f'(1)");
        test.expect_near(bp(0, 2), 0.0, tolerance, "Test 20e: Quintic f''(0)");
        test.expect_near(bp(1, 2), 0.0, tolerance, "Test 20f: Quintic f''(1)");
    }

    // Test 21: from_derivatives multi-interval
    {
        std::vector<double> xi = {0, 1, 2};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, 0}, {0, -1}};

        BsPoly bp = BsPoly::from_derivatives(xi, yi);

        test.expect_near(bp(0), 0.0, tolerance, "Test 21a: Multi-interval f(0)");
        test.expect_near(bp(1), 1.0, tolerance, "Test 21b: Multi-interval f(1)");
        test.expect_near(bp(2), 0.0, tolerance, "Test 21c: Multi-interval f(2)");
    }

    // ============================================================
    // Group 8: Basis Conversions
    // ============================================================
    std::cout << "\n--- Group 8: Basis Conversions ---\n" << std::endl;

    // Test 22: from_power_basis (constant)
    {
        std::vector<std::vector<double>> power_coeffs = {{3}};  // p(x) = 3
        std::vector<double> breaks = {0, 1};
        BsPoly bp = BsPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(bp(0.5), 3.0, tolerance, "Test 22: from_power_basis constant");
    }

    // Test 23: from_power_basis (linear)
    {
        // Power basis: p(x) = 1 + 2*(x-0) = 1 + 2x on [0,1]
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}};
        std::vector<double> breaks = {0, 1};
        BsPoly bp = BsPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(bp(0), 1.0, tolerance, "Test 23a: from_power_basis linear f(0)");
        test.expect_near(bp(0.5), 2.0, tolerance, "Test 23b: from_power_basis linear f(0.5)");
        test.expect_near(bp(1), 3.0, tolerance, "Test 23c: from_power_basis linear f(1)");
    }

    // Test 24: from_power_basis (quadratic)
    {
        // Power basis: p(x) = 1 + 0*x + 1*x^2 = 1 + x^2 on [0,1]
        std::vector<std::vector<double>> power_coeffs = {{1}, {0}, {1}};
        std::vector<double> breaks = {0, 1};
        BsPoly bp = BsPoly::from_power_basis(power_coeffs, breaks);

        test.expect_near(bp(0), 1.0, tolerance, "Test 24a: from_power_basis quadratic f(0)");
        test.expect_near(bp(0.5), 1.25, tolerance, "Test 24b: from_power_basis quadratic f(0.5)");
        test.expect_near(bp(1), 2.0, tolerance, "Test 24c: from_power_basis quadratic f(1)");
    }

    // Test 25: to_power_basis round-trip
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        std::vector<double> breaks = {0, 1};
        BsPoly bp = BsPoly::from_power_basis(power_coeffs, breaks);

        auto recovered = bp.to_power_basis();

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
        BsPoly bp1(coeffs1, breaks1);

        std::vector<std::vector<double>> coeffs2 = {{2}};  // p(x) = 2 on [1,2]
        std::vector<double> breaks2 = {1, 2};

        BsPoly extended = bp1.extend(coeffs2, breaks2, true);

        test.expect_near(extended(0.5), 1.0, tolerance, "Test 26a: Extended f(0.5)");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 26b: Extended f(1.5)");
        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 26c: Extended num_intervals");
    }

    // Test 27: Extend to the left
    {
        std::vector<std::vector<double>> coeffs1 = {{2}};  // p(x) = 2 on [1,2]
        std::vector<double> breaks1 = {1, 2};
        BsPoly bp1(coeffs1, breaks1);

        std::vector<std::vector<double>> coeffs2 = {{1}};  // p(x) = 1 on [0,1]
        std::vector<double> breaks2 = {0, 1};

        BsPoly extended = bp1.extend(coeffs2, breaks2, false);

        test.expect_near(extended(0.5), 1.0, tolerance, "Test 27a: Extended left f(0.5)");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 27b: Extended left f(1.5)");
    }

    // ============================================================
    // Group 10: Root Finding
    // ============================================================
    std::cout << "\n--- Group 10: Root Finding ---\n" << std::endl;

    // Test 28: Linear root
    {
        std::vector<std::vector<double>> coeffs = {{-1}, {1}};  // p(x) = 2x - 1 on [0,1]
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        auto roots = bp.roots();
        test.expect_eq(roots.size(), 1ul, "Test 28a: Linear has 1 root");
        if (!roots.empty()) {
            test.expect_near(roots[0], 0.5, tolerance, "Test 28b: Linear root at 0.5");
        }
    }

    // Test 29: Quadratic roots
    {
        // p(x) = x^2 - 0.25 on [0, 1] has root at x = 0.5 (only within interval)
        BsPoly bp = BsPoly::from_power_basis({{-0.25}, {0}, {1}}, {0, 1});

        auto roots = bp.roots(true, false);  // disable extrapolation to find only roots in [0,1]
        test.expect_eq(roots.size(), 1ul, "Test 29a: Quadratic has 1 root on [0,1]");
        if (roots.size() == 1) {
            test.expect_near(roots[0], 0.5, tolerance, "Test 29b: Quadratic root at 0.5");
        }
    }

    // Test 30: No roots
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        auto roots = bp.roots();
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
        BsPoly bp(coeffs, breaks, ExtrapolateMode::NoExtrapolate, 1, 2);

        test.expect_eq(static_cast<size_t>(bp.degree()), 2ul, "Test 31a: degree");
        test.expect_eq(static_cast<size_t>(bp.num_intervals()), 1ul, "Test 31b: num_intervals");
        test.expect_true(bp.is_ascending(), "Test 31c: is_ascending");
        test.expect_eq(bp.c().size(), 3ul, "Test 31d: c() size");
        test.expect_eq(bp.x().size(), 2ul, "Test 31e: x() size");
        test.expect_true(bp.extrapolate() == ExtrapolateMode::NoExtrapolate, "Test 31f: extrapolate mode");
        test.expect_eq(static_cast<size_t>(bp.extrapolate_order_left()), 1ul, "Test 31g: extrapolate_order_left");
        test.expect_eq(static_cast<size_t>(bp.extrapolate_order_right()), 2ul, "Test 31h: extrapolate_order_right");
    }

    // ============================================================
    // Group 12: Descending Breakpoints
    // ============================================================
    std::cout << "\n--- Group 12: Descending Breakpoints ---\n" << std::endl;

    // Test 32: Descending breakpoints
    // Bernstein basis with descending breakpoints [1, 0]
    // With coeffs {{1}, {0}}: p(t) = 1*(1-t) + 0*t = 1-t
    // For descending [1,0], t = (x - 1)/(0 - 1) = 1 - x
    // So p(x) = 1 - (1-x) = x
    {
        std::vector<std::vector<double>> coeffs = {{1}, {0}};
        std::vector<double> breaks = {1, 0};  // Descending
        BsPoly bp(coeffs, breaks);

        test.expect_true(!bp.is_ascending(), "Test 32a: is_ascending is false");
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 32b: Descending f(0)");
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 32c: Descending f(0.5)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 32d: Descending f(1)");
    }

    // Test 32e-f: Descending with from_power_basis
    // Power basis: p(x) = a_0 + a_1*(x - left) where left=1 for [1,0]
    // For f(x) = x, need: a_0 + a_1*(x - 1) = x => a_0 = 1, a_1 = 1
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {1}}, {1, 0});  // f(x) = 1 + (x-1) = x, descending
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 32e: Descending from_power_basis f(0.5)");
        test.expect_near(bp(0.5, 1), 1.0, tolerance, "Test 32f: Descending derivative");
    }

    // Test 32g-h: Descending integration
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {1}}, {1, 0});  // f(x) = x, descending
        test.expect_near(bp.integrate(0, 1), 0.5, tolerance, "Test 32g: Descending integration 0 to 1");
        test.expect_near(bp.integrate(0.25, 0.75), 0.25, tolerance, "Test 32h: Descending integration partial");
    }

    // Test 32i-j: Multi-interval descending
    {
        std::vector<std::vector<double>> coeffs = {{2, 1}};  // Constant 2 on [2,1], constant 1 on [1,0]
        std::vector<double> breaks = {2, 1, 0};  // Two descending intervals
        BsPoly bp(coeffs, breaks);
        test.expect_near(bp(0.5), 1.0, tolerance, "Test 32i: Multi-interval descending f(0.5)");
        test.expect_near(bp(1.5), 2.0, tolerance, "Test 32j: Multi-interval descending f(1.5)");
    }

    // ============================================================
    // Group 13: Independent Verification
    // ============================================================
    std::cout << "\n--- Group 13: Independent Verification ---\n" << std::endl;

    // Test 33: Numerical integration verification
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {1}}, {0, 1});  // f(x) = x

        double analytical = bp.integrate(0, 1);
        double numerical = numerical_integrate(bp, 0, 1);

        test.expect_near(analytical, 0.5, tolerance, "Test 33a: Analytical integral");
        test.expect_near(numerical, 0.5, 1e-6, "Test 33b: Numerical integral");
    }

    // Test 34: Finite difference derivative verification
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});  // f(x) = x^2

        double analytical = bp(0.5, 1);  // 2*0.5 = 1
        double numerical = finite_diff_derivative(bp, 0.5);

        test.expect_near(analytical, 1.0, tolerance, "Test 34a: Analytical derivative");
        test.expect_near(numerical, 1.0, 1e-5, "Test 34b: Numerical derivative");
    }

    // ============================================================
    // Group 14: Property-Based Tests
    // ============================================================
    std::cout << "\n--- Group 14: Property-Based Tests ---\n" << std::endl;

    // Test 35: Integral additivity
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});

        double int_0_1 = bp.integrate(0, 1);
        double int_0_half = bp.integrate(0, 0.5);
        double int_half_1 = bp.integrate(0.5, 1);

        test.expect_near(int_0_1, int_0_half + int_half_1, tolerance,
                        "Test 35: Integral additivity");
    }

    // Test 36: Derivative-integral relationship
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BsPoly antideriv = bp.antiderivative();
        BsPoly recovered = antideriv.derivative();

        bool all_close = true;
        for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            if (std::abs(bp(x) - recovered(x)) > tolerance) {
                all_close = false;
            }
        }
        test.expect_true(all_close, "Test 36: d/dx[antiderivative] = original");
    }

    // Test 37: Integral reversal
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});

        double int_ab = bp.integrate(0.2, 0.8);
        double int_ba = bp.integrate(0.8, 0.2);

        test.expect_near(int_ab, -int_ba, tolerance, "Test 37: integrate(a,b) = -integrate(b,a)");
    }

    // ============================================================
    // Group 15: High-Degree Polynomials
    // ============================================================
    std::cout << "\n--- Group 15: High-Degree Polynomials ---\n" << std::endl;

    // Test 38: High-degree polynomial stability (degree 50)
    // All Bernstein coefficients = 1.0 means p(t) = sum(B_{i,n}(t)) = 1 (partition of unity)
    // This tests numerical stability of de Casteljau algorithm at high degrees
    {
        const int high_degree = 50;
        std::vector<std::vector<double>> coeffs(high_degree + 1, std::vector<double>(1, 1.0));
        BsPoly bp(coeffs, {0, 1});

        // The polynomial should evaluate to 1.0 everywhere within the domain
        test.expect_near(bp(0.0), 1.0, tolerance, "Test 38a: High-degree (50) at t=0");
        test.expect_near(bp(0.25), 1.0, tolerance, "Test 38b: High-degree (50) at t=0.25");
        test.expect_near(bp(0.5), 1.0, tolerance, "Test 38c: High-degree (50) at t=0.5");
        test.expect_near(bp(0.75), 1.0, tolerance, "Test 38d: High-degree (50) at t=0.75");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 38e: High-degree (50) at t=1");
        test.expect_near(bp(0.001), 1.0, tolerance, "Test 38f: High-degree (50) near 0");
        test.expect_near(bp(0.999), 1.0, tolerance, "Test 38g: High-degree (50) near 1");
    }

    // Test 38h-l: Very high degree (100) with linear polynomial f(x) = x
    // Coefficients for linear t elevated to degree n: c[i] = i/n
    {
        const int very_high_degree = 100;
        std::vector<std::vector<double>> coeffs(very_high_degree + 1, std::vector<double>(1));

        // Coefficients for linear t elevated to degree n: c[i] = i/n
        for (int i = 0; i <= very_high_degree; ++i) {
            coeffs[i][0] = static_cast<double>(i) / very_high_degree;
        }

        BsPoly bp(coeffs, {0, 1});

        // Should evaluate to t at all points
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 38h: Very-high-degree (100) linear at t=0");
        test.expect_near(bp(0.25), 0.25, tolerance, "Test 38i: Very-high-degree (100) linear at t=0.25");
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 38j: Very-high-degree (100) linear at t=0.5");
        test.expect_near(bp(0.75), 0.75, tolerance, "Test 38k: Very-high-degree (100) linear at t=0.75");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 38l: Very-high-degree (100) linear at t=1");
    }

    // Test 38m-q: High-degree polynomial with non-trivial shape
    // Quadratic t^2 at degree 50: tests de Casteljau with curved evaluation
    {
        const int degree = 50;
        std::vector<std::vector<double>> coeffs(degree + 1, std::vector<double>(1));

        // Coefficients for t^2 elevated to degree n: c[i] = i*(i-1) / (n*(n-1))
        for (int i = 0; i <= degree; ++i) {
            coeffs[i][0] = static_cast<double>(i) * (i - 1) / (degree * (degree - 1));
        }

        BsPoly bp(coeffs, {0, 1});

        // Should evaluate to t^2 at all points - verify exact values
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 38m: High-degree quadratic at t=0");
        test.expect_near(bp(0.25), 0.0625, tolerance, "Test 38n: High-degree quadratic at t=0.25");
        test.expect_near(bp(0.5), 0.25, tolerance, "Test 38o: High-degree quadratic at t=0.5");
        test.expect_near(bp(0.75), 0.5625, tolerance, "Test 38p: High-degree quadratic at t=0.75");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 38q: High-degree quadratic at t=1");

        // Additional verification: derivative of t^2 is 2t, verify via finite differences
        double analytical_deriv = bp(0.5, 1);
        double numerical_deriv = finite_diff_derivative(bp, 0.5);
        test.expect_near(analytical_deriv, 1.0, tolerance, "Test 38r: d/dt[t^2] at t=0.5 = 1");
        test.expect_near(analytical_deriv, numerical_deriv, 1e-5,
                        "Test 38s: High-degree derivative matches finite difference");
    }

    // ============================================================
    // Group 16: NaN and Infinity Handling
    // ============================================================
    std::cout << "\n--- Group 16: NaN and Infinity Handling ---\n" << std::endl;

    // Test 39: NaN input
    {
        BsPoly bp({{1}}, {0, 1});
        test.expect_true(std::isnan(bp(std::numeric_limits<double>::quiet_NaN())),
                        "Test 39: NaN input gives NaN output");
    }

    // Test 40: Infinity with NoExtrapolate
    {
        BsPoly bp({{1}}, {0, 1}, ExtrapolateMode::NoExtrapolate);
        test.expect_true(std::isnan(bp(std::numeric_limits<double>::infinity())),
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

        BsPoly bp = BsPoly::from_derivatives(xi, yi, orders);

        // Should match f, f' but not higher derivatives
        test.expect_near(bp(0), 0.0, tolerance, "Test 41a: orders f(0)");
        test.expect_near(bp(1), 1.0, tolerance, "Test 41b: orders f(1)");
        test.expect_near(bp(0, 1), 1.0, tolerance, "Test 41c: orders f'(0)");
        test.expect_near(bp(1, 1), -1.0, tolerance, "Test 41d: orders f'(1)");

        // Verify that HIGHER derivatives were NOT matched (orders limited to 1)
        // If orders worked correctly, f''(0) should NOT equal yi[0][2] = 2
        double second_deriv = bp(0, 2);
        test.expect_true(std::abs(second_deriv - 2.0) > 0.1,
            "Test 41e: orders parameter actually limited derivatives (f''(0) != 2)");
    }

    // ============================================================
    // Group 18: Move Semantics
    // ============================================================
    std::cout << "\n--- Group 18: Move Semantics ---\n" << std::endl;

    // Test 42: Move constructor
    {
        BsPoly bp1({{0}, {1}}, {0, 1});
        double val_before = bp1(0.5);

        BsPoly bp2(std::move(bp1));
        double val_after = bp2(0.5);

        test.expect_near(val_after, val_before, tolerance, "Test 42: Move constructor preserves value");
    }

    // ============================================================
    // Group 19: Controlled Extrapolation (Taylor Order)
    // ============================================================
    std::cout << "\n--- Group 19: Controlled Extrapolation ---\n" << std::endl;

    // Test 43: Constant extrapolation (order 0)
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);  // f(x) = x
        BsPoly bp_const(bp.c(), bp.x(), ExtrapolateMode::Extrapolate, 0, 0);  // Constant extrapolation

        // At x=2 (beyond [0,1]), should use f(1) = 1 (constant)
        test.expect_near(bp_const(2), 1.0, tolerance, "Test 43a: Constant extrapolation right");
        // At x=-1 (before [0,1]), should use f(0) = 0 (constant)
        test.expect_near(bp_const(-1), 0.0, tolerance, "Test 43b: Constant extrapolation left");
    }

    // Test 44: Linear extrapolation (order 1)
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);  // f(x) = x^2
        BsPoly bp_linear(bp.c(), bp.x(), ExtrapolateMode::Extrapolate, 1, 1);

        // At x=2, linear extrapolation from x=1: f(1) + f'(1)*(2-1) = 1 + 2*1 = 3
        test.expect_near(bp_linear(2), 3.0, tolerance, "Test 44a: Linear extrapolation right");
        // At x=-1, linear extrapolation from x=0: f(0) + f'(0)*(-1-0) = 0 + 0*(-1) = 0
        test.expect_near(bp_linear(-1), 0.0, tolerance, "Test 44b: Linear extrapolation left");
    }

    // ============================================================
    // Group 20: BPoly Equivalence Verification
    // ============================================================
    std::cout << "\n--- Group 20: BPoly Equivalence Verification ---\n" << std::endl;

    // Test 45: Verify BsPoly matches BPoly for same Bernstein coefficients
    // Since BsPoly stores Bernstein coefficients internally, it should produce
    // identical results to BPoly with the same coefficients
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0.5}, {1}};
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        // For Bernstein: p(t) = 0*(1-t)^2 + 0.5*2t(1-t) + 1*t^2
        // = t(1-t) + t^2 = t - t^2 + t^2 = t
        // So p(x) = x on [0,1]
        test.expect_near(bp(0.25), 0.25, tolerance, "Test 45a: BsPoly = identity");
        test.expect_near(bp(0.75), 0.75, tolerance, "Test 45b: BsPoly = identity");
    }

    // Test 46: De Casteljau algorithm correctness
    {
        // Linear: c0=1, c1=3 => p(t) = 1*(1-t) + 3*t = 1 + 2t
        // At t=0.5: 1 + 1 = 2
        BsPoly bp({{1}, {3}}, {0, 1});
        test.expect_near(bp(0.5), 2.0, tolerance, "Test 46: De Casteljau linear");
    }

    // ============================================================
    // Group 21: Additional Property Tests
    // ============================================================
    std::cout << "\n--- Group 21: Additional Property Tests ---\n" << std::endl;

    // Test 47: Higher order derivatives eventually become zero
    {
        // Degree 3 polynomial
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});

        // 4th derivative of degree 3 polynomial is 0
        BsPoly d4 = bp.derivative(4);
        test.expect_near(d4(0.5), 0.0, tolerance, "Test 47: 4th derivative of cubic is 0");
    }

    // Test 48: from_derivatives endpoint constraints
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{1, 2, 3}, {4, 5, 6}};

        BsPoly bp = BsPoly::from_derivatives(xi, yi);

        bool all_pass = true;
        if (std::abs(bp(0) - 1) > tolerance) all_pass = false;
        if (std::abs(bp(0, 1) - 2) > tolerance) all_pass = false;
        if (std::abs(bp(0, 2) - 3) > tolerance) all_pass = false;
        if (std::abs(bp(1) - 4) > tolerance) all_pass = false;
        if (std::abs(bp(1, 1) - 5) > tolerance) all_pass = false;
        if (std::abs(bp(1, 2) - 6) > tolerance) all_pass = false;

        test.expect_true(all_pass, "Test 48: from_derivatives matches all endpoint constraints");
    }

    // ============================================================
    // Group 22: Extended Independent Verification
    // ============================================================
    std::cout << "\n--- Group 22: Extended Independent Verification ---\n" << std::endl;

    // Test 49: Multi-point finite difference derivative verification
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {-2}, {3}, {-1}}, {0, 1});

        bool all_pass = true;
        for (double x : {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}) {
            double analytical = bp(x, 1);
            double numerical = finite_diff_derivative(bp, x);
            if (std::abs(analytical - numerical) > 1e-5) {
                all_pass = false;
            }
        }
        test.expect_true(all_pass, "Test 49: Multi-point derivative matches finite differences");
    }

    // Test 50: Multi-point second derivative verification
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});

        bool all_pass = true;
        for (double x : {0.2, 0.4, 0.5, 0.6, 0.8}) {
            double analytical = bp(x, 2);
            double numerical = finite_diff_second_derivative(bp, x);
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
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}, {4}, {5}}, {0, 1});

        bool all_pass = true;
        std::vector<double> points = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

        std::vector<double> results1 = bp(points);
        std::vector<double> results2 = bp(points);
        std::vector<double> results3 = bp(points);

        for (size_t i = 0; i < points.size(); ++i) {
            if (results1[i] != results2[i] || results2[i] != results3[i]) {
                all_pass = false;
            }
        }
        test.expect_true(all_pass, "Test 51: Reproducibility - repeated evaluation identical");
    }

    // Test 52: Thread safety - actual multi-threaded concurrent access
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}, {4}, {5}}, {0, 1});
        const int num_threads = 10;
        std::atomic<bool> all_correct{true};
        std::vector<std::thread> threads;

        auto thread_func = [&bp, &all_correct](unsigned int seed) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            for (int i = 0; i < 100; ++i) {
                double x = dist(rng);
                double result = bp(x);
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
        const BsPoly bp({{0}, {1}}, {0, 1});

        double v1 = bp(0.5);
        double v2 = bp(0.5, 1);
        std::vector<double> v3 = bp({0.1, 0.2, 0.3});
        double v4 = bp.integrate(0, 1);
        int d = bp.degree();
        int n = bp.num_intervals();
        bool asc = bp.is_ascending();
        const auto& c = bp.coefficients();
        const auto& x = bp.breakpoints();
        ExtrapolateMode mode = bp.extrapolate();

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
        BsPoly bp = BsPoly::from_power_basis({{0}, {1}}, {0, 1});  // f(x) = x

        test.expect_near(bp(0.0), 0.0, tolerance, "Test 54a: f(0) at left boundary");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 54b: f(1) at right boundary");
    }

    // Test 55: Very small interval
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {1}}, {0, 1e-10});  // Very small interval: f(x) = 1 + x

        // Should still evaluate correctly
        double val = bp(0.5e-10);
        double expected = 1.0 + 0.5e-10;  // f(x) = 1 + x at x = 0.5e-10
        test.expect_true(!std::isnan(val), "Test 55a: Small interval not NaN");
        test.expect_near(val, expected, 1e-15, "Test 55b: Small interval correct value");
    }

    // Test 56: Cubic polynomial verification
    {
        // f(x) = x^3 on [0, 1]
        BsPoly bp = BsPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1});

        test.expect_near(bp(0), 0.0, tolerance, "Test 56a: x^3(0)");
        test.expect_near(bp(0.5), 0.125, tolerance, "Test 56b: x^3(0.5)");
        test.expect_near(bp(1), 1.0, tolerance, "Test 56c: x^3(1)");
    }

    // ============================================================
    // Group 25: Scipy Reference Verification
    // ============================================================
    std::cout << "\n--- Group 25: Scipy Reference Verification ---\n" << std::endl;

    // Test 57: scipy.interpolate.BPoly.from_derivatives reference values
    // Hermite cubic: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
    // Reference values from scripts/from_derivatives_comparison.json
    {
        BsPoly bp = BsPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});

        // scipy reference values at specific points
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 57a: Hermite cubic scipy f(0)");
        test.expect_near(bp(0.25), 0.34375, tolerance, "Test 57b: Hermite cubic scipy f(0.25)");
        test.expect_near(bp(0.5), 0.75, tolerance, "Test 57c: Hermite cubic scipy f(0.5)");
        test.expect_near(bp(0.75), 1.03125, tolerance, "Test 57d: Hermite cubic scipy f(0.75)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 57e: Hermite cubic scipy f(1)");
    }

    // Test 58: Quintic Hermite scipy reference
    // f(0)=0, f'(0)=1, f''(0)=0, f(1)=1, f'(1)=-1, f''(1)=0
    {
        BsPoly bp = BsPoly::from_derivatives({0, 1}, {{0, 1, 0}, {1, -1, 0}});

        // Verify endpoint constraints
        test.expect_near(bp(0), 0.0, tolerance, "Test 58a: Quintic scipy f(0)");
        test.expect_near(bp(1), 1.0, tolerance, "Test 58b: Quintic scipy f(1)");
        test.expect_near(bp(0, 1), 1.0, tolerance, "Test 58c: Quintic scipy f'(0)");
        test.expect_near(bp(1, 1), -1.0, tolerance, "Test 58d: Quintic scipy f'(1)");
        test.expect_near(bp(0, 2), 0.0, tolerance, "Test 58e: Quintic scipy f''(0)");
        test.expect_near(bp(1, 2), 0.0, tolerance, "Test 58f: Quintic scipy f''(1)");
    }

    // Test 59: Multi-interval scipy reference
    {
        BsPoly bp = BsPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1, 0}, {0, -1}});

        test.expect_near(bp(0), 0.0, tolerance, "Test 59a: Multi-interval scipy f(0)");
        test.expect_near(bp(1), 1.0, tolerance, "Test 59b: Multi-interval scipy f(1)");
        test.expect_near(bp(2), 0.0, tolerance, "Test 59c: Multi-interval scipy f(2)");
    }

    // Test 60: Bernstein basis evaluation matches scipy
    // B_2^2(t) = 2t(1-t), maximum at t=0.5 is 0.5
    {
        BsPoly bp({{0}, {1}, {0}}, {0, 1});  // Middle control point only
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 60a: B_1^2 at t=0");
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 60b: B_1^2 at t=0.5");
        test.expect_near(bp(1.0), 0.0, tolerance, "Test 60c: B_1^2 at t=1");
    }

    // Test 61: Integration scipy reference
    // Integral of x^2 from 0 to 1 is 1/3
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});
        test.expect_near(bp.integrate(0, 1), 1.0/3.0, tolerance, "Test 61: Integral of x^2 matches scipy");
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
        BsPoly bp(tiny_coeffs, breaks);

        // Bernstein linear: f(t) = (1-t)*c0 + t*c1
        test.expect_near(bp(0.0), 1e-100, 1e-110, "Test 62a: Tiny coefficients at 0");
        test.expect_near(bp(0.5), 1.5e-100, 1e-110, "Test 62b: Tiny coefficients at 0.5");
        test.expect_near(bp(1.0), 2e-100, 1e-110, "Test 62c: Tiny coefficients at 1");
    }

    // Test 63: Very large coefficients (1e100 scale)
    {
        std::vector<std::vector<double>> huge_coeffs = {{1e100}, {2e100}};
        std::vector<double> breaks = {0, 1};
        BsPoly bp(huge_coeffs, breaks);

        test.expect_near(bp(0.0), 1e100, 1e90, "Test 63a: Huge coefficients at 0");
        test.expect_near(bp(0.5), 1.5e100, 1e90, "Test 63b: Huge coefficients at 0.5");
        test.expect_near(bp(1.0), 2e100, 1e90, "Test 63c: Huge coefficients at 1");
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
        BsPoly bp(c, x);
        BsPoly ibp = bp.antiderivative();

        // Check continuity at each interior breakpoint
        double eps = 1e-10;
        bool continuous = true;
        for (size_t i = 1; i < x.size() - 1; ++i) {
            double left = ibp(x[i] - eps);
            double right = ibp(x[i] + eps);
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
        BsPoly bp = BsPoly::from_derivatives({0, 1}, {{1, 2, -1}, {3, 0, 2}});

        for (int n = 1; n <= 3; ++n) {
            BsPoly antideriv = bp.antiderivative(n);
            BsPoly recovered = antideriv.derivative(n);

            bool matches = true;
            for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
                if (std::abs(bp(x) - recovered(x)) > tolerance) {
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
        BsPoly bp = BsPoly::from_power_basis({{0, 1}, {1, -0.5}}, {0, 1, 3}, ExtrapolateMode::Periodic);

        // Period = 3 - 0 = 3
        // At x=3.5: wraps to x=0.5, should equal bp(0.5)
        test.expect_near(bp(3.5), bp(0.5), tolerance, "Test 66a: Periodic f(3.5) = f(0.5)");

        // At x=-0.5: wraps to x=2.5 (period=3), should equal bp(2.5)
        test.expect_near(bp(-0.5), bp(2.5), tolerance, "Test 66b: Periodic f(-0.5) = f(2.5)");

        // Test derivative in periodic mode
        test.expect_near(bp(3.5, 1), bp(0.5, 1), tolerance, "Test 66c: Periodic f'(3.5) = f'(0.5)");
        test.expect_near(bp(-0.5, 1), bp(2.5, 1), tolerance, "Test 66d: Periodic f'(-0.5) = f'(2.5)");
    }

    // ============================================================
    // Group 30: Corner Case Tests
    // ============================================================
    std::cout << "\n--- Group 30: Corner Case Tests ---\n" << std::endl;

    // Test 67: Single interval degree 5
    {
        std::vector<std::vector<double>> c = {{1}, {0}, {0}, {0}, {0}, {2}};
        BsPoly bp(c, {0, 1});
        // For Bernstein degree 5: p(t) = c_0*B_0,5(t) + c_1*B_1,5(t) + ... + c_5*B_5,5(t)
        // where B_0,5(t) = (1-t)^5 and B_5,5(t) = t^5
        // Middle coefficients c_1 through c_4 are all 0, so they contribute nothing
        // At t=0: only B_0,5(0)=1 contributes, so p(0) = 1*1 = 1
        // At t=1: only B_5,5(1)=1 contributes, so p(1) = 2*1 = 2
        // At t=0.5: B_0,5(0.5) = (0.5)^5 = 0.03125, B_5,5(0.5) = (0.5)^5 = 0.03125
        //           p(0.5) = 1*0.03125 + 2*0.03125 = 0.09375
        test.expect_near(bp(0.0), 1.0, tolerance, "Test 67a: Degree 5 at 0");
        test.expect_near(bp(0.5), 0.09375, tolerance, "Test 67b: Degree 5 at 0.5");
        test.expect_near(bp(1.0), 2.0, tolerance, "Test 67c: Degree 5 at 1");
    }

    // Test 68: Two intervals with different quadratic shapes
    // First interval [0, 0.5]: bump shape (coeffs [0,1,0] gives max at midpoint)
    // Second interval [0.5, 1]: decreasing from 1 to 0 (coeffs [1,0,0])
    {
        std::vector<std::vector<double>> c = {{0, 1}, {1, 0}, {0, 0}};  // Quadratic coefficients per interval
        BsPoly bp(c, {0, 0.5, 1});
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 68a: Two intervals at 0");
        test.expect_near(bp(0.25), 0.5, tolerance, "Test 68b: Two intervals at 0.25");
        test.expect_near(bp(0.5), 1.0, tolerance, "Test 68c: Two intervals at 0.5");
        test.expect_near(bp(0.75), 0.25, tolerance, "Test 68d: Two intervals at 0.75");
        test.expect_near(bp(1.0), 0.0, tolerance, "Test 68e: Two intervals at 1");
    }

    // Test 69: Many intervals (10 linear pieces)
    {
        std::vector<std::vector<double>> c(2, std::vector<double>(10));
        std::vector<double> breaks(11);
        for (int i = 0; i <= 10; ++i) {
            breaks[i] = static_cast<double>(i);
        }
        // Create linear f(x) = x using Bernstein coefficients on each interval
        for (int i = 0; i < 10; ++i) {
            c[0][i] = breaks[i];      // Left endpoint value
            c[1][i] = breaks[i + 1];  // Right endpoint value
        }
        BsPoly bp(c, breaks);
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 69a: Many intervals at 0.5");
        test.expect_near(bp(5.5), 5.5, tolerance, "Test 69b: Many intervals at 5.5");
        test.expect_near(bp(9.5), 9.5, tolerance, "Test 69c: Many intervals at 9.5");
    }

    // Test 70: Large coefficients (1e10 scale)
    {
        std::vector<std::vector<double>> c = {{1e10}, {2e10}};
        BsPoly bp(c, {0, 1});
        test.expect_near(bp(0.0), 1e10, 1.0, "Test 70a: Large coeffs at 0");
        test.expect_near(bp(0.5), 1.5e10, 1e5, "Test 70b: Large coeffs at 0.5");
        test.expect_near(bp(1.0), 2e10, 1.0, "Test 70c: Large coeffs at 1");
    }

    // Test 71: Small coefficients (1e-15 scale)
    {
        std::vector<std::vector<double>> c = {{1e-15}, {2e-15}};
        BsPoly bp(c, {0, 1});
        test.expect_near(bp(0.0), 1e-15, 1e-25, "Test 71a: Small coeffs at 0");
        test.expect_near(bp(0.5), 1.5e-15, 1e-25, "Test 71b: Small coeffs at 0.5");
        test.expect_near(bp(1.0), 2e-15, 1e-25, "Test 71c: Small coeffs at 1");
    }

    // Test 72: Far extrapolation
    {
        std::vector<std::vector<double>> c = {{0}, {1}};  // f(x) = x
        BsPoly bp(c, {0, 1}, ExtrapolateMode::Extrapolate);
        test.expect_near(bp(10.0), 10.0, tolerance, "Test 72a: Far extrapolation at 10");
        test.expect_near(bp(-10.0), -10.0, tolerance, "Test 72b: Far extrapolation at -10");
        test.expect_near(bp(100.0), 100.0, 1e-8, "Test 72c: Far extrapolation at 100");
    }

    // Test 73: Near-boundary evaluation
    {
        std::vector<std::vector<double>> c = {{0, 1}, {1, 2}};
        std::vector<double> breaks = {0, 1, 2};
        BsPoly bp(c, breaks);

        double eps = 1e-14;
        test.expect_true(std::isfinite(bp(1.0 - eps)), "Test 73a: Near boundary 1-eps finite");
        test.expect_true(std::isfinite(bp(1.0 + eps)), "Test 73b: Near boundary 1+eps finite");
        test.expect_near(bp(1.0 - eps), bp(1.0), 1e-10, "Test 73c: Near boundary 1-eps value");
        test.expect_near(bp(1.0 + eps), bp(1.0), 1e-10, "Test 73d: Near boundary 1+eps value");
    }

    // Test 74: from_derivatives with values only (no derivative constraints)
    {
        BsPoly bp = BsPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}});
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 74a: Values-only f(0)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 74b: Values-only f(1)");
        test.expect_near(bp(2.0), 0.0, tolerance, "Test 74c: Values-only f(2)");
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 74d: Values-only f(0.5)");
    }

    // Test 75: from_derivatives with asymmetric orders
    {
        // 4 derivatives at left, 1 at right
        BsPoly bp = BsPoly::from_derivatives({0, 1}, {{0, 1, 0, 0}, {1}});
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 75a: Asymmetric f(0)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 75b: Asymmetric f(1)");
        // Derivative at 0 should be 1
        double h = 1e-8;
        double deriv_at_0 = (bp(h) - bp(0)) / h;
        test.expect_near(deriv_at_0, 1.0, 1e-5, "Test 75c: Asymmetric f'(0) approx");
    }

    // ============================================================
    // Group 31: Missing Coverage - C0 Continuity at Breakpoints
    // ============================================================
    std::cout << "\n--- Group 31: C0 Continuity at Breakpoints ---\n" << std::endl;

    // Test 76: from_derivatives produces C0 continuous polynomial at interval boundaries
    {
        // Multi-interval Hermite interpolation - polynomial should be continuous
        BsPoly bp = BsPoly::from_derivatives({0, 1, 2, 3}, {{0, 1}, {1, 0}, {0.5, -0.5}, {0, 0}});

        double eps = 1e-12;
        bool continuous = true;

        // Check C0 continuity at each interior breakpoint
        for (double bpt : {1.0, 2.0}) {
            double left = bp(bpt - eps);
            double right = bp(bpt + eps);
            double at_bp = bp(bpt);
            if (std::abs(left - at_bp) > 1e-8 || std::abs(right - at_bp) > 1e-8) {
                test.fail("Test 76: C0 continuity at x=" + std::to_string(bpt));
                continuous = false;
            }
        }
        if (continuous) {
            test.pass("Test 76: from_derivatives produces C0 continuous polynomial");
        }
    }

    // Test 77: from_derivatives with matching derivatives produces C1 continuity
    {
        // C1 continuity: function AND first derivative match at breakpoints
        BsPoly bp = BsPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1, 1}, {3, 1}});

        double eps = 1e-10;
        bool c1_continuous = true;

        // Check C1 continuity at x=1
        double f_left = bp(1.0 - eps);
        double f_right = bp(1.0 + eps);
        double df_left = bp(1.0 - eps, 1);
        double df_right = bp(1.0 + eps, 1);

        if (std::abs(f_left - f_right) > 1e-8) {
            test.fail("Test 77a: C0 continuity at x=1");
            c1_continuous = false;
        }
        if (std::abs(df_left - df_right) > 1e-6) {
            test.fail("Test 77b: C1 continuity (derivative) at x=1");
            c1_continuous = false;
        }
        if (c1_continuous) {
            test.pass("Test 77: from_derivatives with matching f' produces C1 continuity");
        }
    }

    // ============================================================
    // Group 32: Missing Coverage - extend() with Mixed Degrees
    // ============================================================
    std::cout << "\n--- Group 32: extend() with Mixed Degrees ---\n" << std::endl;

    // Test 78: Extend cubic polynomial with linear polynomial
    {
        // Create cubic on [0, 1]: f(x) = x^3
        BsPoly bp_cubic = BsPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1});

        // Create linear on [1, 2]: g(x) = 2x - 1 (matches f(1)=1, since f(1)=1 and g(1)=1)
        // For Bernstein linear on [1,2]: g(t) = c0*(1-t) + c1*t where t = x - 1
        // g(1) = c0 = 1, g(2) = c1 = 3
        BsPoly bp_linear({{1}, {3}}, {1, 2});

        BsPoly extended = bp_cubic.extend(bp_linear.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 78a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.125, tolerance, "Test 78b: Cubic part at 0.5");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 78c: Linear part at 1.5");
        test.expect_near(extended(1.0), 1.0, tolerance, "Test 78d: Continuity at boundary");
    }

    // Test 79: Extend quadratic with quintic
    {
        // Quadratic on [0, 1]
        BsPoly bp_quad = BsPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});

        // Quintic on [1, 2] from_derivatives with 3 constraints each side
        BsPoly bp_quint = BsPoly::from_derivatives({1, 2}, {{1, 2, 2}, {4, 4, 2}});

        BsPoly extended = bp_quad.extend(bp_quint.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 79a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.25, tolerance, "Test 79b: Quadratic part");
        test.expect_near(extended(1.5), bp_quint(1.5), tolerance, "Test 79c: Quintic part");
    }

    // ============================================================
    // Group 33: Missing Coverage - Edge Cases
    // ============================================================
    std::cout << "\n--- Group 33: Edge Case Coverage ---\n" << std::endl;

    // Test 80: Empty vector evaluation
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> empty_input;
        std::vector<double> result = bp(empty_input);
        test.expect_eq(result.size(), 0ul, "Test 80: Empty vector evaluation returns empty");
    }

    // Test 81: Single-point vector evaluation
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> single_point = {0.5};
        std::vector<double> result = bp(single_point);
        test.expect_eq(result.size(), 1ul, "Test 81a: Single-point vector size");
        test.expect_near(result[0], 2.0, tolerance, "Test 81b: Single-point value correct");
    }

    // Test 82: Evaluation exactly at all breakpoints
    {
        // Multi-interval polynomial: linear segments f(x)=x on [0,1], f(x)=x on [1,2], f(x)=3x-4 on [2,3]
        // Note: Continuous at breakpoints (f(2)=2 from both sides) but derivative discontinuous at x=2
        BsPoly bp({{0, 1, 2}, {1, 2, 5}}, {0, 1, 2, 3});
        std::vector<double> breakpts = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> results = bp(breakpts);

        test.expect_near(results[0], 0.0, tolerance, "Test 82a: Eval at bp[0]");
        test.expect_near(results[1], 1.0, tolerance, "Test 82b: Eval at bp[1]");
        test.expect_near(results[2], 2.0, tolerance, "Test 82c: Eval at bp[2]");
        test.expect_near(results[3], 5.0, tolerance, "Test 82d: Eval at bp[3]");
    }

    // ============================================================
    // Group 34: Additional Edge Case Coverage
    // ============================================================
    std::cout << "\n--- Group 34: Additional Edge Case Coverage ---\n" << std::endl;

    // Test 83: Zero polynomial (all coefficients zero)
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {0}};
        std::vector<double> breaks = {0, 1};
        BsPoly bp(coeffs, breaks);

        test.expect_near(bp(0.0), 0.0, tolerance, "Test 83a: Zero polynomial at 0");
        test.expect_near(bp(0.5), 0.0, tolerance, "Test 83b: Zero polynomial at 0.5");
        test.expect_near(bp(1.0), 0.0, tolerance, "Test 83c: Zero polynomial at 1");
        test.expect_near(bp(0.5, 1), 0.0, tolerance, "Test 83d: Zero polynomial derivative");
        test.expect_near(bp.integrate(0, 1), 0.0, tolerance, "Test 83e: Zero polynomial integral");
    }

    // Test 84: Repeated roots - polynomial (x-0.5)^2 = x^2 - x + 0.25
    {
        // f(x) = x^2 - x + 0.25 = (x - 0.5)^2 has a double root at x = 0.5
        BsPoly bp = BsPoly::from_power_basis({{0.25}, {-1}, {1}}, {0, 1});

        auto roots = bp.roots();
        // Should find one root (possibly reported twice, or once with multiplicity)
        test.expect_true(roots.size() >= 1, "Test 84a: Repeated root found");
        if (!roots.empty()) {
            test.expect_near(roots[0], 0.5, 1e-6, "Test 84b: Repeated root at 0.5");
        }
        // Verify the polynomial touches zero at 0.5
        test.expect_near(bp(0.5), 0.0, tolerance, "Test 84c: f(0.5) = 0");
        // Derivative should also be zero at repeated root
        test.expect_near(bp(0.5, 1), 0.0, tolerance, "Test 84d: f'(0.5) = 0");
    }

    // Test 85: Integration beyond domain (extrapolation region)
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);

        // Integration from -1 to 2 (extends beyond [0,1] domain)
        // f(x) = x, so integral from -1 to 2 = [x^2/2] from -1 to 2 = 2 - 0.5 = 1.5
        double integral = bp.integrate(-1, 2);
        test.expect_near(integral, 1.5, tolerance, "Test 85a: Integration beyond domain");

        // Partial integration in extrapolation region
        // Integral from -1 to 0 = [x^2/2] from -1 to 0 = 0 - 0.5 = -0.5
        test.expect_near(bp.integrate(-1, 0), -0.5, tolerance, "Test 85b: Integration in left extrapolation");

        // Integral from 1 to 2 = [x^2/2] from 1 to 2 = 2 - 0.5 = 1.5
        test.expect_near(bp.integrate(1, 2), 1.5, tolerance, "Test 85c: Integration in right extrapolation");
    }

    // ============================================================
    // Group 35: Bernstein Basis-Specific Properties
    // ============================================================
    std::cout << "\n--- Group 35: Bernstein Basis-Specific Properties ---\n" << std::endl;

    // Test 86: Partition of Unity - sum of Bernstein basis functions = 1
    // B_{i,n}(t) sum to 1 for all t in [0,1]
    {
        // For coefficients all equal to 1, the polynomial evaluates to 1 everywhere
        for (int n = 1; n <= 10; ++n) {
            std::vector<std::vector<double>> coeffs(n + 1, {1.0});
            BsPoly bp(coeffs, {0, 1});

            bool all_one = true;
            std::vector<double> test_pts = {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0};
            for (double t : test_pts) {
                if (std::abs(bp(t) - 1.0) > 1e-10) {
                    all_one = false;
                    break;
                }
            }
            std::string test_name = "Test 86" + std::string(1, 'a' + (n-1 > 9 ? 9 : n-1)) + ": Partition of unity degree " + std::to_string(n);
            test.expect_true(all_one, test_name);
        }
    }

    // Test 87: Endpoint Interpolation - p(0) = c_0, p(1) = c_n
    {
        struct TestCase {
            std::vector<double> coeffs;
        };
        std::vector<TestCase> test_cases = {
            {{1.0, 2.0}},
            {{0.0, 5.0, 3.0}},
            {{1.5, 2.5, 3.5, 4.5}},
            {{10.0, 0.0, 0.0, 0.0, 20.0}}
        };

        int idx = 0;
        for (const auto& tc : test_cases) {
            std::vector<std::vector<double>> coeffs;
            for (double c : tc.coeffs) {
                coeffs.push_back({c});
            }
            BsPoly bp(coeffs, {0, 1});

            double expected_0 = tc.coeffs.front();
            double expected_1 = tc.coeffs.back();

            std::string suffix = std::string(1, 'a' + idx);
            test.expect_near(bp(0.0), expected_0, tolerance, "Test 87" + suffix + ": Endpoint p(0) = c_0");
            test.expect_near(bp(1.0), expected_1, tolerance, "Test 87" + suffix + ": Endpoint p(1) = c_n");
            idx++;
        }
    }

    // Test 88: Convex Hull Property - values lie within [min(coeffs), max(coeffs)]
    {
        struct TestCase {
            std::vector<double> coeffs;
        };
        std::vector<TestCase> test_cases = {
            {{1, 5, 2, 4, 3}},
            {{0, 10, 5, 8, 2}},
            {{-5, 10, -3, 7, 0}},
            {{1, 1, 1, 1, 1}}
        };

        int idx = 0;
        for (const auto& tc : test_cases) {
            std::vector<std::vector<double>> coeffs;
            for (double c : tc.coeffs) {
                coeffs.push_back({c});
            }
            BsPoly bp(coeffs, {0, 1});

            double min_c = *std::min_element(tc.coeffs.begin(), tc.coeffs.end());
            double max_c = *std::max_element(tc.coeffs.begin(), tc.coeffs.end());

            bool within_hull = true;
            for (int i = 0; i <= 100; ++i) {
                double t = i / 100.0;
                double val = bp(t);
                if (val < min_c - 1e-10 || val > max_c + 1e-10) {
                    within_hull = false;
                    break;
                }
            }
            std::string suffix = std::string(1, 'a' + idx);
            test.expect_true(within_hull, "Test 88" + suffix + ": Convex hull property");
            idx++;
        }
    }

    // Test 89: Non-negativity Preservation
    // Non-negative coefficients produce non-negative polynomial values
    {
        struct TestCase {
            std::vector<double> coeffs;
        };
        std::vector<TestCase> test_cases = {
            {{0, 1, 0.5, 2, 0.1}},
            {{1, 2, 3, 4, 5}},
            {{0, 0, 0, 0, 1}},
            {{1, 0, 0, 0, 0}},
            {{0.001, 0.002, 0.003, 0.004}}
        };

        int idx = 0;
        for (const auto& tc : test_cases) {
            std::vector<std::vector<double>> coeffs;
            for (double c : tc.coeffs) {
                coeffs.push_back({c});
            }
            BsPoly bp(coeffs, {0, 1});

            bool all_non_negative = true;
            for (int i = 0; i <= 100; ++i) {
                double t = i / 100.0;
                if (bp(t) < -1e-15) {
                    all_non_negative = false;
                    break;
                }
            }
            std::string suffix = std::string(1, 'a' + idx);
            test.expect_true(all_non_negative, "Test 89" + suffix + ": Non-negativity preservation");
            idx++;
        }
    }

    // Test 90: Zero Coefficient Invariance - adding zero high-order term preserves values
    // (Note: This is NOT testing Bernstein degree elevation algorithm, just that
    // adding a zero coefficient to the power basis produces identical polynomial values)
    {
        // Test with linear f(x) = x
        BsPoly bp_linear = BsPoly::from_power_basis({{0}, {1}}, {0, 1});
        // Add zero x^2 term - polynomial is still f(x) = x
        BsPoly bp_quad = BsPoly::from_power_basis({{0}, {1}, {0}}, {0, 1});

        // Both should evaluate to same values (linear function)
        test.expect_near(bp_linear(0.25), bp_quad(0.25), tolerance, "Test 90a: Zero coeff invariance at 0.25");
        test.expect_near(bp_linear(0.5), bp_quad(0.5), tolerance, "Test 90b: Zero coeff invariance at 0.5");
        test.expect_near(bp_linear(0.75), bp_quad(0.75), tolerance, "Test 90c: Zero coeff invariance at 0.75");
    }

    // Test 91: Symmetry - symmetric coefficients produce symmetric curve
    {
        // Symmetric coefficients: [1, 2, 2, 1]
        BsPoly bp({{1}, {2}, {2}, {1}}, {0, 1});

        // Check p(t) = p(1-t)
        test.expect_near(bp(0.2), bp(0.8), tolerance, "Test 91a: Symmetric curve at 0.2 vs 0.8");
        test.expect_near(bp(0.3), bp(0.7), tolerance, "Test 91b: Symmetric curve at 0.3 vs 0.7");
        test.expect_near(bp(0.4), bp(0.6), tolerance, "Test 91c: Symmetric curve at 0.4 vs 0.6");
    }

    // ============================================================
    // Group 36: Derivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 36: Derivative Edge Cases ---\n" << std::endl;

    // Test 92: Derivative of degree-0 polynomial should be zero polynomial
    {
        BsPoly bp({{5}}, {0, 1});  // Constant 5
        BsPoly deriv = bp.derivative();
        test.expect_near(deriv(0.5), 0.0, tolerance, "Test 92a: Derivative of constant is 0");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 0ul, "Test 92b: Derivative degree is 0");
    }

    // Test 93: Over-differentiate (n-th derivative where n > degree)
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});  // Quadratic
        BsPoly d3 = bp.derivative(3);  // Third derivative of quadratic = 0
        test.expect_near(d3(0.5), 0.0, tolerance, "Test 93a: 3rd derivative of quadratic is 0");

        BsPoly d10 = bp.derivative(10);  // Way over-differentiated
        test.expect_near(d10(0.5), 0.0, tolerance, "Test 93b: 10th derivative of quadratic is 0");
    }

    // Test 94: Chained derivatives vs single derivative(n)
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});  // Cubic
        BsPoly d3_chained = bp.derivative().derivative().derivative();
        BsPoly d3_single = bp.derivative(3);

        test.expect_near(d3_chained(0.5), d3_single(0.5), tolerance, "Test 94: Chained vs single derivative(3)");
    }

    // ============================================================
    // Group 37: Antiderivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 37: Antiderivative Edge Cases ---\n" << std::endl;

    // Test 95: Antiderivative of zero polynomial
    {
        BsPoly bp({{0}, {0}}, {0, 1});  // Zero polynomial
        BsPoly anti = bp.antiderivative();
        test.expect_near(anti(0.0), 0.0, tolerance, "Test 95a: Antiderivative of zero at 0");
        test.expect_near(anti(0.5), 0.0, tolerance, "Test 95b: Antiderivative of zero at 0.5");
        test.expect_near(anti(1.0), 0.0, tolerance, "Test 95c: Antiderivative of zero at 1");
    }

    // Test 96: Antiderivative(n).derivative(n) = original
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BsPoly round_trip = bp.antiderivative(2).derivative(2);

        test.expect_near(bp(0.25), round_trip(0.25), tolerance, "Test 96a: antiderivative(2).derivative(2) at 0.25");
        test.expect_near(bp(0.5), round_trip(0.5), tolerance, "Test 96b: antiderivative(2).derivative(2) at 0.5");
        test.expect_near(bp(0.75), round_trip(0.75), tolerance, "Test 96c: antiderivative(2).derivative(2) at 0.75");
    }

    // Test 97: Chained antiderivatives vs single antiderivative(n)
    {
        BsPoly bp = BsPoly::from_power_basis({{2}}, {0, 1});  // f(x) = 2
        BsPoly a2_chained = bp.antiderivative().antiderivative();
        BsPoly a2_single = bp.antiderivative(2);

        test.expect_near(a2_chained(0.5), a2_single(0.5), tolerance, "Test 97: Chained vs single antiderivative(2)");
    }

    // ============================================================
    // Group 37b: Derivative/Antiderivative Structural Properties
    // ============================================================
    std::cout << "\n--- Group 37b: Derivative/Antiderivative Structural Properties ---\n" << std::endl;

    // Test 97b: Derivative reduces degree by 1
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});  // Cubic (degree 3)
        BsPoly deriv = bp.derivative();
        test.expect_eq(static_cast<size_t>(bp.degree()), 3ul, "Test 97b-1: Original degree is 3");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 2ul, "Test 97b-2: Derivative degree is 2");
    }

    // Test 97c: Antiderivative increases degree by 1
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});  // Quadratic (degree 2)
        BsPoly anti = bp.antiderivative();
        test.expect_eq(static_cast<size_t>(bp.degree()), 2ul, "Test 97c-1: Original degree is 2");
        test.expect_eq(static_cast<size_t>(anti.degree()), 3ul, "Test 97c-2: Antiderivative degree is 3");
    }

    // Test 97d: Derivative preserves num_intervals
    {
        BsPoly bp({{1, 2, 3}}, {0, 1, 2, 3});  // 3 intervals
        BsPoly deriv = bp.derivative();
        test.expect_eq(static_cast<size_t>(deriv.num_intervals()), 3ul,
            "Test 97d: Derivative preserves num_intervals");
    }

    // Test 97e: Derivative preserves breakpoints
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        BsPoly deriv = bp.derivative();
        test.expect_near(deriv.x()[0], 0.0, tolerance, "Test 97e-1: Derivative preserves left breakpoint");
        test.expect_near(deriv.x()[1], 1.0, tolerance, "Test 97e-2: Derivative preserves right breakpoint");
    }

    // Test 97f: Antiderivative starts at 0 at left boundary
    {
        BsPoly bp = BsPoly::from_power_basis({{5}}, {0, 1});  // Constant 5
        BsPoly anti = bp.antiderivative();
        test.expect_near(anti(0), 0.0, tolerance, "Test 97f: Antiderivative(left_boundary) = 0");
    }

    // ============================================================
    // Group 38: Integration Edge Cases
    // ============================================================
    std::cout << "\n--- Group 38: Integration Edge Cases ---\n" << std::endl;

    // Test 98: integrate(a, a) = 0
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        test.expect_near(bp.integrate(0.5, 0.5), 0.0, tolerance, "Test 98: integrate(a, a) = 0");
    }

    // Test 99: Integration crossing multiple intervals
    {
        // Simple two-interval polynomial: f(x) = 1 on [0,1], f(x) = 2 on [1,2]
        // Using Bernstein coefficients directly (constant on each interval)
        BsPoly bp({{1, 2}}, {0, 1, 2});

        // Integral from 0 to 2 = 1 * 1 + 2 * 1 = 3.0
        test.expect_near(bp.integrate(0, 2), 3.0, tolerance, "Test 99: Integration across multiple intervals");
    }

    // Test 100: NoExtrapolate mode returns NaN when evaluating beyond bounds
    {
        BsPoly temp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        BsPoly bp(temp.c(), {0, 1}, ExtrapolateMode::NoExtrapolate);
        // NoExtrapolate returns NaN on evaluation beyond bounds
        double result = bp(-1);
        test.expect_true(std::isnan(result), "Test 100: NoExtrapolate evaluation beyond bounds returns NaN");
    }

    // ============================================================
    // Group 39: Root Finding Edge Cases
    // ============================================================
    std::cout << "\n--- Group 39: Root Finding Edge Cases ---\n" << std::endl;

    // Test 101: No roots (always positive polynomial)
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {0}, {1}}, {0, 1});  // 1 + x^2 >= 1
        auto roots = bp.roots();
        test.expect_eq(roots.size(), 0ul, "Test 101: No roots for always-positive polynomial");
    }

    // Test 102: Root at domain boundary
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {1}}, {0, 1});  // f(x) = x, root at x=0
        auto roots = bp.roots();
        test.expect_true(roots.size() >= 1, "Test 102a: Root at boundary found");
        if (!roots.empty()) {
            double min_root = *std::min_element(roots.begin(), roots.end());
            test.expect_near(min_root, 0.0, 1e-6, "Test 102b: Root at x=0");
        }
    }

    // Test 103: Many roots (polynomial with roots at 0.25, 0.5, 0.75)
    {
        // f(x) = (x - 0.25)(x - 0.5)(x - 0.75)
        BsPoly bp = BsPoly::from_power_basis({{-0.09375}, {0.6875}, {-1.5}, {1}}, {0, 1});
        auto roots = bp.roots();
        test.expect_eq(roots.size(), 3ul, "Test 103a: Three roots found");

        if (roots.size() == 3) {
            std::sort(roots.begin(), roots.end());
            test.expect_near(roots[0], 0.25, 1e-6, "Test 103b: Root at 0.25");
            test.expect_near(roots[1], 0.5, 1e-6, "Test 103c: Root at 0.5");
            test.expect_near(roots[2], 0.75, 1e-6, "Test 103d: Root at 0.75");
        }
    }

    // ============================================================
    // Group 40: Extrapolation Order Edge Cases
    // ============================================================
    std::cout << "\n--- Group 40: Extrapolation Order Edge Cases ---\n" << std::endl;

    // Test 104: extrapolation_order > polynomial degree behaves like full extrapolation
    {
        // Get Bernstein coefficients via from_power_basis, then construct with extrapolation order
        BsPoly temp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        BsPoly bp_full(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, -1, -1);
        BsPoly bp_high(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 10, 10);

        test.expect_near(bp_full(-0.5), bp_high(-0.5), tolerance, "Test 104a: High extrapolation order matches full");
        test.expect_near(bp_full(1.5), bp_high(1.5), tolerance, "Test 104b: High extrapolation order matches full");
    }

    // Test 105: extrapolation_order = 0 gives constant extrapolation
    {
        BsPoly temp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        BsPoly bp(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 0, 0);
        // f(x) = 1 + 2x, f(0) = 1, f(1) = 3
        test.expect_near(bp(-0.5), 1.0, tolerance, "Test 105a: Order 0 extrapolation at left");
        test.expect_near(bp(1.5), 3.0, tolerance, "Test 105b: Order 0 extrapolation at right");
    }

    // Test 106: Asymmetric extrapolation orders
    {
        BsPoly temp = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        BsPoly bp(temp.c(), {0, 1}, ExtrapolateMode::Extrapolate, 0, -1);
        test.expect_near(bp(-0.5), 1.0, tolerance, "Test 106a: Order 0 on left");
        test.expect_near(bp(1.5), 1.0 + 2*1.5, tolerance, "Test 106b: Full order on right");
    }

    // ============================================================
    // Group 41: Move/Copy Semantics
    // ============================================================
    std::cout << "\n--- Group 41: Move/Copy Semantics ---\n" << std::endl;

    // Test 107: Copy constructor
    {
        BsPoly bp1 = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BsPoly bp2(bp1);  // Copy

        test.expect_near(bp1(0.5), bp2(0.5), tolerance, "Test 107a: Copy constructor preserves value");
        test.expect_eq(static_cast<size_t>(bp1.degree()), static_cast<size_t>(bp2.degree()), "Test 107b: Copy constructor preserves degree");
    }

    // Test 108: Move constructor
    {
        BsPoly bp1 = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double original_val = bp1(0.5);
        BsPoly bp2(std::move(bp1));

        test.expect_near(bp2(0.5), original_val, tolerance, "Test 108: Move constructor preserves value");
    }

    // ============================================================
    // Group 42: Const-Correctness and Thread Safety
    // ============================================================
    std::cout << "\n--- Group 42: Const-Correctness and Thread Safety ---\n" << std::endl;

    // Test 109: All const methods work on const object
    {
        const BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double v1 = bp(0.5);
        double v2 = bp(0.5, 1);
        std::vector<double> v3 = bp({0.25, 0.5, 0.75});
        int d = bp.degree();
        int n = bp.num_intervals();
        const auto& c = bp.c();
        const auto& x = bp.x();
        double integ = bp.integrate(0, 1);
        auto roots = bp.roots();

        test.expect_near(v1, 1 + 2*0.5 + 3*0.25, tolerance, "Test 109a: Const evaluation");
        test.expect_eq(static_cast<size_t>(d), 2ul, "Test 109b: Const degree()");
        test.expect_eq(static_cast<size_t>(n), 1ul, "Test 109c: Const num_intervals()");
        test.pass("Test 109: All const methods work on const object");
    }

    // Test 110: Thread safety - multiple threads evaluating same polynomial
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        std::atomic<int> correct_count{0};
        const int num_threads = 10;
        const int evals_per_thread = 1000;

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&bp, &correct_count, evals_per_thread]() {
                std::mt19937 gen(std::random_device{}());
                std::uniform_real_distribution<> dis(0.0, 1.0);

                for (int i = 0; i < evals_per_thread; ++i) {
                    double x = dis(gen);
                    double result = bp(x);
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
                      "Test 110: Thread safety - all concurrent evaluations correct");
    }

    // ============================================================
    // Group 43: Property-Based Tests
    // ============================================================
    std::cout << "\n--- Group 43: Property-Based Tests ---\n" << std::endl;

    // Test 111: Linearity of integration
    {
        BsPoly bp1 = BsPoly::from_power_basis({{1}, {2}}, {0, 1});
        BsPoly bp2 = BsPoly::from_power_basis({{3}, {4}}, {0, 1});
        BsPoly bp_sum = BsPoly::from_power_basis({{4}, {6}}, {0, 1});

        double int_sum = bp_sum.integrate(0, 1);
        double sum_int = bp1.integrate(0, 1) + bp2.integrate(0, 1);

        test.expect_near(int_sum, sum_int, tolerance, "Test 111: Linearity of integration");
    }

    // Test 112: Fundamental theorem of calculus
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BsPoly anti = bp.antiderivative();
        BsPoly anti_deriv = anti.derivative();

        for (double x : {0.25, 0.5, 0.75}) {
            test.expect_near(bp(x), anti_deriv(x), tolerance,
                           "Test 112: FTC d/dx(antiderivative(f)) = f at x=" + std::to_string(x));
        }
    }

    // Test 113: Integration reversal
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double int_ab = bp.integrate(0.2, 0.8);
        double int_ba = bp.integrate(0.8, 0.2);

        test.expect_near(int_ab, -int_ba, tolerance, "Test 113: Integration reversal");
    }

    // Test 114: Integration additivity
    {
        BsPoly bp = BsPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double int_ac = bp.integrate(0.2, 0.8);
        double int_ab = bp.integrate(0.2, 0.5);
        double int_bc = bp.integrate(0.5, 0.8);

        test.expect_near(int_ac, int_ab + int_bc, tolerance, "Test 114: Integration additivity");
    }

    // ============================================================
    // Group 44: Symmetry Tests
    // ============================================================
    std::cout << "\n--- Group 44: Symmetry Tests ---\n" << std::endl;

    // Test 115: Symmetric Bernstein curve p(t) = p(1-t) on [0,1]
    {
        // Symmetric coefficients produce a symmetric curve on [0,1]
        // coefficients [1, 3, 3, 1] are symmetric
        BsPoly bp({{1}, {3}, {3}, {1}}, {0, 1});

        test.expect_near(bp(0.2), bp(0.8), tolerance, "Test 115a: Symmetric curve at 0.2 vs 0.8");
        test.expect_near(bp(0.3), bp(0.7), tolerance, "Test 115b: Symmetric curve at 0.3 vs 0.7");
    }

    // Test 116: Anti-symmetric Bernstein curve p(t) + p(1-t) = const on [0,1]
    {
        // For coefficients [a, b, c, d] with a+d = b+c, we get p(t) + p(1-t) = a+d
        // coefficients [0, 2, 2, 4] give anti-symmetric: p(t) + p(1-t) = 4
        BsPoly bp({{0}, {2}, {2}, {4}}, {0, 1});

        double sum_at_02_08 = bp(0.2) + bp(0.8);
        double sum_at_03_07 = bp(0.3) + bp(0.7);
        test.expect_near(sum_at_02_08, 4.0, tolerance, "Test 116a: Anti-symmetric sum at 0.2+0.8");
        test.expect_near(sum_at_03_07, 4.0, tolerance, "Test 116b: Anti-symmetric sum at 0.3+0.7");
    }

    // ============================================================
    // Group 45: Boundary Edge Cases
    // ============================================================
    std::cout << "\n--- Group 45: Boundary Edge Cases ---\n" << std::endl;

    // Test 117: Evaluation at breakpoint + epsilon
    {
        BsPoly bp1 = BsPoly::from_power_basis({{0}, {1}}, {0, 1});
        BsPoly bp2 = BsPoly::from_power_basis({{1}, {2}}, {1, 2});
        BsPoly bp = bp1.extend(bp2.c(), {1, 2}, true);

        double eps = std::numeric_limits<double>::epsilon();
        double at_bp = bp(1.0);
        double after_bp = bp(1.0 + eps);
        double before_bp = bp(1.0 - eps);

        test.expect_near(at_bp, 1.0, tolerance, "Test 117a: At breakpoint");
        test.expect_near(before_bp, 1.0, tolerance, "Test 117b: Just before breakpoint");
        test.expect_near(after_bp, 1.0, tolerance, "Test 117c: Just after breakpoint");
    }

    // Test 118: Very wide interval
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {1}}, {0, 1e10});
        test.expect_near(bp(0), 0.0, tolerance, "Test 118a: Wide interval at left");
        test.expect_near(bp(5e9), 5e9, 1e-3, "Test 118b: Wide interval at midpoint");
        test.expect_near(bp(1e10), 1e10, 1e-3, "Test 118c: Wide interval at right");
    }

    // Test 119: Very narrow interval
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {1}}, {0, 1e-10});
        test.expect_near(bp(0), 0.0, tolerance, "Test 119a: Narrow interval at left");
        test.expect_near(bp(5e-11), 5e-11, 1e-20, "Test 119b: Narrow interval at midpoint");
        test.expect_near(bp(1e-10), 1e-10, 1e-20, "Test 119c: Narrow interval at right");
    }

    // ============================================================
    // Group 46: Reference Data Verification (from scipy)
    // ============================================================
    std::cout << "\n--- Group 46: Reference Data Verification ---\n" << std::endl;

    // Test 120: Verify Bernstein evaluation against scipy reference
    // Quadratic with coefficients [0, 1, 0] (bump at t=0.5)
    {
        BsPoly bp({{0}, {1}, {0}}, {0, 1});
        // Maximum at t=0.5 should be 0.5 (due to B_1,2(0.5) = 2*0.5*0.5 = 0.5)
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 120a: Bernstein at t=0");
        test.expect_near(bp(0.5), 0.5, tolerance, "Test 120b: Bernstein at t=0.5");
        test.expect_near(bp(1.0), 0.0, tolerance, "Test 120c: Bernstein at t=1");
        test.expect_near(bp(0.25), 0.375, tolerance, "Test 120d: Bernstein at t=0.25");
        test.expect_near(bp(0.75), 0.375, tolerance, "Test 120e: Bernstein at t=0.75");
    }

    // Test 121: Verify from_power_basis against scipy (quadratic x^2)
    {
        BsPoly bp = BsPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});
        auto& c = bp.c();

        // Bernstein coefficients for x^2 on [0,1] should be [0, 0, 1]
        test.expect_near(c[0][0], 0.0, tolerance, "Test 121a: x^2 Bernstein c_0 = 0");
        test.expect_near(c[1][0], 0.0, tolerance, "Test 121b: x^2 Bernstein c_1 = 0");
        test.expect_near(c[2][0], 1.0, tolerance, "Test 121c: x^2 Bernstein c_2 = 1");
    }

    // Test 122: Verify high-degree evaluation stability
    // For degree 50 with coefficients [0, 0, ..., 0, 1] (basis function B_{50,50})
    {
        std::vector<std::vector<double>> coeffs(51, {0.0});
        coeffs[50][0] = 1.0;
        BsPoly bp(coeffs, {0, 1});

        // B_{n,n}(1) = 1, B_{n,n}(0) = 0, B_{n,n}(0.5) = (0.5)^50
        test.expect_near(bp(0.0), 0.0, tolerance, "Test 122a: B_{50,50}(0) = 0");
        test.expect_near(bp(1.0), 1.0, 1e-8, "Test 122b: B_{50,50}(1) = 1");
        test.expect_near(bp(0.5), std::pow(0.5, 50), 1e-20, "Test 122c: B_{50,50}(0.5) = (0.5)^50");
    }

    // Test 123: Verify from_derivatives Hermite interpolation against scipy
    // From reference: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
    {
        BsPoly bp = BsPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});

        test.expect_near(bp(0.0), 0.0, tolerance, "Test 123a: Hermite f(0) = 0");
        test.expect_near(bp(1.0), 1.0, tolerance, "Test 123b: Hermite f(1) = 1");
        test.expect_near(bp(0.0, 1), 1.0, tolerance, "Test 123c: Hermite f'(0) = 1");
        test.expect_near(bp(1.0, 1), -1.0, tolerance, "Test 123d: Hermite f'(1) = -1");
        // From scipy reference: f(0.5) = 0.75
        test.expect_near(bp(0.5), 0.75, tolerance, "Test 123e: Hermite f(0.5) = 0.75 (scipy ref)");
    }

    // ============================================================
    // Summary
    // ============================================================
    test.summary();

    return test.all_passed() ? 0 : 1;
}
