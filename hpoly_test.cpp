#include "include/hpoly.h"
#include "test_utils.h"
#include <cassert>
#include <limits>
#include <thread>
#include <atomic>
#include <random>

int main() {
    TestRunner test;
    const double tolerance = 1e-10;
    const double loose_tolerance = 1e-6;

    std::cout << "=== HPoly (Hermite Polynomial) Test Suite ===" << std::endl;

    // ============================================================
    // Group 1: Basic Construction and Evaluation - Physicist's
    // ============================================================
    std::cout << "\n--- Group 1: Basic Construction and Evaluation (Physicist's) ---\n" << std::endl;

    // Test 1: Basic construction with Physicist's Hermite
    test.expect_no_throw([]() {
        // H_0 = 1 (constant), so c = {1} gives p(x) = 1
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);
    }, "Test 1: Basic Physicist construction");

    // Test 2: Constant polynomial H_0 = 1
    {
        std::vector<std::vector<double>> coeffs = {{5}};  // p(x) = 5
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0.0), 5.0, tolerance, "Test 2a: Physicist constant f(0)");
        test.expect_near(hp(0.5), 5.0, tolerance, "Test 2b: Physicist constant f(0.5)");
        test.expect_near(hp(1.0), 5.0, tolerance, "Test 2c: Physicist constant f(1)");
    }

    // Test 3: Verify Hermite kind accessor
    {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 1};
        HPoly hp_phys(coeffs, breaks, HermiteKind::Physicist);
        HPoly hp_prob(coeffs, breaks, HermiteKind::Probabilist);

        test.expect_true(hp_phys.kind() == HermiteKind::Physicist, "Test 3a: kind() returns Physicist");
        test.expect_true(hp_prob.kind() == HermiteKind::Probabilist, "Test 3b: kind() returns Probabilist");
    }

    // Test 4: Linear polynomial with H_1(s) = 2s (Physicist's)
    // On interval [0,1], s = 2x - 1
    // p(x) = c_0 * H_0(s) + c_1 * H_1(s) = c_0 + c_1 * 2s = c_0 + c_1 * 2(2x-1)
    // For p(x) = x, we need c_0 + 2*c_1*(2x-1) = x
    // => c_0 - 2*c_1 + 4*c_1*x = x
    // => c_0 = 2*c_1 and 4*c_1 = 1 => c_1 = 0.25, c_0 = 0.5
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0.0), 0.0, tolerance, "Test 4a: Physicist linear f(0)");
        test.expect_near(hp(0.25), 0.25, tolerance, "Test 4b: Physicist linear f(0.25)");
        test.expect_near(hp(0.5), 0.5, tolerance, "Test 4c: Physicist linear f(0.5)");
        test.expect_near(hp(1.0), 1.0, tolerance, "Test 4d: Physicist linear f(1)");
    }

    // Test 5: Multiple intervals - Physicist's
    {
        // Two constant pieces: [0,1] has value 1, [1,2] has value 2
        std::vector<std::vector<double>> coeffs = {{1, 2}};
        std::vector<double> breaks = {0, 1, 2};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0.5), 1.0, tolerance, "Test 5a: Physicist multi-interval f(0.5)");
        test.expect_near(hp(1.5), 2.0, tolerance, "Test 5b: Physicist multi-interval f(1.5)");
    }

    // ============================================================
    // Group 2: Basic Construction - Probabilist's
    // ============================================================
    std::cout << "\n--- Group 2: Basic Construction (Probabilist's) ---\n" << std::endl;

    // Test 6: Constant polynomial He_0 = 1 (Probabilist's)
    {
        std::vector<std::vector<double>> coeffs = {{5}};
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Probabilist);

        test.expect_near(hp(0.0), 5.0, tolerance, "Test 6a: Probabilist constant f(0)");
        test.expect_near(hp(0.5), 5.0, tolerance, "Test 6b: Probabilist constant f(0.5)");
        test.expect_near(hp(1.0), 5.0, tolerance, "Test 6c: Probabilist constant f(1)");
    }

    // Test 7: Linear polynomial with He_1(s) = s (Probabilist's)
    // On interval [0,1], s = 2x - 1
    // p(x) = c_0 * He_0(s) + c_1 * He_1(s) = c_0 + c_1 * s = c_0 + c_1 * (2x-1)
    // For p(x) = x, we need c_0 + c_1*(2x-1) = x
    // => c_0 - c_1 + 2*c_1*x = x
    // => c_0 = c_1 and 2*c_1 = 1 => c_1 = 0.5, c_0 = 0.5
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Probabilist);

        test.expect_near(hp(0.0), 0.0, tolerance, "Test 7a: Probabilist linear f(0)");
        test.expect_near(hp(0.25), 0.25, tolerance, "Test 7b: Probabilist linear f(0.25)");
        test.expect_near(hp(0.5), 0.5, tolerance, "Test 7c: Probabilist linear f(0.5)");
        test.expect_near(hp(1.0), 1.0, tolerance, "Test 7d: Probabilist linear f(1)");
    }

    // ============================================================
    // Group 3: Error Handling
    // ============================================================
    std::cout << "\n--- Group 3: Error Handling ---\n" << std::endl;

    // Test 8: Empty coefficients error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs;  // Empty
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks);
    }, "Test 8: Empty coefficients error");

    // Test 9: Too few breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0};  // Only 1 breakpoint
        HPoly hp(coeffs, breaks);
    }, "Test 9: Too few breakpoints error");

    // Test 10: Non-monotonic breakpoints error
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0, 0.5, 0.3};  // Not monotonic
        HPoly hp(coeffs, breaks);
    }, "Test 10: Non-monotonic breakpoints error");

    // Test 11: from_derivatives with mismatched xi/yi sizes
    test.expect_throw([]() {
        std::vector<double> xi = {0, 1, 2};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};
        HPoly::from_derivatives(xi, yi);
    }, "Test 11: from_derivatives mismatched xi/yi sizes");

    // Test 12: from_derivatives with single point
    test.expect_throw([]() {
        HPoly::from_derivatives({0}, {{1, 0}});
    }, "Test 12: from_derivatives with single point");

    // Test 13: from_derivatives with empty yi element
    test.expect_throw([]() {
        HPoly::from_derivatives({0, 1}, {{}, {1}});
    }, "Test 13: from_derivatives with empty yi element");

    // Test 14: extend with non-contiguous breakpoints
    test.expect_throw([]() {
        HPoly hp({{1}}, {0, 1});
        hp.extend({{2}}, {5, 6}, true);
    }, "Test 14: extend with non-contiguous breakpoints");

    // Test 15: extend with opposite ordering
    test.expect_throw([]() {
        HPoly hp({{1}}, {0, 1});  // Ascending
        hp.extend({{2}}, {2, 1}, true);  // Descending
    }, "Test 15: extend with opposite ordering");

    // Test 16: from_power_basis with empty coefficients
    test.expect_throw([]() {
        HPoly::from_power_basis({}, {0, 1});
    }, "Test 16: from_power_basis with empty coefficients");

    // Test 17: from_derivatives with invalid orders parameter size
    test.expect_throw([]() {
        HPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}}, {1, 2});
    }, "Test 17: from_derivatives with invalid orders size");

    // ============================================================
    // Group 4: Vector Evaluation
    // ============================================================
    std::cout << "\n--- Group 4: Vector Evaluation ---\n" << std::endl;

    // Test 18: Evaluate Physicist's at multiple points
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        std::vector<double> xs = {0.0, 0.25, 0.5, 0.75, 1.0};
        std::vector<double> results = hp(xs);

        test.expect_eq(results.size(), 5ul, "Test 18a: Vector result size");
        test.expect_near(results[0], 0.0, tolerance, "Test 18b: Vector result[0]");
        test.expect_near(results[2], 0.5, tolerance, "Test 18c: Vector result[2]");
        test.expect_near(results[4], 1.0, tolerance, "Test 18d: Vector result[4]");
    }

    // Test 19: Evaluate Probabilist's at multiple points
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Probabilist);

        std::vector<double> xs = {0.0, 0.25, 0.5, 0.75, 1.0};
        std::vector<double> results = hp(xs);

        test.expect_eq(results.size(), 5ul, "Test 19a: Probabilist vector result size");
        test.expect_near(results[0], 0.0, tolerance, "Test 19b: Probabilist vector result[0]");
        test.expect_near(results[2], 0.5, tolerance, "Test 19c: Probabilist vector result[2]");
        test.expect_near(results[4], 1.0, tolerance, "Test 19d: Probabilist vector result[4]");
    }

    // ============================================================
    // Group 5: Extrapolation Modes
    // ============================================================
    std::cout << "\n--- Group 5: Extrapolation Modes ---\n" << std::endl;

    // Test 20: Extrapolate mode (default)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist, ExtrapolateMode::Extrapolate);

        test.expect_near(hp(-0.5), -0.5, tolerance, "Test 20a: Extrapolate f(-0.5)");
        test.expect_near(hp(1.5), 1.5, tolerance, "Test 20b: Extrapolate f(1.5)");
    }

    // Test 21: NoExtrapolate mode
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist, ExtrapolateMode::NoExtrapolate);

        test.expect_true(std::isnan(hp(-0.5)), "Test 21a: NoExtrapolate f(-0.5) is NaN");
        test.expect_true(std::isnan(hp(1.5)), "Test 21b: NoExtrapolate f(1.5) is NaN");
    }

    // Test 22: Periodic mode
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist, ExtrapolateMode::Periodic);

        test.expect_near(hp(0.5), 0.5, tolerance, "Test 22a: Periodic f(0.5)");
        test.expect_near(hp(1.5), hp(0.5), tolerance, "Test 22b: Periodic f(1.5) = f(0.5)");
        test.expect_near(hp(2.5), hp(0.5), tolerance, "Test 22c: Periodic f(2.5) = f(0.5)");
    }

    // ============================================================
    // Group 6: Derivative Operations - Physicist's
    // ============================================================
    std::cout << "\n--- Group 6: Derivative Operations (Physicist's) ---\n" << std::endl;

    // Test 23: Derivative of linear polynomial (Physicist's)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);
        HPoly dhp = hp.derivative();

        // Derivative of x is 1
        test.expect_near(dhp(0.0), 1.0, tolerance, "Test 23a: Physicist d/dx[x] at 0");
        test.expect_near(dhp(0.5), 1.0, tolerance, "Test 23b: Physicist d/dx[x] at 0.5");
        test.expect_near(dhp(1.0), 1.0, tolerance, "Test 23c: Physicist d/dx[x] at 1");
    }

    // Test 24: Verify derivative against finite differences (Physicist's)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, 0, -2}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        double x = 0.3;
        double analytical = hp(x, 1);
        double numerical = finite_diff_derivative(hp, x);
        test.expect_near(analytical, numerical, 1e-5, "Test 24: Physicist derivative vs finite diff");
    }

    // Test 25: operator()(x, nu) syntax for derivatives (Physicist's)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        test.expect_near(hp(0, 0), 0.0, tolerance, "Test 25a: Physicist f(0, 0)");
        test.expect_near(hp(0, 1), 1.0, tolerance, "Test 25b: Physicist f(0, 1)");
        test.expect_near(hp(1, 0), 1.0, tolerance, "Test 25c: Physicist f(1, 0)");
        test.expect_near(hp(1, 1), -1.0, tolerance, "Test 25d: Physicist f(1, 1)");
    }

    // ============================================================
    // Group 7: Derivative Operations - Probabilist's
    // ============================================================
    std::cout << "\n--- Group 7: Derivative Operations (Probabilist's) ---\n" << std::endl;

    // Test 26: Derivative of linear polynomial (Probabilist's)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Probabilist);
        HPoly dhp = hp.derivative();

        test.expect_near(dhp(0.0), 1.0, tolerance, "Test 26a: Probabilist d/dx[x] at 0");
        test.expect_near(dhp(0.5), 1.0, tolerance, "Test 26b: Probabilist d/dx[x] at 0.5");
        test.expect_near(dhp(1.0), 1.0, tolerance, "Test 26c: Probabilist d/dx[x] at 1");
    }

    // Test 27: Verify derivative against finite differences (Probabilist's)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, 0, -2}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Probabilist);

        double x = 0.7;
        double analytical = hp(x, 1);
        double numerical = finite_diff_derivative(hp, x);
        test.expect_near(analytical, numerical, 1e-5, "Test 27: Probabilist derivative vs finite diff");
    }

    // ============================================================
    // Group 8: Antiderivative and Integration - Physicist's
    // ============================================================
    std::cout << "\n--- Group 8: Antiderivative and Integration (Physicist's) ---\n" << std::endl;

    // Test 28: Antiderivative of constant (Physicist's)
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);
        HPoly ihp = hp.antiderivative();

        // Antiderivative of 2 is 2x (starting at 0)
        test.expect_near(ihp(0), 0.0, tolerance, "Test 28a: Physicist int[2] at 0");
        test.expect_near(ihp(1), 2.0, tolerance, "Test 28b: Physicist int[2] at 1");
        test.expect_near(ihp(0.5), 1.0, tolerance, "Test 28c: Physicist int[2] at 0.5");
    }

    // Test 29: Integration (definite integral) - Physicist's
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        double integral = hp.integrate(0, 1);
        test.expect_near(integral, 0.5, tolerance, "Test 29a: Physicist int_0^1 x dx");
        test.expect_near(hp.integrate(0, 0.5), 0.125, tolerance, "Test 29b: Physicist int_0^0.5 x dx");
    }

    // Test 30: Negative derivative order = antiderivative
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        HPoly ihp = hp.derivative(-1);  // -1 means antiderivative
        test.expect_near(ihp(1), 2.0, tolerance, "Test 30: derivative(-1) = antiderivative");
    }

    // Test 31: Compare integration with numerical integration
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        double analytical = hp.integrate(0.2, 0.8);
        double numerical = numerical_integrate(hp, 0.2, 0.8);
        test.expect_near(analytical, numerical, 1e-4, "Test 31: Physicist integrate vs numerical");
    }

    // ============================================================
    // Group 9: Antiderivative and Integration - Probabilist's
    // ============================================================
    std::cout << "\n--- Group 9: Antiderivative and Integration (Probabilist's) ---\n" << std::endl;

    // Test 32: Antiderivative of constant (Probabilist's)
    {
        std::vector<std::vector<double>> coeffs = {{2}};  // p(x) = 2
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Probabilist);
        HPoly ihp = hp.antiderivative();

        test.expect_near(ihp(0), 0.0, tolerance, "Test 32a: Probabilist int[2] at 0");
        test.expect_near(ihp(1), 2.0, tolerance, "Test 32b: Probabilist int[2] at 1");
        test.expect_near(ihp(0.5), 1.0, tolerance, "Test 32c: Probabilist int[2] at 0.5");
    }

    // Test 33: Integration (definite integral) - Probabilist's
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.5}};  // p(x) = x
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Probabilist);

        double integral = hp.integrate(0, 1);
        test.expect_near(integral, 0.5, tolerance, "Test 33a: Probabilist int_0^1 x dx");
        test.expect_near(hp.integrate(0, 0.5), 0.125, tolerance, "Test 33b: Probabilist int_0^0.5 x dx");
    }

    // ============================================================
    // Group 10: from_derivatives (Hermite Interpolation) - Physicist's
    // ============================================================
    std::cout << "\n--- Group 10: from_derivatives (Physicist's) ---\n" << std::endl;

    // Test 34: Simple cubic from_derivatives (Physicist's)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};  // f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1

        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 34a: Physicist Hermite f(0)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 34b: Physicist Hermite f(1)");
        test.expect_near(hp(0, 1), 1.0, tolerance, "Test 34c: Physicist Hermite f'(0)");
        test.expect_near(hp(1, 1), -1.0, tolerance, "Test 34d: Physicist Hermite f'(1)");
    }

    // Test 35: from_derivatives with second derivatives (Physicist's)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, -1, 0}};

        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 35a: Physicist Quintic f(0)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 35b: Physicist Quintic f(1)");
        test.expect_near(hp(0, 1), 1.0, tolerance, "Test 35c: Physicist Quintic f'(0)");
        test.expect_near(hp(1, 1), -1.0, tolerance, "Test 35d: Physicist Quintic f'(1)");
        test.expect_near(hp(0, 2), 0.0, tolerance, "Test 35e: Physicist Quintic f''(0)");
        test.expect_near(hp(1, 2), 0.0, tolerance, "Test 35f: Physicist Quintic f''(1)");
    }

    // Test 36: from_derivatives multi-interval (Physicist's)
    {
        std::vector<double> xi = {0, 1, 2};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, 0}, {0, -1}};

        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 36a: Physicist multi-interval f(0)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 36b: Physicist multi-interval f(1)");
        test.expect_near(hp(2), 0.0, tolerance, "Test 36c: Physicist multi-interval f(2)");
    }

    // ============================================================
    // Group 11: from_derivatives (Hermite Interpolation) - Probabilist's
    // ============================================================
    std::cout << "\n--- Group 11: from_derivatives (Probabilist's) ---\n" << std::endl;

    // Test 37: Simple cubic from_derivatives (Probabilist's)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};

        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Probabilist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 37a: Probabilist Hermite f(0)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 37b: Probabilist Hermite f(1)");
        test.expect_near(hp(0, 1), 1.0, tolerance, "Test 37c: Probabilist Hermite f'(0)");
        test.expect_near(hp(1, 1), -1.0, tolerance, "Test 37d: Probabilist Hermite f'(1)");
    }

    // Test 38: from_derivatives with second derivatives (Probabilist's)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, -1, 0}};

        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Probabilist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 38a: Probabilist Quintic f(0)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 38b: Probabilist Quintic f(1)");
        test.expect_near(hp(0, 1), 1.0, tolerance, "Test 38c: Probabilist Quintic f'(0)");
        test.expect_near(hp(1, 1), -1.0, tolerance, "Test 38d: Probabilist Quintic f'(1)");
        test.expect_near(hp(0, 2), 0.0, tolerance, "Test 38e: Probabilist Quintic f''(0)");
        test.expect_near(hp(1, 2), 0.0, tolerance, "Test 38f: Probabilist Quintic f''(1)");
    }

    // ============================================================
    // Group 12: Basis Conversions
    // ============================================================
    std::cout << "\n--- Group 12: Basis Conversions ---\n" << std::endl;

    // Test 39: from_power_basis (constant) - Physicist's
    {
        std::vector<std::vector<double>> power_coeffs = {{3}};  // p(x) = 3
        std::vector<double> breaks = {0, 1};
        HPoly hp = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0.5), 3.0, tolerance, "Test 39: Physicist from_power_basis constant");
    }

    // Test 40: from_power_basis (linear) - Physicist's
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}};  // p(x) = 1 + 2x on [0,1]
        std::vector<double> breaks = {0, 1};
        HPoly hp = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0), 1.0, tolerance, "Test 40a: Physicist from_power_basis linear f(0)");
        test.expect_near(hp(0.5), 2.0, tolerance, "Test 40b: Physicist from_power_basis linear f(0.5)");
        test.expect_near(hp(1), 3.0, tolerance, "Test 40c: Physicist from_power_basis linear f(1)");
    }

    // Test 41: from_power_basis (quadratic) - Physicist's
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {0}, {1}};  // p(x) = 1 + x^2 on [0,1]
        std::vector<double> breaks = {0, 1};
        HPoly hp = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0), 1.0, tolerance, "Test 41a: Physicist from_power_basis quadratic f(0)");
        test.expect_near(hp(0.5), 1.25, tolerance, "Test 41b: Physicist from_power_basis quadratic f(0.5)");
        test.expect_near(hp(1), 2.0, tolerance, "Test 41c: Physicist from_power_basis quadratic f(1)");
    }

    // Test 42: from_power_basis (constant) - Probabilist's
    {
        std::vector<std::vector<double>> power_coeffs = {{3}};
        std::vector<double> breaks = {0, 1};
        HPoly hp = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Probabilist);

        test.expect_near(hp(0.5), 3.0, tolerance, "Test 42: Probabilist from_power_basis constant");
    }

    // Test 43: from_power_basis (linear) - Probabilist's
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}};  // p(x) = 1 + 2x
        std::vector<double> breaks = {0, 1};
        HPoly hp = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Probabilist);

        test.expect_near(hp(0), 1.0, tolerance, "Test 43a: Probabilist from_power_basis linear f(0)");
        test.expect_near(hp(0.5), 2.0, tolerance, "Test 43b: Probabilist from_power_basis linear f(0.5)");
        test.expect_near(hp(1), 3.0, tolerance, "Test 43c: Probabilist from_power_basis linear f(1)");
    }

    // Test 44: to_power_basis round-trip - Physicist's
    {
        std::vector<std::vector<double>> power_coeffs_in = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        std::vector<double> breaks = {0, 1};
        HPoly hp = HPoly::from_power_basis(power_coeffs_in, breaks, HermiteKind::Physicist);

        auto power_coeffs_out = hp.to_power_basis();
        test.expect_near(power_coeffs_out[0][0], 1.0, tolerance, "Test 44a: Physicist round-trip c[0]");
        test.expect_near(power_coeffs_out[1][0], 2.0, tolerance, "Test 44b: Physicist round-trip c[1]");
        test.expect_near(power_coeffs_out[2][0], 3.0, tolerance, "Test 44c: Physicist round-trip c[2]");
    }

    // Test 45: to_power_basis round-trip - Probabilist's
    {
        std::vector<std::vector<double>> power_coeffs_in = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        std::vector<double> breaks = {0, 1};
        HPoly hp = HPoly::from_power_basis(power_coeffs_in, breaks, HermiteKind::Probabilist);

        auto power_coeffs_out = hp.to_power_basis();
        test.expect_near(power_coeffs_out[0][0], 1.0, tolerance, "Test 45a: Probabilist round-trip c[0]");
        test.expect_near(power_coeffs_out[1][0], 2.0, tolerance, "Test 45b: Probabilist round-trip c[1]");
        test.expect_near(power_coeffs_out[2][0], 3.0, tolerance, "Test 45c: Probabilist round-trip c[2]");
    }

    // ============================================================
    // Group 13: Root Finding
    // ============================================================
    std::cout << "\n--- Group 13: Root Finding ---\n" << std::endl;

    // Test 46: Simple root - Physicist's
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x on [0,1]
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        auto roots = hp.roots();
        test.expect_eq(roots.size(), 1ul, "Test 46a: Physicist single root count");
        test.expect_near(roots[0], 0.0, tolerance, "Test 46b: Physicist root at 0");
    }

    // Test 47: Multiple roots via from_derivatives
    {
        std::vector<double> xi = {0, 0.5, 1};
        std::vector<std::vector<double>> yi = {{-1}, {0.5}, {-0.5}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {0});

        auto roots = hp.roots();
        test.expect_true(roots.size() >= 1, "Test 47: Has at least one root");
    }

    // Test 48: Root in extrapolation region
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{1, -2}, {-1, -2}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        auto roots = hp.roots(true, true);
        test.expect_true(!roots.empty(), "Test 48: Found roots including extrapolation");
    }

    // ============================================================
    // Group 14: Extend
    // ============================================================
    std::cout << "\n--- Group 14: Extend ---\n" << std::endl;

    // Test 49: Extend right - Physicist's
    {
        HPoly hp1({{1}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp2 = hp1.extend({{2}}, {1, 2}, true);

        test.expect_eq(static_cast<size_t>(hp2.num_intervals()), 2ul, "Test 49a: Extended intervals");
        test.expect_near(hp2(0.5), 1.0, tolerance, "Test 49b: Extended f(0.5)");
        test.expect_near(hp2(1.5), 2.0, tolerance, "Test 49c: Extended f(1.5)");
    }

    // Test 50: Extend left - Physicist's
    {
        HPoly hp1({{2}}, {1, 2}, HermiteKind::Physicist);
        HPoly hp2 = hp1.extend({{1}}, {0, 1}, false);

        test.expect_eq(static_cast<size_t>(hp2.num_intervals()), 2ul, "Test 50a: Extended left intervals");
        test.expect_near(hp2(0.5), 1.0, tolerance, "Test 50b: Extended left f(0.5)");
        test.expect_near(hp2(1.5), 2.0, tolerance, "Test 50c: Extended left f(1.5)");
    }

    // ============================================================
    // Group 15: Thread Safety
    // ============================================================
    std::cout << "\n--- Group 15: Thread Safety ---\n" << std::endl;

    // Test 51: Concurrent evaluation - Physicist's
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        std::atomic<int> errors(0);
        const int num_threads = 4;
        const int evals_per_thread = 1000;

        auto thread_func = [&hp, &errors, evals_per_thread]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);

            for (int i = 0; i < evals_per_thread; ++i) {
                double x = dis(gen);
                double val = hp(x);
                double deriv = hp(x, 1);
                (void)val;
                (void)deriv;

                // Basic sanity checks
                if (std::isnan(val) || std::isnan(deriv)) {
                    errors++;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(thread_func);
        }
        for (auto& t : threads) {
            t.join();
        }

        test.expect_eq(static_cast<size_t>(errors.load()), 0ul, "Test 51: Physicist thread safety no errors");
    }

    // Test 52: Concurrent evaluation - Probabilist's
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};
        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Probabilist);

        std::atomic<int> errors(0);
        const int num_threads = 4;
        const int evals_per_thread = 1000;

        auto thread_func = [&hp, &errors, evals_per_thread]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);

            for (int i = 0; i < evals_per_thread; ++i) {
                double x = dis(gen);
                double val = hp(x);
                double deriv = hp(x, 1);
                (void)val;
                (void)deriv;

                if (std::isnan(val) || std::isnan(deriv)) {
                    errors++;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(thread_func);
        }
        for (auto& t : threads) {
            t.join();
        }

        test.expect_eq(static_cast<size_t>(errors.load()), 0ul, "Test 52: Probabilist thread safety no errors");
    }

    // ============================================================
    // Group 16: Edge Cases and Special Values
    // ============================================================
    std::cout << "\n--- Group 16: Edge Cases and Special Values ---\n" << std::endl;

    // Test 53: NaN input
    {
        HPoly hp({{1}}, {0, 1}, HermiteKind::Physicist);
        test.expect_true(std::isnan(hp(std::numeric_limits<double>::quiet_NaN())),
                        "Test 53: NaN input returns NaN");
    }

    // Test 54: Evaluation at exact breakpoints
    {
        std::vector<std::vector<double>> coeffs = {{1, 2, 3}};
        std::vector<double> breaks = {0, 1, 2, 3};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0), 1.0, tolerance, "Test 54a: Value at breakpoint 0");
        test.expect_near(hp(1), 2.0, tolerance, "Test 54b: Value at breakpoint 1");
        test.expect_near(hp(2), 3.0, tolerance, "Test 54c: Value at breakpoint 2");
        test.expect_near(hp(3), 3.0, tolerance, "Test 54d: Value at breakpoint 3");
    }

    // Test 55: High-order derivative beyond degree
    {
        HPoly hp({{1}, {1}}, {0, 1}, HermiteKind::Physicist);  // degree 1
        HPoly d3hp = hp.derivative(3);  // 3rd derivative of linear = 0

        test.expect_near(d3hp(0.5), 0.0, tolerance, "Test 55: High-order derivative is zero");
    }

    // Test 56: Descending breakpoints
    {
        std::vector<std::vector<double>> coeffs = {{1, 2}};
        std::vector<double> breaks = {2, 1, 0};  // Descending
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(1.5), 1.0, tolerance, "Test 56a: Descending f(1.5)");
        test.expect_near(hp(0.5), 2.0, tolerance, "Test 56b: Descending f(0.5)");
    }

    // ============================================================
    // Group 17: Properties and Accessors
    // ============================================================
    std::cout << "\n--- Group 17: Properties and Accessors ---\n" << std::endl;

    // Test 57: degree()
    {
        std::vector<std::vector<double>> coeffs = {{1}, {2}, {3}};  // degree 2
        HPoly hp(coeffs, {0, 1}, HermiteKind::Physicist);
        test.expect_eq(static_cast<size_t>(hp.degree()), 2ul, "Test 57: degree()");
    }

    // Test 58: num_intervals()
    {
        HPoly hp({{1, 2, 3}}, {0, 1, 2, 3}, HermiteKind::Physicist);
        test.expect_eq(static_cast<size_t>(hp.num_intervals()), 3ul, "Test 58: num_intervals()");
    }

    // Test 59: c() and x() accessors
    {
        std::vector<std::vector<double>> coeffs = {{1}, {2}};
        std::vector<double> breaks = {0, 1};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        auto c = hp.c();
        auto x = hp.x();

        test.expect_eq(c.size(), 2ul, "Test 59a: c() size");
        test.expect_eq(x.size(), 2ul, "Test 59b: x() size");
        test.expect_near(c[0][0], 1.0, tolerance, "Test 59c: c()[0][0]");
        test.expect_near(c[1][0], 2.0, tolerance, "Test 59d: c()[1][0]");
        test.expect_near(x[0], 0.0, tolerance, "Test 59e: x()[0]");
        test.expect_near(x[1], 1.0, tolerance, "Test 59f: x()[1]");
    }

    // Test 60: is_ascending()
    {
        HPoly hp_asc({{1}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp_desc({{1}}, {1, 0}, HermiteKind::Physicist);

        test.expect_true(hp_asc.is_ascending(), "Test 60a: is_ascending() true");
        test.expect_true(!hp_desc.is_ascending(), "Test 60b: is_ascending() false");
    }

    // Test 61: extrapolate() accessor
    {
        HPoly hp_ext({{1}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Extrapolate);
        HPoly hp_no({{1}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::NoExtrapolate);
        HPoly hp_per({{1}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Periodic);

        test.expect_true(hp_ext.extrapolate() == ExtrapolateMode::Extrapolate, "Test 61a: extrapolate mode");
        test.expect_true(hp_no.extrapolate() == ExtrapolateMode::NoExtrapolate, "Test 61b: no extrapolate mode");
        test.expect_true(hp_per.extrapolate() == ExtrapolateMode::Periodic, "Test 61c: periodic mode");
    }

    // ============================================================
    // Group 18: Compare Physicist's vs Probabilist's
    // ============================================================
    std::cout << "\n--- Group 18: Compare Physicist's vs Probabilist's ---\n" << std::endl;

    // Test 62: Both should give same result for same function via from_derivatives
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0}, {1, -1, 0}};

        HPoly hp_phys = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);
        HPoly hp_prob = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Probabilist);

        // Both should give same values since they match the same constraints
        test.expect_near(hp_phys(0.25), hp_prob(0.25), tolerance, "Test 62a: Same function at 0.25");
        test.expect_near(hp_phys(0.5), hp_prob(0.5), tolerance, "Test 62b: Same function at 0.5");
        test.expect_near(hp_phys(0.75), hp_prob(0.75), tolerance, "Test 62c: Same function at 0.75");
    }

    // Test 63: Both should give same result via from_power_basis
    {
        std::vector<std::vector<double>> power = {{1}, {2}, {3}};  // 1 + 2x + 3x^2

        HPoly hp_phys = HPoly::from_power_basis(power, {0, 1}, HermiteKind::Physicist);
        HPoly hp_prob = HPoly::from_power_basis(power, {0, 1}, HermiteKind::Probabilist);

        test.expect_near(hp_phys(0.3), hp_prob(0.3), tolerance, "Test 63a: Same power basis at 0.3");
        test.expect_near(hp_phys(0.7), hp_prob(0.7), tolerance, "Test 63b: Same power basis at 0.7");
    }

    // ============================================================
    // Group 19: Higher Order Polynomials
    // ============================================================
    std::cout << "\n--- Group 19: Higher Order Polynomials ---\n" << std::endl;

    // Test 64: Quintic interpolation - Physicist's
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0, 0}, {1, 0, 0, 0}};

        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 64a: Physicist higher order f(0)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 64b: Physicist higher order f(1)");
        test.expect_near(hp(0, 1), 1.0, tolerance, "Test 64c: Physicist higher order f'(0)");
        test.expect_near(hp(1, 1), 0.0, tolerance, "Test 64d: Physicist higher order f'(1)");
    }

    // Test 65: Quintic interpolation - Probabilist's
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 0, 0}, {1, 0, 0, 0}};

        HPoly hp = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Probabilist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 65a: Probabilist higher order f(0)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 65b: Probabilist higher order f(1)");
        test.expect_near(hp(0, 1), 1.0, tolerance, "Test 65c: Probabilist higher order f'(0)");
        test.expect_near(hp(1, 1), 0.0, tolerance, "Test 65d: Probabilist higher order f'(1)");
    }

    // ============================================================
    // Group 20: Integration with Periodic Boundaries
    // ============================================================
    std::cout << "\n--- Group 20: Integration with Periodic Boundaries ---\n" << std::endl;

    // Test 66: Periodic integration - Physicist's
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x on [0,1]
        HPoly hp(coeffs, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Periodic);

        double one_period = hp.integrate(0, 1);
        double two_periods = hp.integrate(0, 2);

        test.expect_near(two_periods, 2 * one_period, tolerance, "Test 66: Periodic integration");
    }

    // ============================================================
    // Group 21: Extrapolation Orders
    // ============================================================
    std::cout << "\n--- Group 21: Extrapolation Orders ---\n" << std::endl;

    // Test 67: Extrapolation with constant order (0)
    {
        std::vector<std::vector<double>> coeffs = {{0.5}, {0.25}};  // p(x) = x
        HPoly hp(coeffs, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Extrapolate, 0, 0);

        // Constant extrapolation: f(-0.5) = f(0) = 0
        test.expect_near(hp(-0.5), hp(0), tolerance, "Test 67a: Constant left extrapolation");
        // f(1.5) = f(1) = 1
        test.expect_near(hp(1.5), hp(1), tolerance, "Test 67b: Constant right extrapolation");
    }

    // Test 68: Extrapolation with linear order (1)
    {
        std::vector<std::vector<double>> coeffs = {{1.0/3.0}, {0.5}, {1.0/6.0}};  // approx x^2
        HPoly hp(coeffs, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Extrapolate, 1, 1);

        // Linear extrapolation uses tangent line at boundary
        double val_at_boundary = hp(1);
        double deriv_at_boundary = hp(1, 1);
        double expected_extrap = val_at_boundary + deriv_at_boundary * 0.5;

        test.expect_near(hp(1.5), expected_extrap, tolerance, "Test 68: Linear extrapolation");
    }

    // ============================================================
    // Group 22: Copy Construction
    // ============================================================
    std::cout << "\n--- Group 22: Copy Construction ---\n" << std::endl;

    // Test 69: Copy constructor
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};
        HPoly hp1 = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);
        HPoly hp2(hp1);  // Copy

        test.expect_near(hp2(0.5), hp1(0.5), tolerance, "Test 69a: Copy gives same values");
        test.expect_near(hp2(0.5, 1), hp1(0.5, 1), tolerance, "Test 69b: Copy gives same derivatives");
    }

    // Test 70: Move constructor
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};
        HPoly hp1 = HPoly::from_derivatives(xi, yi, {}, HermiteKind::Physicist);
        double val_before = hp1(0.5);

        HPoly hp2(std::move(hp1));  // Move

        test.expect_near(hp2(0.5), val_before, tolerance, "Test 70: Move preserves values");
    }

    // ============================================================
    // Group 23: from_array2d factory
    // ============================================================
    std::cout << "\n--- Group 23: from_array2d factory ---\n" << std::endl;

    // Test 71: from_array2d basic usage
    {
        std::vector<std::vector<double>> vv = {{0.5}, {0.25}};
        ndarray::array2d<double> arr(vv);
        HPoly hp = HPoly::from_array2d(std::move(arr), {0, 1}, HermiteKind::Physicist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 71a: from_array2d f(0)");
        test.expect_near(hp(0.5), 0.5, tolerance, "Test 71b: from_array2d f(0.5)");
        test.expect_near(hp(1), 1.0, tolerance, "Test 71c: from_array2d f(1)");
    }

    // ============================================================
    // Group 24: Independent Verification (manual math checks)
    // ============================================================
    std::cout << "\n--- Group 24: Independent Verification (manual math checks) ---\n" << std::endl;

    // Test 72: Verify derivative with finite differences for p(x) = x^3 via from_power_basis
    {
        // Physicist's
        std::vector<std::vector<double>> power_coeffs = {{0}, {0}, {0}, {1}};
        std::vector<double> breaks = {0, 1};
        HPoly hp_phys = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Physicist);

        double x = 0.5;
        double analytical_phys = hp_phys(x, 1);  // Should be 3*x^2 = 0.75
        double numerical_phys = finite_diff_derivative(hp_phys, x);

        test.expect_near(analytical_phys, 0.75, tolerance, "Test 72a: Physicist analytical derivative of x^3 at 0.5");
        test.expect_near(numerical_phys, 0.75, 1e-5, "Test 72b: Physicist numerical derivative of x^3 at 0.5");

        // Probabilist's
        HPoly hp_prob = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Probabilist);

        double analytical_prob = hp_prob(x, 1);
        double numerical_prob = finite_diff_derivative(hp_prob, x);

        test.expect_near(analytical_prob, 0.75, tolerance, "Test 72c: Probabilist analytical derivative of x^3 at 0.5");
        test.expect_near(numerical_prob, 0.75, 1e-5, "Test 72d: Probabilist numerical derivative of x^3 at 0.5");
    }

    // Test 73: Verify antiderivative with numerical integration
    {
        // Physicist's
        std::vector<std::vector<double>> power_coeffs = {{0}, {1}};  // p(x) = x on [0,1]
        HPoly hp_phys = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Physicist);
        HPoly anti_phys = hp_phys.antiderivative();

        double analytical_phys = anti_phys(0.5) - anti_phys(0);  // Should be 0.125
        double numerical_phys = numerical_integrate(hp_phys, 0, 0.5);

        test.expect_near(analytical_phys, 0.125, tolerance, "Test 73a: Physicist analytical antiderivative");
        test.expect_near(numerical_phys, 0.125, 1e-6, "Test 73b: Physicist numerical antiderivative");

        // Probabilist's
        HPoly hp_prob = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Probabilist);
        HPoly anti_prob = hp_prob.antiderivative();

        double analytical_prob = anti_prob(0.5) - anti_prob(0);
        double numerical_prob = numerical_integrate(hp_prob, 0, 0.5);

        test.expect_near(analytical_prob, 0.125, tolerance, "Test 73c: Probabilist analytical antiderivative");
        test.expect_near(numerical_prob, 0.125, 1e-6, "Test 73d: Probabilist numerical antiderivative");
    }

    // ============================================================
    // Group 25: Property-Based Tests (derivative-antiderivative roundtrip)
    // ============================================================
    std::cout << "\n--- Group 25: Property-Based Tests ---\n" << std::endl;

    // Test 74: Integral additivity
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {-1}};  // 1 + 2x - x^2
        HPoly hp = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Physicist);

        double int_0_1 = hp.integrate(0, 1);
        double int_0_half = hp.integrate(0, 0.5);
        double int_half_1 = hp.integrate(0.5, 1);

        test.expect_near(int_0_1, int_0_half + int_half_1, tolerance,
                        "Test 74: Integral additivity");
    }

    // Test 75: Derivative-integral relationship (d/dx[antiderivative] = original)
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        HPoly hp = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Physicist);

        HPoly antideriv = hp.antiderivative();
        HPoly recovered = antideriv.derivative();

        bool all_close = true;
        for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            if (std::abs(hp(x) - recovered(x)) > tolerance) {
                all_close = false;
                break;
            }
        }
        test.expect_true(all_close, "Test 75: d/dx[antiderivative] = original");
    }

    // Test 76: Integral reversal
    {
        HPoly hp = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Physicist);

        double int_ab = hp.integrate(0.2, 0.8);
        double int_ba = hp.integrate(0.8, 0.2);

        test.expect_near(int_ab, -int_ba, tolerance, "Test 76: Integral reversal");
    }

    // ============================================================
    // Group 26: High-Degree Polynomials (degree 10+)
    // ============================================================
    std::cout << "\n--- Group 26: High-Degree Polynomials ---\n" << std::endl;

    // Test 77: High-degree polynomial (degree 10) via from_power_basis
    {
        // Physicist's
        std::vector<std::vector<double>> power_coeffs(11, std::vector<double>(1, 0.0));
        power_coeffs[10][0] = 1.0;  // x^10
        std::vector<double> breaks = {0, 1};

        HPoly hp_phys = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp_phys(0), 0.0, tolerance, "Test 77a: Physicist x^10 at 0");
        test.expect_near(hp_phys(1), 1.0, tolerance, "Test 77b: Physicist x^10 at 1");
        test.expect_near(hp_phys(0.5), std::pow(0.5, 10), tolerance, "Test 77c: Physicist x^10 at 0.5");

        // Probabilist's
        HPoly hp_prob = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Probabilist);

        test.expect_near(hp_prob(0), 0.0, tolerance, "Test 77d: Probabilist x^10 at 0");
        test.expect_near(hp_prob(1), 1.0, tolerance, "Test 77e: Probabilist x^10 at 1");
        test.expect_near(hp_prob(0.5), std::pow(0.5, 10), tolerance, "Test 77f: Probabilist x^10 at 0.5");
    }

    // ============================================================
    // Group 27: NaN and Infinity Handling
    // ============================================================
    std::cout << "\n--- Group 27: NaN and Infinity Handling ---\n" << std::endl;

    // Test 78: NaN input returns NaN
    {
        HPoly hp_phys({{1}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp_prob({{1}}, {0, 1}, HermiteKind::Probabilist);

        test.expect_true(std::isnan(hp_phys(std::numeric_limits<double>::quiet_NaN())),
                        "Test 78a: Physicist NaN input gives NaN output");
        test.expect_true(std::isnan(hp_prob(std::numeric_limits<double>::quiet_NaN())),
                        "Test 78b: Probabilist NaN input gives NaN output");
    }

    // Test 79: Infinity input with NoExtrapolate gives NaN
    {
        HPoly hp_phys({{1}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::NoExtrapolate);
        HPoly hp_prob({{1}}, {0, 1}, HermiteKind::Probabilist, ExtrapolateMode::NoExtrapolate);

        test.expect_true(std::isnan(hp_phys(std::numeric_limits<double>::infinity())),
                        "Test 79a: Physicist Inf input with NoExtrapolate gives NaN");
        test.expect_true(std::isnan(hp_prob(std::numeric_limits<double>::infinity())),
                        "Test 79b: Probabilist Inf input with NoExtrapolate gives NaN");
    }

    // ============================================================
    // Group 28: Orders Parameter in from_derivatives
    // ============================================================
    std::cout << "\n--- Group 28: Orders Parameter ---\n" << std::endl;

    // Test 80: from_derivatives with limited orders
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1, 5}, {1, -1, 3}};  // More derivatives than needed
        std::vector<int> orders = {0};  // Only use function values

        // Physicist's
        HPoly hp_phys = HPoly::from_derivatives(xi, yi, orders, HermiteKind::Physicist);

        test.expect_near(hp_phys(0), 0.0, tolerance, "Test 80a: Physicist limited orders f(0)");
        test.expect_near(hp_phys(1), 1.0, tolerance, "Test 80b: Physicist limited orders f(1)");
        test.expect_eq(static_cast<size_t>(hp_phys.degree()), 1ul, "Test 80c: Physicist limited orders degree");

        // Probabilist's
        HPoly hp_prob = HPoly::from_derivatives(xi, yi, orders, HermiteKind::Probabilist);

        test.expect_near(hp_prob(0), 0.0, tolerance, "Test 80d: Probabilist limited orders f(0)");
        test.expect_near(hp_prob(1), 1.0, tolerance, "Test 80e: Probabilist limited orders f(1)");
        test.expect_eq(static_cast<size_t>(hp_prob.degree()), 1ul, "Test 80f: Probabilist limited orders degree");
    }

    // ============================================================
    // Group 29: Move Semantics
    // ============================================================
    std::cout << "\n--- Group 29: Move Semantics ---\n" << std::endl;

    // Test 81: Move constructor (check moved-from is valid/empty state)
    {
        // Physicist's
        std::vector<std::vector<double>> coeffs = {{5}};
        std::vector<double> breaks = {0, 1};
        HPoly hp1(coeffs, breaks, HermiteKind::Physicist);
        HPoly hp2(std::move(hp1));

        test.expect_near(hp2(0.5), 5.0, tolerance, "Test 81a: Physicist move constructor preserves data");
        test.expect_true(hp1.coefficients().empty(), "Test 81b: Physicist moved-from coefficients empty");
        test.expect_true(hp1.breakpoints().empty(), "Test 81c: Physicist moved-from breakpoints empty");

        // Probabilist's
        HPoly hp3(coeffs, breaks, HermiteKind::Probabilist);
        HPoly hp4(std::move(hp3));

        test.expect_near(hp4(0.5), 5.0, tolerance, "Test 81d: Probabilist move constructor preserves data");
        test.expect_true(hp3.coefficients().empty(), "Test 81e: Probabilist moved-from coefficients empty");
        test.expect_true(hp3.breakpoints().empty(), "Test 81f: Probabilist moved-from breakpoints empty");
    }

    // ============================================================
    // Group 30: Controlled Extrapolation (Taylor order)
    // ============================================================
    std::cout << "\n--- Group 30: Controlled Extrapolation ---\n" << std::endl;

    // Test 82: Constant extrapolation (order 0)
    {
        // Physicist's: p(x) = x on [0,1] with constant extrapolation
        HPoly hp_phys({{0.5}, {0.25}}, {0, 1}, HermiteKind::Physicist,
                      ExtrapolateMode::Extrapolate, 0, 0);

        test.expect_near(hp_phys(-0.5), 0.0, tolerance, "Test 82a: Physicist constant extrapolation left");
        test.expect_near(hp_phys(1.5), 1.0, tolerance, "Test 82b: Physicist constant extrapolation right");

        // Probabilist's: p(x) = x on [0,1] with constant extrapolation
        HPoly hp_prob({{0.5}, {0.5}}, {0, 1}, HermiteKind::Probabilist,
                      ExtrapolateMode::Extrapolate, 0, 0);

        test.expect_near(hp_prob(-0.5), 0.0, tolerance, "Test 82c: Probabilist constant extrapolation left");
        test.expect_near(hp_prob(1.5), 1.0, tolerance, "Test 82d: Probabilist constant extrapolation right");
    }

    // Test 83: Linear extrapolation (order 1)
    {
        // p(x) = x^2 on [0,1] via from_power_basis
        HPoly hp_phys = HPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, HermiteKind::Physicist,
                                                 ExtrapolateMode::Extrapolate);
        // Rebuild with linear extrapolation order
        HPoly hp_lin(hp_phys.c(), {0, 1}, HermiteKind::Physicist,
                     ExtrapolateMode::Extrapolate, 1, 1);

        // At x=0: f(0)=0, f'(0)=0, so linear extrapolation left is 0
        test.expect_near(hp_lin(-0.5), 0.0, tolerance, "Test 83a: Linear extrapolation left");
        // At x=1: f(1)=1, f'(1)=2, so linear extrapolation is 1 + 2*(x-1)
        test.expect_near(hp_lin(1.5), 2.0, tolerance, "Test 83b: Linear extrapolation right");
    }

    // ============================================================
    // Group 31: Numpy Reference Verification
    // ============================================================
    // Values verified against numpy.polynomial.hermite / hermite_e
    std::cout << "\n--- Group 31: Numpy Reference Verification ---\n" << std::endl;

    // Test 84: Constant verification - Both variants
    // numpy: H.hermval(s, [5.0]) = 5.0 for all s
    // numpy: HE.hermeval(s, [5.0]) = 5.0 for all s
    {
        HPoly hp_phys({{5.0}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp_prob({{5.0}}, {0, 1}, HermiteKind::Probabilist);

        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            test.expect_near(hp_phys(x), 5.0, tolerance,
                "Test 84a: Physicist constant at " + std::to_string(x));
            test.expect_near(hp_prob(x), 5.0, tolerance,
                "Test 84b: Probabilist constant at " + std::to_string(x));
        }
    }

    // Test 85: Linear verification via from_power_basis
    // p(x) = x on [0,1], verified values: [0.0, 0.25, 0.5, 0.75, 1.0]
    {
        HPoly hp_phys = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp_prob = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Probabilist);

        std::vector<double> xs = {0.0, 0.25, 0.5, 0.75, 1.0};
        for (double x : xs) {
            test.expect_near(hp_phys(x), x, tolerance,
                "Test 85a: Physicist linear at " + std::to_string(x));
            test.expect_near(hp_prob(x), x, tolerance,
                "Test 85b: Probabilist linear at " + std::to_string(x));
        }
    }

    // Test 86: Quadratic verification via from_power_basis
    // p(x) = x^2 on [0,1], values: [0.0, 0.0625, 0.25, 0.5625, 1.0]
    {
        HPoly hp_phys = HPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp_prob = HPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, HermiteKind::Probabilist);

        test.expect_near(hp_phys(0.0), 0.0, tolerance, "Test 86a: Physicist x^2 at 0");
        test.expect_near(hp_phys(0.25), 0.0625, tolerance, "Test 86b: Physicist x^2 at 0.25");
        test.expect_near(hp_phys(0.5), 0.25, tolerance, "Test 86c: Physicist x^2 at 0.5");
        test.expect_near(hp_phys(0.75), 0.5625, tolerance, "Test 86d: Physicist x^2 at 0.75");
        test.expect_near(hp_phys(1.0), 1.0, tolerance, "Test 86e: Physicist x^2 at 1.0");

        test.expect_near(hp_prob(0.0), 0.0, tolerance, "Test 86f: Probabilist x^2 at 0");
        test.expect_near(hp_prob(0.25), 0.0625, tolerance, "Test 86g: Probabilist x^2 at 0.25");
        test.expect_near(hp_prob(0.5), 0.25, tolerance, "Test 86h: Probabilist x^2 at 0.5");
        test.expect_near(hp_prob(0.75), 0.5625, tolerance, "Test 86i: Probabilist x^2 at 0.75");
        test.expect_near(hp_prob(1.0), 1.0, tolerance, "Test 86j: Probabilist x^2 at 1.0");
    }

    // ============================================================
    // Group 32: Comprehensive Coefficient Tests
    // ============================================================
    std::cout << "\n--- Group 32: Comprehensive Coefficient Tests ---\n" << std::endl;

    // Test 87: from_power_basis coefficient verification
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        HPoly hp_phys = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Physicist);
        HPoly hp_prob = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Probabilist);

        // Verify evaluation matches expected power basis values
        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            double expected = 1 + 2*x + 3*x*x;
            test.expect_near(hp_phys(x), expected, tolerance,
                "Test 87a: Physicist from_power eval at " + std::to_string(x));
            test.expect_near(hp_prob(x), expected, tolerance,
                "Test 87b: Probabilist from_power eval at " + std::to_string(x));
        }
    }

    // Test 88: Derivative coefficient verification
    {
        // p(x) = x^2 on [0,1], derivative should be 2x
        HPoly hp_phys = HPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, HermiteKind::Physicist);
        HPoly dhp_phys = hp_phys.derivative();

        test.expect_near(dhp_phys(0.0), 0.0, tolerance, "Test 88a: Physicist deriv eval at 0");
        test.expect_near(dhp_phys(0.5), 1.0, tolerance, "Test 88b: Physicist deriv eval at 0.5");
        test.expect_near(dhp_phys(1.0), 2.0, tolerance, "Test 88c: Physicist deriv eval at 1");

        HPoly hp_prob = HPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, HermiteKind::Probabilist);
        HPoly dhp_prob = hp_prob.derivative();

        test.expect_near(dhp_prob(0.0), 0.0, tolerance, "Test 88d: Probabilist deriv eval at 0");
        test.expect_near(dhp_prob(0.5), 1.0, tolerance, "Test 88e: Probabilist deriv eval at 0.5");
        test.expect_near(dhp_prob(1.0), 2.0, tolerance, "Test 88f: Probabilist deriv eval at 1");
    }

    // Test 89: Round-trip coefficient verification (power->hermite->power)
    {
        std::vector<std::vector<double>> power_coeffs = {{1}, {-2}, {3}, {-4}};

        HPoly hp_phys = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Physicist);
        auto recovered_phys = hp_phys.to_power_basis();

        test.expect_near(recovered_phys[0][0], 1.0, tolerance, "Test 89a: Physicist round-trip c0");
        test.expect_near(recovered_phys[1][0], -2.0, tolerance, "Test 89b: Physicist round-trip c1");
        test.expect_near(recovered_phys[2][0], 3.0, tolerance, "Test 89c: Physicist round-trip c2");
        test.expect_near(recovered_phys[3][0], -4.0, tolerance, "Test 89d: Physicist round-trip c3");

        HPoly hp_prob = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Probabilist);
        auto recovered_prob = hp_prob.to_power_basis();

        test.expect_near(recovered_prob[0][0], 1.0, tolerance, "Test 89e: Probabilist round-trip c0");
        test.expect_near(recovered_prob[1][0], -2.0, tolerance, "Test 89f: Probabilist round-trip c1");
        test.expect_near(recovered_prob[2][0], 3.0, tolerance, "Test 89g: Probabilist round-trip c2");
        test.expect_near(recovered_prob[3][0], -4.0, tolerance, "Test 89h: Probabilist round-trip c3");
    }

    // ============================================================
    // Group 33: Extended Root Finding Tests
    // ============================================================
    std::cout << "\n--- Group 33: Extended Root Finding Tests ---\n" << std::endl;

    // Test 90: Roots with discontinuity=false
    {
        // Two intervals with different constants, discontinuity at x=1
        std::vector<std::vector<double>> coeffs = {{-1, 1}};  // -1 on [0,1], +1 on [1,2]
        std::vector<double> breaks = {0, 1, 2};
        HPoly hp(coeffs, breaks, HermiteKind::Physicist);

        auto r_with = hp.roots(true, false);    // Include discontinuity
        auto r_without = hp.roots(false, false); // Exclude discontinuity

        test.expect_eq(r_with.size(), 1ul, "Test 90a: Roots with discontinuity finds boundary root");
        test.expect_eq(r_without.size(), 0ul, "Test 90b: Roots without discontinuity excludes boundary");
    }

    // Test 91: Roots with extrapolate=false
    {
        // p(x) = x - 2 on [0,1]: root at x=2, outside domain
        HPoly hp = HPoly::from_power_basis({{-2}, {1}}, {0, 1}, HermiteKind::Physicist);

        auto r_extrap = hp.roots(true, true);     // With extrapolation
        auto r_no_extrap = hp.roots(true, false);  // No extrapolation

        test.expect_eq(r_extrap.size(), 1ul, "Test 91a: Roots with extrapolation finds root");
        if (r_extrap.size() >= 1) {
            test.expect_near(r_extrap[0], 2.0, 1e-10, "Test 91b: Extrapolated root value");
        }
        test.expect_eq(r_no_extrap.size(), 0ul, "Test 91c: Roots without extrapolation excludes");
    }

    // Test 92: Root at exact breakpoint
    {
        // p(x) = x on [0,1] has root at x=0 (breakpoint)
        HPoly hp = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Physicist);

        auto r = hp.roots(true, false);
        test.expect_eq(r.size(), 1ul, "Test 92a: Root at breakpoint count");
        if (!r.empty()) {
            test.expect_near(r[0], 0.0, 1e-10, "Test 92b: Root at left breakpoint");
        }
    }

    // Test 93: Multi-interval with multiple roots
    {
        // Two linear functions crossing zero in each interval
        // Interval [0,1]: p(x) = x - 0.5 (root at 0.5)
        // Interval [1,2]: p(x) = x - 1.5 (root at 1.5)
        // Using from_power_basis per interval: [{-0.5, -0.5}, {1, 1}]
        HPoly hp = HPoly::from_power_basis({{-0.5, -0.5}, {1, 1}}, {0, 1, 2}, HermiteKind::Physicist);

        auto r = hp.roots(true, false);
        test.expect_true(r.size() >= 2, "Test 93a: Multi-interval root count >= 2");
        if (r.size() >= 2) {
            std::sort(r.begin(), r.end());
            test.expect_near(r[0], 0.5, 1e-10, "Test 93b: First interval root");
            // Depending on discontinuity handling, there may be a boundary root at 1.0
            test.expect_near(r.back(), 1.5, 1e-10, "Test 93c: Second interval root");
        }
    }

    // ============================================================
    // Group 34: Periodic Mode Calculus Tests
    // ============================================================
    std::cout << "\n--- Group 34: Periodic Mode Calculus Tests ---\n" << std::endl;

    // Test 94: Periodic mode derivative
    {
        // p(x) = x on [0,1], Physicist's
        HPoly hp({{0.5}, {0.25}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Periodic);
        HPoly dhp = hp.derivative();

        // Derivative of x is 1, should be 1 everywhere (periodic)
        test.expect_near(dhp(0.5), 1.0, tolerance, "Test 94a: Periodic derivative at 0.5");
        test.expect_near(dhp(1.5), 1.0, tolerance, "Test 94b: Periodic derivative at 1.5 (wrapped)");
        test.expect_near(dhp(2.5), 1.0, tolerance, "Test 94c: Periodic derivative at 2.5 (wrapped)");
    }

    // Test 95: Periodic mode antiderivative
    {
        // p(x) = 2 (constant), Physicist's
        HPoly hp({{2}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Periodic);
        HPoly ihp = hp.antiderivative();

        // Antiderivative of 2 is 2x, with periodic wrapping
        test.expect_near(ihp(0), 0.0, tolerance, "Test 95a: Periodic antiderivative at 0");
        test.expect_near(ihp(0.5), 1.0, tolerance, "Test 95b: Periodic antiderivative at 0.5");
    }

    // Test 96: Periodic mode integration across boundary
    {
        // p(x) = 1 (constant), Physicist's
        HPoly hp({{1}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Periodic);

        // Integral of 1 over any interval of length L should be L
        double int_0_1 = hp.integrate(0, 1);
        double int_0_half = hp.integrate(0, 0.5);

        test.expect_near(int_0_1, 1.0, tolerance, "Test 96a: Periodic integral [0,1]");
        test.expect_near(int_0_half, 0.5, tolerance, "Test 96b: Periodic integral [0,0.5]");

        // Cross boundary: f(x) = x on [0,1] periodic
        HPoly hp2({{0.5}, {0.25}}, {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Periodic);
        double int_across = hp2.integrate(0.5, 1.5);
        // From 0.5 to 1.0: integral of x = 0.375; From 0.0 to 0.5 (wrapped): integral of x = 0.125
        test.expect_near(int_across, 0.5, tolerance, "Test 96c: Periodic integral [0.5,1.5] crossing boundary");
    }

    // ============================================================
    // Group 35: Corner Cases (Extreme Values)
    // ============================================================
    std::cout << "\n--- Group 35: Corner Cases (Extreme Values) ---\n" << std::endl;

    // Test 97: High-degree polynomial (degree 20)
    {
        std::vector<std::vector<double>> power_coeffs(21, std::vector<double>(1, 0.0));
        power_coeffs[20][0] = 1.0;  // x^20

        HPoly hp = HPoly::from_power_basis(power_coeffs, {0, 1}, HermiteKind::Physicist);

        test.expect_near(hp(0), 0.0, tolerance, "Test 97a: x^20 at 0");
        test.expect_near(hp(1), 1.0, tolerance, "Test 97b: x^20 at 1");
        test.expect_near(hp(0.5), std::pow(0.5, 20), tolerance, "Test 97c: x^20 at 0.5");
    }

    // Test 98: Large coefficients
    {
        HPoly hp({{1e10}, {2e10}}, {0, 1}, HermiteKind::Physicist);

        // At x=0: s=-1, Physicist H_0=1, H_1(-1)=-2
        // p = 1e10 * 1 + 2e10 * (-2) = 1e10 - 4e10 = -3e10
        // At x=1: s=1, H_0=1, H_1(1)=2
        // p = 1e10 * 1 + 2e10 * 2 = 1e10 + 4e10 = 5e10
        test.expect_near(hp(0.0), -3e10, 1e5, "Test 98a: Physicist large coeffs at 0");
        test.expect_near(hp(1.0), 5e10, 1e5, "Test 98b: Physicist large coeffs at 1");
    }

    // Test 99: Small coefficients
    {
        HPoly hp({{1e-15}, {2e-15}}, {0, 1}, HermiteKind::Physicist);

        test.expect_near(hp(0.0), 1e-15 - 4e-15, 1e-25, "Test 99a: Physicist small coeffs at 0");
        test.expect_near(hp(1.0), 1e-15 + 4e-15, 1e-25, "Test 99b: Physicist small coeffs at 1");
    }

    // Test 100: Very small interval
    {
        double h = 1e-10;
        // p(x) = x on [0, h], use from_power_basis for accuracy
        HPoly hp = HPoly::from_power_basis({{0}, {1}}, {0, h}, HermiteKind::Physicist);

        double mid = h / 2;
        test.expect_near(hp(mid), mid, 1e-20, "Test 100: Small interval evaluation");
    }

    // Test 101: Far extrapolation
    {
        HPoly hp = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Physicist,
                                            ExtrapolateMode::Extrapolate);

        test.expect_near(hp(100.0), 100.0, tolerance, "Test 101a: Far extrapolation right");
        test.expect_near(hp(-100.0), -100.0, tolerance, "Test 101b: Far extrapolation left");
    }

    // ============================================================
    // Group 36: Extended Independent Verification
    // ============================================================
    std::cout << "\n--- Group 36: Extended Independent Verification ---\n" << std::endl;

    // Test 102: Multi-point finite difference derivative verification
    {
        HPoly hp = HPoly::from_power_basis({{1}, {-2}, {3}, {-1}}, {0, 1}, HermiteKind::Physicist);

        bool all_pass = true;
        for (double x : {0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9}) {
            double analytical = hp(x, 1);
            double numerical = finite_diff_derivative(hp, x);
            if (std::abs(analytical - numerical) > 1e-5) {
                test.fail("Test 102: finite diff derivative at x=" + std::to_string(x));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 102: Multi-point derivative matches finite differences");
        }
    }

    // Test 103: Multi-point second derivative verification
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1}, HermiteKind::Physicist);

        bool all_pass = true;
        for (double x : {0.2, 0.4, 0.5, 0.6, 0.8}) {
            double analytical = hp(x, 2);
            double numerical = finite_diff_second_derivative(hp, x);
            if (std::abs(analytical - numerical) > 1e-4) {
                test.fail("Test 103: finite diff 2nd derivative at x=" + std::to_string(x));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 103: Multi-point 2nd derivative matches finite differences");
        }
    }

    // Test 104: Multi-interval numerical integration verification
    {
        HPoly hp = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Physicist);

        bool all_pass = true;
        std::vector<std::pair<double, double>> intervals = {{0, 0.5}, {0.25, 0.75}, {0, 1}};
        for (const auto& [a, b] : intervals) {
            double analytical = hp.integrate(a, b);
            double numerical = numerical_integrate(hp, a, b);
            double expected = (b*b - a*a) / 2.0;  // exact integral of x

            if (std::abs(analytical - expected) > tolerance) {
                test.fail("Test 104: analytical integral [" + std::to_string(a) + "," + std::to_string(b) + "]");
                all_pass = false;
            }
            if (std::abs(numerical - expected) > 1e-6) {
                test.fail("Test 104: numerical integral [" + std::to_string(a) + "," + std::to_string(b) + "]");
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("Test 104: Multi-interval integration verification");
        }
    }

    // ============================================================
    // Group 37: Periodic Derivative Evaluation
    // ============================================================
    std::cout << "\n--- Group 37: Periodic Derivative Evaluation ---\n" << std::endl;

    // Test 105: Periodic mode with derivative evaluation (f(x+period) = f(x) and f'(x+period) = f'(x))
    {
        HPoly hp = HPoly::from_power_basis({{0, 1}, {1, -0.5}}, {0, 1, 3},
                                            HermiteKind::Physicist, ExtrapolateMode::Periodic);

        // Period = 3 - 0 = 3
        test.expect_near(hp(3.5), hp(0.5), tolerance, "Test 105a: Periodic f(3.5) = f(0.5)");
        test.expect_near(hp(-0.5), hp(2.5), tolerance, "Test 105b: Periodic f(-0.5) = f(2.5)");
        test.expect_near(hp(3.5, 1), hp(0.5, 1), tolerance, "Test 105c: Periodic f'(3.5) = f'(0.5)");
        test.expect_near(hp(-0.5, 1), hp(2.5, 1), tolerance, "Test 105d: Periodic f'(-0.5) = f'(2.5)");
    }

    // ============================================================
    // Group 38: Corner Case Tests (multi-interval)
    // ============================================================
    std::cout << "\n--- Group 38: Corner Case Tests (multi-interval) ---\n" << std::endl;

    // Test 106: Two intervals with from_power_basis
    {
        // Interval [0, 0.5]: p(x) = x, Interval [0.5, 1]: p(x) = 0.5 (constant)
        HPoly hp = HPoly::from_power_basis({{0, 0.5}, {1, 0}}, {0, 0.5, 1}, HermiteKind::Physicist);
        test.expect_near(hp(0.0), 0.0, tolerance, "Test 106a: Two intervals at 0");
        test.expect_near(hp(0.25), 0.25, tolerance, "Test 106b: Two intervals at 0.25");
        test.expect_near(hp(0.5), 0.5, tolerance, "Test 106c: Two intervals at 0.5");
        test.expect_near(hp(0.75), 0.5, tolerance, "Test 106d: Two intervals at 0.75");
        test.expect_near(hp(1.0), 0.5, tolerance, "Test 106e: Two intervals at 1");
    }

    // Test 107: Many intervals (10 linear pieces)
    {
        std::vector<double> breaks(11);
        for (int i = 0; i <= 10; ++i) {
            breaks[i] = static_cast<double>(i);
        }
        // Build f(x) = x using from_power_basis for each interval
        // Power coeffs per interval: constant = a (left breakpoint), linear = 1
        std::vector<std::vector<double>> c0(1, std::vector<double>(10));
        std::vector<std::vector<double>> c1(1, std::vector<double>(10));
        std::vector<std::vector<double>> power_coeffs(2, std::vector<double>(10));
        for (int i = 0; i < 10; ++i) {
            power_coeffs[0][i] = breaks[i];  // constant term = left breakpoint
            power_coeffs[1][i] = 1.0;        // linear term = 1
        }
        HPoly hp = HPoly::from_power_basis(power_coeffs, breaks, HermiteKind::Physicist);

        test.expect_near(hp(0.5), 0.5, tolerance, "Test 107a: Many intervals at 0.5");
        test.expect_near(hp(5.5), 5.5, tolerance, "Test 107b: Many intervals at 5.5");
        test.expect_near(hp(9.5), 9.5, tolerance, "Test 107c: Many intervals at 9.5");
    }

    // Test 108: from_derivatives with values only
    {
        HPoly hp_phys = HPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}}, {}, HermiteKind::Physicist);
        test.expect_near(hp_phys(0.0), 0.0, tolerance, "Test 108a: Physicist values-only f(0)");
        test.expect_near(hp_phys(1.0), 1.0, tolerance, "Test 108b: Physicist values-only f(1)");
        test.expect_near(hp_phys(2.0), 0.0, tolerance, "Test 108c: Physicist values-only f(2)");
        test.expect_near(hp_phys(0.5), 0.5, tolerance, "Test 108d: Physicist values-only f(0.5)");

        HPoly hp_prob = HPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}}, {}, HermiteKind::Probabilist);
        test.expect_near(hp_prob(0.0), 0.0, tolerance, "Test 108e: Probabilist values-only f(0)");
        test.expect_near(hp_prob(1.0), 1.0, tolerance, "Test 108f: Probabilist values-only f(1)");
        test.expect_near(hp_prob(2.0), 0.0, tolerance, "Test 108g: Probabilist values-only f(2)");
    }

    // Test 109: from_derivatives with asymmetric orders
    {
        HPoly hp = HPoly::from_derivatives({0, 1}, {{0, 1, 0, 0}, {1}}, {}, HermiteKind::Physicist);
        test.expect_near(hp(0.0), 0.0, tolerance, "Test 109a: Asymmetric f(0)");
        test.expect_near(hp(1.0), 1.0, tolerance, "Test 109b: Asymmetric f(1)");
        double h = 1e-8;
        double deriv_at_0 = (hp(h) - hp(0)) / h;
        test.expect_near(deriv_at_0, 1.0, 1e-5, "Test 109c: Asymmetric f'(0) approx");
    }

    // ============================================================
    // Group 39: C0 Continuity at Breakpoints
    // ============================================================
    std::cout << "\n--- Group 39: C0 Continuity at Breakpoints ---\n" << std::endl;

    // Test 110: from_derivatives produces C0 continuous polynomial
    {
        HPoly hp = HPoly::from_derivatives({0, 1, 2, 3}, {{0, 1}, {1, 0}, {0.5, -0.5}, {0, 0}},
                                            {}, HermiteKind::Physicist);

        double eps = 1e-12;
        bool continuous = true;

        for (double bp : {1.0, 2.0}) {
            double left = hp(bp - eps);
            double right = hp(bp + eps);
            double at_bp = hp(bp);
            if (std::abs(left - at_bp) > 1e-8 || std::abs(right - at_bp) > 1e-8) {
                test.fail("Test 110: C0 continuity at x=" + std::to_string(bp));
                continuous = false;
            }
        }
        if (continuous) {
            test.pass("Test 110: from_derivatives produces C0 continuous polynomial");
        }
    }

    // Test 111: from_derivatives with matching derivatives produces C1 continuity
    {
        HPoly hp = HPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1, 1}, {3, 1}},
                                            {}, HermiteKind::Physicist);

        double eps = 1e-10;
        bool c1_continuous = true;

        double f_left = hp(1.0 - eps);
        double f_right = hp(1.0 + eps);
        double df_left = hp(1.0 - eps, 1);
        double df_right = hp(1.0 + eps, 1);

        if (std::abs(f_left - f_right) > 1e-8) {
            test.fail("Test 111a: C0 continuity at x=1");
            c1_continuous = false;
        }
        if (std::abs(df_left - df_right) > 1e-6) {
            test.fail("Test 111b: C1 continuity (derivative) at x=1");
            c1_continuous = false;
        }
        if (c1_continuous) {
            test.pass("Test 111: from_derivatives with matching f' produces C1 continuity");
        }
    }

    // ============================================================
    // Group 40: extend() with Mixed Degrees
    // ============================================================
    std::cout << "\n--- Group 40: extend() with Mixed Degrees ---\n" << std::endl;

    // Test 112: Extend cubic polynomial with linear polynomial
    {
        HPoly hp_cubic = HPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1}, HermiteKind::Physicist);

        // Linear on [1,2]: g(x) = 2*(x-1) + 1 = 2x - 1
        HPoly hp_linear = HPoly::from_power_basis({{1}, {2}}, {1, 2}, HermiteKind::Physicist);

        HPoly extended = hp_cubic.extend(hp_linear.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 112a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.125, tolerance, "Test 112b: Cubic part at 0.5");
        test.expect_near(extended(1.5), 2.0, tolerance, "Test 112c: Linear part at 1.5");
        test.expect_near(extended(1.0), 1.0, tolerance, "Test 112d: Continuity at boundary");
    }

    // Test 113: Extend quadratic with quintic
    {
        HPoly hp_quad = HPoly::from_power_basis({{0}, {0}, {1}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp_quint = HPoly::from_derivatives({1, 2}, {{1, 2, 2}, {4, 4, 2}},
                                                  {}, HermiteKind::Physicist);

        HPoly extended = hp_quad.extend(hp_quint.c(), {1, 2}, true);

        test.expect_eq(static_cast<size_t>(extended.num_intervals()), 2ul, "Test 113a: Extended has 2 intervals");
        test.expect_near(extended(0.5), 0.25, tolerance, "Test 113b: Quadratic part");
        test.expect_near(extended(1.5), hp_quint(1.5), tolerance, "Test 113c: Quintic part");
    }

    // ============================================================
    // Group 41: Edge Case Coverage
    // ============================================================
    std::cout << "\n--- Group 41: Edge Case Coverage ---\n" << std::endl;

    // Test 114: Empty vector evaluation
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}}, {0, 1}, HermiteKind::Physicist);
        std::vector<double> empty_input;
        std::vector<double> result = hp(empty_input);
        test.expect_eq(result.size(), 0ul, "Test 114: Empty vector evaluation returns empty");
    }

    // Test 115: Single-point vector evaluation
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}}, {0, 1}, HermiteKind::Physicist);
        std::vector<double> single_point = {0.5};
        std::vector<double> result = hp(single_point);
        test.expect_eq(result.size(), 1ul, "Test 115a: Single-point vector size");
        test.expect_near(result[0], 2.0, tolerance, "Test 115b: Single-point value correct");
    }

    // Test 116: Evaluation at all breakpoints
    {
        HPoly hp = HPoly::from_power_basis({{0, 1, 2}, {1, 1, 1}}, {0, 1, 2, 3},
                                            HermiteKind::Physicist);
        std::vector<double> breakpts = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> results = hp(breakpts);

        test.expect_near(results[0], 0.0, tolerance, "Test 116a: Eval at bp[0]");
        test.expect_near(results[1], 1.0, tolerance, "Test 116b: Eval at bp[1]");
        test.expect_near(results[2], 2.0, tolerance, "Test 116c: Eval at bp[2]");
        test.expect_near(results[3], 3.0, tolerance, "Test 116d: Eval at bp[3]");
    }

    // ============================================================
    // Group 42: Zero/Edge Polynomials
    // ============================================================
    std::cout << "\n--- Group 42: Zero/Edge Polynomials ---\n" << std::endl;

    // Test 117: Zero polynomial
    {
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {0}};
        HPoly hp(coeffs, {0, 1}, HermiteKind::Physicist);

        test.expect_near(hp(0.0), 0.0, tolerance, "Test 117a: Zero polynomial at 0");
        test.expect_near(hp(0.5), 0.0, tolerance, "Test 117b: Zero polynomial at 0.5");
        test.expect_near(hp(1.0), 0.0, tolerance, "Test 117c: Zero polynomial at 1");
        test.expect_near(hp(0.5, 1), 0.0, tolerance, "Test 117d: Zero polynomial derivative");
        test.expect_near(hp.integrate(0, 1), 0.0, tolerance, "Test 117e: Zero polynomial integral");
    }

    // Test 118: Repeated roots - (x-0.5)^2
    {
        HPoly hp = HPoly::from_power_basis({{0.25}, {-1}, {1}}, {0, 1}, HermiteKind::Physicist);

        auto roots = hp.roots();
        test.expect_true(roots.size() >= 1, "Test 118a: Repeated root found");
        if (!roots.empty()) {
            test.expect_near(roots[0], 0.5, 1e-6, "Test 118b: Repeated root at 0.5");
        }
        test.expect_near(hp(0.5), 0.0, tolerance, "Test 118c: f(0.5) = 0");
        test.expect_near(hp(0.5, 1), 0.0, tolerance, "Test 118d: f'(0.5) = 0");
    }

    // Test 119: Integration beyond domain
    {
        HPoly hp = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Physicist,
                                            ExtrapolateMode::Extrapolate);

        double integral = hp.integrate(-1, 2);
        test.expect_near(integral, 1.5, tolerance, "Test 119a: Integration beyond domain");
        test.expect_near(hp.integrate(-1, 0), -0.5, tolerance, "Test 119b: Integration in left extrapolation");
        test.expect_near(hp.integrate(1, 2), 1.5, tolerance, "Test 119c: Integration in right extrapolation");
    }

    // ============================================================
    // Group 43: Derivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 43: Derivative Edge Cases ---\n" << std::endl;

    // Test 120: Derivative of constant is 0
    {
        HPoly hp_phys({{5}}, {0, 1}, HermiteKind::Physicist);
        HPoly deriv_phys = hp_phys.derivative();
        test.expect_near(deriv_phys(0.5), 0.0, tolerance, "Test 120a: Physicist derivative of constant is 0");
        test.expect_eq(static_cast<size_t>(deriv_phys.degree()), 0ul, "Test 120b: Physicist derivative degree is 0");

        HPoly hp_prob({{5}}, {0, 1}, HermiteKind::Probabilist);
        HPoly deriv_prob = hp_prob.derivative();
        test.expect_near(deriv_prob(0.5), 0.0, tolerance, "Test 120c: Probabilist derivative of constant is 0");
        test.expect_eq(static_cast<size_t>(deriv_prob.degree()), 0ul, "Test 120d: Probabilist derivative degree is 0");
    }

    // Test 121: Over-differentiate (higher order than degree)
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}, {3}}, {0, 1}, HermiteKind::Physicist);
        HPoly d3 = hp.derivative(3);
        test.expect_near(d3(0.5), 0.0, tolerance, "Test 121a: 3rd derivative of quadratic is 0");

        HPoly d10 = hp.derivative(10);
        test.expect_near(d10(0.5), 0.0, tolerance, "Test 121b: 10th derivative of quadratic is 0");
    }

    // Test 122: Chained derivatives vs single derivative(n)
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1}, HermiteKind::Physicist);
        HPoly d3_chained = hp.derivative().derivative().derivative();
        HPoly d3_single = hp.derivative(3);

        test.expect_near(d3_chained(0.5), d3_single(0.5), tolerance,
                        "Test 122: Chained vs single derivative(3)");
    }

    // ============================================================
    // Group 44: Antiderivative Edge Cases
    // ============================================================
    std::cout << "\n--- Group 44: Antiderivative Edge Cases ---\n" << std::endl;

    // Test 123: Antiderivative of zero polynomial
    {
        HPoly hp({{0}, {0}}, {0, 1}, HermiteKind::Physicist);
        HPoly anti = hp.antiderivative();
        test.expect_near(anti(0.0), 0.0, tolerance, "Test 123a: Antiderivative of zero at 0");
        test.expect_near(anti(0.5), 0.0, tolerance, "Test 123b: Antiderivative of zero at 0.5");
        test.expect_near(anti(1.0), 0.0, tolerance, "Test 123c: Antiderivative of zero at 1");
    }

    // Test 124: Antiderivative(n).derivative(n) = original
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}, {3}}, {0, 1}, HermiteKind::Physicist);
        HPoly round_trip = hp.antiderivative(2).derivative(2);

        test.expect_near(hp(0.25), round_trip(0.25), tolerance, "Test 124a: antiderivative(2).derivative(2) at 0.25");
        test.expect_near(hp(0.5), round_trip(0.5), tolerance, "Test 124b: antiderivative(2).derivative(2) at 0.5");
        test.expect_near(hp(0.75), round_trip(0.75), tolerance, "Test 124c: antiderivative(2).derivative(2) at 0.75");
    }

    // Test 125: Chained antiderivatives vs single antiderivative(n)
    {
        HPoly hp = HPoly::from_power_basis({{2}}, {0, 1}, HermiteKind::Physicist);
        HPoly a2_chained = hp.antiderivative().antiderivative();
        HPoly a2_single = hp.antiderivative(2);

        test.expect_near(a2_chained(0.5), a2_single(0.5), tolerance,
                        "Test 125: Chained vs single antiderivative(2)");
    }

    // Test 126: Derivative/Antiderivative structural properties (degree, num_intervals, breakpoints)
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1}, HermiteKind::Physicist);
        HPoly deriv = hp.derivative();
        HPoly anti = hp.antiderivative();

        // Derivative reduces degree by 1
        test.expect_eq(static_cast<size_t>(hp.degree()), 3ul, "Test 126a: Original degree is 3");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 2ul, "Test 126b: Derivative degree is 2");

        // Antiderivative increases degree by 1
        test.expect_eq(static_cast<size_t>(anti.degree()), 4ul, "Test 126c: Antiderivative degree is 4");

        // Derivative preserves num_intervals
        HPoly hp_multi({{1, 2, 3}}, {0, 1, 2, 3}, HermiteKind::Physicist);
        HPoly deriv_multi = hp_multi.derivative();
        test.expect_eq(static_cast<size_t>(deriv_multi.num_intervals()), 3ul,
            "Test 126d: Derivative preserves num_intervals");

        // Derivative preserves breakpoints
        test.expect_near(deriv.breakpoints()[0], 0.0, tolerance, "Test 126e: Derivative preserves left breakpoint");
        test.expect_near(deriv.breakpoints()[1], 1.0, tolerance, "Test 126f: Derivative preserves right breakpoint");

        // Antiderivative starts at 0 at left boundary
        HPoly hp_const = HPoly::from_power_basis({{5}}, {0, 1}, HermiteKind::Physicist);
        HPoly anti_const = hp_const.antiderivative();
        test.expect_near(anti_const(0), 0.0, tolerance, "Test 126g: Antiderivative(left_boundary) = 0");
    }

    // ============================================================
    // Group 45: Integration Edge Cases
    // ============================================================
    std::cout << "\n--- Group 45: Integration Edge Cases ---\n" << std::endl;

    // Test 127: integrate(a, a) = 0
    {
        HPoly hp = HPoly::from_power_basis({{1}, {2}, {3}}, {0, 1}, HermiteKind::Physicist);
        test.expect_near(hp.integrate(0.5, 0.5), 0.0, tolerance, "Test 127: integrate(a, a) = 0");
    }

    // Test 128: Integration crossing multiple intervals
    {
        HPoly hp({{1, 2}}, {0, 1, 2}, HermiteKind::Physicist);
        test.expect_near(hp.integrate(0, 2), 3.0, tolerance, "Test 128: Integration across multiple intervals");
    }

    // Test 129: NoExtrapolate mode returns NaN beyond bounds
    {
        HPoly temp = HPoly::from_power_basis({{1}, {2}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp(temp.c(), {0, 1}, HermiteKind::Physicist, ExtrapolateMode::NoExtrapolate);
        double result = hp(-1);
        test.expect_true(std::isnan(result), "Test 129: NoExtrapolate evaluation beyond bounds returns NaN");
    }

    // ============================================================
    // Group 46: Root Finding Edge Cases
    // ============================================================
    std::cout << "\n--- Group 46: Root Finding Edge Cases ---\n" << std::endl;

    // Test 130: No roots (always positive polynomial)
    {
        HPoly hp = HPoly::from_power_basis({{1}, {0}, {1}}, {0, 1}, HermiteKind::Physicist);
        auto roots = hp.roots();
        test.expect_eq(roots.size(), 0ul, "Test 130: No roots for always-positive polynomial");
    }

    // Test 131: Root at domain boundary
    {
        HPoly hp = HPoly::from_power_basis({{0}, {1}}, {0, 1}, HermiteKind::Physicist);
        auto roots = hp.roots();
        test.expect_true(roots.size() >= 1, "Test 131a: Root at boundary found");
        if (!roots.empty()) {
            double min_root = *std::min_element(roots.begin(), roots.end());
            test.expect_near(min_root, 0.0, 1e-6, "Test 131b: Root at x=0");
        }
    }

    // Test 132: Root at internal breakpoint with sign change
    {
        // Two intervals: [-1, 0] and [0, 1], with values crossing zero at x=0
        // Interval [-1, 0]: p(x) = x (root at 0)
        // Interval [0, 1]: p(x) = x (root at 0)
        HPoly hp = HPoly::from_power_basis({{-1, 0}, {1, 1}}, {-1, 0, 1}, HermiteKind::Physicist);

        auto roots = hp.roots(true, false);
        test.expect_true(roots.size() >= 1, "Test 132a: Root at internal breakpoint found");
        if (!roots.empty()) {
            // Find root nearest to 0
            double closest = *std::min_element(roots.begin(), roots.end(),
                [](double a, double b) { return std::abs(a) < std::abs(b); });
            test.expect_near(closest, 0.0, 1e-6, "Test 132b: Root at internal breakpoint x=0");
        }
    }

    // ============================================================
    // Group 47: Extrapolation Order Edge Cases
    // ============================================================
    std::cout << "\n--- Group 47: Extrapolation Order Edge Cases ---\n" << std::endl;

    // Test 133: extrapolation_order > polynomial degree (matches full)
    {
        HPoly temp = HPoly::from_power_basis({{1}, {2}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp_full(temp.c(), {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Extrapolate, -1, -1);
        HPoly hp_high(temp.c(), {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Extrapolate, 10, 10);

        test.expect_near(hp_full(-0.5), hp_high(-0.5), tolerance, "Test 133a: High extrapolation order matches full left");
        test.expect_near(hp_full(1.5), hp_high(1.5), tolerance, "Test 133b: High extrapolation order matches full right");
    }

    // Test 134: extrapolation_order = 0 gives constant extrapolation
    {
        HPoly temp = HPoly::from_power_basis({{1}, {2}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp(temp.c(), {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Extrapolate, 0, 0);
        test.expect_near(hp(-0.5), 1.0, tolerance, "Test 134a: Order 0 extrapolation at left");
        test.expect_near(hp(1.5), 3.0, tolerance, "Test 134b: Order 0 extrapolation at right");
    }

    // Test 135: Asymmetric extrapolation orders
    {
        HPoly temp = HPoly::from_power_basis({{1}, {2}}, {0, 1}, HermiteKind::Physicist);
        HPoly hp(temp.c(), {0, 1}, HermiteKind::Physicist, ExtrapolateMode::Extrapolate, 0, -1);

        // Left: constant extrapolation, f(0) = 1
        test.expect_near(hp(-0.5), 1.0, tolerance, "Test 135a: Order 0 on left");
        // Right: full polynomial extrapolation, f(x) = 1 + 2x
        test.expect_near(hp(1.5), 1.0 + 2*1.5, tolerance, "Test 135b: Full order on right");

        // Verify accessor values
        test.expect_eq(static_cast<size_t>(hp.extrapolate_order_left()), 0ul, "Test 135c: extrapolate_order_left() = 0");
        test.expect_true(hp.extrapolate_order_right() == -1, "Test 135d: extrapolate_order_right() = -1");
    }

    // ============================================================
    // Summary
    // ============================================================
    test.summary();

    return test.all_passed() ? 0 : 1;
}
