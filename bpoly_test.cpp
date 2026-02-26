#include "include/bpoly.h"
#include "test_utils.h"
#include <cassert>
#include <limits>
#include <thread>
#include <atomic>
#include <random>

int main() {
    TestRunner test;
    const double tolerance = 1e-12;
    
    std::cout << "=== BPoly Manual Test Suite ===" << std::endl;
    
    // Test 1: Basic construction
    test.expect_no_throw([]() {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);
    }, "Basic construction");
    
    // Test 2: Linear polynomial evaluation
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};  // Linear: f(x) = x
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);
        
        test.expect_near(bp(0.0), 0.0, tolerance, "Linear f(0)");
        test.expect_near(bp(0.25), 0.25, tolerance, "Linear f(0.25)");
        test.expect_near(bp(0.5), 0.5, tolerance, "Linear f(0.5)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Linear f(1)");
    }
    
    // Test 3: Quadratic polynomial
    {
        // Quadratic: f(x) = x² on [0,1] using Bernstein basis
        // f(x) = x² = 0*B₀₂(t) + 0*B₁₂(t) + 1*B₂₂(t)
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);
        
        test.expect_near(bp(0.0), 0.0, tolerance, "Quadratic f(0)");
        test.expect_near(bp(0.5), 0.25, tolerance, "Quadratic f(0.5)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Quadratic f(1)");
    }
    
    // Test 4: Multiple intervals
    {
        // Two linear pieces: [0,1] and [1,2]
        std::vector<std::vector<double>> coeffs = {{0, 1}, {1, 2}};  // f₁(x) = x, f₂(x) = 1 + (x-1) = x
        std::vector<double> breaks = {0, 1, 2};
        BPoly bp(coeffs, breaks);
        
        test.expect_near(bp(0.5), 0.5, tolerance, "Multi-interval f(0.5)");
        test.expect_near(bp(1.0), 1.0, tolerance, "Multi-interval f(1.0)");
        test.expect_near(bp(1.5), 1.5, tolerance, "Multi-interval f(1.5)");
    }
    
    // Test 5: from_derivatives - simple Hermite cubic
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {{0, 1}, {1, -1}};  // f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        
        test.expect_no_throw([&]() {
            BPoly bp = BPoly::from_derivatives(xi, yi);
            
            // Verify endpoint conditions
            double f0 = bp(0);
            double f1 = bp(1);
            test.expect_near(f0, 0.0, tolerance, "Hermite f(0)");
            test.expect_near(f1, 1.0, tolerance, "Hermite f(1)");
            
            // Test derivative at endpoints (approximate)
            double h = 1e-8;
            double df0 = (bp(h) - bp(0)) / h;
            double df1 = (bp(1) - bp(1-h)) / h;
            test.expect_near(df0, 1.0, 1e-6, "Hermite f'(0) approx");
            test.expect_near(df1, -1.0, 1e-6, "Hermite f'(1) approx");
            
        }, "from_derivatives Hermite");
    }
    
    // Test 6: Error conditions
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs;  // Empty
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);
    }, "Empty coefficients error");
    
    test.expect_throw([]() {
        std::vector<std::vector<double>> coeffs = {{1}};
        std::vector<double> breaks = {0};  // Too few breakpoints
        BPoly bp(coeffs, breaks);
    }, "Too few breakpoints error");
    
    // Test 7: Vector evaluation
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);
        
        std::vector<double> x_vals = {0.0, 0.25, 0.5, 0.75, 1.0};
        std::vector<double> results = bp(x_vals);
        
        for (size_t i = 0; i < x_vals.size(); i++) {
            test.expect_near(results[i], x_vals[i], tolerance, 
                           "Vector evaluation [" + std::to_string(i) + "]");
        }
    }
    
    // Test 8: Breakpoint boundary evaluation (scipy bug-inspired)
    {
        std::vector<std::vector<double>> coeffs = {{1, 2, 3}, {4, 5, 6}};  // Linear
        std::vector<double> breaks = {0.0, 1.0, 2.0, 3.0};
        BPoly bp(coeffs, breaks);
        
        // Evaluate exactly at each breakpoint
        // For linear Bernstein polynomials: f(x) = c0*(1-t) + c1*t where t = (x-x0)/(x1-x0)
        test.expect_near(bp(0.0), 1.0, tolerance, "Breakpoint at 0.0");  // First interval start
        test.expect_near(bp(1.0), 2.0, tolerance, "Breakpoint at 1.0");  // Second interval start  
        test.expect_near(bp(2.0), 3.0, tolerance, "Breakpoint at 2.0");  // Third interval start
        test.expect_near(bp(3.0), 6.0, tolerance, "Breakpoint at 3.0");  // Third interval end
    }
    
    // Test 9: Repeated breakpoint handling (scipy validation issue)
    {
        std::vector<std::vector<double>> coeffs = {{1, 2}};
        std::vector<double> breaks = {0.0, 1e-10, 1.0};  // Very small interval
        
        test.expect_no_throw([&]() {
            BPoly bp(coeffs, breaks);
            
            // Should handle tiny intervals gracefully
            double val1 = bp(0.5e-10);  // Middle of tiny interval
            double val2 = bp(0.5);      // Middle of normal interval
            
            if (!std::isfinite(val1) || !std::isfinite(val2)) {
                throw std::runtime_error("Non-finite values in tiny interval test");
            }
        }, "Tiny interval handling");
        
        // Test actual repeated breakpoints (should throw)
        test.expect_throw([]() {
            BPoly bp({{1}}, {0.0, 0.0, 1.0});
        }, "Repeated breakpoints validation");
    }
    
    // Test 10: from_derivatives empty derivatives (validation bug)
    {
        std::vector<double> xi = {0, 1};
        std::vector<std::vector<double>> yi = {
            {},      // Empty derivative array
            {1}      // f(1) = 1
        };
        
        test.expect_throw([&]() {
            BPoly::from_derivatives(xi, yi);
        }, "Empty derivatives validation");
    }
    
    // Test 11: Extrapolation consistency (scipy boundary handling)
    {
        std::vector<std::vector<double>> coeffs = {{1, 3}, {2, -1}};  // Linear on two intervals
        std::vector<double> breaks = {0, 1, 2};
        BPoly bp(coeffs, breaks);

        // Interval 0: c=[1,2], p(t) = 1*(1-t) + 2*t = 1 + t
        // Interval 1: c=[3,-1], p(t) = 3*(1-t) + (-1)*t = 3 - 4t

        // With proper extrapolation (not clamping):
        // At x=-1: t=-1, p(-1) = 1 + (-1) = 0
        // At x=3: t=2, p(2) = 3 - 4*2 = -5

        // Test that extrapolation uses the polynomial formula (not clamping)
        test.expect_near(bp(-1.0), 0.0, tolerance, "Left extrapolation value");
        test.expect_near(bp(3.0), -5.0, tolerance, "Right extrapolation value");

        // Test that the function is continuous at boundaries
        test.expect_near(bp(0.0), 1.0, tolerance, "Left boundary");
        test.expect_near(bp(1.0), 3.0, tolerance, "Middle boundary");
        test.expect_near(bp(2.0), -1.0, tolerance, "Right boundary");
    }
    
    // Test 12: Coefficient matrix validation (scipy construction issue)
    {
        // Jagged coefficient matrix
        test.expect_throw([]() {
            BPoly bp({{1, 2}, {3}}, {0, 1, 2});  // Different sizes
        }, "Jagged coefficient matrix");
        
        // Wrong number of intervals
        test.expect_throw([]() {
            BPoly bp({{1, 2, 3}, {4, 5, 6}}, {0, 1, 2});  // 3 intervals, 2 breaks
        }, "Wrong interval count");
        
        // Empty coefficient row
        test.expect_throw([]() {
            BPoly bp({{/*empty*/}}, {0, 1});
        }, "Empty coefficient row");
    }
    
    // Test 13: NaN and infinity handling (scipy special values)
    {
        // Test NaN in coefficients
        test.expect_no_throw([]() {
            std::vector<std::vector<double>> coeffs_nan = {{1}, {std::numeric_limits<double>::quiet_NaN()}};
            BPoly bp(coeffs_nan, {0, 1});
            
            double result = bp(0.5);
            if (!std::isnan(result)) {
                throw std::runtime_error("Expected NaN result from NaN coefficient");
            }
        }, "NaN coefficient handling");
        
        // Test infinity in coefficients
        test.expect_no_throw([]() {
            std::vector<std::vector<double>> coeffs_inf = {{1}, {std::numeric_limits<double>::infinity()}};
            BPoly bp(coeffs_inf, {0, 1});
            
            double result = bp(0.5);
            if (!std::isinf(result)) {
                throw std::runtime_error("Expected infinite result from infinite coefficient");
            }
        }, "Infinity coefficient handling");
        
        // Test NaN input
        std::vector<std::vector<double>> normal_coeffs = {{1}, {2}};
        BPoly bp(normal_coeffs, {0, 1});
        
        test.expect_no_throw([&]() {
            double nan_eval = bp(std::numeric_limits<double>::quiet_NaN());
            if (!std::isnan(nan_eval)) {
                throw std::runtime_error("NaN input should return NaN");
            }
        }, "NaN input handling");
    }
    
    // Test 14: High-order derivative stability (scipy numerical issue)
    {
        // Create degree 5 polynomial with Bernstein coefficients [0, 1, 2, 3, 4, 5]
        // Derivative coeffs: n*(c[i+1] - c[i]) = 5*[1,1,1,1,1] = [5,5,5,5,5]
        // Since all derivative coeffs are equal, f'(x) = 5 (constant)
        // Second derivative and higher should all be 0
        std::vector<std::vector<double>> coeffs;
        for (int j = 0; j <= 5; ++j) {
            coeffs.push_back({static_cast<double>(j)});
        }
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);

        // First derivative should be 5 everywhere
        BPoly deriv1 = bp.derivative(1);
        test.expect_near(deriv1(0.0), 5.0, tolerance, "Degree 5 poly: f'(0) = 5");
        test.expect_near(deriv1(0.5), 5.0, tolerance, "Degree 5 poly: f'(0.5) = 5");
        test.expect_near(deriv1(1.0), 5.0, tolerance, "Degree 5 poly: f'(1) = 5");

        // Second and higher derivatives should be 0
        for (int order = 2; order <= 7; ++order) {
            BPoly deriv = bp.derivative(order);
            test.expect_near(deriv(0.5), 0.0, tolerance,
                           "Degree 5 poly: f^(" + std::to_string(order) + ")(0.5) = 0");
        }
    }
    
    // Test 15: Numerical precision at extremes (scipy precision issue)
    {
        // Linear Bernstein polynomial [c0, c1] evaluates to c0*(1-t) + c1*t
        // At t=0.5: (c0 + c1)/2 = 0.5*c0 + 0.5*c1
        std::vector<double> breaks = {0, 1};

        // Very small coefficients: [1e-100, 2e-100]
        // Expected at t=0.5: 0.5*1e-100 + 0.5*2e-100 = 1.5e-100
        std::vector<std::vector<double>> tiny_coeffs = {{1e-100}, {2e-100}};
        BPoly bp_tiny(tiny_coeffs, breaks);
        test.expect_near(bp_tiny(0.5), 1.5e-100, 1e-110, "Tiny coefficients: f(0.5) = 1.5e-100");
        test.expect_near(bp_tiny(0.0), 1e-100, 1e-110, "Tiny coefficients: f(0) = 1e-100");
        test.expect_near(bp_tiny(1.0), 2e-100, 1e-110, "Tiny coefficients: f(1) = 2e-100");

        // Very large coefficients: [1e100, 2e100]
        // Expected at t=0.5: 0.5*1e100 + 0.5*2e100 = 1.5e100
        std::vector<std::vector<double>> huge_coeffs = {{1e100}, {2e100}};
        BPoly bp_huge(huge_coeffs, breaks);
        test.expect_near(bp_huge(0.5), 1.5e100, 1e90, "Huge coefficients: f(0.5) = 1.5e100");
        test.expect_near(bp_huge(0.0), 1e100, 1e90, "Huge coefficients: f(0) = 1e100");
        test.expect_near(bp_huge(1.0), 2e100, 1e90, "Huge coefficients: f(1) = 2e100");
    }

    // Test 16: ExtrapolateMode::NoExtrapolate
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks, ExtrapolateMode::NoExtrapolate);

        // Inside domain should work
        test.expect_near(bp(0.5), 0.5, tolerance, "NoExtrapolate inside domain");

        // Outside domain should return NaN
        test.expect_no_throw([&]() {
            double left = bp(-0.5);
            if (!std::isnan(left)) {
                throw std::runtime_error("NoExtrapolate: left OOB should be NaN");
            }
        }, "NoExtrapolate left OOB returns NaN");

        test.expect_no_throw([&]() {
            double right = bp(1.5);
            if (!std::isnan(right)) {
                throw std::runtime_error("NoExtrapolate: right OOB should be NaN");
            }
        }, "NoExtrapolate right OOB returns NaN");
    }

    // Test 17: ExtrapolateMode::Periodic
    {
        // Linear from 0 to 2 on [0, 1]
        std::vector<std::vector<double>> coeffs = {{0}, {2}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks, ExtrapolateMode::Periodic);

        // Inside domain should work normally
        test.expect_near(bp(0.0), 0.0, tolerance, "Periodic at x=0");
        test.expect_near(bp(0.5), 1.0, tolerance, "Periodic at x=0.5");

        // Periodic wrapping: x=1.5 wraps to x=0.5
        test.expect_near(bp(1.5), 1.0, tolerance, "Periodic wrap right (1.5 -> 0.5)");

        // Periodic wrapping: x=-0.5 wraps to x=0.5
        test.expect_near(bp(-0.5), 1.0, tolerance, "Periodic wrap left (-0.5 -> 0.5)");

        // Multiple periods
        test.expect_near(bp(2.5), 1.0, tolerance, "Periodic wrap 2 periods right");
        test.expect_near(bp(-1.5), 1.0, tolerance, "Periodic wrap 2 periods left");
    }

    // Test 18: extend() method - extend right
    {
        // Original: linear from 0 to 1 on [0, 1]
        std::vector<std::vector<double>> coeffs1 = {{0}, {1}};
        std::vector<double> breaks1 = {0, 1};
        BPoly bp1(coeffs1, breaks1);

        // Extension: linear from 1 to 0 on [1, 2]
        std::vector<std::vector<double>> coeffs2 = {{1}, {0}};
        std::vector<double> breaks2 = {1, 2};

        BPoly extended = bp1.extend(coeffs2, breaks2, true);

        test.expect_near(extended(0.5), 0.5, tolerance, "Extended f(0.5)");
        test.expect_near(extended(1.0), 1.0, tolerance, "Extended f(1.0)");
        test.expect_near(extended(1.5), 0.5, tolerance, "Extended f(1.5)");
        test.expect_near(extended(2.0), 0.0, tolerance, "Extended f(2.0)");

        // Check breakpoints
        test.expect_eq(extended.num_intervals(), 2, "extend() created correct number of intervals");
    }

    // Test 19: extend() method - extend left
    {
        // Original: linear from 0 to 1 on [1, 2]
        std::vector<std::vector<double>> coeffs1 = {{0}, {1}};
        std::vector<double> breaks1 = {1, 2};
        BPoly bp1(coeffs1, breaks1);

        // Extension: linear from 1 to 0 on [0, 1]
        std::vector<std::vector<double>> coeffs2 = {{1}, {0}};
        std::vector<double> breaks2 = {0, 1};

        BPoly extended = bp1.extend(coeffs2, breaks2, false);

        test.expect_near(extended(0.5), 0.5, tolerance, "Extended left f(0.5)");
        test.expect_near(extended(1.0), 0.0, tolerance, "Extended left f(1.0)");
        test.expect_near(extended(1.5), 0.5, tolerance, "Extended left f(1.5)");
    }

    // Test 20: extend() validation
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);

        // Mismatched breakpoint
        test.expect_throw([&]() {
            std::vector<std::vector<double>> bad_coeffs = {{1}, {0}};
            std::vector<double> bad_breaks = {2, 3};  // Should start at 1
            bp.extend(bad_coeffs, bad_breaks, true);
        }, "extend() mismatched breakpoint");
    }

    // Test 21: Property aliases (scipy compatibility)
    {
        std::vector<std::vector<double>> coeffs = {{1, 2}, {3, 4}};
        std::vector<double> breaks = {0, 0.5, 1};
        BPoly bp(coeffs, breaks);

        // Test c() alias for coefficients()
        // Note: both c() and coefficients() return copies now (not references)
        auto c_alias = bp.c();
        auto coeffs_ref = bp.coefficients();
        test.expect_true(c_alias == coeffs_ref, "c() alias for coefficients()");

        // Test x() alias for breakpoints()
        const auto& x_alias = bp.x();
        const auto& breaks_ref = bp.breakpoints();
        test.expect_true(&x_alias == &breaks_ref, "x() alias for breakpoints()");
    }

    // Test 22: extrapolate() getter
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};

        BPoly bp_default(coeffs, breaks);
        test.expect_true(bp_default.extrapolate() == ExtrapolateMode::Extrapolate,
                        "extrapolate() getter default mode");

        BPoly bp_no_extrap(coeffs, breaks, ExtrapolateMode::NoExtrapolate);
        test.expect_true(bp_no_extrap.extrapolate() == ExtrapolateMode::NoExtrapolate,
                        "extrapolate() getter NoExtrapolate mode");
    }

    // ============================================================
    // scipy-inspired tests from scipy/interpolate/tests/test_interpolate.py
    // ============================================================

    // Test 23: scipy test_simple (degree 0)
    {
        std::vector<std::vector<double>> c = {{3}};
        std::vector<double> x = {0, 1};
        BPoly bp(c, x);
        test.expect_near(bp(0.1), 3.0, tolerance, "scipy test_simple: constant");
    }

    // Test 24: scipy test_simple2 (degree 1)
    {
        std::vector<std::vector<double>> c = {{3}, {1}};
        std::vector<double> x = {0, 1};
        BPoly bp(c, x);
        // At t=0.1: 3*(1-0.1) + 1*0.1 = 3*0.9 + 0.1 = 2.8
        test.expect_near(bp(0.1), 2.8, tolerance, "scipy test_simple2: linear");
    }

    // Test 25: scipy test_simple3 (degree 2)
    {
        std::vector<std::vector<double>> c = {{3}, {1}, {4}};
        std::vector<double> x = {0, 1};
        BPoly bp(c, x);
        // At t=0.2: 3*C(2,0)*0.2^0*0.8^2 + 1*C(2,1)*0.2^1*0.8^1 + 4*C(2,2)*0.2^2*0.8^0
        //         = 3*1*1*0.64 + 1*2*0.2*0.8 + 4*1*0.04*1
        //         = 1.92 + 0.32 + 0.16 = 2.4
        test.expect_near(bp(0.2), 2.4, tolerance, "scipy test_simple3: quadratic");
    }

    // Test 26: scipy test_simple4 (degree 3)
    {
        std::vector<std::vector<double>> c = {{1}, {1}, {1}, {2}};
        std::vector<double> x = {0, 1};
        BPoly bp(c, x);
        // At t=0.3: sum(c[j]*C(3,j)*t^j*(1-t)^(3-j))
        // = 1*1*0.7^3 + 1*3*0.3*0.7^2 + 1*3*0.09*0.7 + 2*1*0.027
        // = 0.343 + 0.441 + 0.189 + 0.054 = 1.027
        double expected = 0.343 + 0.441 + 0.189 + 0.054;
        test.expect_near(bp(0.3), expected, tolerance, "scipy test_simple4: cubic");
    }

    // Test 27: scipy test_simple5 (degree 4)
    {
        std::vector<std::vector<double>> c = {{1}, {1}, {8}, {2}, {1}};
        std::vector<double> x = {0, 1};
        BPoly bp(c, x);
        // At t=0.3:
        // 1*C(4,0)*0.7^4 + 1*C(4,1)*0.3*0.7^3 + 8*C(4,2)*0.09*0.49 +
        // 2*C(4,3)*0.027*0.7 + 1*C(4,4)*0.0081
        double expected = 1*0.2401 + 4*0.1029 + 8*6*0.0441 + 2*4*0.0189 + 1*0.0081;
        test.expect_near(bp(0.3), expected, tolerance, "scipy test_simple5: quartic");
    }

    // Test 28: scipy test_interval_length (non-unit interval)
    {
        std::vector<std::vector<double>> c = {{3}, {1}, {4}};
        std::vector<double> x = {0, 2};
        BPoly bp(c, x);
        // At x=0.1, t=0.05: 3*0.95^2 + 1*2*0.05*0.95 + 4*0.05^2
        double s = 0.05;
        double expected = 3*std::pow(1-s, 2) + 1*2*s*(1-s) + 4*s*s;
        test.expect_near(bp(0.1), expected, tolerance, "scipy test_interval_length");
    }

    // Test 29: scipy test_two_intervals
    {
        std::vector<std::vector<double>> c = {{3, 0}, {0, 0}, {0, 2}};
        std::vector<double> x = {0, 1, 3};
        BPoly bp(c, x);
        // At x=0.4, interval 0, t=0.4: 3*(1-0.4)^2 = 3*0.36 = 1.08
        test.expect_near(bp(0.4), 3*0.36, tolerance, "scipy test_two_intervals first");
        // At x=1.7, interval 1, t=(1.7-1)/2=0.35: 2*(0.35)^2 = 0.245
        test.expect_near(bp(1.7), 2*0.35*0.35, tolerance, "scipy test_two_intervals second");
    }

    // Test 30: scipy test_derivative (TestBPolyCalculus)
    {
        std::vector<std::vector<double>> c = {{3, 0}, {0, 0}, {0, 2}};
        std::vector<double> x = {0, 1, 3};
        BPoly bp(c, x);
        BPoly bp_der = bp.derivative();

        // At x=0.4: d/dx[3*(1-t)^2] = d/dt[3*(1-t)^2] / h = -6*(1-t) / 1 at t=0.4
        // = -6*0.6 = -3.6
        test.expect_near(bp_der(0.4), -6*0.6, tolerance, "scipy derivative first interval");

        // At x=1.7: d/dx[2*t^2] = d/dt[2*t^2] / h = 4*t / 2 at t=0.35
        // = 4*0.35/2 = 0.7
        test.expect_near(bp_der(1.7), 0.7, tolerance, "scipy derivative second interval");
    }

    // Test 31: scipy from_derivatives test_make_poly_1
    {
        BPoly bp = BPoly::from_derivatives({0, 1}, {{2}, {3}});
        auto& c = bp.coefficients();
        test.expect_near(c[0][0], 2.0, tolerance, "scipy make_poly_1 c[0]");
        test.expect_near(c[1][0], 3.0, tolerance, "scipy make_poly_1 c[1]");
    }

    // Test 32: scipy from_derivatives test_make_poly_2a
    {
        // ya=[1,0], yb=[1] -> linear polynomial matching f(0)=1, f'(0)=0, f(1)=1
        BPoly bp = BPoly::from_derivatives({0, 1}, {{1, 0}, {1}});
        auto& c = bp.coefficients();
        // Expected: c=[1.0, 1.0, 1.0] (degree 2)
        test.expect_near(c[0][0], 1.0, tolerance, "scipy make_poly_2a c[0]");
        test.expect_near(c[1][0], 1.0, tolerance, "scipy make_poly_2a c[1]");
        test.expect_near(c[2][0], 1.0, tolerance, "scipy make_poly_2a c[2]");
    }

    // Test 33: scipy from_derivatives test_make_poly_2b
    {
        // ya=[2,3], yb=[1] -> f(0)=2, f'(0)=3, f(1)=1
        BPoly bp = BPoly::from_derivatives({0, 1}, {{2, 3}, {1}});
        auto& c = bp.coefficients();
        // Expected: c=[2.0, 3.5, 1.0] (degree 2)
        test.expect_near(c[0][0], 2.0, tolerance, "scipy make_poly_2b c[0]");
        test.expect_near(c[1][0], 3.5, tolerance, "scipy make_poly_2b c[1]");
        test.expect_near(c[2][0], 1.0, tolerance, "scipy make_poly_2b c[2]");
    }

    // Test 34: scipy from_derivatives test_make_poly_3
    {
        // ya=[1,2,3], yb=[4] -> f(0)=1, f'(0)=2, f''(0)=3, f(1)=4
        BPoly bp = BPoly::from_derivatives({0, 1}, {{1, 2, 3}, {4}});
        auto& c = bp.coefficients();
        // Degree 3, expected: c=[1.0, 5/3, 17/6, 4.0]
        test.expect_near(c[0][0], 1.0, tolerance, "scipy make_poly_3 c[0]");
        test.expect_near(c[1][0], 5.0/3.0, tolerance, "scipy make_poly_3 c[1]");
        test.expect_near(c[2][0], 17.0/6.0, tolerance, "scipy make_poly_3 c[2]");
        test.expect_near(c[3][0], 4.0, tolerance, "scipy make_poly_3 c[3]");
    }

    // Test 35: Quintic from_derivatives with second derivatives
    {
        // f(0)=0, f'(0)=1, f''(0)=0, f(1)=1, f'(1)=1, f''(1)=0
        // This should be close to x (linear) but with zero curvature at endpoints
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1, 0}, {1, 1, 0}});
        test.expect_near(bp(0.0), 0.0, tolerance, "Quintic endpoints left");
        test.expect_near(bp(1.0), 1.0, tolerance, "Quintic endpoints right");
        test.expect_near(bp(0.5), 0.5, tolerance, "Quintic midpoint");
    }

    // Test 36: derivative() and antiderivative() preserve extrapolate mode
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}, {0}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks, ExtrapolateMode::NoExtrapolate);

        BPoly bp_deriv = bp.derivative();
        test.expect_true(bp_deriv.extrapolate() == ExtrapolateMode::NoExtrapolate,
                        "derivative() preserves extrapolate mode");

        BPoly bp_antideriv = bp.antiderivative();
        test.expect_true(bp_antideriv.extrapolate() == ExtrapolateMode::NoExtrapolate,
                        "antiderivative() preserves extrapolate mode");

        // Test that NoExtrapolate derivative returns NaN outside domain
        test.expect_no_throw([&]() {
            double val = bp_deriv(-0.5);
            if (!std::isnan(val)) {
                throw std::runtime_error("derivative with NoExtrapolate should return NaN OOB");
            }
        }, "derivative NoExtrapolate OOB returns NaN");
    }

    // Test 37: extend() validates sorted breakpoints
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);

        test.expect_throw([&]() {
            // Unsorted new breakpoints
            bp.extend({{1}, {0}}, {1, 0.5}, true);  // 0.5 < 1, not increasing
        }, "extend() rejects unsorted breakpoints");
    }

    // Test 38: Extrapolation uses polynomial formula, not clamping
    {
        // Linear polynomial p(t) = t on [0, 1]
        // Bernstein coefficients: c0 = 0, c1 = 1
        // For extrapolation at x = 1.5, t = 1.5, should give:
        // p(1.5) = 0*(1-1.5) + 1*1.5 = 1.5 (not clamped to 1)
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks, ExtrapolateMode::Extrapolate);

        double val_at_1_5 = bp(1.5);
        test.expect_near(val_at_1_5, 1.5, tolerance, "extrapolation at x=1.5 gives 1.5 (not clamped)");

        double val_at_neg = bp(-0.5);
        test.expect_near(val_at_neg, -0.5, tolerance, "extrapolation at x=-0.5 gives -0.5 (not clamped)");

        double val_at_2 = bp(2.0);
        test.expect_near(val_at_2, 2.0, tolerance, "extrapolation at x=2.0 gives 2.0");
    }

    // Test 39: Quadratic extrapolation
    {
        // Quadratic polynomial p(t) = t^2 on [0, 1]
        // Bernstein coefficients for t^2: c0 = 0, c1 = 0, c2 = 1
        // p(t) = B_{0,2}(t)*0 + B_{1,2}(t)*0 + B_{2,2}(t)*1 = t^2
        std::vector<std::vector<double>> coeffs = {{0}, {0}, {1}};
        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks, ExtrapolateMode::Extrapolate);

        // At t=2: p(2) = (1-2)^2*0 + 2*2*(1-2)*0 + 2^2*1 = 4
        double val_at_2 = bp(2.0);
        test.expect_near(val_at_2, 4.0, tolerance, "quadratic extrapolation at x=2 gives 4");

        // At t=-1: p(-1) = (1-(-1))^2*0 + 2*(-1)*(1-(-1))*0 + (-1)^2*1 = 1
        double val_at_neg1 = bp(-1.0);
        test.expect_near(val_at_neg1, 1.0, tolerance, "quadratic extrapolation at x=-1 gives 1");
    }

    // Test 40: extend() with degree elevation preserves polynomial values
    {
        // Linear polynomial p1(t) = t on [0, 1], coeffs [0, 1]
        // Quadratic polynomial p2(t) = t^2 on [1, 2], coeffs [0, 0, 1]
        // When extended, the linear should be elevated to quadratic
        // Elevated coeffs for [0,1]: [0, 0.5, 1] (from degree elevation formula)
        //
        // Note: At x=1, find_interval returns interval 1 (quadratic), so p(1)=0 there.
        // This is correct behavior - the polynomials don't need to match at boundaries.

        std::vector<std::vector<double>> linear_coeffs = {{0}, {1}};
        std::vector<double> linear_breaks = {0, 1};
        BPoly linear(linear_coeffs, linear_breaks);

        // Extend with quadratic
        std::vector<std::vector<double>> quad_coeffs = {{0}, {0}, {1}};
        BPoly extended = linear.extend(quad_coeffs, {1, 2}, true);

        // Verify the linear part is preserved (test slightly before boundary)
        test.expect_near(extended(0.0), 0.0, tolerance, "extend degree elevation: p(0) = 0");
        test.expect_near(extended(0.5), 0.5, tolerance, "extend degree elevation: p(0.5) = 0.5");
        // At x = 1-eps, linear p(x)=x gives 1-eps, so use larger tolerance
        test.expect_near(extended(1.0 - 1e-8), 1.0, 1e-7, "extend degree elevation: p(1-eps) ≈ 1");

        // Verify the quadratic part
        // At x=1 (boundary), interval 1 is used, t=0, p(0) = 0
        test.expect_near(extended(1.0), 0.0, tolerance, "extend degree elevation: p(1) = 0 (quadratic)");
        // At x=1.5, in interval [1,2], t = 0.5, p(0.5) = 0.5^2 = 0.25
        test.expect_near(extended(1.5), 0.25, tolerance, "extend degree elevation: p(1.5) = 0.25");
        // At x=2, t=1, p(1) = 1
        test.expect_near(extended(2.0), 1.0, tolerance, "extend degree elevation: p(2) = 1");
    }

    // Test 41: extend() left with degree elevation
    {
        // Start with quadratic on [1, 2], extend left with linear on [0, 1]
        std::vector<std::vector<double>> quad_coeffs = {{0}, {0}, {1}};
        std::vector<double> quad_breaks = {1, 2};
        BPoly quad(quad_coeffs, quad_breaks);

        std::vector<std::vector<double>> linear_coeffs = {{0}, {1}};
        BPoly extended = quad.extend(linear_coeffs, {0, 1}, false);  // extend left

        // Verify the linear part is preserved after elevation (test slightly before boundary)
        test.expect_near(extended(0.0), 0.0, tolerance, "extend left: p(0) = 0");
        test.expect_near(extended(0.5), 0.5, tolerance, "extend left: p(0.5) = 0.5");
        // At x = 1-eps, linear p(x)=x gives 1-eps, so use larger tolerance
        test.expect_near(extended(1.0 - 1e-8), 1.0, 1e-7, "extend left: p(1-eps) ≈ 1");

        // Verify the quadratic part (at boundary x=1, we're in quadratic interval)
        test.expect_near(extended(1.0), 0.0, tolerance, "extend left: p(1) = 0 (quadratic)");
        test.expect_near(extended(1.5), 0.25, tolerance, "extend left: p(1.5) = 0.25");
        test.expect_near(extended(2.0), 1.0, tolerance, "extend left: p(2) = 1");
    }

    // Test 42: antiderivative() is continuous at interval boundaries
    {
        // Create a multi-interval polynomial
        // p(x) = 1 on [0, 1], p(x) = 2 on [1, 2]
        std::vector<std::vector<double>> coeffs = {{1, 2}};  // constant polynomials
        std::vector<double> breaks = {0, 1, 2};
        BPoly bp(coeffs, breaks);

        BPoly antideriv = bp.antiderivative();

        // Antiderivative should be continuous at x=1
        // On [0,1]: integral of 1 is x, so at x=1-, value is 1
        // On [1,2]: should start at 1 (continuity), then add integral of 2*(x-1)
        double left_of_1 = antideriv(1.0 - 1e-10);
        double right_of_1 = antideriv(1.0 + 1e-10);

        // The difference should be very small (continuous)
        double discontinuity = std::abs(right_of_1 - left_of_1);
        test.expect_true(discontinuity <= 1e-8, "antiderivative continuous at x=1");

        // Also check the integral values
        // Integral of 1 from 0 to 1 = 1
        test.expect_near(antideriv(1.0), 1.0, tolerance, "antiderivative at x=1 is 1");

        // Integral of 1 from 0 to 1 = 1, plus integral of 2 from 1 to 2 = 2, total = 3
        test.expect_near(antideriv(2.0), 3.0, tolerance, "antiderivative at x=2 is 3");

        // Integral from 0 to 0.5 = 0.5
        test.expect_near(antideriv(0.5), 0.5, tolerance, "antiderivative at x=0.5 is 0.5");
    }

    // Test 43: antiderivative() with varying polynomial - continuity check
    {
        // from_derivatives creates Hermite cubic matching:
        // Interval [0,1]: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        // This is NOT simply p(x)=x; it's a cubic Hermite polynomial.
        //
        // Hermite cubic: f(t) = -2t³ + 2t² + t
        // Integral from 0 to 1: [-t⁴/2 + 2t³/3 + t²/2]₀¹ = -0.5 + 2/3 + 0.5 = 2/3 ≈ 0.6667

        BPoly bp = BPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1, -1}, {0}});

        BPoly antideriv = bp.antiderivative();

        // Check continuity at x=1
        double left_of_1 = antideriv(1.0 - 1e-10);
        double at_1 = antideriv(1.0);
        double right_of_1 = antideriv(1.0 + 1e-10);

        double disc1 = std::abs(at_1 - left_of_1);
        double disc2 = std::abs(right_of_1 - at_1);

        test.expect_true(disc1 <= 1e-8 && disc2 <= 1e-8,
                        "varying polynomial antiderivative continuous at x=1");

        // Integral of Hermite cubic from 0 to 1 = 2/3
        test.expect_near(antideriv(1.0), 2.0/3.0, tolerance, "antiderivative of Hermite cubic from 0 to 1 is 2/3");
    }

    // Test 44: degree elevation mathematical correctness
    {
        // Create linear [0, 1] and verify manual elevation
        // Linear: c = [0, 1], evaluates to t
        // Elevated to quadratic: c' = [0, 0.5, 1]
        // Both should evaluate to the same values
        std::vector<std::vector<double>> linear_coeffs = {{0}, {1}};
        std::vector<std::vector<double>> elevated_coeffs = {{0}, {0.5}, {1}};
        std::vector<double> breaks = {0, 1};

        BPoly linear(linear_coeffs, breaks);
        BPoly elevated(elevated_coeffs, breaks);

        // They should evaluate to the same at all points
        bool all_match = true;
        for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            double linear_val = linear(t);
            double elevated_val = elevated(t);
            if (std::abs(linear_val - elevated_val) > tolerance) {
                test.fail("degree elevation: linear(" + std::to_string(t) + ") != elevated(" + std::to_string(t) + ")");
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("degree elevation preserves polynomial values");
        }
    }

    // Test 45: Infinity input handling
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {0, 1};

        // Extrapolate mode: infinity should produce infinity (polynomial extrapolation)
        BPoly bp_extrap(coeffs, breaks, ExtrapolateMode::Extrapolate);
        double pos_inf = std::numeric_limits<double>::infinity();
        double neg_inf = -std::numeric_limits<double>::infinity();

        // For Extrapolate mode, ±infinity input should produce ±infinity or NaN
        double result_pos = bp_extrap(pos_inf);
        test.expect_true(std::isinf(result_pos) || std::isnan(result_pos),
                        "Extrapolate mode handles +infinity");

        double result_neg = bp_extrap(neg_inf);
        test.expect_true(std::isinf(result_neg) || std::isnan(result_neg),
                        "Extrapolate mode handles -infinity");

        // NoExtrapolate mode: infinity should return NaN
        BPoly bp_no_extrap(coeffs, breaks, ExtrapolateMode::NoExtrapolate);
        test.expect_true(std::isnan(bp_no_extrap(pos_inf)),
                        "NoExtrapolate mode returns NaN for +infinity");

        // Periodic mode: infinity should return NaN (can't wrap infinity periodically)
        BPoly bp_periodic(coeffs, breaks, ExtrapolateMode::Periodic);
        test.expect_true(std::isnan(bp_periodic(pos_inf)),
                        "Periodic mode returns NaN for +infinity");
        test.expect_true(std::isnan(bp_periodic(neg_inf)),
                        "Periodic mode returns NaN for -infinity");
    }

    // Test 46: from_derivatives with mixed-degree intervals (degree elevation)
    {
        // Create a polynomial with different derivative counts at each point:
        // Point 0: f(0)=0, f'(0)=1 (2 constraints)
        // Point 1: f(1)=1 (1 constraint)
        // Point 2: f(2)=0 (1 constraint)
        //
        // Interval [0,1]: degree = 2+1-1 = 2 (quadratic)
        // Interval [1,2]: degree = 1+1-1 = 1 (linear)
        // max_degree = 2
        //
        // Interval 1 should be degree-elevated from linear to quadratic

        BPoly bp = BPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1}, {0}});

        // Check that degree is 2
        test.expect_eq(static_cast<size_t>(bp.degree()), 2, "from_derivatives mixed: degree is 2");

        // Interval 0: quadratic with f(0)=0, f'(0)=1, f(1)=1
        // Using from_derivatives, this should be correct
        test.expect_near(bp(0.0), 0.0, tolerance, "from_derivatives mixed: f(0)=0");
        test.expect_near(bp(1.0), 1.0, tolerance, "from_derivatives mixed: f(1)=1");
        test.expect_near(bp(2.0), 0.0, tolerance, "from_derivatives mixed: f(2)=0");

        // The key test: interval 1 is linear from 1 to 0
        // At x=1.5, should be 0.5 (midpoint between 1 and 0)
        // If degree elevation is wrong, we'd get incorrect values
        test.expect_near(bp(1.5), 0.5, tolerance, "from_derivatives mixed: f(1.5)=0.5 (linear interval)");

        // Also test that f(0.5) is correct for the quadratic interval
        // For quadratic with f(0)=0, f'(0)=1, f(1)=1:
        // Hermite gives coeffs [0, 0.5, 1] which is p(t) = t (linear)
        // So f(0.5) = 0.5
        test.expect_near(bp(0.5), 0.5, tolerance, "from_derivatives mixed: f(0.5)=0.5 (quadratic interval)");
    }

    // Test 47: from_derivatives asymmetric derivatives (regression for degree elevation)
    {
        // More complex case: f(0)=0, f'(0)=1, f''(0)=0, f(1)=1, f(2)=0
        // Interval [0,1]: degree = 3+1-1 = 3 (cubic)
        // Interval [1,2]: degree = 1+1-1 = 1 (linear)
        // max_degree = 3

        BPoly bp = BPoly::from_derivatives({0, 1, 2}, {{0, 1, 0}, {1}, {0}});

        test.expect_eq(static_cast<size_t>(bp.degree()), 3, "from_derivatives asymmetric: degree is 3");

        // Verify boundary values
        test.expect_near(bp(0.0), 0.0, tolerance, "from_derivatives asymmetric: f(0)=0");
        test.expect_near(bp(1.0), 1.0, tolerance, "from_derivatives asymmetric: f(1)=1");
        test.expect_near(bp(2.0), 0.0, tolerance, "from_derivatives asymmetric: f(2)=0");

        // Critical test: linear interval [1,2] should still be linear after degree elevation
        // f(1)=1, f(2)=0, so f(1.5)=0.5
        test.expect_near(bp(1.5), 0.5, tolerance, "from_derivatives asymmetric: linear part f(1.5)=0.5");
        test.expect_near(bp(1.25), 0.75, tolerance, "from_derivatives asymmetric: linear part f(1.25)=0.75");
        test.expect_near(bp(1.75), 0.25, tolerance, "from_derivatives asymmetric: linear part f(1.75)=0.25");
    }

    // Test 48: High-degree polynomial numerical stability (de Casteljau algorithm)
    {
        // Create a high-degree polynomial (degree 50) that represents 1.0 everywhere
        // All coefficients = 1.0 means p(t) = sum(B_{i,n}(t)) = 1 (partition of unity)
        const int high_degree = 50;
        std::vector<std::vector<double>> coeffs(high_degree + 1, std::vector<double>(1, 1.0));
        std::vector<double> breaks = {0, 1};

        BPoly bp(coeffs, breaks);

        // The polynomial should evaluate to 1.0 everywhere within the domain
        // This tests numerical stability - the old power-based method could have issues
        test.expect_near(bp(0.0), 1.0, tolerance, "high-degree (50) at t=0");
        test.expect_near(bp(0.25), 1.0, tolerance, "high-degree (50) at t=0.25");
        test.expect_near(bp(0.5), 1.0, tolerance, "high-degree (50) at t=0.5");
        test.expect_near(bp(0.75), 1.0, tolerance, "high-degree (50) at t=0.75");
        test.expect_near(bp(1.0), 1.0, tolerance, "high-degree (50) at t=1");

        // Also test at boundaries where numerical issues are more likely
        test.expect_near(bp(0.001), 1.0, tolerance, "high-degree (50) near 0");
        test.expect_near(bp(0.999), 1.0, tolerance, "high-degree (50) near 1");
    }

    // Test 49: Very high degree with linear polynomial
    {
        // Create a very high-degree linear polynomial (all elevated from linear)
        // p(t) = t, but represented as degree 100
        // The de Casteljau algorithm should still give correct results
        const int very_high_degree = 100;
        std::vector<std::vector<double>> coeffs(very_high_degree + 1, std::vector<double>(1));

        // Coefficients for linear t elevated to degree n: c[i] = i/n
        for (int i = 0; i <= very_high_degree; ++i) {
            coeffs[i][0] = static_cast<double>(i) / very_high_degree;
        }

        std::vector<double> breaks = {0, 1};
        BPoly bp(coeffs, breaks);

        // Should evaluate to t at all points
        test.expect_near(bp(0.0), 0.0, tolerance, "very-high-degree (100) linear at t=0");
        test.expect_near(bp(0.25), 0.25, tolerance, "very-high-degree (100) linear at t=0.25");
        test.expect_near(bp(0.5), 0.5, tolerance, "very-high-degree (100) linear at t=0.5");
        test.expect_near(bp(0.75), 0.75, tolerance, "very-high-degree (100) linear at t=0.75");
        test.expect_near(bp(1.0), 1.0, tolerance, "very-high-degree (100) linear at t=1");
    }

    // Test 50: Move semantics efficiency
    {
        // Create a moderately large polynomial
        const int degree = 20;
        const int num_intervals = 100;
        std::vector<std::vector<double>> coeffs(degree + 1,
            std::vector<double>(num_intervals, 1.0));
        std::vector<double> breaks(num_intervals + 1);
        for (int i = 0; i <= num_intervals; ++i) {
            breaks[i] = static_cast<double>(i);
        }

        BPoly bp1(coeffs, breaks);

        // Get pointers to internal data before move
        const double* data_ptr_before = bp1.coefficients()[0].data();
        const double* breaks_ptr_before = bp1.breakpoints().data();

        // Move construct
        BPoly bp2(std::move(bp1));

        // After move, bp2 should have the same data pointers (no copy)
        const double* data_ptr_after = bp2.coefficients()[0].data();
        const double* breaks_ptr_after = bp2.breakpoints().data();

        if (data_ptr_before == data_ptr_after && breaks_ptr_before == breaks_ptr_after) {
            test.pass("move constructor actually moves (no copy)");
        } else {
            test.fail("move constructor copied data instead of moving");
        }

        // Verify the moved-to object works correctly
        test.expect_near(bp2(0.5), 1.0, tolerance, "moved object evaluates correctly");
    }

    // ============================================================
    // Descending breakpoints tests (scipy test_descending)
    // ============================================================

    // Test 51: Descending breakpoints basic
    {
        // Linear descending: x = [1, 0], c = [0, 1]
        // At t=0 (x=1): p(0) = 0
        // At t=1 (x=0): p(1) = 1
        // At t=0.5 (x=0.5): p(0.5) = 0.5
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        std::vector<double> breaks = {1, 0};  // Descending
        BPoly bp(coeffs, breaks);

        test.expect_near(bp(1.0), 0.0, tolerance, "descending f(1)=0");
        test.expect_near(bp(0.5), 0.5, tolerance, "descending f(0.5)=0.5");
        test.expect_near(bp(0.0), 1.0, tolerance, "descending f(0)=1");
    }

    // Test 52: Descending matches ascending equivalent
    {
        // Ascending: [0,1] with coeffs [a,b] -> p(t) = a*(1-t) + b*t
        // Descending: [1,0] with coeffs [b,a] (reversed)
        // For x in [0,1], both should give same values
        std::vector<std::vector<double>> ca = {{1}, {3}};  // p(t)=1*(1-t)+3*t at t=(x-0)/(1-0)=x
        std::vector<std::vector<double>> cd = {{3}, {1}};  // p(t)=3*(1-t)+1*t at t=(x-1)/(0-1)=(1-x)

        BPoly pa(ca, {0, 1});
        BPoly pd(cd, {1, 0});  // Reversed breaks and coeffs

        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            test.expect_near(pa(x), pd(x), tolerance, "ascending==descending at x=" + std::to_string(x));
        }
    }

    // Test 53: Descending derivative
    {
        // For descending [1,0] with coeffs [0,1]:
        // p(t) = 0*(1-t) + 1*t = t, where t = (x-1)/(0-1) = 1-x
        // So p(x) = 1-x, and dp/dx = -1
        // But wait, with our formula: h = 0-1 = -1, scale = 1/(-1) = -1
        // deriv_coeffs = -1 * (1-0) = -1
        // So derivative evaluates to -1 everywhere
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        BPoly bp(coeffs, {1, 0});  // Descending
        BPoly deriv = bp.derivative();

        // The polynomial on [1,0] is: p(t) = t where t = (x-1)/(0-1) = (x-1)/(-1) = 1-x
        // So p(x) = 1-x, derivative = -1
        test.expect_near(deriv(0.5), -1.0, tolerance, "descending derivative = -1");
    }

    // Test 54: Descending with multiple intervals
    {
        // Three breakpoints descending: [2, 1, 0]
        // Interval 0: [2,1], Interval 1: [1,0]
        std::vector<std::vector<double>> coeffs = {{0, 1}, {1, 2}};  // Linear in each
        std::vector<double> breaks = {2, 1, 0};

        BPoly bp(coeffs, breaks);

        // Interval 0: [2,1], coeffs [0,1], t=(x-2)/(1-2)=(x-2)/(-1)=2-x
        // p(t) = 0*(1-t) + 1*t = t = 2-x
        // At x=1.5: t=0.5, p=0.5
        test.expect_near(bp(1.5), 0.5, tolerance, "descending multi f(1.5)");

        // Interval 1: [1,0], coeffs [1,2], t=(x-1)/(0-1)=(x-1)/(-1)=1-x
        // p(t) = 1*(1-t) + 2*t = 1 + t = 1 + (1-x) = 2-x
        // At x=0.5: p = 2-0.5 = 1.5
        test.expect_near(bp(0.5), 1.5, tolerance, "descending multi f(0.5)");
    }

    // Test 55: Descending extrapolation
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        BPoly bp(coeffs, {1, 0}, ExtrapolateMode::Extrapolate);

        // Polynomial: p(x) = 1-x
        // At x=2: p(2) = 1-2 = -1
        // At x=-1: p(-1) = 1-(-1) = 2
        test.expect_near(bp(2.0), -1.0, tolerance, "descending extrapolate right");
        test.expect_near(bp(-1.0), 2.0, tolerance, "descending extrapolate left");
    }

    // Test 56: Descending NoExtrapolate
    {
        std::vector<std::vector<double>> coeffs = {{0}, {1}};
        BPoly bp(coeffs, {1, 0}, ExtrapolateMode::NoExtrapolate);

        // Inside domain should work
        test.expect_near(bp(0.5), 0.5, tolerance, "descending NoExtrap inside");

        // Outside domain should be NaN
        test.expect_no_throw([&]() {
            if (!std::isnan(bp(2.0))) {
                throw std::runtime_error("Should return NaN for x>1");
            }
        }, "descending NoExtrap right OOB");

        test.expect_no_throw([&]() {
            if (!std::isnan(bp(-0.5))) {
                throw std::runtime_error("Should return NaN for x<0");
            }
        }, "descending NoExtrap left OOB");
    }

    // Test 57: is_ascending() accessor
    {
        BPoly asc({{0}, {1}}, {0, 1});
        BPoly desc({{0}, {1}}, {1, 0});

        test.expect_true(asc.is_ascending(), "is_ascending() returns true for ascending");
        test.expect_true(!desc.is_ascending(), "is_ascending() returns false for descending");
    }

    // Test 58: from_derivatives rejects descending
    test.expect_throw([&]() {
        BPoly::from_derivatives({1, 0}, {{0}, {1}});  // Descending xi
    }, "from_derivatives rejects descending");

    // Test 59: extend() rejects mismatched order
    {
        BPoly asc({{0}, {1}}, {0, 1});
        test.expect_throw([&]() {
            asc.extend({{1}, {0}}, {1, 0}, true);  // Descending extension
        }, "extend rejects descending when ascending");
    }

    // ============================================================
    // Additional scipy-inspired tests
    // ============================================================

    // Test 60: xi/yi length mismatch (scipy test_xi_yi)
    test.expect_throw([&]() {
        BPoly::from_derivatives({0, 1}, {{0}});  // 2 xi, 1 yi
    }, "xi/yi length mismatch");

    // Test 61: All-zero function (scipy test_zeros)
    {
        std::vector<double> xi = {0, 1, 2, 3};
        std::vector<std::vector<double>> yi = {{0, 0}, {0}, {0, 0}, {0, 0}};
        BPoly pp = BPoly::from_derivatives(xi, yi);

        // Degree should be 3 (elevated from mixed 2,1,2,2)
        test.expect_eq(static_cast<size_t>(pp.degree()), 3, "test_zeros degree is 3");

        // All evaluations should be zero
        for (double x : {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}) {
            test.expect_near(pp(x), 0.0, tolerance, "test_zeros f(" + std::to_string(x) + ")=0");
        }

        // Derivative should also be zero
        BPoly ppd = pp.derivative();
        test.expect_near(ppd(1.5), 0.0, tolerance, "test_zeros derivative=0");
    }

    // Test 62: 6th order derivatives (scipy test_make_poly_12)
    {
        // Test with 6 derivatives at each endpoint
        std::vector<double> ya = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};  // f=0, f'=1, higher=0
        std::vector<double> yb = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0};  // f=1, f'=1, higher=0

        BPoly pp = BPoly::from_derivatives({0, 1}, {ya, yb});

        // Verify f(0)=ya[0]=0, f(1)=yb[0]=1
        test.expect_near(pp(0), 0.0, tolerance, "6th-order f(0)=0");
        test.expect_near(pp(1), 1.0, tolerance, "6th-order f(1)=1");

        // With f(0)=0, f'(0)=1, f(1)=1, f'(1)=1, and all higher derivatives=0,
        // the polynomial should pass through f(0.5) = 0.5 (since it's close to linear)
        // The exact scipy value for this configuration is 0.5
        test.expect_near(pp(0.5), 0.5, tolerance, "6th-order f(0.5)=0.5");
    }

    // Test 63: k=12 derivatives, multi-interval (scipy test_random_12)
    {
        int m = 5, k = 12;
        std::vector<double> xi = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
        std::vector<std::vector<double>> yi(m + 1);

        for (int i = 0; i <= m; ++i) {
            yi[i].resize(k);
            for (int j = 0; j < k; ++j) {
                yi[i][j] = std::sin(i + j * 0.1);  // Deterministic values
            }
        }

        BPoly pp = BPoly::from_derivatives(xi, yi);

        // Verify pp(xi[i]) == yi[i][0] for all breakpoints
        bool all_match = true;
        for (int i = 0; i <= m; ++i) {
            double expected = yi[i][0];
            double actual = pp(xi[i]);
            if (std::abs(actual - expected) > tolerance) {
                test.fail("k=12 f(xi[" + std::to_string(i) + "]) mismatch");
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("k=12 all breakpoint values match");
        }
    }

    // ============================================================
    // Integration and derivative(-N) tests (scipy TestBPolyCalculus)
    // ============================================================

    // Test 64: test_der_antider (scipy) - derivative of antiderivative = original
    {
        BPoly bp = BPoly::from_derivatives({0, 1}, {{1, 2}, {3, -1}});
        BPoly antideriv = bp.antiderivative();
        BPoly recovered = antideriv.derivative();

        bool all_match = true;
        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            if (std::abs(bp(x) - recovered(x)) > tolerance) {
                test.fail("der_antider at x=" + std::to_string(x));
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("der_antider: derivative(antiderivative) = original");
        }

        // Independent verification: compare antiderivative against numerical integration
        // The antiderivative at x equals the integral from 0 to x (since antideriv(0) = 0)
        double numerical_val = numerical_integrate(bp, 0, 0.5);
        test.expect_near(antideriv(0.5), numerical_val, 1e-6,
                        "der_antider: antiderivative vs numerical integration at x=0.5");

        double numerical_val_full = numerical_integrate(bp, 0, 1.0);
        test.expect_near(antideriv(1.0), numerical_val_full, 1e-6,
                        "der_antider: antiderivative vs numerical integration at x=1.0");
    }

    // Test 65: integrate(a, b) constant
    {
        // Constant polynomial: f(x) = 2, integral from 0 to 1 = 2
        BPoly bp({{2}}, {0, 1});
        test.expect_near(bp.integrate(0, 1), 2.0, tolerance, "integrate constant");
    }

    // Test 66: integrate(a, b) linear
    {
        // Linear: f(x) = x on [0,1], integral = 0.5
        BPoly bp({{0}, {1}}, {0, 1});
        test.expect_near(bp.integrate(0, 1), 0.5, tolerance, "integrate linear");
    }

    // Test 67: integrate(a, b) quadratic
    {
        // Quadratic: f(x) = x^2 on [0,1], integral = 1/3
        // Bernstein form of x^2: c = [0, 0, 1]
        BPoly bp({{0}, {0}, {1}}, {0, 1});
        test.expect_near(bp.integrate(0, 1), 1.0/3.0, tolerance, "integrate quadratic");
    }

    // Test 68: integrate partial interval
    {
        BPoly bp({{0}, {1}}, {0, 1});  // f(x) = x
        // Integral of x from 0.25 to 0.75 = (0.75^2 - 0.25^2)/2 = 0.25
        test.expect_near(bp.integrate(0.25, 0.75), 0.25, tolerance, "integrate partial");
    }

    // Test 69: integrate with extrapolation
    {
        BPoly bp({{0}, {1}}, {0, 1});  // f(x) = x
        // Integral from -1 to 2 (extrapolated)
        // For linear, extrapolation continues as f(x) = x
        // Integral = 2^2/2 - (-1)^2/2 = 2 - 0.5 = 1.5
        test.expect_near(bp.integrate(-1, 2), 1.5, tolerance, "integrate extrap");
    }

    // Test 70: integrate multi-interval
    {
        // Two intervals: [0,1] and [1,2], linear f(x) = x
        BPoly bp({{0, 1}, {1, 2}}, {0, 1, 2});
        // Integral of x from 0 to 2 = 2^2/2 = 2
        test.expect_near(bp.integrate(0, 2), 2.0, tolerance, "integrate multi");
    }

    // Test 71: derivative(-1) equals antiderivative
    {
        // f(x) = 1 + x (linear Bernstein: c0=1, c1=2)
        // Antiderivative: F(x) = x + x²/2 with F(0) = 0
        BPoly bp({{1}, {2}}, {0, 1});
        BPoly d_neg1 = bp.derivative(-1);
        BPoly antid = bp.antiderivative();

        bool all_match = true;
        for (double x : {0.0, 0.5, 1.0}) {
            if (std::abs(d_neg1(x) - antid(x)) > tolerance) {
                test.fail("derivative(-1) at x=" + std::to_string(x));
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("derivative(-1) equals antiderivative");
        }

        // Independent verification: check against analytical formula F(x) = x + x²/2
        // At x=0.5: F(0.5) = 0.5 + 0.125 = 0.625
        test.expect_near(antid(0.5), 0.625, tolerance, "derivative(-1) independent: F(0.5)=0.625");
        // At x=1.0: F(1.0) = 1.0 + 0.5 = 1.5
        test.expect_near(antid(1.0), 1.5, tolerance, "derivative(-1) independent: F(1.0)=1.5");
    }

    // Test 72: derivative(-2) equals antiderivative(2)
    {
        // f(x) = 1 + x
        // First antiderivative: F1(x) = x + x²/2
        // Second antiderivative: F2(x) = x²/2 + x³/6 with F2(0) = 0
        BPoly bp({{1}, {2}}, {0, 1});
        BPoly d_neg2 = bp.derivative(-2);
        BPoly antid2 = bp.antiderivative(2);

        bool all_match = true;
        for (double x : {0.0, 0.5, 1.0}) {
            if (std::abs(d_neg2(x) - antid2(x)) > tolerance) {
                test.fail("derivative(-2) at x=" + std::to_string(x));
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("derivative(-2) equals antiderivative(2)");
        }

        // Independent verification using numerical double integration
        double numerical_f2 = numerical_integrate(bp.antiderivative(), 0, 0.5);
        test.expect_near(antid2(0.5), numerical_f2, 1e-6, "derivative(-2) numerical verification");
    }

    // Test 73: operator()(x, nu) in-place derivative
    {
        BPoly bp({{0}, {1}}, {0, 1});  // f(x) = x
        // f(0.5) = 0.5
        test.expect_near(bp(0.5, 0), 0.5, tolerance, "bp(x, nu=0)");
        // f'(x) = 1
        test.expect_near(bp(0.5, 1), 1.0, tolerance, "bp(x, nu=1)");
        // f''(x) = 0
        test.expect_near(bp(0.5, 2), 0.0, tolerance, "bp(x, nu=2)");
    }

    // Test 74: operator()(x, nu) negative (antiderivative)
    {
        BPoly bp({{2}}, {0, 1});  // f(x) = 2 (constant)
        // Antiderivative F(x) = 2x, F(0.5) = 1
        test.expect_near(bp(0.5, -1), 1.0, tolerance, "bp(x, nu=-1)");
    }

    // Test 75: integrate descending breakpoints
    {
        BPoly bp({{1}, {0}}, {1, 0});  // Descending, coeffs reversed
        // This represents f(x) = x on descending [1,0]
        // Integral of x from 0 to 1 = 0.5
        test.expect_near(bp.integrate(0, 1), 0.5, tolerance, "integrate descending");
    }

    // Test 76: integrate reversed bounds (a > b)
    {
        BPoly bp({{0}, {1}}, {0, 1});  // f(x) = x
        // Integral from 1 to 0 = -0.5
        test.expect_near(bp.integrate(1, 0), -0.5, tolerance, "integrate reversed bounds");
    }

    // Test 77: integrate Hermite cubic (scipy antiderivative test)
    {
        // f(x) with f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});
        // We computed earlier: antiderivative at x=1 minus x=0 should be 2/3
        test.expect_near(bp.integrate(0, 1), 2.0/3.0, tolerance, "integrate Hermite cubic");
    }

    // ============================================================
    // scipy test_integrate_extrap tests
    // ============================================================

    // Test 78: integrate with extrapolate=false parameter (scipy test_integrate_extrap)
    {
        // BPoly with default extrapolation
        BPoly bp({{0}, {1}}, {0, 1});  // f(x) = x on [0,1]

        // integrate(0, 2) with extrapolate=false should return NaN
        double result = bp.integrate(0, 2, false);
        test.expect_no_throw([&]() {
            if (!std::isnan(result)) {
                throw std::runtime_error("Should return NaN");
            }
        }, "integrate extrapolate=false returns NaN for OOB");
    }

    // Test 79: integrate with extrapolate=true parameter
    {
        // BPoly with NoExtrapolate mode
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::NoExtrapolate);

        // integrate(0, 2) with extrapolate=true should work
        double result = bp.integrate(0, 2, true);
        // For linear f(x)=x, integral from 0 to 2 = 2
        test.expect_near(result, 2.0, tolerance, "integrate extrapolate=true overrides NoExtrap");
    }

    // Test 80: integrate with default extrapolation (NoExtrapolate mode)
    {
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::NoExtrapolate);

        // integrate(0, 2) without override should return NaN
        double result = bp.integrate(0, 2);
        test.expect_no_throw([&]() {
            if (!std::isnan(result)) {
                throw std::runtime_error("Should return NaN");
            }
        }, "integrate respects NoExtrapolate default");
    }

    // Test 81: integrate with default extrapolation (Extrapolate mode)
    {
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);

        // integrate(0, 2) should work with Extrapolate mode
        test.expect_near(bp.integrate(0, 2), 2.0, tolerance, "integrate respects Extrapolate default");
    }

    // ============================================================
    // scipy test_integrate_periodic tests
    // ============================================================

    // Test 82: integrate periodic - full period
    {
        // f(x) = x on [0,1] with periodic mode
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Periodic);

        // Integral over one full period [0,1] = 0.5
        test.expect_near(bp.integrate(0, 1), 0.5, tolerance, "integrate periodic full period");
    }

    // Test 83: integrate periodic - shifted period
    {
        // f(x) = 1 (constant) on [0,1] with periodic mode
        BPoly bp({{1}}, {0, 1}, ExtrapolateMode::Periodic);

        // Integral from 1 to 2 (one period shifted) = 1
        test.expect_near(bp.integrate(1, 2), 1.0, tolerance, "integrate periodic shifted period");
    }

    // Test 84: integrate periodic - multiple periods
    {
        // f(x) = 1 (constant) on [0,1] with periodic mode
        BPoly bp({{1}}, {0, 1}, ExtrapolateMode::Periodic);

        // Integral from 0 to 3 (three periods) = 3
        test.expect_near(bp.integrate(0, 3), 3.0, tolerance, "integrate periodic multiple periods");
    }

    // Test 85: integrate periodic - negative range
    {
        // f(x) = 1 (constant) on [0,1] with periodic mode
        BPoly bp({{1}}, {0, 1}, ExtrapolateMode::Periodic);

        // Integral from -1 to 0 (one period before) = 1
        test.expect_near(bp.integrate(-1, 0), 1.0, tolerance, "integrate periodic negative range");
    }

    // Test 86: integrate periodic - partial period
    {
        // f(x) = 1 (constant) on [0,1] with periodic mode
        BPoly bp({{1}}, {0, 1}, ExtrapolateMode::Periodic);

        // Integral from 0.5 to 1.5 (crossing boundary) = 1
        test.expect_near(bp.integrate(0.5, 1.5), 1.0, tolerance, "integrate periodic partial crossing");
    }

    // Test 87: integrate periodic - linear over multiple periods
    {
        // f(x) = x on [0,1], periodic
        // The function wraps, so f(1.5) = f(0.5) = 0.5
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Periodic);

        // Integral over two periods = 2 * 0.5 = 1
        test.expect_near(bp.integrate(0, 2), 1.0, tolerance, "integrate periodic linear 2 periods");
    }

    // Test 88: integrate periodic - reversed bounds
    {
        // f(x) = 1 (constant) on [0,1] with periodic mode
        BPoly bp({{1}}, {0, 1}, ExtrapolateMode::Periodic);

        // Integral from 2 to 0 = -2
        test.expect_near(bp.integrate(2, 0), -2.0, tolerance, "integrate periodic reversed");
    }

    // ========================================================================
    // from_power_basis() tests
    // ========================================================================

    // Test 89: from_power_basis constant
    {
        // Power basis: p(x) = 3 (constant)
        // Power coeffs: c[0] = 3
        BPoly bp = BPoly::from_power_basis({{3}}, {0, 1});
        test.expect_near(bp(0.0), 3.0, tolerance, "from_power_basis constant at 0");
        test.expect_near(bp(0.5), 3.0, tolerance, "from_power_basis constant at 0.5");
        test.expect_near(bp(1.0), 3.0, tolerance, "from_power_basis constant at 1");
    }

    // Test 90: from_power_basis linear
    {
        // Power basis: p(x) = 1 + 2*(x-a) on [0,1]
        // At x=0: p(0) = 1
        // At x=1: p(1) = 1 + 2*1 = 3
        // Power coeffs: c[0] = 1, c[1] = 2
        BPoly bp = BPoly::from_power_basis({{1}, {2}}, {0, 1});
        test.expect_near(bp(0.0), 1.0, tolerance, "from_power_basis linear at 0");
        test.expect_near(bp(0.5), 2.0, tolerance, "from_power_basis linear at 0.5");
        test.expect_near(bp(1.0), 3.0, tolerance, "from_power_basis linear at 1");
    }

    // Test 91: from_power_basis quadratic
    {
        // Power basis: p(x) = 1 + (x-0)^2 = 1 + x^2 on [0,1]
        // Actually for PPoly-style: p(x) = c[0] + c[1]*(x-a) + c[2]*(x-a)^2
        // For interval [0,1]: p(x) = 0 + 0*x + 1*x^2
        // At x=0: p(0) = 0
        // At x=0.5: p(0.5) = 0.25
        // At x=1: p(1) = 1
        BPoly bp = BPoly::from_power_basis({{0}, {0}, {1}}, {0, 1});
        test.expect_near(bp(0.0), 0.0, tolerance, "from_power_basis quadratic at 0");
        test.expect_near(bp(0.5), 0.25, tolerance, "from_power_basis quadratic at 0.5");
        test.expect_near(bp(1.0), 1.0, tolerance, "from_power_basis quadratic at 1");
    }

    // Test 92: from_power_basis non-unit interval
    {
        // Power basis: p(x) = 2 + 1*(x-0) on [0,2]
        // At x=0: p(0) = 2
        // At x=1: p(1) = 3
        // At x=2: p(2) = 4
        BPoly bp = BPoly::from_power_basis({{2}, {1}}, {0, 2});
        test.expect_near(bp(0.0), 2.0, tolerance, "from_power_basis non-unit at 0");
        test.expect_near(bp(1.0), 3.0, tolerance, "from_power_basis non-unit at 1");
        test.expect_near(bp(2.0), 4.0, tolerance, "from_power_basis non-unit at 2");
    }

    // Test 93: from_power_basis two intervals
    {
        // Interval [0,1]: p(x) = x (linear)
        // Interval [1,2]: p(x) = 1 + 0*(x-1) = 1 (constant)
        // Power coeffs: [[0, 1], [1, 0]] means:
        //   c[0] = [0, 1] (constant terms for intervals 0 and 1)
        //   c[1] = [1, 0] (linear terms for intervals 0 and 1)
        BPoly bp = BPoly::from_power_basis({{0, 1}, {1, 0}}, {0, 1, 2});
        test.expect_near(bp(0.0), 0.0, tolerance, "from_power_basis two intervals at 0");
        test.expect_near(bp(0.5), 0.5, tolerance, "from_power_basis two intervals at 0.5");
        test.expect_near(bp(1.0), 1.0, tolerance, "from_power_basis two intervals at 1");
        test.expect_near(bp(1.5), 1.0, tolerance, "from_power_basis two intervals at 1.5");
        test.expect_near(bp(2.0), 1.0, tolerance, "from_power_basis two intervals at 2");
    }

    // Test 94: from_power_basis cubic
    {
        // Power basis: p(x) = x^3 on [0,1]
        // c[0] = 0, c[1] = 0, c[2] = 0, c[3] = 1
        BPoly bp = BPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1});
        test.expect_near(bp(0.0), 0.0, tolerance, "from_power_basis cubic at 0");
        test.expect_near(bp(0.5), 0.125, tolerance, "from_power_basis cubic at 0.5");
        test.expect_near(bp(1.0), 1.0, tolerance, "from_power_basis cubic at 1");
    }

    // Test 95: from_power_basis error - empty coeffs
    {
        try {
            BPoly::from_power_basis({}, {0, 1});
            test.fail("from_power_basis empty should throw");
        } catch (const std::invalid_argument& e) {
            test.pass("from_power_basis empty throws");
        }
    }

    // ========================================================================
    // roots() tests
    // ========================================================================

    // Test 96: roots of linear - single root
    {
        // f(x) = x - 0.5 on [0,1], root at x=0.5
        // In Bernstein basis: need coeffs such that f(0) = -0.5, f(1) = 0.5
        // Linear Bernstein: f(t) = (1-t)*c0 + t*c1
        // f(0) = c0 = -0.5, f(1) = c1 = 0.5
        BPoly bp({{-0.5}, {0.5}}, {0, 1});
        auto r = bp.roots();
        test.expect_eq(r.size(), 1, "roots linear count");
        if (!r.empty()) {
            test.expect_near(r[0], 0.5, tolerance, "roots linear value");
        }
    }

    // Test 97: roots of quadratic - two roots
    {
        // f(x) = (x-0.25)(x-0.75) = x^2 - x + 0.1875 on [0,1]
        // Using from_power_basis: c[0] = 0.1875, c[1] = -1, c[2] = 1
        BPoly bp = BPoly::from_power_basis({{0.1875}, {-1}, {1}}, {0, 1});
        auto r = bp.roots();
        test.expect_eq(r.size(), 2, "roots quadratic count");
        if (r.size() >= 2) {
            test.expect_near(r[0], 0.25, tolerance, "roots quadratic first");
            test.expect_near(r[1], 0.75, tolerance, "roots quadratic second");
        }
    }

    // Test 98: roots at endpoints
    {
        // f(x) = x*(x-1) on [0,1], roots at x=0 and x=1
        // Power basis: c[0] = 0, c[1] = -1, c[2] = 1
        BPoly bp = BPoly::from_power_basis({{0}, {-1}, {1}}, {0, 1});
        auto r = bp.roots();
        test.expect_eq(r.size(), 2, "roots endpoints count");
        if (r.size() >= 2) {
            test.expect_near(r[0], 0.0, tolerance, "roots endpoint at 0");
            test.expect_near(r[1], 1.0, tolerance, "roots endpoint at 1");
        }
    }

    // Test 99: roots no real roots
    {
        // f(x) = x^2 + 1 on [0,1], no real roots
        BPoly bp = BPoly::from_power_basis({{1}, {0}, {1}}, {0, 1});
        auto r = bp.roots();
        test.expect_eq(r.size(), 0, "roots none count");
    }

    // Test 100: roots constant zero - without extrapolation
    {
        // f(x) = 0 everywhere - technically infinite roots
        // The roots() implementation samples at many points and reports all where |f| < tol.
        // For a constant zero, this returns all sample points (many more than 2).
        // We verify the domain boundaries are included.
        BPoly bp({{0}}, {0, 1});
        auto r = bp.roots(true, false);  // discontinuity=true, extrapolate=false
        // For a zero function, expect many sample points to be reported as roots
        test.expect_true(r.size() >= 2, "roots zero function finds at least 2 roots");
        // First root should be at or near 0
        if (!r.empty()) {
            test.expect_near(r[0], 0.0, tolerance, "roots zero function: first root at 0");
        }
        // Last root should be at or near 1
        if (r.size() >= 2) {
            test.expect_near(r.back(), 1.0, tolerance, "roots zero function: last root at 1");
        }
    }

    // Test 101: roots multi-interval
    {
        // Two intervals: f(x) = x on [0,1], f(x) = 2-x on [1,2]
        // Roots at x=0 and x=2
        // Interval 1: f(t) = t, Bernstein: c0=0, c1=1
        // Interval 2: f(t) = 1 - t, Bernstein: c0=1, c1=0
        BPoly bp({{0, 1}, {1, 0}}, {0, 1, 2});
        auto r = bp.roots();
        test.expect_eq(r.size(), 2, "roots multi-interval count");
        if (r.size() >= 2) {
            test.expect_near(r[0], 0.0, tolerance, "roots multi at 0");
            test.expect_near(r[1], 2.0, tolerance, "roots multi at 2");
        }
    }

    // Test 102: roots with extrapolation
    {
        // f(x) = x - 2 on [0,1], root at x=2 (in extrapolated region)
        BPoly bp({{-2}, {-1}}, {0, 1}, ExtrapolateMode::Extrapolate);
        auto r = bp.roots(true, true);  // discontinuity=true, extrapolate=true
        test.expect_eq(r.size(), 1, "roots extrapolate count");
        if (!r.empty()) {
            test.expect_near(r[0], 2.0, tolerance, "roots extrapolate value");
        }
    }

    // Test 103: roots without extrapolation
    {
        // Same as above but extrapolate=false
        BPoly bp({{-2}, {-1}}, {0, 1}, ExtrapolateMode::Extrapolate);
        auto r = bp.roots(true, false);  // extrapolate=false
        test.expect_eq(r.size(), 0, "roots no extrapolate count");
    }

    // Test 104: roots discontinuity
    {
        // Two intervals with sign change at discontinuity
        // Interval 1: f(x) = 1 on [0,1]
        // Interval 2: f(x) = -1 on [1,2]
        // There's a sign change at x=1 (discontinuity)
        BPoly bp({{1, -1}}, {0, 1, 2});
        auto r = bp.roots(true, false);  // discontinuity=true
        test.expect_eq(r.size(), 1, "roots discontinuity count");
        if (!r.empty()) {
            test.expect_near(r[0], 1.0, tolerance, "roots discontinuity at 1");
        }
    }

    // Test 105: roots no discontinuity flag
    {
        // Same as above but discontinuity=false
        BPoly bp({{1, -1}}, {0, 1, 2});
        auto r = bp.roots(false, false);  // discontinuity=false
        test.expect_eq(r.size(), 0, "roots no discontinuity flag");
    }

    // Test 106: roots from_derivatives polynomial
    {
        // Hermite cubic passing through (0,0) and (1,0) with derivatives
        // This creates a polynomial that starts at 0, goes up, comes back to 0
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1}, {0, -1}});
        auto r = bp.roots();
        // Should find roots at 0 and 1
        test.expect_eq(r.size(), 2, "roots from_derivatives count");
        if (r.size() >= 2) {
            test.expect_near(r[0], 0.0, tolerance, "roots from_derivatives at 0");
            test.expect_near(r[1], 1.0, tolerance, "roots from_derivatives at 1");
        }
    }

    // Test 107: roots left extrapolation
    {
        // f(x) = x + 1 on [0,1], root at x=-1 (left extrapolation)
        BPoly bp({{1}, {2}}, {0, 1}, ExtrapolateMode::Extrapolate);
        auto r = bp.roots(true, true);
        test.expect_eq(r.size(), 1, "roots left extrapolate count");
        if (!r.empty()) {
            test.expect_near(r[0], -1.0, tolerance, "roots left extrapolate value");
        }
    }

    // ========================================================================
    // orders parameter tests for from_derivatives()
    // ========================================================================

    // Test 108: orders parameter - global limit
    {
        // With 3 derivatives each [f, f', f''] but orders=0 uses only f
        // This creates a linear interpolation
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1, 2}, {1, -1, 0}}, {0});
        // With orders=0, only function values used, degree = 0+0+1 = 1 (linear)
        test.expect_eq(static_cast<size_t>(bp.degree()), 1, "orders global: degree is 1");
        test.expect_near(bp(0.0), 0.0, tolerance, "orders global: f(0)=0");
        test.expect_near(bp(1.0), 1.0, tolerance, "orders global: f(1)=1");
        test.expect_near(bp(0.5), 0.5, tolerance, "orders global: f(0.5)=0.5 (linear)");
    }

    // Test 109: orders parameter - global limit with higher value
    {
        // With 3 derivatives each but orders=1 uses f and f'
        // This creates a cubic Hermite interpolation
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1, 2}, {1, -1, 0}}, {1});
        // With orders=1, use f and f', degree = 1+1+1 = 3 (cubic)
        test.expect_eq(static_cast<size_t>(bp.degree()), 3, "orders=1: degree is 3");
        test.expect_near(bp(0.0), 0.0, tolerance, "orders=1: f(0)=0");
        test.expect_near(bp(1.0), 1.0, tolerance, "orders=1: f(1)=1");
    }

    // Test 110: orders parameter - per-point limits
    {
        // Different limits at each point: orders=[1, 0]
        // Left: use f and f', Right: use only f
        // degree = (1+1) + (0+1) - 1 = 2
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}}, {1, 0});
        test.expect_eq(static_cast<size_t>(bp.degree()), 2, "orders per-point: degree is 2");
        test.expect_near(bp(0.0), 0.0, tolerance, "orders per-point: f(0)=0");
        test.expect_near(bp(1.0), 1.0, tolerance, "orders per-point: f(1)=1");
    }

    // Test 111: orders parameter - empty (default behavior)
    {
        // Empty orders = use all available derivatives
        BPoly bp1 = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}}, {});
        BPoly bp2 = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});
        test.expect_eq(static_cast<size_t>(bp1.degree()), static_cast<size_t>(bp2.degree()),
                      "orders empty: same as default");
        test.expect_near(bp1(0.5), bp2(0.5), tolerance, "orders empty: same values");
    }

    // Test 112: orders parameter - limit higher than available
    {
        // orders=10 but only 2 derivatives available - should use all available
        BPoly bp1 = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}}, {10});
        BPoly bp2 = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});
        test.expect_eq(static_cast<size_t>(bp1.degree()), static_cast<size_t>(bp2.degree()),
                      "orders too high: uses available");
    }

    // Test 113: orders parameter - validation error
    {
        try {
            // orders size doesn't match xi size and isn't 1
            BPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}}, {0, 1});  // 3 points, 2 orders
            test.fail("orders wrong size should throw");
        } catch (const std::invalid_argument&) {
            test.pass("orders wrong size throws");
        }
    }

    // Test 114: orders parameter - multi-interval
    {
        // Three points with different orders at each
        BPoly bp = BPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1}, {0, -1}}, {1, 0, 1});
        test.expect_near(bp(0.0), 0.0, tolerance, "orders multi: f(0)=0");
        test.expect_near(bp(1.0), 1.0, tolerance, "orders multi: f(1)=1");
        test.expect_near(bp(2.0), 0.0, tolerance, "orders multi: f(2)=0");
    }

    // ========================================================================
    // to_power_basis() tests
    // ========================================================================

    // Test 115: to_power_basis constant
    {
        // Constant polynomial f(x) = 3
        BPoly bp({{3}}, {0, 1});
        auto power = bp.to_power_basis();
        test.expect_eq(power.size(), 1, "to_power_basis constant: 1 row");
        test.expect_near(power[0][0], 3.0, tolerance, "to_power_basis constant: c[0]=3");
    }

    // Test 116: to_power_basis linear
    {
        // Linear: f(x) = x on [0,1], Bernstein coeffs: c0=0, c1=1
        BPoly bp({{0}, {1}}, {0, 1});
        auto power = bp.to_power_basis();
        // Power form: f(x) = 0 + 1*(x-0) = x
        test.expect_near(power[0][0], 0.0, tolerance, "to_power_basis linear: c[0]=0");
        test.expect_near(power[1][0], 1.0, tolerance, "to_power_basis linear: c[1]=1");
    }

    // Test 117: to_power_basis quadratic
    {
        // Quadratic: f(x) = x^2 on [0,1], Bernstein coeffs: c0=0, c1=0, c2=1
        BPoly bp({{0}, {0}, {1}}, {0, 1});
        auto power = bp.to_power_basis();
        // Power form: f(x) = 0 + 0*(x-0) + 1*(x-0)^2
        test.expect_near(power[0][0], 0.0, tolerance, "to_power_basis quadratic: c[0]=0");
        test.expect_near(power[1][0], 0.0, tolerance, "to_power_basis quadratic: c[1]=0");
        test.expect_near(power[2][0], 1.0, tolerance, "to_power_basis quadratic: c[2]=1");
    }

    // Test 118: to_power_basis non-unit interval
    {
        // Linear f(x) = x on [0,2], Bernstein coeffs: c0=0, c1=2
        BPoly bp({{0}, {2}}, {0, 2});
        auto power = bp.to_power_basis();
        // Power form: f(x) = 0 + 1*(x-0)
        test.expect_near(power[0][0], 0.0, tolerance, "to_power_basis non-unit: c[0]=0");
        test.expect_near(power[1][0], 1.0, tolerance, "to_power_basis non-unit: c[1]=1");
    }

    // Test 119: to_power_basis round-trip
    {
        // Start with power basis, convert to BPoly, convert back
        std::vector<std::vector<double>> orig_power = {{1}, {2}, {3}};  // 1 + 2x + 3x^2
        BPoly bp = BPoly::from_power_basis(orig_power, {0, 1});
        auto recovered = bp.to_power_basis();
        test.expect_near(recovered[0][0], orig_power[0][0], tolerance, "round-trip: c[0]");
        test.expect_near(recovered[1][0], orig_power[1][0], tolerance, "round-trip: c[1]");
        test.expect_near(recovered[2][0], orig_power[2][0], tolerance, "round-trip: c[2]");
    }

    // Test 120: to_power_basis multi-interval
    {
        // Two intervals with different polynomials
        // Interval [0,1]: f(x) = x (linear)
        // Interval [1,2]: f(x) = 1 (constant, but elevated to linear)
        BPoly bp({{0, 1}, {1, 1}}, {0, 1, 2});
        auto power = bp.to_power_basis();
        // First interval: 0 + 1*(x-0)
        test.expect_near(power[0][0], 0.0, tolerance, "to_power_basis multi [0]: c[0]=0");
        test.expect_near(power[1][0], 1.0, tolerance, "to_power_basis multi [0]: c[1]=1");
        // Second interval: 1 + 0*(x-1)
        test.expect_near(power[0][1], 1.0, tolerance, "to_power_basis multi [1]: c[0]=1");
        test.expect_near(power[1][1], 0.0, tolerance, "to_power_basis multi [1]: c[1]=0");
    }

    // Test 121: to_power_basis from Hermite interpolation
    {
        // Hermite cubic: f(0)=0, f'(0)=1, f(1)=1, f'(1)=-1
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});
        auto power = bp.to_power_basis();
        // Verify by evaluating the power basis polynomial
        auto eval_power = [&](double x) {
            double result = 0.0;
            double x_power = 1.0;
            for (size_t k = 0; k < power.size(); ++k) {
                result += power[k][0] * x_power;
                x_power *= x;
            }
            return result;
        };
        test.expect_near(eval_power(0.0), bp(0.0), tolerance, "to_power_basis Hermite at 0");
        test.expect_near(eval_power(0.5), bp(0.5), tolerance, "to_power_basis Hermite at 0.5");
        test.expect_near(eval_power(1.0), bp(1.0), tolerance, "to_power_basis Hermite at 1");
    }

    // Test 122: to_power_basis cubic
    {
        // Cubic: f(x) = x^3 on [0,1]
        BPoly bp = BPoly::from_power_basis({{0}, {0}, {0}, {1}}, {0, 1});
        auto power = bp.to_power_basis();
        test.expect_near(power[0][0], 0.0, tolerance, "to_power_basis cubic: c[0]=0");
        test.expect_near(power[1][0], 0.0, tolerance, "to_power_basis cubic: c[1]=0");
        test.expect_near(power[2][0], 0.0, tolerance, "to_power_basis cubic: c[2]=0");
        test.expect_near(power[3][0], 1.0, tolerance, "to_power_basis cubic: c[3]=1");
    }

    // ========================================================================
    // Additional scipy tests (from scipy/interpolate/tests/test_interpolate.py)
    // ========================================================================

    // Test 123: scipy test_periodic with derivative evaluation
    {
        // scipy: x = [0, 1, 3], c = [[3, 0], [0, 0], [0, 2]]
        // Interval 0: [0,1], quadratic with c=[3,0,0], f(t) = 3*(1-t)^2
        // Interval 1: [1,3], quadratic with c=[0,0,2], f(t) = 2*t^2
        std::vector<std::vector<double>> c = {{3, 0}, {0, 0}, {0, 2}};
        std::vector<double> x = {0, 1, 3};
        BPoly bp(c, x, ExtrapolateMode::Periodic);

        // Period = 3 - 0 = 3
        // At x=3.4: wraps to x=0.4, interval 0, t=0.4
        // f(0.4) = 3*(1-0.4)^2 = 3*0.36 = 1.08
        test.expect_near(bp(3.4), 3 * 0.6 * 0.6, tolerance, "scipy periodic: bp(3.4)");

        // At x=-1.3: wraps to x=1.7 (period=3, -1.3+3=1.7), interval 1, t=(1.7-1)/2=0.35
        // f(0.35) = 2*0.35^2 = 0.245
        test.expect_near(bp(-1.3), 2 * 0.35 * 0.35, tolerance, "scipy periodic: bp(-1.3)");

        // Derivative evaluation in periodic mode
        // At x=3.4 (wraps to 0.4), d/dx[3*(1-t)^2] = -6*(1-t)/h = -6*0.6/1 = -3.6
        test.expect_near(bp(3.4, 1), -6 * 0.6, tolerance, "scipy periodic: bp(3.4, 1)");

        // At x=-1.3 (wraps to 1.7), d/dx[2*t^2] = 4*t/h = 4*0.35/2 = 0.7
        test.expect_near(bp(-1.3, 1), 0.7, tolerance, "scipy periodic: bp(-1.3, 1)");
    }

    // Test 124: scipy test_make_poly_2c - f'(1)=3 case
    {
        // ya=[2], yb=[1, 3] -> f(0)=2, f(1)=1, f'(1)=3
        // Expected: c = [2., -0.5, 1.]
        BPoly bp = BPoly::from_derivatives({0, 1}, {{2}, {1, 3}});
        auto& c = bp.coefficients();
        test.expect_near(c[0][0], 2.0, tolerance, "scipy make_poly_2c: c[0]=2");
        test.expect_near(c[1][0], -0.5, tolerance, "scipy make_poly_2c: c[1]=-0.5");
        test.expect_near(c[2][0], 1.0, tolerance, "scipy make_poly_2c: c[2]=1");
    }

    // Test 125: scipy test_make_poly_3b - f'(1)=2, f''(1)=3
    {
        // ya=[1], yb=[4, 2, 3] -> f(0)=1, f(1)=4, f'(1)=2, f''(1)=3
        // Expected: c = [1., 19./6, 10./3, 4.]
        BPoly bp = BPoly::from_derivatives({0, 1}, {{1}, {4, 2, 3}});
        auto& c = bp.coefficients();
        test.expect_near(c[0][0], 1.0, tolerance, "scipy make_poly_3b: c[0]=1");
        test.expect_near(c[1][0], 19.0/6.0, tolerance, "scipy make_poly_3b: c[1]=19/6");
        test.expect_near(c[2][0], 10.0/3.0, tolerance, "scipy make_poly_3b: c[2]=10/3");
        test.expect_near(c[3][0], 4.0, tolerance, "scipy make_poly_3b: c[3]=4");
    }

    // Test 126: scipy test_make_poly_3c - f'(0)=2, f'(1)=3
    {
        // ya=[1, 2], yb=[4, 3] -> f(0)=1, f'(0)=2, f(1)=4, f'(1)=3
        // Expected: c = [1., 5./3, 3., 4.]
        BPoly bp = BPoly::from_derivatives({0, 1}, {{1, 2}, {4, 3}});
        auto& c = bp.coefficients();
        test.expect_near(c[0][0], 1.0, tolerance, "scipy make_poly_3c: c[0]=1");
        test.expect_near(c[1][0], 5.0/3.0, tolerance, "scipy make_poly_3c: c[1]=5/3");
        test.expect_near(c[2][0], 3.0, tolerance, "scipy make_poly_3c: c[2]=3");
        test.expect_near(c[3][0], 4.0, tolerance, "scipy make_poly_3c: c[3]=4");
    }

    // Test 127: scipy test_antiderivative_simple - multi-interval with known integral
    {
        // scipy: x = [0, 1, 3], c = [[0, 0], [1, 1]]
        // f(x) = x on [0,1], f(x) = (x-1) on [1,3] (both linear with slope 1)
        // Actually: interval 0: t=(x-0)/1=x, f(t)=t -> f(x)=x
        //          interval 1: t=(x-1)/2, f(t)=t -> f(x)=(x-1)/2
        std::vector<std::vector<double>> c = {{0, 0}, {1, 1}};
        std::vector<double> x_breaks = {0, 1, 3};
        BPoly bp(c, x_breaks);
        BPoly bi = bp.antiderivative();

        // scipy formula: for xx<1: xx^2/2, for xx>=1: 0.5*xx*(xx/2 - 1) + 3/4
        // At x=0: 0
        test.expect_near(bi(0.0), 0.0, tolerance, "scipy antideriv_simple: bi(0)");
        // At x=0.5: 0.5^2/2 = 0.125
        test.expect_near(bi(0.5), 0.125, tolerance, "scipy antideriv_simple: bi(0.5)");
        // At x=1: 1^2/2 = 0.5
        test.expect_near(bi(1.0), 0.5, tolerance, "scipy antideriv_simple: bi(1)");
        // At x=2: 0.5*2*(2/2-1) + 3/4 = 0.5*2*0 + 0.75 = 0.75
        test.expect_near(bi(2.0), 0.75, tolerance, "scipy antideriv_simple: bi(2)");
        // At x=3: 0.5*3*(3/2-1) + 3/4 = 0.5*3*0.5 + 0.75 = 0.75 + 0.75 = 1.5
        test.expect_near(bi(3.0), 1.5, tolerance, "scipy antideriv_simple: bi(3)");
    }

    // Test 128: scipy test_derivative_ppoly - Bernstein/power basis derivative consistency
    {
        // Create random-ish BPoly, convert to power basis, differentiate both, compare
        std::vector<std::vector<double>> c = {{1}, {2}, {-1}, {3}, {0.5}};  // degree 4
        std::vector<double> x = {0, 1};
        BPoly bp(c, x);

        // Differentiate k times and compare with power basis evaluation
        bool all_match = true;
        for (int d = 1; d <= 4; ++d) {
            BPoly bp_deriv = bp.derivative(d);
            auto power = bp_deriv.to_power_basis();

            // Evaluate power basis at test points
            auto eval_power = [&](double t) {
                double result = 0.0;
                double t_power = 1.0;
                for (size_t k = 0; k < power.size(); ++k) {
                    result += power[k][0] * t_power;
                    t_power *= t;
                }
                return result;
            };

            for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
                double bp_val = bp_deriv(t);
                double power_val = eval_power(t);
                if (std::abs(bp_val - power_val) > tolerance) {
                    test.fail("derivative_ppoly d=" + std::to_string(d) + " at t=" + std::to_string(t));
                    all_match = false;
                }
            }
        }
        if (all_match) {
            test.pass("scipy derivative_ppoly consistency");
        }
    }

    // Test 129: scipy test_raise_degree - degree elevation preserves polynomial
    {
        // Create a polynomial and verify degree elevation works
        // Linear f(t)=t elevated to degree 5 should still evaluate to t
        const int original_degree = 1;
        const int target_degree = 5;

        // Original linear coefficients
        std::vector<std::vector<double>> linear_c = {{0}, {1}};
        BPoly linear(linear_c, {0, 1});

        // Manually create elevated coefficients: c[i] = i/n for linear t
        std::vector<std::vector<double>> elevated_c(target_degree + 1, std::vector<double>(1));
        for (int i = 0; i <= target_degree; ++i) {
            elevated_c[i][0] = static_cast<double>(i) / target_degree;
        }
        BPoly elevated(elevated_c, {0, 1});

        // Both should evaluate to same values
        bool all_match = true;
        for (double t : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
            if (std::abs(linear(t) - elevated(t)) > tolerance) {
                test.fail("raise_degree at t=" + std::to_string(t));
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("scipy test_raise_degree");
        }
    }

    // Test 130: scipy test_coords_order - repeated breakpoints validation
    {
        test.expect_throw([]() {
            std::vector<double> xi = {0, 0, 1};  // Repeated value
            std::vector<std::vector<double>> yi = {{0}, {0}, {0}};
            BPoly::from_derivatives(xi, yi);
        }, "scipy coords_order: repeated breakpoints rejected");
    }

    // Test 131: scipy test_orders_local - verify continuity with local orders
    {
        // Create polynomial with different orders at each interior point
        // Test that the specified number of derivatives are continuous
        int m = 5;
        std::vector<double> xi = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
        std::vector<std::vector<double>> yi(m + 1);

        // Fill with deterministic values (6 derivatives at each point)
        for (int i = 0; i <= m; ++i) {
            yi[i].resize(6);
            for (int j = 0; j < 6; ++j) {
                yi[i][j] = std::cos(i * 0.5 + j * 0.3);
            }
        }

        // Use orders = [2, 3, 2, 3, 2, 3] (different at each point)
        std::vector<int> orders = {2, 3, 2, 3, 2, 3};
        BPoly pp = BPoly::from_derivatives(xi, yi, orders);

        // At interior point xi[2]=1.0, orders[2]=2 means f, f', f'' should be continuous
        // Check continuity of f and f'
        bool all_continuous = true;
        double eps = 1e-10;
        double left = pp(xi[2] - eps);
        double right = pp(xi[2] + eps);
        if (std::abs(left - right) > 1e-8) {
            test.fail("orders_local: f discontinuous at xi[2]");
            all_continuous = false;
        }

        double left_d = pp(xi[2] - eps, 1);
        double right_d = pp(xi[2] + eps, 1);
        if (std::abs(left_d - right_d) > 1e-8) {
            test.fail("orders_local: f' discontinuous at xi[2]");
            all_continuous = false;
        }

        if (all_continuous) {
            test.pass("scipy test_orders_local");
        }
    }

    // Test 132: scipy test_integrate_periodic - fractional periods
    {
        // f(x) = x on [0,2] with periodic mode
        // Period = 2, integral over one period = 2
        std::vector<std::vector<double>> c = {{0}, {2}};  // f(t) = 2t, so f(x) = x
        BPoly bp(c, {0, 2}, ExtrapolateMode::Periodic);

        // Integral from 0.5 to 2.5 (one full period, shifted)
        // Should equal integral from 0 to 2 = 2
        test.expect_near(bp.integrate(0.5, 2.5), 2.0, tolerance, "scipy periodic integrate shifted period");

        // Integral from -0.5 to 0.5 (crossing lower boundary)
        // Wraps: [-0.5,0) maps to [1.5,2), [0,0.5) stays
        // Integral of x from 1.5 to 2 = 2^2/2 - 1.5^2/2 = 2 - 1.125 = 0.875
        // Integral of x from 0 to 0.5 = 0.5^2/2 = 0.125
        // Total = 1.0
        test.expect_near(bp.integrate(-0.5, 0.5), 1.0, tolerance, "scipy periodic integrate crossing boundary");

        // Integral from 1 to 5 (two full periods)
        // = 2 * (integral of x from 0 to 2) = 2 * 2 = 4
        test.expect_near(bp.integrate(1, 5), 4.0, tolerance, "scipy periodic integrate two periods");
    }

    // Test 133: scipy test_descending random data consistency
    {
        // Create ascending polynomial with specific coefficients
        // Then create equivalent descending, verify they match
        const int degree = 3;
        std::vector<std::vector<double>> ca = {{1}, {-0.5}, {2}, {0.3}};  // Ascending coeffs
        std::vector<double> xa = {0, 10};  // Ascending breaks

        // For descending: reverse coefficients and breaks
        std::vector<std::vector<double>> cd = {{0.3}, {2}, {-0.5}, {1}};  // Reversed coeffs
        std::vector<double> xd = {10, 0};  // Reversed breaks

        BPoly pa(ca, xa);
        BPoly pd(cd, xd);

        // Test at multiple points - they should evaluate to the same values
        bool all_match = true;
        for (double x : {0.0, 2.5, 5.0, 7.5, 10.0}) {
            double va = pa(x);
            double vd = pd(x);
            if (std::abs(va - vd) > 1e-10) {
                test.fail("descending consistency at x=" + std::to_string(x));
                all_match = false;
            }
        }

        // Also test first derivative
        for (double x : {0.0, 2.5, 5.0, 7.5, 10.0}) {
            double va = pa(x, 1);
            double vd = pd(x, 1);
            if (std::abs(va - vd) > 1e-10) {
                test.fail("descending derivative at x=" + std::to_string(x));
                all_match = false;
            }
        }

        if (all_match) {
            test.pass("scipy test_descending consistency");
        }
    }

    // Test 134: scipy test_make_poly_12 - 6th order derivatives matching
    {
        // Test with 6 derivatives at each endpoint, verify all match
        std::vector<double> ya = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> yb = {1.0, -1.0, 0.5, -0.5, 0.25, -0.25};

        BPoly pp = BPoly::from_derivatives({0, 1}, {ya, yb});

        // Verify f^(j)(0) = ya[j] and f^(j)(1) = yb[j] for j=0..5
        // Use operator()(x, nu) to evaluate j-th derivative at points
        // Note: Higher derivatives accumulate numerical error, so use relaxed tolerance
        for (int j = 0; j < 6; ++j) {
            double val_at_0 = pp(0.0, j);
            double val_at_1 = pp(1.0, j);
            // Tolerance increases with derivative order due to numerical accumulation
            double deriv_tol = tolerance * std::pow(10.0, j);
            test.expect_near(val_at_0, ya[j], deriv_tol,
                           "make_poly_12: f^(" + std::to_string(j) + ")(0)");
            test.expect_near(val_at_1, yb[j], deriv_tol,
                           "make_poly_12: f^(" + std::to_string(j) + ")(1)");
        }
    }

    // Test 135: scipy test_antider_continuous - antiderivative continuity at breakpoints
    {
        // Multi-interval polynomial, verify antiderivative is C0 continuous
        std::vector<std::vector<double>> c = {{1, 2, 3}, {0, 1, -1}, {2, 0, 1}};
        std::vector<double> x = {0, 1, 2, 3};
        BPoly bp(c, x);
        BPoly bi = bp.antiderivative();

        // Check continuity at each interior breakpoint
        double eps = 1e-10;
        bool continuous = true;
        for (size_t i = 1; i < x.size() - 1; ++i) {
            double left = bi(x[i] - eps);
            double right = bi(x[i] + eps);
            if (std::abs(left - right) > 1e-8) {
                test.fail("antider_continuous at x=" + std::to_string(x[i]));
                continuous = false;
            }
        }
        if (continuous) {
            test.pass("scipy test_antider_continuous");
        }
    }

    // Test 136: scipy test_der_antider with higher orders
    {
        // Test derivative(antiderivative(n)) = original for n > 1
        BPoly bp = BPoly::from_derivatives({0, 1}, {{1, 2, -1}, {3, 0, 2}});

        for (int n = 1; n <= 3; ++n) {
            BPoly antideriv = bp.antiderivative(n);
            BPoly recovered = antideriv.derivative(n);

            bool matches = true;
            for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
                if (std::abs(bp(x) - recovered(x)) > tolerance) {
                    test.fail("der_antider n=" + std::to_string(n) + " at x=" + std::to_string(x));
                    matches = false;
                }
            }
            if (matches) {
                test.pass("scipy der_antider n=" + std::to_string(n));
            }
        }
    }

    // Test 137: scipy multi-shape evaluation (1D version - we don't support ND)
    {
        // Test evaluation at array of points with multi-interval polynomial
        // Verified against scipy.interpolate.BPoly (scipy 1.16.3)
        std::vector<std::vector<double>> c = {{3, 0}, {0, 0}, {0, 2}};
        std::vector<double> x = {0, 1, 3};
        BPoly bp(c, x);

        // Evaluate at array of points spanning both intervals
        std::vector<double> xx = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
        std::vector<double> results = bp(xx);

        // Expected values from scipy (NOT from our implementation - avoids circular testing)
        // Generated by: python scripts/generate_test_values.py
        std::vector<double> scipy_expected = {3.0, 0.75, 0.0, 0.125, 0.5, 1.125, 2.0};

        // Verify each result against scipy reference
        bool all_correct = true;
        for (size_t i = 0; i < xx.size(); ++i) {
            if (std::abs(results[i] - scipy_expected[i]) > tolerance) {
                test.fail("multi-shape at x=" + std::to_string(xx[i]) +
                         " expected=" + std::to_string(scipy_expected[i]) +
                         " got=" + std::to_string(results[i]));
                all_correct = false;
            }
        }
        if (all_correct) {
            test.pass("scipy multi-shape evaluation (verified against scipy)");
        }
    }

    // Test 138: scipy test_gh_5430 - integer types for orders parameter
    {
        // Test that orders parameter works with different integer representations
        // (In C++ this is less of an issue, but test edge cases)

        // orders = 1 (single value)
        BPoly p1 = BPoly::from_derivatives({0, 1}, {{0, 1}, {0, -1}}, {1});
        test.expect_near(p1(0), 0.0, tolerance, "gh_5430: orders=1 f(0)");

        // orders = {1, 1} (per-point)
        BPoly p2 = BPoly::from_derivatives({0, 1}, {{0, 1}, {0, -1}}, {1, 1});
        test.expect_near(p2(0), 0.0, tolerance, "gh_5430: orders={1,1} f(0)");

        // Verify both produce same result
        test.expect_near(p1(0.5), p2(0.5), tolerance, "gh_5430: both orders forms match");
    }

    // Test 139: scipy extrapolate attribute behavior
    {
        // Test extrapolate mode is correctly used for evaluation
        std::vector<std::vector<double>> c = {{0}, {1}};
        std::vector<double> x = {0, 1};

        // Default should be Extrapolate
        BPoly bp_default(c, x);
        test.expect_near(bp_default(1.5), 1.5, tolerance, "extrap_attr: default extrapolates");

        // Explicit Extrapolate
        BPoly bp_extrap(c, x, ExtrapolateMode::Extrapolate);
        test.expect_near(bp_extrap(1.5), 1.5, tolerance, "extrap_attr: explicit extrapolate");

        // NoExtrapolate should return NaN
        BPoly bp_no(c, x, ExtrapolateMode::NoExtrapolate);
        test.expect_true(std::isnan(bp_no(1.5)), "extrap_attr: NoExtrapolate returns NaN");

        // Periodic should wrap
        BPoly bp_periodic(c, x, ExtrapolateMode::Periodic);
        test.expect_near(bp_periodic(1.5), 0.5, tolerance, "extrap_attr: Periodic wraps");
    }

    // Test 140: scipy test_antider_neg - negative derivative = antiderivative
    {
        BPoly bp({{1}, {2}, {3}}, {0, 1});

        // derivative(-1) should equal antiderivative(1)
        BPoly d_neg1 = bp.derivative(-1);
        BPoly antid1 = bp.antiderivative(1);

        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            test.expect_near(d_neg1(x), antid1(x), tolerance,
                           "antider_neg: d(-1)==antid(1) at x=" + std::to_string(x));
        }

        // derivative(-2) should equal antiderivative(2)
        BPoly d_neg2 = bp.derivative(-2);
        BPoly antid2 = bp.antiderivative(2);

        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            test.expect_near(d_neg2(x), antid2(x), tolerance,
                           "antider_neg: d(-2)==antid(2) at x=" + std::to_string(x));
        }
    }

    // ========================================================================
    // Independent verification tests using finite differences
    // ========================================================================

    // Test 141: Verify derivatives using finite differences
    {
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});
        BPoly deriv = bp.derivative();

        bool all_match = true;
        for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            double analytical = deriv(x);
            double numerical = finite_diff_derivative(bp, x);
            if (std::abs(analytical - numerical) > 1e-5) {
                test.fail("finite diff derivative at x=" + std::to_string(x) +
                         " analytical=" + std::to_string(analytical) +
                         " numerical=" + std::to_string(numerical));
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("derivatives match finite differences");
        }
    }

    // Test 142: Verify second derivatives using finite differences
    {
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1, 0}, {1, -1, 0}});
        BPoly deriv2 = bp.derivative(2);

        bool all_match = true;
        for (double x : {0.2, 0.4, 0.6, 0.8}) {
            double analytical = deriv2(x);
            double numerical = finite_diff_second_derivative(bp, x);
            if (std::abs(analytical - numerical) > 1e-3) {  // Looser tolerance for 2nd derivative
                test.fail("finite diff 2nd deriv at x=" + std::to_string(x));
                all_match = false;
            }
        }
        if (all_match) {
            test.pass("second derivatives match finite differences");
        }
    }

    // Test 143: Verify integration using numerical integration
    {
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}});

        // Test integrate() against numerical_integrate()
        double analytical = bp.integrate(0.2, 0.8);
        double numerical = numerical_integrate(bp, 0.2, 0.8);
        test.expect_near(analytical, numerical, 1e-6, "integrate vs numerical integration");
    }

    // ========================================================================
    // Corner cases from corner_case_test_data.json
    // ========================================================================

    // Test 144: single_interval_degree5
    {
        std::vector<std::vector<double>> c = {{1}, {0}, {0}, {0}, {0}, {2}};
        BPoly bp(c, {0, 1});
        test.expect_near(bp(0.0), 1.0, tolerance, "corner degree5 at 0");
        test.expect_near(bp(0.25), 0.2392578125, tolerance, "corner degree5 at 0.25");
        test.expect_near(bp(0.5), 0.09375, tolerance, "corner degree5 at 0.5");
        test.expect_near(bp(0.75), 0.4755859375, tolerance, "corner degree5 at 0.75");
        test.expect_near(bp(1.0), 2.0, tolerance, "corner degree5 at 1");
    }

    // Test 145: two_intervals_degree2
    {
        std::vector<std::vector<double>> c = {{0, 1}, {1, 0}, {0, 0}};
        BPoly bp(c, {0, 0.5, 1});
        test.expect_near(bp(0.0), 0.0, tolerance, "corner two_int at 0");
        test.expect_near(bp(0.25), 0.5, tolerance, "corner two_int at 0.25");
        test.expect_near(bp(0.5), 1.0, tolerance, "corner two_int at 0.5");
        test.expect_near(bp(0.75), 0.25, tolerance, "corner two_int at 0.75");
        test.expect_near(bp(1.0), 0.0, tolerance, "corner two_int at 1");
    }

    // Test 146: many_intervals (10 linear pieces)
    {
        std::vector<std::vector<double>> c = {
            {0,1,2,3,4,5,6,7,8,9}, {1,2,3,4,5,6,7,8,9,10}
        };
        std::vector<double> breaks = {0,1,2,3,4,5,6,7,8,9,10};
        BPoly bp(c, breaks);
        test.expect_near(bp(0.5), 0.5, tolerance, "corner many_int at 0.5");
        test.expect_near(bp(5.5), 5.5, tolerance, "corner many_int at 5.5");
        test.expect_near(bp(9.5), 9.5, tolerance, "corner many_int at 9.5");
    }

    // Test 147: large_coefficients (1e10 scale)
    {
        std::vector<std::vector<double>> c = {{1e10}, {2e10}};
        BPoly bp(c, {0, 1});
        test.expect_near(bp(0.0), 1e10, 1.0, "corner large_coeff at 0");
        test.expect_near(bp(0.5), 1.5e10, 1e5, "corner large_coeff at 0.5");
        test.expect_near(bp(1.0), 2e10, 1.0, "corner large_coeff at 1");
    }

    // Test 148: small_coefficients (1e-15 scale)
    {
        std::vector<std::vector<double>> c = {{1e-15}, {2e-15}};
        BPoly bp(c, {0, 1});
        test.expect_near(bp(0.0), 1e-15, 1e-25, "corner small_coeff at 0");
        test.expect_near(bp(0.5), 1.5e-15, 1e-25, "corner small_coeff at 0.5");
        test.expect_near(bp(1.0), 2e-15, 1e-25, "corner small_coeff at 1");
    }

    // Test 149: linear extrapolation (far outside domain)
    {
        std::vector<std::vector<double>> c = {{0}, {1}};  // f(x) = x
        BPoly bp(c, {0, 1}, ExtrapolateMode::Extrapolate);
        test.expect_near(bp(10.0), 10.0, tolerance, "corner far_extrap at 10");
        test.expect_near(bp(-10.0), -10.0, tolerance, "corner far_extrap at -10");
        test.expect_near(bp(100.0), 100.0, tolerance, "corner far_extrap at 100");
    }

    // Test 150: near boundary evaluation (epsilon-close to breakpoints)
    {
        std::vector<std::vector<double>> c = {{0, 1}, {1, 2}};
        std::vector<double> breaks = {0, 1, 2};
        BPoly bp(c, breaks);

        double eps = 1e-14;
        // Should not crash or give NaN
        test.expect_true(std::isfinite(bp(1.0 - eps)), "corner near_boundary 1-eps finite");
        test.expect_true(std::isfinite(bp(1.0 + eps)), "corner near_boundary 1+eps finite");
        // Values should be close to boundary value
        test.expect_near(bp(1.0 - eps), 1.0, 1e-10, "corner near_boundary 1-eps value");
        test.expect_near(bp(1.0 + eps), 1.0, 1e-10, "corner near_boundary 1+eps value");
    }

    // Test 151: from_derivatives with only function values (no derivatives)
    {
        BPoly bp = BPoly::from_derivatives({0, 1, 2}, {{0}, {1}, {0}});
        test.expect_near(bp(0.0), 0.0, tolerance, "corner from_deriv_values f(0)");
        test.expect_near(bp(1.0), 1.0, tolerance, "corner from_deriv_values f(1)");
        test.expect_near(bp(2.0), 0.0, tolerance, "corner from_deriv_values f(2)");
        test.expect_near(bp(0.5), 0.5, tolerance, "corner from_deriv_values f(0.5)");
    }

    // Test 152: from_derivatives with high-order derivatives at one point
    {
        // f(0)=0, f'(0)=1, f''(0)=0, f'''(0)=0 and f(1)=1
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1, 0, 0}, {1}});
        test.expect_near(bp(0.0), 0.0, tolerance, "corner high_order f(0)");
        test.expect_near(bp(1.0), 1.0, tolerance, "corner high_order f(1)");
        // Derivative at 0 should be 1
        double h = 1e-8;
        double deriv_at_0 = (bp(h) - bp(0)) / h;
        test.expect_near(deriv_at_0, 1.0, 1e-5, "corner high_order f'(0) approx");
    }

    // ========================================================================
    // Property-based tests (mathematical invariants that must always hold)
    // These test mathematical identities rather than specific values
    // ========================================================================

    // Test 153: Property - integrate(a,b) == antiderivative(b) - antiderivative(a)
    {
        // This property must hold for ANY polynomial
        std::vector<std::pair<std::string, BPoly>> test_polys;
        test_polys.push_back({"linear", BPoly({{0}, {1}}, {0, 1})});
        test_polys.push_back({"quadratic", BPoly({{0}, {0}, {1}}, {0, 1})});
        test_polys.push_back({"cubic", BPoly({{1}, {2}, {3}, {4}}, {0, 1})});
        test_polys.push_back({"multi_interval", BPoly({{1, 2}, {3, 4}}, {0, 1, 2})});
        test_polys.push_back({"hermite", BPoly::from_derivatives({0, 1}, {{0, 1}, {1, -1}})});

        bool all_pass = true;
        for (const auto& [name, bp] : test_polys) {
            BPoly anti = bp.antiderivative();

            // Test multiple intervals
            std::vector<std::pair<double, double>> intervals;
            if (name == "multi_interval") {
                intervals = {{0.0, 0.5}, {0.5, 1.5}, {0.0, 2.0}};
            } else {
                intervals = {{0.0, 0.5}, {0.25, 0.75}, {0.0, 1.0}};
            }

            for (const auto& [a, b] : intervals) {
                double via_integrate = bp.integrate(a, b);
                double via_antideriv = anti(b) - anti(a);
                double error = std::abs(via_integrate - via_antideriv);

                if (error > 1e-10) {
                    test.fail("property integrate==antideriv_diff " + name +
                             " [" + std::to_string(a) + "," + std::to_string(b) + "]");
                    all_pass = false;
                }
            }
        }
        if (all_pass) {
            test.pass("property: integrate(a,b) == antiderivative(b) - antiderivative(a)");
        }
    }

    // Test 154: Property - derivative(antiderivative(f)) == f
    {
        std::vector<std::pair<std::string, BPoly>> test_polys;
        test_polys.push_back({"constant", BPoly({{5}}, {0, 1})});
        test_polys.push_back({"linear", BPoly({{1}, {3}}, {0, 1})});
        test_polys.push_back({"quadratic", BPoly({{1}, {2}, {3}}, {0, 1})});
        test_polys.push_back({"multi", BPoly({{1, 2, 3}, {4, 5, 6}}, {0, 1, 2, 3})});

        bool all_pass = true;
        for (const auto& [name, bp] : test_polys) {
            BPoly recovered = bp.antiderivative().derivative();

            double domain_end = bp.breakpoints().back();
            std::vector<double> test_pts;
            for (int i = 0; i <= 10; ++i) {
                test_pts.push_back(i * domain_end / 10.0);
            }

            for (double x : test_pts) {
                double error = std::abs(bp(x) - recovered(x));
                if (error > tolerance) {
                    test.fail("property deriv(antideriv)==original " + name +
                             " at x=" + std::to_string(x));
                    all_pass = false;
                }
            }
        }
        if (all_pass) {
            test.pass("property: derivative(antiderivative(f)) == f");
        }
    }

    // Test 155: Property - Bernstein partition of unity (all-1 coeffs -> 1)
    {
        bool all_pass = true;
        for (int degree : {1, 2, 5, 10, 20, 50}) {
            std::vector<std::vector<double>> coeffs(degree + 1, std::vector<double>(1, 1.0));
            BPoly bp(coeffs, {0, 1});

            for (double t : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
                double error = std::abs(bp(t) - 1.0);
                if (error > 1e-10) {
                    test.fail("property partition_of_unity degree=" + std::to_string(degree) +
                             " at t=" + std::to_string(t));
                    all_pass = false;
                }
            }
        }
        if (all_pass) {
            test.pass("property: Bernstein partition of unity");
        }
    }

    // Test 156: Property - from_derivatives matches endpoint constraints
    {
        struct TestCase {
            std::string name;
            std::vector<double> xi;
            std::vector<std::vector<double>> yi;
        };

        std::vector<TestCase> cases = {
            {"linear", {0, 1}, {{0}, {1}}},
            {"cubic_hermite", {0, 1}, {{0, 1}, {1, -1}}},
            {"quintic", {0, 1}, {{0, 1, 0}, {1, -1, 0}}},
            {"asymmetric", {0, 1}, {{0, 1, 2}, {1}}},
            {"three_point", {0, 1, 2}, {{0, 1}, {1}, {0, -1}}},
        };

        bool all_pass = true;
        for (const auto& tc : cases) {
            BPoly bp = BPoly::from_derivatives(tc.xi, tc.yi);

            for (size_t i = 0; i < tc.xi.size(); ++i) {
                double x = tc.xi[i];
                const auto& y_derivs = tc.yi[i];

                for (size_t order = 0; order < y_derivs.size(); ++order) {
                    double expected = y_derivs[order];
                    double actual = bp(x, static_cast<int>(order));
                    double deriv_tol = tolerance * std::pow(10.0, static_cast<double>(order));

                    if (std::abs(actual - expected) > deriv_tol) {
                        test.fail("property from_deriv_endpoints " + tc.name +
                                 " order=" + std::to_string(order) +
                                 " at x=" + std::to_string(x));
                        all_pass = false;
                    }
                }
            }
        }
        if (all_pass) {
            test.pass("property: from_derivatives matches endpoint constraints");
        }
    }

    // Test 157: Property - derivative order composition: d^m(d^n(f)) == d^(m+n)(f)
    {
        BPoly bp = BPoly::from_derivatives({0, 1}, {{0, 1, 2, 0}, {1, -1, 0, 0}});

        bool all_pass = true;
        for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
            // d^1(d^1(f)) == d^2(f)
            double via_composition = bp.derivative(1).derivative(1)(x);
            double via_direct = bp.derivative(2)(x);
            if (std::abs(via_composition - via_direct) > tolerance) {
                test.fail("property deriv_composition d^1(d^1) at x=" + std::to_string(x));
                all_pass = false;
            }

            // d^2(d^1(f)) == d^3(f)
            via_composition = bp.derivative(2).derivative(1)(x);
            via_direct = bp.derivative(3)(x);
            if (std::abs(via_composition - via_direct) > tolerance) {
                test.fail("property deriv_composition d^2(d^1) at x=" + std::to_string(x));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("property: derivative order composition d^m(d^n(f)) == d^(m+n)(f)");
        }
    }

    // Test 158: Property - antiderivative order composition: A^m(A^n(f)) == A^(m+n)(f)
    {
        BPoly bp({{1}, {2}, {3}}, {0, 1});

        bool all_pass = true;
        for (double x : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            // A^1(A^1(f)) == A^2(f)
            double via_composition = bp.antiderivative(1).antiderivative(1)(x);
            double via_direct = bp.antiderivative(2)(x);
            if (std::abs(via_composition - via_direct) > tolerance) {
                test.fail("property antideriv_composition A^1(A^1) at x=" + std::to_string(x));
                all_pass = false;
            }

            // A^2(A^1(f)) == A^3(f)
            via_composition = bp.antiderivative(2).antiderivative(1)(x);
            via_direct = bp.antiderivative(3)(x);
            if (std::abs(via_composition - via_direct) > tolerance) {
                test.fail("property antideriv_composition A^2(A^1) at x=" + std::to_string(x));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("property: antiderivative order composition A^m(A^n(f)) == A^(m+n)(f)");
        }
    }

    // Test 159: Property - integrate is additive: integrate(a,c) == integrate(a,b) + integrate(b,c)
    {
        BPoly bp = BPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1}, {0, -1}});

        bool all_pass = true;
        std::vector<std::tuple<double, double, double>> test_points = {
            {0.0, 0.5, 1.0},
            {0.0, 1.0, 2.0},
            {0.25, 0.75, 1.5},
        };

        for (const auto& [a, b, c] : test_points) {
            double left = bp.integrate(a, c);
            double right = bp.integrate(a, b) + bp.integrate(b, c);
            if (std::abs(left - right) > tolerance) {
                test.fail("property integrate_additive a=" + std::to_string(a) +
                         " b=" + std::to_string(b) + " c=" + std::to_string(c));
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("property: integrate is additive");
        }
    }

    // Test 160: Property - integrate reversal: integrate(a,b) == -integrate(b,a)
    {
        BPoly bp({{1}, {2}, {3}}, {0, 1});

        bool all_pass = true;
        std::vector<std::pair<double, double>> intervals = {
            {0.0, 0.5}, {0.25, 0.75}, {0.0, 1.0}, {-0.5, 1.5}
        };

        for (const auto& [a, b] : intervals) {
            double forward = bp.integrate(a, b);
            double reverse = bp.integrate(b, a);
            if (std::abs(forward + reverse) > tolerance) {
                test.fail("property integrate_reversal [" + std::to_string(a) +
                         "," + std::to_string(b) + "]");
                all_pass = false;
            }
        }
        if (all_pass) {
            test.pass("property: integrate(a,b) == -integrate(b,a)");
        }
    }

    // ========================================================================
    // Extrapolation order tests (controlled extrapolation via Taylor expansion)
    // ========================================================================

    // Test 161: Constant extrapolation (order=0) - clamp to boundary value
    {
        // Quadratic polynomial: f(x) = x^2 on [0,1]
        BPoly bp({{0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 0, 0);

        // At boundaries
        test.expect_near(bp(0.0), 0.0, tolerance, "extrap_order=0: f(0)");
        test.expect_near(bp(1.0), 1.0, tolerance, "extrap_order=0: f(1)");

        // Extrapolation should clamp to boundary values
        test.expect_near(bp(-1.0), 0.0, tolerance, "extrap_order=0: f(-1) clamped to f(0)");
        test.expect_near(bp(2.0), 1.0, tolerance, "extrap_order=0: f(2) clamped to f(1)");
        test.expect_near(bp(-10.0), 0.0, tolerance, "extrap_order=0: f(-10) clamped to f(0)");
        test.expect_near(bp(10.0), 1.0, tolerance, "extrap_order=0: f(10) clamped to f(1)");
    }

    // Test 162: Linear extrapolation (order=1) - tangent line at boundary
    {
        // Quadratic polynomial: f(x) = x^2 on [0,1]
        // f'(0) = 0, f'(1) = 2
        BPoly bp({{0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 1, 1);

        // At boundaries
        test.expect_near(bp(0.0), 0.0, tolerance, "extrap_order=1: f(0)");
        test.expect_near(bp(1.0), 1.0, tolerance, "extrap_order=1: f(1)");

        // Left extrapolation: f(x) ≈ f(0) + f'(0)*(x-0) = 0 + 0*x = 0
        test.expect_near(bp(-1.0), 0.0, tolerance, "extrap_order=1: f(-1) linear from left");
        test.expect_near(bp(-5.0), 0.0, tolerance, "extrap_order=1: f(-5) linear from left");

        // Right extrapolation: f(x) ≈ f(1) + f'(1)*(x-1) = 1 + 2*(x-1)
        test.expect_near(bp(2.0), 1.0 + 2.0*(2.0-1.0), tolerance, "extrap_order=1: f(2) linear from right");
        test.expect_near(bp(3.0), 1.0 + 2.0*(3.0-1.0), tolerance, "extrap_order=1: f(3) linear from right");
    }

    // Test 163: Quadratic extrapolation (order=2)
    {
        // Cubic polynomial: f(x) = x^3 on [0,1]
        // Bernstein coeffs for x^3: [0, 0, 0, 1]
        BPoly bp({{0}, {0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 2, 2);

        // f(1) = 1, f'(1) = 3, f''(1) = 6
        // Quadratic extrapolation from right: f(1) + f'(1)*(x-1) + f''(1)/2*(x-1)^2
        // At x=2: 1 + 3*1 + 6/2*1 = 1 + 3 + 3 = 7
        test.expect_near(bp(2.0), 7.0, tolerance, "extrap_order=2: f(2) quadratic from right");

        // f(0) = 0, f'(0) = 0, f''(0) = 0
        // Quadratic extrapolation from left: 0 + 0*(x-0) + 0/2*(x)^2 = 0
        test.expect_near(bp(-1.0), 0.0, tolerance, "extrap_order=2: f(-1) quadratic from left");
    }

    // Test 164: Asymmetric extrapolation orders (different left and right)
    {
        // Linear polynomial: f(x) = x on [0,1]
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 0, 1);

        // Left: constant extrapolation (clamp to f(0)=0)
        test.expect_near(bp(-1.0), 0.0, tolerance, "asymmetric extrap: left constant");

        // Right: linear extrapolation (tangent line: f(1) + f'(1)*(x-1) = 1 + 1*(x-1) = x)
        test.expect_near(bp(2.0), 2.0, tolerance, "asymmetric extrap: right linear");
    }

    // Test 165: Order > degree defaults to full polynomial
    {
        // Linear polynomial: f(x) = x
        BPoly bp_full({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, -1, -1);
        BPoly bp_high({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 5, 5);

        // Both should give same (full polynomial) extrapolation
        test.expect_near(bp_full(2.0), bp_high(2.0), tolerance, "order>degree: same as full poly right");
        test.expect_near(bp_full(-1.0), bp_high(-1.0), tolerance, "order>degree: same as full poly left");
    }

    // Test 166: Accessors for extrapolation orders
    {
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 2, 3);
        test.expect_eq(bp.extrapolate_order_left(), 2, "accessor: extrapolate_order_left");
        test.expect_eq(bp.extrapolate_order_right(), 3, "accessor: extrapolate_order_right");

        BPoly bp_default({{0}, {1}}, {0, 1});
        test.expect_eq(bp_default.extrapolate_order_left(), -1, "accessor: default left is -1");
        test.expect_eq(bp_default.extrapolate_order_right(), -1, "accessor: default right is -1");
    }

    // Test 167: Extrapolation order with derivative() - order decreases by 1
    {
        // Start with linear extrapolation (order=1)
        BPoly bp({{0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 2, 2);
        BPoly deriv = bp.derivative();

        // Derivative should have order-1 = 1 (linear extrapolation)
        test.expect_eq(deriv.extrapolate_order_left(), 1, "derivative: order decreases left");
        test.expect_eq(deriv.extrapolate_order_right(), 1, "derivative: order decreases right");

        // Second derivative should have order 0 (constant extrapolation)
        BPoly deriv2 = deriv.derivative();
        test.expect_eq(deriv2.extrapolate_order_left(), 0, "derivative2: order is 0 left");
        test.expect_eq(deriv2.extrapolate_order_right(), 0, "derivative2: order is 0 right");
    }

    // Test 168: Extrapolation order with antiderivative() - order increases by 1
    {
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 1, 1);
        BPoly anti = bp.antiderivative();

        test.expect_eq(anti.extrapolate_order_left(), 2, "antiderivative: order increases left");
        test.expect_eq(anti.extrapolate_order_right(), 2, "antiderivative: order increases right");
    }

    // Test 169: Full polynomial extrapolation (-1) preserved through derivative/antiderivative
    {
        BPoly bp({{0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, -1, -1);
        BPoly deriv = bp.derivative();
        BPoly anti = bp.antiderivative();

        test.expect_eq(deriv.extrapolate_order_left(), -1, "full poly preserved in derivative left");
        test.expect_eq(deriv.extrapolate_order_right(), -1, "full poly preserved in derivative right");
        test.expect_eq(anti.extrapolate_order_left(), -1, "full poly preserved in antiderivative left");
        test.expect_eq(anti.extrapolate_order_right(), -1, "full poly preserved in antiderivative right");
    }

    // Test 170: extend() preserves extrapolation orders
    {
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 1, 2);
        BPoly extended = bp.extend({{1}, {0}}, {1, 2});

        test.expect_eq(extended.extrapolate_order_left(), 1, "extend: preserves order left");
        test.expect_eq(extended.extrapolate_order_right(), 2, "extend: preserves order right");
    }

    // Test 171: Linear extrapolation prevents explosion for high-degree polynomial
    {
        // Degree 10 polynomial that would explode with full extrapolation
        std::vector<std::vector<double>> coeffs(11, std::vector<double>(1, 0.0));
        coeffs[0][0] = 1.0;  // Start at 1
        coeffs[10][0] = 1.0; // End at 1
        for (int i = 1; i < 10; ++i) coeffs[i][0] = 0.5;  // Middle coefficients

        BPoly bp_full(coeffs, {0, 1}, ExtrapolateMode::Extrapolate, -1, -1);
        BPoly bp_linear(coeffs, {0, 1}, ExtrapolateMode::Extrapolate, 1, 1);

        // Far extrapolation with full polynomial explodes
        double full_at_10 = bp_full(10.0);

        // Linear extrapolation stays bounded
        double linear_at_10 = bp_linear(10.0);

        // Full extrapolation magnitude should be much larger than linear
        test.expect_true(std::abs(full_at_10) > std::abs(linear_at_10) * 100,
                        "linear extrap prevents explosion");
    }

    // Test 172: Constant extrapolation with multi-interval polynomial
    {
        BPoly bp({{0, 1}, {1, 0}}, {0, 1, 2}, ExtrapolateMode::Extrapolate, 0, 0);

        // f(0) = 0, f(2) = 0
        test.expect_near(bp(-1.0), 0.0, tolerance, "multi-interval constant extrap left");
        test.expect_near(bp(3.0), 0.0, tolerance, "multi-interval constant extrap right");
    }

    // Test 173: Extrapolation order only applies to Extrapolate mode
    {
        // NoExtrapolate should still return NaN regardless of order
        BPoly bp_no({{0}, {1}}, {0, 1}, ExtrapolateMode::NoExtrapolate, 1, 1);
        test.expect_true(std::isnan(bp_no(-0.5)), "NoExtrapolate ignores order left");
        test.expect_true(std::isnan(bp_no(1.5)), "NoExtrapolate ignores order right");

        // Periodic should still wrap regardless of order
        BPoly bp_per({{0}, {1}}, {0, 1}, ExtrapolateMode::Periodic, 1, 1);
        test.expect_near(bp_per(1.5), 0.5, tolerance, "Periodic ignores order");
    }

    // Test 174: Edge case - order=0 with zero boundary value
    {
        BPoly bp({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 0, 0);
        test.expect_near(bp(-100.0), 0.0, tolerance, "order=0 with zero boundary");
    }

    // Test 175: Taylor expansion correctness for cubic
    {
        // f(x) = x^3 -> f(1)=1, f'(1)=3, f''(1)=6, f'''(1)=6
        // Order 3 extrapolation at x=2:
        // f(1) + f'(1)*1 + f''(1)/2*1 + f'''(1)/6*1 = 1 + 3 + 3 + 1 = 8 = 2^3
        BPoly bp({{0}, {0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 3, 3);
        test.expect_near(bp(2.0), 8.0, tolerance, "Taylor order 3: exact for cubic at x=2");

        // Order 4 should also give exact result (order >= degree+1)
        BPoly bp4({{0}, {0}, {0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate, 4, 4);
        test.expect_near(bp4(2.0), 8.0, tolerance, "Taylor order 4: exact for cubic at x=2");
    }

    // ========================================================================
    // Group 29: Periodic Derivative Evaluation
    // ========================================================================
    std::cout << "\n--- Group 29: Periodic Derivative Evaluation ---\n" << std::endl;

    // Test 176: Periodic mode with derivative evaluation
    {
        BPoly bp = BPoly::from_power_basis({{0, 1}, {1, -0.5}}, {0, 1, 3}, ExtrapolateMode::Periodic);

        test.expect_near(bp(3.5), bp(0.5), tolerance, "Test 176a: Periodic f(3.5) = f(0.5)");
        test.expect_near(bp(-0.5), bp(2.5), tolerance, "Test 176b: Periodic f(-0.5) = f(2.5)");
        test.expect_near(bp(3.5, 1), bp(0.5, 1), tolerance, "Test 176c: Periodic f'(3.5) = f'(0.5)");
        test.expect_near(bp(-0.5, 1), bp(2.5, 1), tolerance, "Test 176d: Periodic f'(-0.5) = f'(2.5)");
    }

    // Test 177: Periodic derivative and antiderivative consistency
    {
        BPoly bp = BPoly::from_power_basis({{1, 2}, {2, -1}}, {0, 1, 3}, ExtrapolateMode::Periodic);
        BPoly deriv = bp.derivative();
        BPoly anti = bp.antiderivative();

        // Antiderivative's derivative should match original
        BPoly anti_deriv = anti.derivative();
        test.expect_near(bp(0.5), anti_deriv(0.5), tolerance, "Test 177a: Periodic anti-deriv round-trip at 0.5");
        test.expect_near(bp(2.0), anti_deriv(2.0), tolerance, "Test 177b: Periodic anti-deriv round-trip at 2.0");
    }

    // ========================================================================
    // Group 31: C0 Continuity at Breakpoints
    // ========================================================================
    std::cout << "\n--- Group 31: C0 Continuity at Breakpoints ---\n" << std::endl;

    // Test 178: from_derivatives produces C0 continuous polynomial
    {
        BPoly bp = BPoly::from_derivatives({0, 1, 2, 3}, {{0, 1}, {1, 0}, {0.5, -0.5}, {0, 0}});

        double eps = 1e-12;
        bool continuous = true;

        for (double bk : {1.0, 2.0}) {
            double left = bp(bk - eps);
            double right = bp(bk + eps);
            double at_bp = bp(bk);
            if (std::abs(left - at_bp) > 1e-8 || std::abs(right - at_bp) > 1e-8) {
                test.fail("Test 178: C0 continuity at x=" + std::to_string(bk));
                continuous = false;
            }
        }
        if (continuous) {
            test.pass("Test 178: from_derivatives produces C0 continuous polynomial");
        }
    }

    // Test 179: C1 continuity with matching derivatives
    {
        BPoly bp = BPoly::from_derivatives({0, 1, 2}, {{0, 1}, {1, 1}, {3, 1}});

        double eps = 1e-10;
        bool c1_continuous = true;

        double f_left = bp(1.0 - eps);
        double f_right = bp(1.0 + eps);
        double df_left = bp(1.0 - eps, 1);
        double df_right = bp(1.0 + eps, 1);

        if (std::abs(f_left - f_right) > 1e-8) {
            test.fail("Test 179a: C0 continuity at x=1");
            c1_continuous = false;
        }
        if (std::abs(df_left - df_right) > 1e-6) {
            test.fail("Test 179b: C1 continuity (derivative) at x=1");
            c1_continuous = false;
        }
        if (c1_continuous) {
            test.pass("Test 179: from_derivatives with matching f' produces C1 continuity");
        }
    }

    // ========================================================================
    // Group 33: Edge Case Coverage
    // ========================================================================
    std::cout << "\n--- Group 33: Edge Case Coverage ---\n" << std::endl;

    // Test 180: Empty vector evaluation
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> empty_input;
        std::vector<double> result = bp(empty_input);
        test.expect_eq(result.size(), 0ul, "Test 180: Empty vector evaluation returns empty");
    }

    // Test 181: Single-point vector evaluation
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}}, {0, 1});
        std::vector<double> single_point = {0.5};
        std::vector<double> result = bp(single_point);
        test.expect_eq(result.size(), 1ul, "Test 181a: Single-point vector size");
        test.expect_near(result[0], 2.0, tolerance, "Test 181b: Single-point value correct");
    }

    // Test 182: Evaluation exactly at all breakpoints
    {
        BPoly bp = BPoly::from_power_basis({{0, 1, 2}, {1, 1, 1}}, {0, 1, 2, 3});
        std::vector<double> breakpts = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> results = bp(breakpts);

        test.expect_near(results[0], 0.0, tolerance, "Test 182a: Eval at bp[0]");
        test.expect_near(results[1], 1.0, tolerance, "Test 182b: Eval at bp[1]");
        test.expect_near(results[2], 2.0, tolerance, "Test 182c: Eval at bp[2]");
        test.expect_near(results[3], 3.0, tolerance, "Test 182d: Eval at bp[3]");
    }

    // ========================================================================
    // Group 34: Additional Edge Cases
    // ========================================================================
    std::cout << "\n--- Group 34: Additional Edge Cases ---\n" << std::endl;

    // Test 183: Zero polynomial
    {
        BPoly bp({{0}, {0}, {0}}, {0, 1});

        test.expect_near(bp(0.0), 0.0, tolerance, "Test 183a: Zero polynomial at 0");
        test.expect_near(bp(0.5), 0.0, tolerance, "Test 183b: Zero polynomial at 0.5");
        test.expect_near(bp(1.0), 0.0, tolerance, "Test 183c: Zero polynomial at 1");
        test.expect_near(bp(0.5, 1), 0.0, tolerance, "Test 183d: Zero polynomial derivative");
        test.expect_near(bp.integrate(0, 1), 0.0, tolerance, "Test 183e: Zero polynomial integral");
    }

    // Test 184: Repeated roots - (x-0.5)^2
    {
        BPoly bp = BPoly::from_power_basis({{0.25}, {-1}, {1}}, {0, 1});

        auto roots = bp.roots();
        test.expect_true(roots.size() >= 1, "Test 184a: Repeated root found");
        if (!roots.empty()) {
            test.expect_near(roots[0], 0.5, 1e-6, "Test 184b: Repeated root at 0.5");
        }
        test.expect_near(bp(0.5), 0.0, tolerance, "Test 184c: f(0.5) = 0");
        test.expect_near(bp(0.5, 1), 0.0, tolerance, "Test 184d: f'(0.5) = 0");
    }

    // Test 185: Integration beyond domain
    {
        BPoly bp = BPoly::from_power_basis({{0}, {1}}, {0, 1}, ExtrapolateMode::Extrapolate);

        test.expect_near(bp.integrate(-1, 2), 1.5, tolerance, "Test 185a: Integration beyond domain");
        test.expect_near(bp.integrate(-1, 0), -0.5, tolerance, "Test 185b: Integration in left extrapolation");
        test.expect_near(bp.integrate(1, 2), 1.5, tolerance, "Test 185c: Integration in right extrapolation");
    }

    // ========================================================================
    // Group 35: Bernstein Basis-Specific Properties
    // ========================================================================
    std::cout << "\n--- Group 35: Bernstein Basis-Specific Properties ---\n" << std::endl;

    // Test 186: Partition of unity - sum of B_{i,n}(t) = 1 for all t
    {
        for (int n = 1; n <= 5; ++n) {
            for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
                double sum = 0.0;
                for (int i = 0; i <= n; ++i) {
                    // B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
                    // Create BPoly with 1 in position i, 0 elsewhere
                    std::vector<std::vector<double>> coeffs(n + 1, {0.0});
                    coeffs[i][0] = 1.0;
                    BPoly bp(coeffs, {0, 1});
                    sum += bp(t);
                }
                std::string test_name = "Test 186: Partition of unity n=" + std::to_string(n) + " t=" + std::to_string(t);
                test.expect_near(sum, 1.0, 1e-10, test_name);
            }
        }
    }

    // Test 187: Non-negativity B_{i,n}(t) >= 0 for t in [0,1]
    {
        for (int n = 1; n <= 4; ++n) {
            for (int i = 0; i <= n; ++i) {
                std::vector<std::vector<double>> coeffs(n + 1, {0.0});
                coeffs[i][0] = 1.0;
                BPoly bp(coeffs, {0, 1});

                bool non_neg = true;
                for (double t : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
                    if (bp(t) < -1e-15) {
                        non_neg = false;
                    }
                }
                std::string test_name = "Test 187: Non-negativity B_{" + std::to_string(i) + "," + std::to_string(n) + "}";
                test.expect_true(non_neg, test_name);
            }
        }
    }

    // Test 188: Endpoint values B_{i,n}(0) = delta_{i,0}, B_{i,n}(1) = delta_{i,n}
    {
        for (int n = 1; n <= 4; ++n) {
            for (int i = 0; i <= n; ++i) {
                std::vector<std::vector<double>> coeffs(n + 1, {0.0});
                coeffs[i][0] = 1.0;
                BPoly bp(coeffs, {0, 1});

                double val_at_0 = bp(0.0);
                double val_at_1 = bp(1.0);
                double exp_0 = (i == 0) ? 1.0 : 0.0;
                double exp_1 = (i == n) ? 1.0 : 0.0;

                test.expect_near(val_at_0, exp_0, 1e-10,
                    "Test 188: B_{" + std::to_string(i) + "," + std::to_string(n) + "}(0) = " + std::to_string(exp_0));
                test.expect_near(val_at_1, exp_1, 1e-10,
                    "Test 188: B_{" + std::to_string(i) + "," + std::to_string(n) + "}(1) = " + std::to_string(exp_1));
            }
        }
    }

    // Test 189: Symmetry B_{i,n}(t) = B_{n-i,n}(1-t)
    {
        for (int n = 2; n <= 4; ++n) {
            for (int i = 0; i <= n; ++i) {
                std::vector<std::vector<double>> coeffs_i(n + 1, {0.0});
                coeffs_i[i][0] = 1.0;
                BPoly bp_i(coeffs_i, {0, 1});

                std::vector<std::vector<double>> coeffs_ni(n + 1, {0.0});
                coeffs_ni[n - i][0] = 1.0;
                BPoly bp_ni(coeffs_ni, {0, 1});

                for (double t : {0.2, 0.5, 0.8}) {
                    test.expect_near(bp_i(t), bp_ni(1.0 - t), 1e-10,
                        "Test 189: B_{" + std::to_string(i) + "," + std::to_string(n) + "}(" + std::to_string(t) + ") symmetry");
                }
            }
        }
    }

    // ========================================================================
    // Group 36: Derivative Edge Cases
    // ========================================================================
    std::cout << "\n--- Group 36: Derivative Edge Cases ---\n" << std::endl;

    // Test 190: Derivative of constant is 0
    {
        BPoly bp({{5}}, {0, 1});
        BPoly deriv = bp.derivative();
        test.expect_near(deriv(0.5), 0.0, tolerance, "Test 190a: Derivative of constant is 0");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 0ul, "Test 190b: Derivative degree is 0");
    }

    // Test 191: Over-differentiate
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BPoly d3 = bp.derivative(3);
        test.expect_near(d3(0.5), 0.0, tolerance, "Test 191a: 3rd derivative of quadratic is 0");

        BPoly d10 = bp.derivative(10);
        test.expect_near(d10(0.5), 0.0, tolerance, "Test 191b: 10th derivative of quadratic is 0");
    }

    // Test 192: Chained derivatives vs single derivative(n)
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});
        BPoly d3_chained = bp.derivative().derivative().derivative();
        BPoly d3_single = bp.derivative(3);

        test.expect_near(d3_chained(0.5), d3_single(0.5), tolerance, "Test 192: Chained vs single derivative(3)");
    }

    // ========================================================================
    // Group 37: Antiderivative Edge Cases
    // ========================================================================
    std::cout << "\n--- Group 37: Antiderivative Edge Cases ---\n" << std::endl;

    // Test 193: Antiderivative of zero polynomial
    {
        BPoly bp({{0}, {0}}, {0, 1});
        BPoly anti = bp.antiderivative();
        test.expect_near(anti(0.0), 0.0, tolerance, "Test 193a: Antiderivative of zero at 0");
        test.expect_near(anti(0.5), 0.0, tolerance, "Test 193b: Antiderivative of zero at 0.5");
        test.expect_near(anti(1.0), 0.0, tolerance, "Test 193c: Antiderivative of zero at 1");
    }

    // Test 194: Antiderivative(n).derivative(n) = original
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BPoly round_trip = bp.antiderivative(2).derivative(2);

        test.expect_near(bp(0.25), round_trip(0.25), tolerance, "Test 194a: antiderivative(2).derivative(2) at 0.25");
        test.expect_near(bp(0.5), round_trip(0.5), tolerance, "Test 194b: antiderivative(2).derivative(2) at 0.5");
        test.expect_near(bp(0.75), round_trip(0.75), tolerance, "Test 194c: antiderivative(2).derivative(2) at 0.75");
    }

    // Test 195: Chained antiderivatives vs single antiderivative(n)
    {
        BPoly bp = BPoly::from_power_basis({{2}}, {0, 1});
        BPoly a2_chained = bp.antiderivative().antiderivative();
        BPoly a2_single = bp.antiderivative(2);

        test.expect_near(a2_chained(0.5), a2_single(0.5), tolerance, "Test 195: Chained vs single antiderivative(2)");
    }

    // ========================================================================
    // Group 37b: Derivative/Antiderivative Structural Properties
    // ========================================================================
    std::cout << "\n--- Group 37b: Derivative/Antiderivative Structural Properties ---\n" << std::endl;

    // Test 195b: Derivative reduces degree by 1
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}, {4}}, {0, 1});
        BPoly deriv = bp.derivative();
        test.expect_eq(static_cast<size_t>(bp.degree()), 3ul, "Test 195b-1: Original degree is 3");
        test.expect_eq(static_cast<size_t>(deriv.degree()), 2ul, "Test 195b-2: Derivative degree is 2");
    }

    // Test 195c: Antiderivative increases degree by 1
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BPoly anti = bp.antiderivative();
        test.expect_eq(static_cast<size_t>(bp.degree()), 2ul, "Test 195c-1: Original degree is 2");
        test.expect_eq(static_cast<size_t>(anti.degree()), 3ul, "Test 195c-2: Antiderivative degree is 3");
    }

    // Test 195d: Derivative preserves num_intervals
    {
        BPoly bp({{1, 2, 3}}, {0, 1, 2, 3});
        BPoly deriv = bp.derivative();
        test.expect_eq(static_cast<size_t>(deriv.num_intervals()), 3ul,
            "Test 195d: Derivative preserves num_intervals");
    }

    // Test 195e: Derivative preserves breakpoints
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}}, {0, 1});
        BPoly deriv = bp.derivative();
        test.expect_near(deriv.breakpoints()[0], 0.0, tolerance, "Test 195e-1: Derivative preserves left breakpoint");
        test.expect_near(deriv.breakpoints()[1], 1.0, tolerance, "Test 195e-2: Derivative preserves right breakpoint");
    }

    // Test 195f: Antiderivative starts at 0 at left boundary
    {
        BPoly bp = BPoly::from_power_basis({{5}}, {0, 1});
        BPoly anti = bp.antiderivative();
        test.expect_near(anti(0), 0.0, tolerance, "Test 195f: Antiderivative(left_boundary) = 0");
    }

    // ========================================================================
    // Group 38: Integration Edge Cases
    // ========================================================================
    std::cout << "\n--- Group 38: Integration Edge Cases ---\n" << std::endl;

    // Test 196: integrate(a, a) = 0
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        test.expect_near(bp.integrate(0.5, 0.5), 0.0, tolerance, "Test 196: integrate(a, a) = 0");
    }

    // Test 197: Integration crossing multiple intervals
    {
        BPoly bp({{1, 2}}, {0, 1, 2});
        test.expect_near(bp.integrate(0, 2), 3.0, tolerance, "Test 197: Integration across multiple intervals");
    }

    // Test 198: NoExtrapolate returns NaN beyond bounds
    {
        BPoly bp({{1}, {2}}, {0, 1}, ExtrapolateMode::NoExtrapolate);
        double result = bp(-1);
        test.expect_true(std::isnan(result), "Test 198: NoExtrapolate evaluation beyond bounds returns NaN");
    }

    // ========================================================================
    // Group 39: Root Finding Edge Cases
    // ========================================================================
    std::cout << "\n--- Group 39: Root Finding Edge Cases ---\n" << std::endl;

    // Test 199: No roots (always positive polynomial)
    {
        BPoly bp = BPoly::from_power_basis({{1}, {0}, {1}}, {0, 1});
        auto roots = bp.roots();
        test.expect_eq(roots.size(), 0ul, "Test 199: No roots for always-positive polynomial");
    }

    // Test 200: Root at domain boundary
    {
        BPoly bp = BPoly::from_power_basis({{0}, {1}}, {0, 1});
        auto roots = bp.roots();
        test.expect_true(roots.size() >= 1, "Test 200a: Root at boundary found");
        if (!roots.empty()) {
            double min_root = *std::min_element(roots.begin(), roots.end());
            test.expect_near(min_root, 0.0, 1e-6, "Test 200b: Root at x=0");
        }
    }

    // Test 201: Many roots
    {
        BPoly bp = BPoly::from_power_basis({{-0.09375}, {0.6875}, {-1.5}, {1}}, {0, 1});
        auto roots = bp.roots();
        test.expect_eq(roots.size(), 3ul, "Test 201a: Three roots found");

        if (roots.size() == 3) {
            std::sort(roots.begin(), roots.end());
            test.expect_near(roots[0], 0.25, 1e-6, "Test 201b: Root at 0.25");
            test.expect_near(roots[1], 0.5, 1e-6, "Test 201c: Root at 0.5");
            test.expect_near(roots[2], 0.75, 1e-6, "Test 201d: Root at 0.75");
        }
    }

    // ========================================================================
    // Group 41: Move/Copy Semantics
    // ========================================================================
    std::cout << "\n--- Group 41: Move/Copy Semantics ---\n" << std::endl;

    // Test 202: Copy constructor
    {
        BPoly bp1 = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        BPoly bp2(bp1);

        test.expect_near(bp1(0.5), bp2(0.5), tolerance, "Test 202a: Copy constructor preserves value");
        test.expect_eq(static_cast<size_t>(bp1.degree()), static_cast<size_t>(bp2.degree()), "Test 202b: Copy constructor preserves degree");
    }

    // Test 203: Move constructor
    {
        BPoly bp1 = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double original_val = bp1(0.5);
        BPoly bp2(std::move(bp1));

        test.expect_near(bp2(0.5), original_val, tolerance, "Test 203: Move constructor preserves value");
    }

    // ========================================================================
    // Group 42: Const-Correctness and Thread Safety
    // ========================================================================
    std::cout << "\n--- Group 42: Const-Correctness and Thread Safety ---\n" << std::endl;

    // Test 204: All const methods work on const object
    {
        const BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
        double v1 = bp(0.5);
        double v2 = bp(0.5, 1);
        std::vector<double> v3 = bp({0.25, 0.5, 0.75});
        int d = bp.degree();
        int n = bp.num_intervals();
        const auto& c = bp.coefficients();
        const auto& x = bp.breakpoints();
        double integ = bp.integrate(0, 1);
        auto roots = bp.roots();

        test.expect_near(v1, 1 + 2*0.5 + 3*0.25, tolerance, "Test 204a: Const evaluation");
        test.expect_near(v2, 2 + 6*0.5, tolerance, "Test 204b: Const derivative evaluation");
        test.expect_eq(v3.size(), 3ul, "Test 204c: Const vector evaluation");
        test.expect_eq(static_cast<size_t>(d), 2ul, "Test 204d: Const degree()");
        test.expect_eq(static_cast<size_t>(n), 1ul, "Test 204e: Const num_intervals()");
        test.expect_true(!c.empty(), "Test 204f: Const coefficients()");
        test.expect_true(!x.empty(), "Test 204g: Const breakpoints()");
        test.expect_near(integ, 3.0, tolerance, "Test 204h: Const integrate()");
        (void)roots;
        test.pass("Test 204: All const methods work on const object");
    }

    // Test 205: Thread safety - multiple threads evaluating same polynomial
    {
        BPoly bp = BPoly::from_power_basis({{1}, {2}, {3}}, {0, 1});
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
                      "Test 205: Thread safety - all concurrent evaluations correct");
    }

    // ========================================================================
    // Group 44: Symmetry Tests
    // ========================================================================
    std::cout << "\n--- Group 44: Symmetry Tests ---\n" << std::endl;

    // Test 206: Even polynomial p(-x) = p(x)
    {
        // f(x) = x^2 on [-1,1] — even function
        // Power basis relative to a=-1: x = (x+1) - 1, x^2 = (x+1)^2 - 2(x+1) + 1
        // c_0 = 1, c_1 = -2, c_2 = 1
        BPoly bp = BPoly::from_power_basis({{1}, {-2}, {1}}, {-1, 1});

        test.expect_near(bp(-0.5), bp(0.5), tolerance, "Test 206a: Even polynomial at +/-0.5");
        test.expect_near(bp(-0.8), bp(0.8), tolerance, "Test 206b: Even polynomial at +/-0.8");
    }

    // Test 207: Odd polynomial p(-x) = -p(x)
    {
        // f(x) = x^3 on [-1,1] — odd function
        // Power basis relative to a=-1: x = (x+1) - 1
        // x^3 = ((x+1)-1)^3 = (x+1)^3 - 3(x+1)^2 + 3(x+1) - 1
        // c_0 = -1, c_1 = 3, c_2 = -3, c_3 = 1
        BPoly bp = BPoly::from_power_basis({{-1}, {3}, {-3}, {1}}, {-1, 1});

        test.expect_near(bp(-0.5), -bp(0.5), tolerance, "Test 207a: Odd polynomial at +/-0.5");
        test.expect_near(bp(-0.8), -bp(0.8), tolerance, "Test 207b: Odd polynomial at +/-0.8");
    }

    // ========================================================================
    // Group 45: Boundary Edge Cases
    // ========================================================================
    std::cout << "\n--- Group 45: Boundary Edge Cases ---\n" << std::endl;

    // Test 208: Evaluation at breakpoint + epsilon
    {
        BPoly bp1 = BPoly::from_power_basis({{0}, {1}}, {0, 1});
        BPoly bp2 = BPoly::from_power_basis({{1}, {2}}, {1, 2});
        BPoly bp = bp1.extend(bp2.coefficients(), {1, 2}, true);

        double eps = std::numeric_limits<double>::epsilon();
        double at_bp = bp(1.0);
        double after_bp = bp(1.0 + eps);
        double before_bp = bp(1.0 - eps);

        test.expect_near(at_bp, 1.0, tolerance, "Test 208a: At breakpoint");
        test.expect_near(before_bp, 1.0, tolerance, "Test 208b: Just before breakpoint");
        test.expect_near(after_bp, 1.0, tolerance, "Test 208c: Just after breakpoint");
    }

    // Test 209: Very wide interval
    {
        BPoly bp = BPoly::from_power_basis({{0}, {1}}, {0, 1e10});
        test.expect_near(bp(0), 0.0, tolerance, "Test 209a: Wide interval at left");
        test.expect_near(bp(5e9), 5e9, 1e-3, "Test 209b: Wide interval at midpoint");
        test.expect_near(bp(1e10), 1e10, 1e-3, "Test 209c: Wide interval at right");
    }

    // Test 210: Very narrow interval
    {
        BPoly bp = BPoly::from_power_basis({{0}, {1}}, {0, 1e-10});
        test.expect_near(bp(0), 0.0, tolerance, "Test 210a: Narrow interval at left");
        test.expect_near(bp(5e-11), 5e-11, 1e-20, "Test 210b: Narrow interval at midpoint");
        test.expect_near(bp(1e-10), 1e-10, 1e-20, "Test 210c: Narrow interval at right");
    }

    // ========================================================================
    // Group 46: Reference Data Verification
    // ========================================================================
    std::cout << "\n--- Group 46: Reference Data Verification ---\n" << std::endl;

    // Test 211: Bernstein basis B_{i,n}(t) at specific points vs C(n,i)*t^i*(1-t)^(n-i)
    {
        auto binom = [](int n, int k) -> double {
            double result = 1.0;
            for (int i = 0; i < k; ++i) {
                result *= static_cast<double>(n - i) / (i + 1);
            }
            return result;
        };

        std::vector<double> test_points = {0.0, 0.25, 0.5, 0.75, 1.0};
        for (int n = 2; n <= 4; ++n) {
            for (int i = 0; i <= n; ++i) {
                std::vector<std::vector<double>> coeffs(n + 1, {0.0});
                coeffs[i][0] = 1.0;
                BPoly bp(coeffs, {0, 1});

                for (double t : test_points) {
                    double expected = binom(n, i) * std::pow(t, i) * std::pow(1.0 - t, n - i);
                    std::string test_name = "Test 211: B_{" + std::to_string(i) + "," + std::to_string(n) + "}(" + std::to_string(t) + ")";
                    test.expect_near(bp(t), expected, 1e-10, test_name);
                }
            }
        }
    }

    test.summary();
    return test.all_passed() ? 0 : 1;
}