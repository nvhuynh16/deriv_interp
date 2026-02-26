#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <string>

// =============================================================================
// Shared test infrastructure for polynomial test suites
// =============================================================================

// Manual test implementation without GTest dependency
class TestRunner {
private:
    int tests_run = 0;
    int tests_passed = 0;

public:
    void expect_near(double actual, double expected, double tolerance, const std::string& test_name) {
        tests_run++;
        if (std::abs(actual - expected) <= tolerance) {
            tests_passed++;
            std::cout << "[PASS] " << test_name << ": " << actual << " ~ " << expected << std::endl;
        } else {
            std::cout << "[FAIL] " << test_name << ": Expected " << expected << ", got " << actual
                      << " (error: " << std::abs(actual - expected) << ")" << std::endl;
        }
    }

    void expect_no_throw(std::function<void()> func, const std::string& test_name) {
        tests_run++;
        try {
            func();
            tests_passed++;
            std::cout << "[PASS] " << test_name << ": No exception thrown" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[FAIL] " << test_name << ": Unexpected exception: " << e.what() << std::endl;
        }
    }

    void expect_throw(std::function<void()> func, const std::string& test_name) {
        tests_run++;
        try {
            func();
            std::cout << "[FAIL] " << test_name << ": Expected exception but none thrown" << std::endl;
        } catch (const std::exception& e) {
            tests_passed++;
            std::cout << "[PASS] " << test_name << ": Expected exception caught: " << e.what() << std::endl;
        }
    }

    void expect_eq(size_t actual, size_t expected, const std::string& test_name) {
        tests_run++;
        if (actual == expected) {
            tests_passed++;
            std::cout << "[PASS] " << test_name << ": " << actual << " == " << expected << std::endl;
        } else {
            std::cout << "[FAIL] " << test_name << ": Expected " << expected << ", got " << actual << std::endl;
        }
    }

    void expect_true(bool condition, const std::string& test_name) {
        tests_run++;
        if (condition) {
            tests_passed++;
            std::cout << "[PASS] " << test_name << std::endl;
        } else {
            std::cout << "[FAIL] " << test_name << ": Expected true, got false" << std::endl;
        }
    }

    void pass(const std::string& test_name) {
        tests_run++;
        tests_passed++;
        std::cout << "[PASS] " << test_name << std::endl;
    }

    void fail(const std::string& test_name) {
        tests_run++;
        std::cout << "[FAIL] " << test_name << std::endl;
    }

    void summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Tests run: " << tests_run << std::endl;
        std::cout << "Tests passed: " << tests_passed << std::endl;
        std::cout << "Tests failed: " << (tests_run - tests_passed) << std::endl;
        if (tests_passed == tests_run) {
            std::cout << "ALL TESTS PASSED!" << std::endl;
        } else {
            std::cout << "SOME TESTS FAILED!" << std::endl;
        }
    }

    bool all_passed() const { return tests_passed == tests_run; }
};

// =============================================================================
// Numerical verification helpers (template versions)
// =============================================================================

// Numerical integration using trapezoidal rule for independent verification
template <typename T>
double numerical_integrate(const T& poly, double a, double b, int n = 10000) {
    double h = (b - a) / n;
    double sum = 0.5 * (poly(a) + poly(b));
    for (int i = 1; i < n; ++i) {
        sum += poly(a + i * h);
    }
    return sum * h;
}

// Finite difference derivative approximation (central difference)
template <typename T>
double finite_diff_derivative(const T& poly, double x, double h = 1e-7) {
    return (poly(x + h) - poly(x - h)) / (2 * h);
}

// Second derivative via finite differences
template <typename T>
double finite_diff_second_derivative(const T& poly, double x, double h = 1e-5) {
    return (poly(x + h) - 2 * poly(x) + poly(x - h)) / (h * h);
}
