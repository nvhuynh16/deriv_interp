/**
 * Compare C++ BPoly::from_derivatives against scipy reference data.
 * Identifies accuracy gaps in various derivative configurations.
 *
 * Test cases use 5-point evaluation grid [0.0, 0.25, 0.5, 0.75, 1.0]
 * (or equivalent for non-unit intervals).
 *
 * Values verified against scipy.interpolate.BPoly (scipy 1.16.3)
 * Run scripts/verify_cpp_values.py to regenerate and verify reference values.
 *
 * All hardcoded scipy values have been verified to match within 1e-10.
 */

#include "include/bpoly.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

struct TestCase {
    std::string name;
    std::vector<double> xi;
    std::vector<std::vector<double>> yi;
    std::vector<double> scipy_coeffs;  // Flattened row-wise
    std::vector<double> eval_points;
    std::vector<double> scipy_results;
    int expected_degree;
};

bool run_comparison(const TestCase& tc) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Test: " << tc.name << "\n";
    std::cout << "  xi: [";
    for (size_t i = 0; i < tc.xi.size(); ++i) {
        std::cout << tc.xi[i] << (i < tc.xi.size()-1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "  yi: [";
    for (size_t i = 0; i < tc.yi.size(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < tc.yi[i].size(); ++j) {
            std::cout << tc.yi[i][j] << (j < tc.yi[i].size()-1 ? ", " : "");
        }
        std::cout << "]" << (i < tc.yi.size()-1 ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "  Expected degree: " << tc.expected_degree << "\n";
    std::cout << std::string(70, '-') << "\n";

    try {
        BPoly bp = BPoly::from_derivatives(tc.xi, tc.yi);

        int degree = bp.degree();
        std::cout << "  C++ degree: " << degree;
        if (degree != tc.expected_degree) {
            std::cout << " [MISMATCH - expected " << tc.expected_degree << "]";
        }
        std::cout << "\n";

        // Compare coefficients
        std::cout << "\n  Coefficients comparison:\n";
        auto cpp_coeffs = bp.coefficients();

        std::cout << "    Scipy: [";
        for (size_t i = 0; i < tc.scipy_coeffs.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << tc.scipy_coeffs[i];
            if (i < tc.scipy_coeffs.size()-1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "    C++:   [";
        for (size_t j = 0; j < cpp_coeffs.size(); ++j) {
            for (size_t i = 0; i < cpp_coeffs[j].size(); ++i) {
                std::cout << std::fixed << std::setprecision(6) << cpp_coeffs[j][i];
                if (j < cpp_coeffs.size()-1 || i < cpp_coeffs[j].size()-1) std::cout << ", ";
            }
        }
        std::cout << "]\n";

        // Compare evaluation results
        std::cout << "\n  Evaluation comparison (tolerance = 1e-10):\n";
        std::cout << "    " << std::setw(10) << "x"
                  << std::setw(15) << "scipy"
                  << std::setw(15) << "C++"
                  << std::setw(15) << "error"
                  << std::setw(10) << "status" << "\n";
        std::cout << "    " << std::string(55, '-') << "\n";

        double max_error = 0.0;
        int mismatches = 0;
        const double tolerance = 1e-10;

        for (size_t i = 0; i < tc.eval_points.size(); ++i) {
            double x = tc.eval_points[i];
            double scipy_val = tc.scipy_results[i];
            double cpp_val = bp(x);
            double error = std::abs(cpp_val - scipy_val);

            max_error = std::max(max_error, error);

            std::string status = (error < tolerance) ? "OK" : "FAIL";
            if (error >= tolerance) mismatches++;

            std::cout << "    " << std::setw(10) << std::fixed << std::setprecision(4) << x
                      << std::setw(15) << std::setprecision(8) << scipy_val
                      << std::setw(15) << cpp_val
                      << std::setw(15) << std::scientific << std::setprecision(2) << error
                      << std::setw(10) << status << "\n";
        }

        std::cout << "\n  Summary:\n";
        std::cout << "    Max error: " << std::scientific << std::setprecision(4) << max_error << "\n";
        std::cout << "    Mismatches: " << mismatches << "/" << tc.eval_points.size() << "\n";
        std::cout << "    Status: " << (mismatches == 0 ? "PASS" : "FAIL") << "\n";

        return mismatches == 0;

    } catch (const std::exception& e) {
        std::cout << "  ERROR: " << e.what() << "\n";
        return false;
    }
}

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "BPoly::from_derivatives Accuracy Comparison vs SciPy\n";
    std::cout << std::string(70, '=') << "\n";

    std::vector<TestCase> cases;

    // Case 1: Linear (n0=1, n1=1)
    cases.push_back({
        "linear_2pt",
        {0, 1},
        {{0}, {1}},
        {0.0, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        1
    });

    // Case 2: Cubic Hermite (n0=2, n1=2)
    cases.push_back({
        "hermite_cubic",
        {0, 1},
        {{0, 1}, {1, -1}},
        {0.0, 0.333333, 1.333333, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.34375, 0.75, 1.03125, 1.0},
        3
    });

    // Case 3: Quintic Hermite (n0=3, n1=3) - exact scipy values
    cases.push_back({
        "hermite_quintic",
        {0, 1},
        {{0, 1, 0}, {1, -1, 0}},
        {0.0, 0.2, 0.4, 1.4, 1.2, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.326171875, 0.8125, 1.119140625, 1.0},  // Exact scipy values
        5
    });

    // Case 4: Asymmetric (n0=1, n1=2)
    cases.push_back({
        "asymmetric_1_2",
        {0, 1},
        {{0}, {1, 0}},
        {0.0, 1.0, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.4375, 0.75, 0.9375, 1.0},
        2
    });

    // Case 5: Asymmetric (n0=2, n1=1) - exact scipy values
    cases.push_back({
        "asymmetric_2_1",
        {0, 1},
        {{0, 1}, {1}},
        {0.0, 0.5, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},  // Exact scipy values
        2
    });

    // Case 6: Asymmetric (n0=3, n1=1) - exact scipy values
    cases.push_back({
        "asymmetric_3_1",
        {0, 1},
        {{0, 1, 2}, {1}},
        {0.0, 0.333333, 1.0, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.296875, 0.625, 0.890625, 1.0},  // Exact scipy values
        3
    });

    // Case 7: Asymmetric (n0=1, n1=3) - exact scipy values
    cases.push_back({
        "asymmetric_1_3",
        {0, 1},
        {{0}, {1, -1, 0}},
        {0.0, 1.666667, 1.333333, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.90625, 1.25, 1.21875, 1.0},  // Exact scipy values
        3
    });

    // Case 8: Higher order (n0=4, n1=4) - degree 7 - exact scipy values
    cases.push_back({
        "higher_order_4",
        {0, 1},
        {{0, 1, 0, 0}, {1, -1, 0, 0}},
        {0.0, 0.142857, 0.285714, 0.428571, 1.428571, 1.285714, 1.142857, 1.0},
        {0.0, 0.25, 0.5, 0.75, 1.0},
        {0.0, 0.3063964844, 0.84375, 1.1652832031, 1.0},  // Exact scipy values
        7
    });

    // Case 9: Non-unit interval [0, 2] - exact scipy values
    cases.push_back({
        "non_unit_interval",
        {0, 2},
        {{0, 0.5}, {4, 2}},
        {0.0, 0.333333, 2.666667, 4.0},
        {0.0, 0.5, 1.0, 1.5, 2.0},
        {0.0, 0.578125, 1.625, 2.859375, 4.0},  // Exact scipy values
        3
    });

    // Case 10: Negative interval [-1, 1]
    cases.push_back({
        "negative_interval",
        {-1, 1},
        {{1, 0}, {1, 0}},
        {1.0, 1.0, 1.0, 1.0},
        {-1.0, -0.5, 0.0, 0.5, 1.0},
        {1.0, 1.0, 1.0, 1.0, 1.0},
        3
    });

    // Case 11: Three points mixed derivatives
    cases.push_back({
        "three_points_mixed",
        {0, 1, 2},
        {{0, 1}, {1}, {0, -1}},
        {0.0, 1.0, 0.5, 0.5, 1.0, 0.0},
        {0.0, 0.5, 1.0, 1.5, 2.0},
        {0.0, 0.5, 1.0, 0.5, 0.0},
        2
    });

    // Run all comparisons
    int total_pass = 0;
    int total_fail = 0;

    for (const auto& tc : cases) {
        if (run_comparison(tc)) {
            total_pass++;
        } else {
            total_fail++;
        }
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "OVERALL SUMMARY\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Total test cases: " << cases.size() << "\n";
    std::cout << "Passed: " << total_pass << "\n";
    std::cout << "Failed: " << total_fail << "\n";
    std::cout << "Result: " << (total_fail == 0 ? "ALL PASS" : "SOME FAILURES") << "\n";

    return total_fail == 0 ? 0 : 1;
}
