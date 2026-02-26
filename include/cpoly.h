#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>
#include <limits>
#include <functional>
#include <optional>

#include "extrapolate_mode.h"
#include "ndarray.h"
#include "poly_base.h"

// =============================================================================
// REFERENCES
// =============================================================================
// Primary reference:
//   numpy.polynomial.chebyshev
//   https://github.com/numpy/numpy/blob/main/numpy/polynomial/chebyshev.py
//   https://numpy.org/doc/stable/reference/routines.polynomials.chebyshev.html
//
// Algorithm references:
//   [1] Clenshaw algorithm for Chebyshev polynomial evaluation
//       https://en.wikipedia.org/wiki/Clenshaw_algorithm
//       numpy.polynomial.chebyshev.chebval()
//
//   [2] Chebyshev derivative formula (chebder)
//       c'_{n-1} = 2n * c_n
//       c'_k = c'_{k+2} + 2(k+1) * c_{k+1}  for k = n-2, ..., 0
//       numpy.polynomial.chebyshev.chebder()
//
//   [3] Chebyshev antiderivative formula (chebint)
//       C_k = (c_{k-1} - c_{k+1}) / (2k)  for k >= 2
//       numpy.polynomial.chebyshev.chebint()
//
//   [4] Chebyshev basis conversion (cheb2poly, poly2cheb)
//       numpy.polynomial.chebyshev.cheb2poly(), poly2cheb()
// =============================================================================

/**
 * Chebyshev polynomial piecewise interpolant
 *
 * Thread-safe: All methods are const after construction, no shared state.
 *
 * Mathematical foundation:
 * - Chebyshev basis: T_n(s) = cos(n * arccos(s)) on [-1, 1]
 * - Polynomial on [x_i, x_{i+1}] evaluated using normalized s = (2x - x_i - x_{i+1}) / (x_{i+1} - x_i)
 * - Coefficients stored as [degree+1][num_intervals]
 * - Evaluation uses Clenshaw algorithm for numerical stability
 */
class CPoly : public PolyBase<CPoly> {
    friend class PolyBase<CPoly>;

public:
    CPoly(std::vector<std::vector<double>> coefficients,
          std::vector<double> breakpoints,
          ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
          int extrapolate_order_left = -1,
          int extrapolate_order_right = -1);

    static CPoly from_array2d(ndarray::array2d<double> coefficients,
                              std::vector<double> breakpoints,
                              ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
                              int extrapolate_order_left = -1,
                              int extrapolate_order_right = -1);

    CPoly(CPoly&& other) noexcept = default;
    CPoly(const CPoly& other) = default;
    CPoly& operator=(const CPoly&) = delete;
    CPoly& operator=(CPoly&&) = delete;

    static CPoly from_derivatives(std::vector<double> xi,
                                  std::vector<std::vector<double>> yi,
                                  std::vector<int> orders = {});

    static CPoly from_power_basis(std::vector<std::vector<double>> power_coeffs,
                                  std::vector<double> breakpoints,
                                  ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate);

    std::vector<std::vector<double>> to_power_basis() const;

private:
    // =========================================================================
    // CRTP hooks for PolyBase
    // =========================================================================

    double evaluate_basis(int interval_idx, double s) const;
    double map_to_local(double x, double a, double b) const;
    static constexpr double breakpoint_left_eval_param() { return -1.0; }
    static constexpr double breakpoint_right_eval_param() { return 1.0; }
    ndarray::array2d<double> compute_derivative_coefficients() const;
    ndarray::array2d<double> compute_antiderivative_coefficients(double& running_integral) const;
    static std::vector<double> elevate_degree_impl(const std::vector<double>& coeffs, int target_degree);
    CPoly make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                    ExtrapolateMode extrap, int ol, int or_) const;
    CPoly make_zero_poly() const;

    // =========================================================================
    // CPoly-specific helpers
    // =========================================================================

    double evaluate_chebyshev(int interval_idx, double s) const;
    static double chebyshev_T(int n, double s);
    static double chebyshev_T_deriv(int n, double s);

    static std::vector<double> compute_derivative_coeffs(
        const std::vector<double>& c, double h);
    static std::vector<double> compute_antiderivative_coeffs(
        const std::vector<double>& c, double h, double integration_constant);

    static std::vector<double> chebyshev_hermite_interpolation(
        double x0, double x1,
        const std::vector<double>& y0,
        const std::vector<double>& y1);

    static std::vector<double> elevate_degree(
        const std::vector<double>& coeffs,
        int target_degree);

    struct FromArray2DTag {};
    CPoly(FromArray2DTag,
          ndarray::array2d<double> coefficients,
          std::vector<double> breakpoints,
          ExtrapolateMode extrapolate,
          int extrapolate_order_left,
          int extrapolate_order_right);
};

// =============================================================================
// Implementation
// =============================================================================

// --- Constructors ---

inline CPoly::CPoly(std::vector<std::vector<double>> coefficients,
                   std::vector<double> breakpoints,
                   ExtrapolateMode extrapolate,
                   int extrapolate_order_left,
                   int extrapolate_order_right)
    : PolyBase(ndarray::array2d<double>(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right) {}

inline CPoly::CPoly(FromArray2DTag,
                   ndarray::array2d<double> coefficients,
                   std::vector<double> breakpoints,
                   ExtrapolateMode extrapolate,
                   int extrapolate_order_left,
                   int extrapolate_order_right)
    : PolyBase(std::move(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right) {}

inline CPoly CPoly::from_array2d(ndarray::array2d<double> coefficients,
                                 std::vector<double> breakpoints,
                                 ExtrapolateMode extrapolate,
                                 int extrapolate_order_left,
                                 int extrapolate_order_right) {
    return CPoly(FromArray2DTag{}, std::move(coefficients), std::move(breakpoints),
                 extrapolate, extrapolate_order_left, extrapolate_order_right);
}

// --- CRTP hooks ---

inline double CPoly::evaluate_basis(int interval_idx, double s) const {
    return evaluate_chebyshev(interval_idx, s);
}

inline double CPoly::map_to_local(double x, double a, double b) const {
    return (2.0 * x - a - b) / (b - a);
}

inline ndarray::array2d<double> CPoly::compute_derivative_coefficients() const {
    int new_degree = degree() - 1;
    ndarray::array2d<double> deriv_coeffs(new_degree + 1, num_intervals());

    for (int interval = 0; interval < num_intervals(); ++interval) {
        double h = breakpoints_[interval + 1] - breakpoints_[interval];

        std::vector<double> interval_coeffs(degree() + 1);
        for (int j = 0; j <= degree(); ++j) {
            interval_coeffs[j] = coefficients_(j, interval);
        }

        std::vector<double> dc = compute_derivative_coeffs(interval_coeffs, h);

        for (int j = 0; j <= new_degree; ++j) {
            deriv_coeffs(j, interval) = dc[j];
        }
    }

    return deriv_coeffs;
}

inline ndarray::array2d<double> CPoly::compute_antiderivative_coefficients(double& running_integral) const {
    int new_degree = degree() + 1;
    ndarray::array2d<double> antideriv_coeffs(new_degree + 1, num_intervals());

    for (int interval = 0; interval < num_intervals(); ++interval) {
        double h = breakpoints_[interval + 1] - breakpoints_[interval];

        std::vector<double> interval_coeffs(degree() + 1);
        for (int j = 0; j <= degree(); ++j) {
            interval_coeffs[j] = coefficients_(j, interval);
        }

        std::vector<double> C = compute_antiderivative_coeffs(interval_coeffs, h, running_integral);

        for (int j = 0; j <= new_degree; ++j) {
            antideriv_coeffs(j, interval) = C[j];
        }

        // Update running integral: value at right endpoint (s = 1)
        // F(1) = sum_k C_k * T_k(1) = sum_k C_k (since T_k(1) = 1)
        running_integral = 0.0;
        for (int k = 0; k <= new_degree; ++k) {
            running_integral += C[k];
        }
    }

    return antideriv_coeffs;
}

inline std::vector<double> CPoly::elevate_degree_impl(const std::vector<double>& coeffs, int target_degree) {
    return elevate_degree(coeffs, target_degree);
}

inline CPoly CPoly::make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                              ExtrapolateMode extrap, int ol, int or_) const {
    return CPoly::from_array2d(std::move(coeffs), std::move(breaks), extrap, ol, or_);
}

inline CPoly CPoly::make_zero_poly() const {
    ndarray::array2d<double> zero_coeffs(1, num_intervals());
    for (int i = 0; i < num_intervals(); ++i) {
        zero_coeffs(0, i) = 0.0;
    }
    return CPoly::from_array2d(std::move(zero_coeffs), breakpoints_, extrapolate_,
                               extrapolate_order_left_, extrapolate_order_right_);
}

// --- CPoly-specific methods ---

inline double CPoly::chebyshev_T(int n, double s) {
    if (n == 0) return 1.0;
    if (n == 1) return s;

    double T_prev2 = 1.0;
    double T_prev1 = s;

    for (int k = 2; k <= n; ++k) {
        double T_curr = 2.0 * s * T_prev1 - T_prev2;
        T_prev2 = T_prev1;
        T_prev1 = T_curr;
    }

    return T_prev1;
}

inline double CPoly::chebyshev_T_deriv(int n, double s) {
    if (n == 0) return 0.0;
    if (n == 1) return 1.0;

    double U_prev2 = 1.0;
    double U_prev1 = 2.0 * s;

    for (int k = 2; k <= n - 1; ++k) {
        double U_curr = 2.0 * s * U_prev1 - U_prev2;
        U_prev2 = U_prev1;
        U_prev1 = U_curr;
    }

    return static_cast<double>(n) * U_prev1;
}

inline double CPoly::evaluate_chebyshev(int interval_idx, double s) const {
    // Clenshaw algorithm [1] for evaluating sum_{k=0}^{n} c_k * T_k(s)
    const int n = degree();

    if (n < 0) return 0.0;
    if (n == 0) return coefficients_(0, interval_idx);

    double b_kplus1 = 0.0;
    double b_kplus2 = 0.0;

    for (int k = n; k >= 1; --k) {
        double b_k = 2.0 * s * b_kplus1 - b_kplus2 + coefficients_(k, interval_idx);
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }

    return coefficients_(0, interval_idx) + s * b_kplus1 - b_kplus2;
}

inline std::vector<double> CPoly::compute_derivative_coeffs(
    const std::vector<double>& c, double h) {
    // Chebyshev derivative formula [2]
    int n = static_cast<int>(c.size()) - 1;
    if (n <= 0) {
        return {0.0};
    }

    std::vector<double> dc(n, 0.0);

    dc[n-1] = 2.0 * n * c[n];

    for (int k = n - 2; k >= 1; --k) {
        double c_prime_kplus2 = (k + 2 <= n - 1) ? dc[k + 2] : 0.0;
        dc[k] = c_prime_kplus2 + 2.0 * (k + 1) * c[k + 1];
    }

    double c_prime_2 = (2 <= n - 1) ? dc[2] : 0.0;
    dc[0] = c_prime_2 / 2.0 + c[1];

    double scale = 2.0 / h;
    for (auto& coef : dc) {
        coef *= scale;
    }

    return dc;
}

inline std::vector<double> CPoly::compute_antiderivative_coeffs(
    const std::vector<double>& c, double h, double integration_constant) {
    // Chebyshev antiderivative formula [3]
    int n = static_cast<int>(c.size()) - 1;
    std::vector<double> C(n + 2, 0.0);

    double scale = h / 2.0;

    double c_0 = c[0];
    double c_2 = (2 <= n) ? c[2] : 0.0;
    C[1] = scale * (c_0 - c_2 / 2.0);

    for (int k = 2; k <= n + 1; ++k) {
        double c_prev = (k - 1 <= n) ? c[k - 1] : 0.0;
        double c_next = (k + 1 <= n) ? c[k + 1] : 0.0;
        C[k] = scale * (c_prev - c_next) / (2.0 * k);
    }

    double sum_at_minus1 = 0.0;
    for (int k = 1; k <= n + 1; ++k) {
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        sum_at_minus1 += C[k] * sign;
    }
    C[0] = integration_constant - sum_at_minus1;

    return C;
}

inline std::vector<double> CPoly::chebyshev_hermite_interpolation(
    double x0, double x1,
    const std::vector<double>& y0,
    const std::vector<double>& y1) {

    double h = x1 - x0;
    int n0 = static_cast<int>(y0.size());
    int n1 = static_cast<int>(y1.size());
    int n = n0 + n1 - 1;

    if (n0 == 0 || n1 == 0) {
        throw std::invalid_argument("Each point must have at least function value");
    }

    std::vector<std::vector<double>> A(n + 1, std::vector<double>(n + 1, 0.0));
    std::vector<double> b(n + 1, 0.0);

    auto T_deriv_at = [](int j, int k, double s) -> double {
        if (j < k) return 0.0;

        double prod = 1.0;
        for (int m = 0; m < k; ++m) {
            prod *= (static_cast<double>(j * j) - static_cast<double>(m * m)) / (2.0 * m + 1.0);
        }

        if (s > 0) {
            return prod;
        } else {
            double sign = ((j + k) % 2 == 0) ? 1.0 : -1.0;
            return sign * prod;
        }
    };

    int eq = 0;

    for (int k = 0; k < n0; ++k) {
        double scale = std::pow(2.0 / h, k);
        for (int j = 0; j <= n; ++j) {
            A[eq][j] = scale * T_deriv_at(j, k, -1.0);
        }
        b[eq] = y0[k];
        ++eq;
    }

    for (int k = 0; k < n1; ++k) {
        double scale = std::pow(2.0 / h, k);
        for (int j = 0; j <= n; ++j) {
            A[eq][j] = scale * T_deriv_at(j, k, +1.0);
        }
        b[eq] = y1[k];
        ++eq;
    }

    return poly_util::solve_linear_system(A, b, n);
}

inline std::vector<double> CPoly::elevate_degree(
    const std::vector<double>& coeffs,
    int target_degree) {
    int n = static_cast<int>(coeffs.size()) - 1;

    if (target_degree < n) {
        throw std::invalid_argument("Target degree must be >= current degree");
    }
    if (target_degree == n) {
        return coeffs;
    }

    std::vector<double> elevated(target_degree + 1, 0.0);
    for (int i = 0; i <= n; ++i) {
        elevated[i] = coeffs[i];
    }
    return elevated;
}

// --- Static factories ---

inline CPoly CPoly::from_derivatives(std::vector<double> xi,
                                     std::vector<std::vector<double>> yi,
                                     std::vector<int> orders) {
    auto setup = poly_util::validate_from_derivatives(xi, yi, orders);
    int num_intervals = setup.num_intervals;
    int max_degree = setup.max_degree;
    auto& get_num_derivs = setup.get_num_derivs;

    std::vector<std::vector<double>> coefficients(max_degree + 1,
                                                  std::vector<double>(num_intervals, 0.0));

    for (int i = 0; i < num_intervals; ++i) {
        int left_num = get_num_derivs(i);
        int right_num = get_num_derivs(i + 1);

        std::vector<double> left_derivs(yi[i].begin(), yi[i].begin() + left_num);
        std::vector<double> right_derivs(yi[i + 1].begin(), yi[i + 1].begin() + right_num);

        std::vector<double> interval_coeffs = chebyshev_hermite_interpolation(
            xi[i], xi[i + 1], left_derivs, right_derivs);

        int interval_degree = static_cast<int>(interval_coeffs.size()) - 1;
        if (interval_degree < max_degree) {
            interval_coeffs = elevate_degree(interval_coeffs, max_degree);
        }

        for (int j = 0; j <= max_degree; ++j) {
            coefficients[j][i] = interval_coeffs[j];
        }
    }

    return CPoly(coefficients, xi);
}

inline CPoly CPoly::from_power_basis(std::vector<std::vector<double>> power_coeffs,
                                     std::vector<double> breakpoints,
                                     ExtrapolateMode extrapolate) {
    if (power_coeffs.empty()) {
        throw std::invalid_argument("Power coefficients cannot be empty");
    }
    if (breakpoints.size() < 2) {
        throw std::invalid_argument("Need at least 2 breakpoints");
    }

    int num_intervals = static_cast<int>(breakpoints.size()) - 1;
    int degree = static_cast<int>(power_coeffs.size()) - 1;

    for (const auto& row : power_coeffs) {
        if (static_cast<int>(row.size()) != num_intervals) {
            throw std::invalid_argument("Power coefficients must have num_intervals columns");
        }
    }

    std::vector<std::vector<double>> cheb_coeffs(degree + 1, std::vector<double>(num_intervals));

    auto binomial = [](int n, int k) -> double {
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        if (k > n - k) k = n - k;
        double result = 1.0;
        for (int i = 0; i < k; ++i) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    };

    for (int interval = 0; interval < num_intervals; ++interval) {
        double a = breakpoints[interval];
        double b = breakpoints[interval + 1];
        double h = b - a;

        std::vector<double> power_t(degree + 1);
        double h_power = 1.0;
        for (int k = 0; k <= degree; ++k) {
            power_t[k] = power_coeffs[k][interval] * h_power;
            h_power *= h;
        }

        std::vector<double> power_s(degree + 1, 0.0);
        for (int k = 0; k <= degree; ++k) {
            double scale = std::pow(0.5, k);
            for (int j = 0; j <= k; ++j) {
                power_s[j] += power_t[k] * binomial(k, j) * scale;
            }
        }

        std::vector<std::vector<double>> T_power(degree + 1);
        T_power[0] = {1.0};
        if (degree >= 1) T_power[1] = {0.0, 1.0};
        for (int k = 2; k <= degree; ++k) {
            T_power[k].resize(k + 1, 0.0);
            for (int j = 0; j < static_cast<int>(T_power[k-1].size()); ++j) {
                T_power[k][j + 1] += 2.0 * T_power[k-1][j];
            }
            for (int j = 0; j < static_cast<int>(T_power[k-2].size()); ++j) {
                T_power[k][j] -= T_power[k-2][j];
            }
        }

        std::vector<std::vector<double>> A(degree + 1, std::vector<double>(degree + 1, 0.0));
        for (int k = 0; k <= degree; ++k) {
            for (int j = 0; j < static_cast<int>(T_power[k].size()); ++j) {
                A[j][k] = T_power[k][j];
            }
        }

        std::vector<double> cheb(degree + 1, 0.0);
        for (int k = degree; k >= 0; --k) {
            double sum = power_s[k];
            for (int j = k + 1; j <= degree; ++j) {
                sum -= A[k][j] * cheb[j];
            }
            cheb[k] = (std::abs(A[k][k]) < 1e-15) ? 0.0 : sum / A[k][k];
        }

        for (int j = 0; j <= degree; ++j) {
            cheb_coeffs[j][interval] = cheb[j];
        }
    }

    return CPoly(cheb_coeffs, breakpoints, extrapolate);
}

inline std::vector<std::vector<double>> CPoly::to_power_basis() const {
    int n = degree();
    int n_intervals = num_intervals();

    std::vector<std::vector<double>> power_coeffs(n + 1, std::vector<double>(n_intervals));

    auto binomial = [](int n, int k) -> double {
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        if (k > n - k) k = n - k;
        double result = 1.0;
        for (int i = 0; i < k; ++i) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    };

    for (int interval = 0; interval < n_intervals; ++interval) {
        double a = breakpoints_[interval];
        double b = breakpoints_[interval + 1];
        double h = b - a;

        std::vector<double> cheb(n + 1);
        for (int j = 0; j <= n; ++j) {
            cheb[j] = coefficients_(j, interval);
        }

        std::vector<std::vector<double>> T_power(n + 1);
        T_power[0] = {1.0};
        if (n >= 1) T_power[1] = {0.0, 1.0};
        for (int k = 2; k <= n; ++k) {
            T_power[k].resize(k + 1, 0.0);
            for (int j = 0; j < static_cast<int>(T_power[k-1].size()); ++j) {
                T_power[k][j + 1] += 2.0 * T_power[k-1][j];
            }
            for (int j = 0; j < static_cast<int>(T_power[k-2].size()); ++j) {
                T_power[k][j] -= T_power[k-2][j];
            }
        }

        std::vector<double> power_s(n + 1, 0.0);
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k < static_cast<int>(T_power[j].size()); ++k) {
                power_s[k] += cheb[j] * T_power[j][k];
            }
        }

        std::vector<double> power_t(n + 1, 0.0);
        for (int k = 0; k <= n; ++k) {
            for (int j = 0; j <= k; ++j) {
                double coef = binomial(k, j) * std::pow(2.0, j) * (((k-j) % 2 == 0) ? 1.0 : -1.0);
                power_t[j] += power_s[k] * coef;
            }
        }

        double h_power = 1.0;
        for (int k = 0; k <= n; ++k) {
            power_coeffs[k][interval] = power_t[k] / h_power;
            h_power *= h;
        }
    }

    return power_coeffs;
}
