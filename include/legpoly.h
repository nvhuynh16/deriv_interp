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
//   numpy.polynomial.legendre
//   https://github.com/numpy/numpy/blob/main/numpy/polynomial/legendre.py
//   https://numpy.org/doc/stable/reference/routines.polynomials.legendre.html
//
// Algorithm references:
//   [1] Clenshaw algorithm for Legendre polynomial evaluation
//       https://en.wikipedia.org/wiki/Clenshaw_algorithm
//       numpy.polynomial.legendre.legval()
//
//   [2] Legendre derivative formula (legder)
//       numpy.polynomial.legendre.legder()
//
//   [3] Legendre antiderivative formula (legint)
//       numpy.polynomial.legendre.legint()
// =============================================================================

/**
 * Legendre polynomial piecewise interpolant
 *
 * Thread-safe: All methods are const after construction, no shared state.
 *
 * Mathematical foundation:
 * - Legendre basis: P_n(s) orthogonal on [-1,1] with weight w(s) = 1
 * - Recurrence: (n+1)*P_{n+1}(s) = (2n+1)*s*P_n(s) - n*P_{n-1}(s)
 * - Domain mapping: s = (2x - a - b) / (b - a)
 */
class LegPoly : public PolyBase<LegPoly> {
    friend class PolyBase<LegPoly>;

public:
    LegPoly(std::vector<std::vector<double>> coefficients,
            std::vector<double> breakpoints,
            ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
            int extrapolate_order_left = -1,
            int extrapolate_order_right = -1);

    static LegPoly from_array2d(ndarray::array2d<double> coefficients,
                                std::vector<double> breakpoints,
                                ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
                                int extrapolate_order_left = -1,
                                int extrapolate_order_right = -1);

    LegPoly(LegPoly&& other) noexcept = default;
    LegPoly(const LegPoly& other) = default;
    LegPoly& operator=(const LegPoly&) = delete;
    LegPoly& operator=(LegPoly&&) = delete;

    static LegPoly from_derivatives(std::vector<double> xi,
                                    std::vector<std::vector<double>> yi,
                                    std::vector<int> orders = {});

    static LegPoly from_power_basis(std::vector<std::vector<double>> power_coeffs,
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
    LegPoly make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                      ExtrapolateMode extrap, int ol, int or_) const;
    LegPoly make_zero_poly() const;

    // =========================================================================
    // LegPoly-specific helpers
    // =========================================================================

    double evaluate_legendre(int interval_idx, double s) const;
    static double legendre_P(int n, double s);
    static double legendre_P_deriv_at_endpoint(int n, int k, double s);

    static std::vector<double> compute_derivative_coeffs(
        const std::vector<double>& c, double h);
    static std::vector<double> compute_antiderivative_coeffs(
        const std::vector<double>& c, double h, double integration_constant);

    static std::vector<double> legendre_hermite_interpolation(
        double x0, double x1,
        const std::vector<double>& y0,
        const std::vector<double>& y1);

    static std::vector<double> elevate_degree(
        const std::vector<double>& coeffs,
        int target_degree);

    struct FromArray2DTag {};
    LegPoly(FromArray2DTag,
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

inline LegPoly::LegPoly(std::vector<std::vector<double>> coefficients,
                         std::vector<double> breakpoints,
                         ExtrapolateMode extrapolate,
                         int extrapolate_order_left,
                         int extrapolate_order_right)
    : PolyBase(ndarray::array2d<double>(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right) {}

inline LegPoly::LegPoly(FromArray2DTag,
                         ndarray::array2d<double> coefficients,
                         std::vector<double> breakpoints,
                         ExtrapolateMode extrapolate,
                         int extrapolate_order_left,
                         int extrapolate_order_right)
    : PolyBase(std::move(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right) {}

inline LegPoly LegPoly::from_array2d(ndarray::array2d<double> coefficients,
                                     std::vector<double> breakpoints,
                                     ExtrapolateMode extrapolate,
                                     int extrapolate_order_left,
                                     int extrapolate_order_right) {
    return LegPoly(FromArray2DTag{}, std::move(coefficients), std::move(breakpoints),
                   extrapolate, extrapolate_order_left, extrapolate_order_right);
}

// --- CRTP hooks ---

inline double LegPoly::evaluate_basis(int interval_idx, double s) const {
    return evaluate_legendre(interval_idx, s);
}

inline double LegPoly::map_to_local(double x, double a, double b) const {
    return (2.0 * x - a - b) / (b - a);
}

inline ndarray::array2d<double> LegPoly::compute_derivative_coefficients() const {
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

inline ndarray::array2d<double> LegPoly::compute_antiderivative_coefficients(double& running_integral) const {
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
        // F(1) = sum_k C_k * P_k(1) = sum_k C_k (since P_k(1) = 1)
        running_integral = 0.0;
        for (int k = 0; k <= new_degree; ++k) {
            running_integral += C[k];
        }
    }

    return antideriv_coeffs;
}

inline std::vector<double> LegPoly::elevate_degree_impl(const std::vector<double>& coeffs, int target_degree) {
    return elevate_degree(coeffs, target_degree);
}

inline LegPoly LegPoly::make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                                  ExtrapolateMode extrap, int ol, int or_) const {
    return LegPoly::from_array2d(std::move(coeffs), std::move(breaks), extrap, ol, or_);
}

inline LegPoly LegPoly::make_zero_poly() const {
    ndarray::array2d<double> zero_coeffs(1, num_intervals());
    for (int i = 0; i < num_intervals(); ++i) {
        zero_coeffs(0, i) = 0.0;
    }
    return LegPoly::from_array2d(std::move(zero_coeffs), breakpoints_, extrapolate_,
                                 extrapolate_order_left_, extrapolate_order_right_);
}

// --- LegPoly-specific methods ---

inline double LegPoly::legendre_P(int n, double s) {
    // Evaluate P_n(s) using recurrence: (n+1)*P_{n+1} = (2n+1)*s*P_n - n*P_{n-1}
    if (n == 0) return 1.0;
    if (n == 1) return s;

    double P_prev2 = 1.0;  // P_0
    double P_prev1 = s;     // P_1

    for (int k = 1; k < n; ++k) {
        // (k+1)*P_{k+1} = (2k+1)*s*P_k - k*P_{k-1}
        double P_curr = ((2.0 * k + 1.0) * s * P_prev1 - k * P_prev2) / (k + 1.0);
        P_prev2 = P_prev1;
        P_prev1 = P_curr;
    }

    return P_prev1;
}

inline double LegPoly::legendre_P_deriv_at_endpoint(int n, int k, double s) {
    // Compute d^k/ds^k P_n(s) at s = +1 or s = -1
    //
    // For Legendre polynomials:
    // P_n(1) = 1
    // P_n(-1) = (-1)^n
    //
    // Higher derivatives at s = 1:
    // P_n^(k)(1) = prod_{m=0}^{k-1} (n-m)(n+m+1) / (2(m+1))
    //
    // At s = -1:
    // P_n^(k)(-1) = (-1)^{n+k} * P_n^(k)(1)

    if (k > n) return 0.0;

    if (k == 0) {
        if (s > 0) return 1.0;
        return (n % 2 == 0) ? 1.0 : -1.0;
    }

    double prod = 1.0;
    for (int m = 0; m < k; ++m) {
        prod *= static_cast<double>((n - m) * (n + m + 1)) / (2.0 * (m + 1));
    }

    if (s > 0) {
        return prod;
    } else {
        double sign = ((n + k) % 2 == 0) ? 1.0 : -1.0;
        return sign * prod;
    }
}

inline double LegPoly::evaluate_legendre(int interval_idx, double s) const {
    // Legendre Clenshaw algorithm [1] for series: sum_{k=0}^{n} c_k * P_k(s)
    // Reference: numpy.polynomial.legendre.legval()
    //
    // The Legendre recurrence is: (k+1)*P_{k+1} = (2k+1)*s*P_k - k*P_{k-1}
    // Rewritten: P_{k+1} = alpha_k * s * P_k - beta_k * P_{k-1}
    // where alpha_k = (2k+1)/(k+1), beta_k = k/(k+1)
    //
    // Clenshaw algorithm (backwards iteration):
    // b_{n+1} = 0, b_{n+2} = 0
    // For k = n, n-1, ..., 1:
    //   b_k = c_k + alpha_k * s * b_{k+1} - beta_{k+1} * b_{k+2}
    // Result = c_0 + s * b_1 - beta_1 * b_2 = c_0 + s * b_1 - (1/2) * b_2

    const int n = degree();

    if (n < 0) return 0.0;
    if (n == 0) return coefficients_(0, interval_idx);
    if (n == 1) return coefficients_(0, interval_idx) + coefficients_(1, interval_idx) * s;

    double b_kplus1 = 0.0;
    double b_kplus2 = 0.0;

    // Iterate from k = n down to k = 1
    for (int k = n; k >= 1; --k) {
        double alpha_k = (2.0 * k + 1.0) / (k + 1.0);
        double beta_kplus1 = (k + 1.0) / (k + 2.0);
        double b_k = coefficients_(k, interval_idx) + alpha_k * s * b_kplus1 - beta_kplus1 * b_kplus2;
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }

    // Final step: result = c_0 + s*b_1 - beta_1*b_2
    // beta_1 = 1/2
    return coefficients_(0, interval_idx) + s * b_kplus1 - 0.5 * b_kplus2;
}

inline std::vector<double> LegPoly::compute_derivative_coeffs(
    const std::vector<double>& c, double h) {
    // Legendre derivative formula [2]
    // Reference: numpy.polynomial.legendre.legder()
    //
    // Algorithm (from numpy source):
    // Step 1: Accumulate higher terms: c[i-1] += c[i+1] for i = n-1 down to 2
    // Step 2: Scale: d[i] = (2i+1) * c[i+1] for i = 0 to n-2
    // Then scale by 2/h to convert from d/ds to d/dx

    int n = static_cast<int>(c.size()) - 1;
    if (n <= 0) {
        return {0.0};
    }

    // Step 1: Accumulate higher terms (working on a copy)
    std::vector<double> c_work = c;
    for (int i = n; i >= 2; --i) {
        if (i + 1 <= n) {
            c_work[i - 1] += c_work[i + 1];
        }
    }

    // Step 2: Compute derivative coefficients
    std::vector<double> dc(n);
    for (int i = 0; i < n; ++i) {
        dc[i] = (2.0 * i + 1.0) * c_work[i + 1];
    }

    // Scale by 2/h for chain rule: df/dx = df/ds * ds/dx = df/ds * (2/h)
    double scale = 2.0 / h;
    for (auto& coef : dc) {
        coef *= scale;
    }

    return dc;
}

inline std::vector<double> LegPoly::compute_antiderivative_coeffs(
    const std::vector<double>& c, double h, double integration_constant) {
    // Legendre antiderivative formula [3]
    // Reference: numpy.polynomial.legendre.legint()
    //
    // integral P_k ds = (P_{k+1} - P_{k-1}) / (2k+1) for k >= 1
    // integral P_0 ds = s = P_1
    //
    // So:
    // C_1 += c_0 (from integral P_0)
    // C_{k+1} += c_k / (2k+1) for k >= 1
    // C_{k-1} -= c_k / (2k+1) for k >= 1
    //
    // Then scale by h/2 for the x-domain.
    // C_0 is determined by the integration constant at left endpoint.

    int n = static_cast<int>(c.size()) - 1;
    std::vector<double> C(n + 2, 0.0);

    double scale = h / 2.0;

    // From integral P_0 ds = P_1
    C[1] += scale * c[0];

    // From integral P_k ds = (P_{k+1} - P_{k-1}) / (2k+1) for k >= 1
    for (int k = 1; k <= n; ++k) {
        double factor = scale * c[k] / (2.0 * k + 1.0);
        if (k + 1 <= n + 1) {
            C[k + 1] += factor;
        }
        if (k - 1 >= 0) {
            C[k - 1] -= factor;
        }
    }

    // C_0 is determined by the integration constant at left endpoint (s = -1)
    // F(-1) = integration_constant
    // F(-1) = sum_k C_k * P_k(-1) = sum_k C_k * (-1)^k
    // So C_0 = integration_constant - sum_{k>=1} C_k * (-1)^k
    double sum_at_minus1 = 0.0;
    for (int k = 1; k <= n + 1; ++k) {
        double sign = (k % 2 == 0) ? 1.0 : -1.0;
        sum_at_minus1 += C[k] * sign;
    }
    C[0] = integration_constant - sum_at_minus1;

    return C;
}

inline std::vector<double> LegPoly::legendre_hermite_interpolation(
    double x0, double x1,
    const std::vector<double>& y0,
    const std::vector<double>& y1) {
    // Solve for Legendre coefficients that satisfy derivative constraints at endpoints
    //
    // At s = -1 (x = x0): f^(k)(x0) = y0[k]
    // At s = +1 (x = x1): f^(k)(x1) = y1[k]
    //
    // f(s) = sum_{j=0}^{n} c_j P_j(s)
    //
    // For derivatives, chain rule: df/dx = (2/h) df/ds
    // f^(k)(x) = (2/h)^k * d^k f/ds^k

    double h = x1 - x0;
    int n0 = static_cast<int>(y0.size());
    int n1 = static_cast<int>(y1.size());
    int n = n0 + n1 - 1;

    if (n0 == 0 || n1 == 0) {
        throw std::invalid_argument("Each point must have at least function value");
    }

    // Build linear system: A * c = b
    std::vector<std::vector<double>> A(n + 1, std::vector<double>(n + 1, 0.0));
    std::vector<double> b(n + 1, 0.0);

    int eq = 0;

    // Left endpoint equations (s = -1, x = x0)
    for (int k = 0; k < n0; ++k) {
        double scale = std::pow(2.0 / h, k);
        for (int j = 0; j <= n; ++j) {
            A[eq][j] = scale * legendre_P_deriv_at_endpoint(j, k, -1.0);
        }
        b[eq] = y0[k];
        ++eq;
    }

    // Right endpoint equations (s = +1, x = x1)
    for (int k = 0; k < n1; ++k) {
        double scale = std::pow(2.0 / h, k);
        for (int j = 0; j <= n; ++j) {
            A[eq][j] = scale * legendre_P_deriv_at_endpoint(j, k, +1.0);
        }
        b[eq] = y1[k];
        ++eq;
    }

    return poly_util::solve_linear_system(A, b, n);
}

inline std::vector<double> LegPoly::elevate_degree(
    const std::vector<double>& coeffs,
    int target_degree) {
    // For Legendre polynomials, degree elevation is trivial:
    // Just pad with zeros for higher-order coefficients
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

inline LegPoly LegPoly::from_derivatives(std::vector<double> xi,
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

        std::vector<double> interval_coeffs = legendre_hermite_interpolation(
            xi[i], xi[i + 1], left_derivs, right_derivs);

        int interval_degree = static_cast<int>(interval_coeffs.size()) - 1;
        if (interval_degree < max_degree) {
            interval_coeffs = elevate_degree(interval_coeffs, max_degree);
        }

        for (int j = 0; j <= max_degree; ++j) {
            coefficients[j][i] = interval_coeffs[j];
        }
    }

    return LegPoly(coefficients, xi);
}

inline LegPoly LegPoly::from_power_basis(std::vector<std::vector<double>> power_coeffs,
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

    std::vector<std::vector<double>> leg_coeffs(degree + 1, std::vector<double>(num_intervals));

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

        // Step 1: Scale power coefficients from (x-a) to normalized t = (x-a)/h
        std::vector<double> power_t(degree + 1);
        double h_power = 1.0;
        for (int k = 0; k <= degree; ++k) {
            power_t[k] = power_coeffs[k][interval] * h_power;
            h_power *= h;
        }

        // Step 2: Convert from t in [0,1] to s in [-1,1]
        // t = (s+1)/2, so t^k = sum_j binomial(k,j) * s^j * (1/2)^k
        std::vector<double> power_s(degree + 1, 0.0);
        for (int k = 0; k <= degree; ++k) {
            double scale = std::pow(0.5, k);
            for (int j = 0; j <= k; ++j) {
                power_s[j] += power_t[k] * binomial(k, j) * scale;
            }
        }

        // Step 3: Convert from power basis in s to Legendre basis
        // Build Legendre polynomials in power form
        std::vector<std::vector<double>> P_power(degree + 1);
        P_power[0] = {1.0};
        if (degree >= 1) P_power[1] = {0.0, 1.0};

        for (int k = 2; k <= degree; ++k) {
            P_power[k].resize(k + 1, 0.0);
            double coef1 = (2.0 * k - 1.0) / k;
            double coef2 = (k - 1.0) / k;
            for (int j = 0; j < static_cast<int>(P_power[k-1].size()); ++j) {
                P_power[k][j + 1] += coef1 * P_power[k-1][j];
            }
            for (int j = 0; j < static_cast<int>(P_power[k-2].size()); ++j) {
                P_power[k][j] -= coef2 * P_power[k-2][j];
            }
        }

        // Build matrix where A[j][k] = coefficient of s^j in P_k
        std::vector<std::vector<double>> A(degree + 1, std::vector<double>(degree + 1, 0.0));
        for (int k = 0; k <= degree; ++k) {
            for (int j = 0; j < static_cast<int>(P_power[k].size()); ++j) {
                A[j][k] = P_power[k][j];
            }
        }

        // Solve for Legendre coefficients
        std::vector<double> leg(degree + 1, 0.0);
        for (int k = degree; k >= 0; --k) {
            double sum = power_s[k];
            for (int j = k + 1; j <= degree; ++j) {
                sum -= A[k][j] * leg[j];
            }
            leg[k] = (std::abs(A[k][k]) < 1e-15) ? 0.0 : sum / A[k][k];
        }

        for (int j = 0; j <= degree; ++j) {
            leg_coeffs[j][interval] = leg[j];
        }
    }

    return LegPoly(leg_coeffs, breakpoints, extrapolate);
}

inline std::vector<std::vector<double>> LegPoly::to_power_basis() const {
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

        // Get Legendre coefficients for this interval
        std::vector<double> leg(n + 1);
        for (int j = 0; j <= n; ++j) {
            leg[j] = coefficients_(j, interval);
        }

        // Build Legendre polynomials in power form (for s in [-1,1])
        std::vector<std::vector<double>> P_power(n + 1);
        P_power[0] = {1.0};
        if (n >= 1) P_power[1] = {0.0, 1.0};
        for (int k = 2; k <= n; ++k) {
            P_power[k].resize(k + 1, 0.0);
            double coef1 = (2.0 * k - 1.0) / k;
            double coef2 = (k - 1.0) / k;
            for (int j = 0; j < static_cast<int>(P_power[k-1].size()); ++j) {
                P_power[k][j + 1] += coef1 * P_power[k-1][j];
            }
            for (int j = 0; j < static_cast<int>(P_power[k-2].size()); ++j) {
                P_power[k][j] -= coef2 * P_power[k-2][j];
            }
        }

        // Convert Legendre to power basis in s
        std::vector<double> power_s(n + 1, 0.0);
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k < static_cast<int>(P_power[j].size()); ++k) {
                power_s[k] += leg[j] * P_power[j][k];
            }
        }

        // Convert from s in [-1,1] to t in [0,1]: s = 2t - 1
        std::vector<double> power_t(n + 1, 0.0);
        for (int k = 0; k <= n; ++k) {
            for (int j = 0; j <= k; ++j) {
                double coef = binomial(k, j) * std::pow(2.0, j) * (((k-j) % 2 == 0) ? 1.0 : -1.0);
                power_t[j] += power_s[k] * coef;
            }
        }

        // Convert from t = (x-a)/h to (x-a)
        double h_power = 1.0;
        for (int k = 0; k <= n; ++k) {
            power_coeffs[k][interval] = power_t[k] / h_power;
            h_power *= h;
        }
    }

    return power_coeffs;
}
