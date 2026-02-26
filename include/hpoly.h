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
//   numpy.polynomial.hermite (Physicist's H_n)
//   numpy.polynomial.hermite_e (Probabilist's He_n)
//   https://numpy.org/doc/stable/reference/routines.polynomials.hermite.html
//   https://numpy.org/doc/stable/reference/routines.polynomials.hermite_e.html
//
// Algorithm references:
//   [1] Clenshaw algorithm adapted for Hermite polynomial evaluation
//       https://en.wikipedia.org/wiki/Clenshaw_algorithm
//
//   [2] Hermite derivative formula
//       Physicist: d/ds H_n = 2n * H_{n-1}, so c'_k = 2(k+1) * c_{k+1}
//       Probabilist: d/ds He_n = n * He_{n-1}, so c'_k = (k+1) * c_{k+1}
//
//   [3] Hermite antiderivative formula
//       Physicist: integral H_n ds = H_{n+1} / (2(n+1))
//       Probabilist: integral He_n ds = He_{n+1} / (n+1)
// =============================================================================

enum class HermiteKind {
    Physicist,    // H_n: H_{n+1}(s) = 2s*H_n(s) - 2n*H_{n-1}(s)
    Probabilist   // He_n: He_{n+1}(s) = s*He_n(s) - n*He_{n-1}(s)
};

/**
 * Hermite polynomial piecewise interpolant
 *
 * Thread-safe: All methods are const after construction, no shared state.
 *
 * Supports both Physicist's (H_n) and Probabilist's (He_n) Hermite polynomials.
 * - Physicist's: H_{n+1}(s) = 2s*H_n(s) - 2n*H_{n-1}(s), H_0=1, H_1=2s
 * - Probabilist's: He_{n+1}(s) = s*He_n(s) - n*He_{n-1}(s), He_0=1, He_1=s
 * - Domain mapping: s = (2x - a - b) / (b - a)
 */
class HPoly : public PolyBase<HPoly> {
    friend class PolyBase<HPoly>;

public:
    HPoly(std::vector<std::vector<double>> coefficients,
          std::vector<double> breakpoints,
          HermiteKind kind = HermiteKind::Physicist,
          ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
          int extrapolate_order_left = -1,
          int extrapolate_order_right = -1);

    static HPoly from_array2d(ndarray::array2d<double> coefficients,
                              std::vector<double> breakpoints,
                              HermiteKind kind = HermiteKind::Physicist,
                              ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
                              int extrapolate_order_left = -1,
                              int extrapolate_order_right = -1);

    HPoly(HPoly&& other) noexcept = default;
    HPoly(const HPoly& other) = default;
    HPoly& operator=(const HPoly&) = delete;
    HPoly& operator=(HPoly&&) = delete;

    static HPoly from_derivatives(std::vector<double> xi,
                                  std::vector<std::vector<double>> yi,
                                  std::vector<int> orders = {},
                                  HermiteKind kind = HermiteKind::Physicist);

    static HPoly from_power_basis(std::vector<std::vector<double>> power_coeffs,
                                  std::vector<double> breakpoints,
                                  HermiteKind kind = HermiteKind::Physicist,
                                  ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate);

    std::vector<std::vector<double>> to_power_basis() const;

    HermiteKind kind() const { return kind_; }

private:
    HermiteKind kind_;

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
    HPoly make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                    ExtrapolateMode extrap, int ol, int or_) const;
    HPoly make_zero_poly() const;

    // =========================================================================
    // HPoly-specific helpers
    // =========================================================================

    double evaluate_hermite(int interval_idx, double s) const;
    static double hermite_H(int n, double s, HermiteKind kind);
    static double hermite_H_deriv_at_endpoint(int n, int k, double s, HermiteKind kind);

    static std::vector<double> compute_derivative_coeffs(
        const std::vector<double>& c, double h, HermiteKind kind);
    static std::vector<double> compute_antiderivative_coeffs(
        const std::vector<double>& c, double h, double integration_constant, HermiteKind kind);

    static std::vector<double> hermite_interpolation(
        double x0, double x1,
        const std::vector<double>& y0,
        const std::vector<double>& y1,
        HermiteKind kind);

    static std::vector<double> elevate_degree(
        const std::vector<double>& coeffs,
        int target_degree);

    struct FromArray2DTag {};
    HPoly(FromArray2DTag,
          ndarray::array2d<double> coefficients,
          std::vector<double> breakpoints,
          HermiteKind kind,
          ExtrapolateMode extrapolate,
          int extrapolate_order_left,
          int extrapolate_order_right);
};

// =============================================================================
// Implementation
// =============================================================================

// --- Constructors ---

inline HPoly::HPoly(std::vector<std::vector<double>> coefficients,
                   std::vector<double> breakpoints,
                   HermiteKind kind,
                   ExtrapolateMode extrapolate,
                   int extrapolate_order_left,
                   int extrapolate_order_right)
    : PolyBase(ndarray::array2d<double>(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right),
      kind_(kind) {}

inline HPoly::HPoly(FromArray2DTag,
                   ndarray::array2d<double> coefficients,
                   std::vector<double> breakpoints,
                   HermiteKind kind,
                   ExtrapolateMode extrapolate,
                   int extrapolate_order_left,
                   int extrapolate_order_right)
    : PolyBase(std::move(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right),
      kind_(kind) {}

inline HPoly HPoly::from_array2d(ndarray::array2d<double> coefficients,
                                 std::vector<double> breakpoints,
                                 HermiteKind kind,
                                 ExtrapolateMode extrapolate,
                                 int extrapolate_order_left,
                                 int extrapolate_order_right) {
    return HPoly(FromArray2DTag{}, std::move(coefficients), std::move(breakpoints),
                 kind, extrapolate, extrapolate_order_left, extrapolate_order_right);
}

// --- CRTP hooks ---

inline double HPoly::evaluate_basis(int interval_idx, double s) const {
    return evaluate_hermite(interval_idx, s);
}

inline double HPoly::map_to_local(double x, double a, double b) const {
    return (2.0 * x - a - b) / (b - a);
}

inline ndarray::array2d<double> HPoly::compute_derivative_coefficients() const {
    int new_degree = degree() - 1;
    ndarray::array2d<double> deriv_coeffs(new_degree + 1, num_intervals());

    for (int interval = 0; interval < num_intervals(); ++interval) {
        double h = breakpoints_[interval + 1] - breakpoints_[interval];

        std::vector<double> interval_coeffs(degree() + 1);
        for (int j = 0; j <= degree(); ++j) {
            interval_coeffs[j] = coefficients_(j, interval);
        }

        std::vector<double> dc = compute_derivative_coeffs(interval_coeffs, h, kind_);

        for (int j = 0; j <= new_degree; ++j) {
            deriv_coeffs(j, interval) = dc[j];
        }
    }

    return deriv_coeffs;
}

inline ndarray::array2d<double> HPoly::compute_antiderivative_coefficients(double& running_integral) const {
    int new_degree = degree() + 1;
    ndarray::array2d<double> antideriv_coeffs(new_degree + 1, num_intervals());

    for (int interval = 0; interval < num_intervals(); ++interval) {
        double h = breakpoints_[interval + 1] - breakpoints_[interval];

        std::vector<double> interval_coeffs(degree() + 1);
        for (int j = 0; j <= degree(); ++j) {
            interval_coeffs[j] = coefficients_(j, interval);
        }

        std::vector<double> C = compute_antiderivative_coeffs(interval_coeffs, h, running_integral, kind_);

        for (int j = 0; j <= new_degree; ++j) {
            antideriv_coeffs(j, interval) = C[j];
        }

        // Update running integral: value at right endpoint (s = 1)
        // F(1) = sum_k C_k * H_k(1) or He_k(1)
        running_integral = 0.0;
        for (int k = 0; k <= new_degree; ++k) {
            running_integral += C[k] * hermite_H(k, 1.0, kind_);
        }
    }

    return antideriv_coeffs;
}

inline std::vector<double> HPoly::elevate_degree_impl(const std::vector<double>& coeffs, int target_degree) {
    return elevate_degree(coeffs, target_degree);
}

inline HPoly HPoly::make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                              ExtrapolateMode extrap, int ol, int or_) const {
    return HPoly::from_array2d(std::move(coeffs), std::move(breaks), kind_, extrap, ol, or_);
}

inline HPoly HPoly::make_zero_poly() const {
    ndarray::array2d<double> zero_coeffs(1, num_intervals());
    for (int i = 0; i < num_intervals(); ++i) {
        zero_coeffs(0, i) = 0.0;
    }
    return HPoly::from_array2d(std::move(zero_coeffs), breakpoints_, kind_, extrapolate_,
                               extrapolate_order_left_, extrapolate_order_right_);
}

// --- HPoly-specific methods ---

inline double HPoly::hermite_H(int n, double s, HermiteKind kind) {
    // Evaluate H_n(s) or He_n(s) using recurrence
    // Physicist's: H_{n+1} = 2s*H_n - 2n*H_{n-1}, H_0 = 1, H_1 = 2s
    // Probabilist's: He_{n+1} = s*He_n - n*He_{n-1}, He_0 = 1, He_1 = s
    if (n == 0) return 1.0;

    if (kind == HermiteKind::Physicist) {
        if (n == 1) return 2.0 * s;

        double H_prev2 = 1.0;      // H_0
        double H_prev1 = 2.0 * s;  // H_1

        for (int k = 1; k < n; ++k) {
            // H_{k+1} = 2s*H_k - 2k*H_{k-1}
            double H_curr = 2.0 * s * H_prev1 - 2.0 * k * H_prev2;
            H_prev2 = H_prev1;
            H_prev1 = H_curr;
        }
        return H_prev1;
    } else {
        // Probabilist's
        if (n == 1) return s;

        double He_prev2 = 1.0;  // He_0
        double He_prev1 = s;     // He_1

        for (int k = 1; k < n; ++k) {
            // He_{k+1} = s*He_k - k*He_{k-1}
            double He_curr = s * He_prev1 - k * He_prev2;
            He_prev2 = He_prev1;
            He_prev1 = He_curr;
        }
        return He_prev1;
    }
}

inline double HPoly::hermite_H_deriv_at_endpoint(int n, int k, double s, HermiteKind kind) {
    // Compute d^k/ds^k H_n(s) or He_n(s) at s = +1 or s = -1
    //
    // For Physicist's Hermite:
    // d/ds H_n = 2n * H_{n-1}
    // d^k/ds^k H_n = 2^k * n!/(n-k)! * H_{n-k}
    //
    // For Probabilist's Hermite:
    // d/ds He_n = n * He_{n-1}
    // d^k/ds^k He_n = n!/(n-k)! * He_{n-k}

    if (k > n) return 0.0;

    // Compute factorial ratio n!/(n-k)! = n*(n-1)*...*(n-k+1)
    double factorial_ratio = 1.0;
    for (int i = 0; i < k; ++i) {
        factorial_ratio *= (n - i);
    }

    if (kind == HermiteKind::Physicist) {
        // d^k/ds^k H_n = 2^k * (n!/(n-k)!) * H_{n-k}(s)
        double scale = std::pow(2.0, k);
        double H_val = hermite_H(n - k, s, kind);
        return scale * factorial_ratio * H_val;
    } else {
        // d^k/ds^k He_n = (n!/(n-k)!) * He_{n-k}(s)
        double He_val = hermite_H(n - k, s, kind);
        return factorial_ratio * He_val;
    }
}

inline double HPoly::evaluate_hermite(int interval_idx, double s) const {
    // Hermite Clenshaw algorithm [1] for series: sum_{k=0}^{n} c_k * H_k(s) or He_k(s)
    //
    // Physicist's: H_{k+1} = 2s*H_k - 2k*H_{k-1}
    //   Clenshaw: b_k = c_k + 2s*b_{k+1} - 2(k+1)*b_{k+2}
    //
    // Probabilist's: He_{k+1} = s*He_k - k*He_{k-1}
    //   Clenshaw: b_k = c_k + s*b_{k+1} - (k+1)*b_{k+2}

    const int n = degree();

    if (n < 0) return 0.0;
    if (n == 0) return coefficients_(0, interval_idx);

    double b_kplus1 = 0.0;
    double b_kplus2 = 0.0;

    if (kind_ == HermiteKind::Physicist) {
        // Iterate from k = n down to k = 0
        for (int k = n; k >= 0; --k) {
            double b_k = coefficients_(k, interval_idx) + 2.0 * s * b_kplus1 - 2.0 * (k + 1) * b_kplus2;
            b_kplus2 = b_kplus1;
            b_kplus1 = b_k;
        }
    } else {
        // Probabilist's
        for (int k = n; k >= 0; --k) {
            double b_k = coefficients_(k, interval_idx) + s * b_kplus1 - (k + 1) * b_kplus2;
            b_kplus2 = b_kplus1;
            b_kplus1 = b_k;
        }
    }

    return b_kplus1;
}

inline std::vector<double> HPoly::compute_derivative_coeffs(
    const std::vector<double>& c, double h, HermiteKind kind) {
    // Hermite derivative formula [2]
    //
    // Physicist's: d/ds H_n = 2n * H_{n-1}
    //   So for f(s) = sum c_k H_k(s): f'(s) = sum c'_k H_k(s)
    //   c'_k = 2(k+1) * c_{k+1}
    //
    // Probabilist's: d/ds He_n = n * He_{n-1}
    //   c'_k = (k+1) * c_{k+1}
    //
    // Then scale by 2/h for chain rule: df/dx = df/ds * ds/dx = df/ds * (2/h)

    int n = static_cast<int>(c.size()) - 1;  // degree
    if (n <= 0) {
        return {0.0};
    }

    std::vector<double> dc(n, 0.0);

    if (kind == HermiteKind::Physicist) {
        for (int k = 0; k < n; ++k) {
            dc[k] = 2.0 * (k + 1) * c[k + 1];
        }
    } else {
        for (int k = 0; k < n; ++k) {
            dc[k] = (k + 1) * c[k + 1];
        }
    }

    // Scale by 2/h for chain rule
    double scale = 2.0 / h;
    for (auto& coef : dc) {
        coef *= scale;
    }

    return dc;
}

inline std::vector<double> HPoly::compute_antiderivative_coeffs(
    const std::vector<double>& c, double h, double integration_constant, HermiteKind kind) {
    // Hermite antiderivative formula [3]
    //
    // Physicist's: integral H_n ds = H_{n+1} / (2(n+1))
    //   So for f(s) = sum c_k H_k(s): F(s) = sum C_k H_k(s)
    //   C_{k+1} = c_k / (2(k+1))
    //
    // Probabilist's: integral He_n ds = He_{n+1} / (n+1)
    //   C_{k+1} = c_k / (k+1)
    //
    // Then scale by h/2 for chain rule: integral f(x) dx = (h/2) * integral f(s) ds
    // C_0 is determined by the integration constant at left endpoint.

    int n = static_cast<int>(c.size()) - 1;
    std::vector<double> C(n + 2, 0.0);  // Degree increases by 1

    // Scale factor for conversion from s to x
    double scale = h / 2.0;

    if (kind == HermiteKind::Physicist) {
        for (int k = 0; k <= n; ++k) {
            C[k + 1] = scale * c[k] / (2.0 * (k + 1));
        }
    } else {
        for (int k = 0; k <= n; ++k) {
            C[k + 1] = scale * c[k] / (k + 1);
        }
    }

    // C_0 is determined by the integration constant at left endpoint (s = -1)
    // F(-1) = integration_constant
    // F(-1) = sum_k C_k * H_k(-1) or He_k(-1)
    double sum_at_minus1 = 0.0;
    for (int k = 1; k <= n + 1; ++k) {
        sum_at_minus1 += C[k] * hermite_H(k, -1.0, kind);
    }
    C[0] = integration_constant - sum_at_minus1;

    return C;
}

inline std::vector<double> HPoly::hermite_interpolation(
    double x0, double x1,
    const std::vector<double>& y0,
    const std::vector<double>& y1,
    HermiteKind kind) {
    // Solve for Hermite coefficients that satisfy derivative constraints at endpoints
    //
    // At s = -1 (x = x0): f^(k)(x0) = y0[k]
    // At s = +1 (x = x1): f^(k)(x1) = y1[k]
    //
    // f(s) = sum_{j=0}^{n} c_j H_j(s) or He_j(s)
    //
    // For derivatives, we need to track the chain rule: df/dx = (2/h) df/ds
    // f^(k)(x) = (2/h)^k * d^k f/ds^k

    double h = x1 - x0;
    int n0 = static_cast<int>(y0.size());
    int n1 = static_cast<int>(y1.size());
    int n = n0 + n1 - 1;  // Polynomial degree

    if (n0 == 0 || n1 == 0) {
        throw std::invalid_argument("Each point must have at least function value");
    }

    // Build linear system: A * c = b
    std::vector<std::vector<double>> A(n + 1, std::vector<double>(n + 1, 0.0));
    std::vector<double> b(n + 1, 0.0);

    int eq = 0;

    // Left endpoint equations (s = -1, x = x0)
    for (int k = 0; k < n0; ++k) {
        double scale = std::pow(2.0 / h, k);  // Chain rule factor
        for (int j = 0; j <= n; ++j) {
            A[eq][j] = scale * hermite_H_deriv_at_endpoint(j, k, -1.0, kind);
        }
        b[eq] = y0[k];
        ++eq;
    }

    // Right endpoint equations (s = +1, x = x1)
    for (int k = 0; k < n1; ++k) {
        double scale = std::pow(2.0 / h, k);  // Chain rule factor
        for (int j = 0; j <= n; ++j) {
            A[eq][j] = scale * hermite_H_deriv_at_endpoint(j, k, +1.0, kind);
        }
        b[eq] = y1[k];
        ++eq;
    }

    return poly_util::solve_linear_system(A, b, n);
}

inline std::vector<double> HPoly::elevate_degree(
    const std::vector<double>& coeffs,
    int target_degree) {
    // For Hermite polynomials, degree elevation is trivial:
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

inline HPoly HPoly::from_derivatives(std::vector<double> xi,
                                     std::vector<std::vector<double>> yi,
                                     std::vector<int> orders,
                                     HermiteKind kind) {
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

        std::vector<double> interval_coeffs = hermite_interpolation(
            xi[i], xi[i + 1], left_derivs, right_derivs, kind);

        int interval_degree = static_cast<int>(interval_coeffs.size()) - 1;
        if (interval_degree < max_degree) {
            interval_coeffs = elevate_degree(interval_coeffs, max_degree);
        }

        for (int j = 0; j <= max_degree; ++j) {
            coefficients[j][i] = interval_coeffs[j];
        }
    }

    return HPoly(coefficients, xi, kind);
}

inline HPoly HPoly::from_power_basis(std::vector<std::vector<double>> power_coeffs,
                                     std::vector<double> breakpoints,
                                     HermiteKind kind,
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

    // Convert each interval from power basis to Hermite basis
    // Power basis on [a,b]: p(x) = sum_{k=0}^n c_k * (x-a)^k
    // First scale to normalized variable t = (x-a)/(b-a):
    // p(t) = sum_{k=0}^n c_k * (h*t)^k = sum_{k=0}^n (c_k * h^k) * t^k
    // Then convert from t on [0,1] to s on [-1,1]: t = (s+1)/2
    // Then convert from power basis in s to Hermite basis

    std::vector<std::vector<double>> herm_coeffs(degree + 1, std::vector<double>(num_intervals));

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

        // Step 3: Convert from power basis in s to Hermite basis
        // Build Hermite polynomials in power form
        std::vector<std::vector<double>> H_power(degree + 1);
        H_power[0] = {1.0};  // H_0 = 1 or He_0 = 1

        if (kind == HermiteKind::Physicist) {
            if (degree >= 1) H_power[1] = {0.0, 2.0};  // H_1 = 2s
            for (int k = 2; k <= degree; ++k) {
                // H_k = 2s*H_{k-1} - 2(k-1)*H_{k-2}
                H_power[k].resize(k + 1, 0.0);
                for (int j = 0; j < static_cast<int>(H_power[k-1].size()); ++j) {
                    H_power[k][j + 1] += 2.0 * H_power[k-1][j];
                }
                for (int j = 0; j < static_cast<int>(H_power[k-2].size()); ++j) {
                    H_power[k][j] -= 2.0 * (k - 1) * H_power[k-2][j];
                }
            }
        } else {
            // Probabilist's
            if (degree >= 1) H_power[1] = {0.0, 1.0};  // He_1 = s
            for (int k = 2; k <= degree; ++k) {
                // He_k = s*He_{k-1} - (k-1)*He_{k-2}
                H_power[k].resize(k + 1, 0.0);
                for (int j = 0; j < static_cast<int>(H_power[k-1].size()); ++j) {
                    H_power[k][j + 1] += H_power[k-1][j];
                }
                for (int j = 0; j < static_cast<int>(H_power[k-2].size()); ++j) {
                    H_power[k][j] -= (k - 1) * H_power[k-2][j];
                }
            }
        }

        // Build matrix where A[j][k] = coefficient of s^j in H_k or He_k
        std::vector<std::vector<double>> A(degree + 1, std::vector<double>(degree + 1, 0.0));
        for (int k = 0; k <= degree; ++k) {
            for (int j = 0; j < static_cast<int>(H_power[k].size()); ++j) {
                A[j][k] = H_power[k][j];
            }
        }

        // Solve A^T * herm = power_s
        std::vector<double> herm(degree + 1, 0.0);
        for (int k = degree; k >= 0; --k) {
            double sum = power_s[k];
            for (int j = k + 1; j <= degree; ++j) {
                sum -= A[k][j] * herm[j];
            }
            herm[k] = (std::abs(A[k][k]) < 1e-15) ? 0.0 : sum / A[k][k];
        }

        for (int j = 0; j <= degree; ++j) {
            herm_coeffs[j][interval] = herm[j];
        }
    }

    return HPoly(herm_coeffs, breakpoints, kind, extrapolate);
}

inline std::vector<std::vector<double>> HPoly::to_power_basis() const {
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

        // Get Hermite coefficients for this interval
        std::vector<double> herm(n + 1);
        for (int j = 0; j <= n; ++j) {
            herm[j] = coefficients_(j, interval);
        }

        // Build Hermite polynomials in power form (for s in [-1,1])
        std::vector<std::vector<double>> H_power(n + 1);
        H_power[0] = {1.0};

        if (kind_ == HermiteKind::Physicist) {
            if (n >= 1) H_power[1] = {0.0, 2.0};
            for (int k = 2; k <= n; ++k) {
                H_power[k].resize(k + 1, 0.0);
                for (int j = 0; j < static_cast<int>(H_power[k-1].size()); ++j) {
                    H_power[k][j + 1] += 2.0 * H_power[k-1][j];
                }
                for (int j = 0; j < static_cast<int>(H_power[k-2].size()); ++j) {
                    H_power[k][j] -= 2.0 * (k - 1) * H_power[k-2][j];
                }
            }
        } else {
            if (n >= 1) H_power[1] = {0.0, 1.0};
            for (int k = 2; k <= n; ++k) {
                H_power[k].resize(k + 1, 0.0);
                for (int j = 0; j < static_cast<int>(H_power[k-1].size()); ++j) {
                    H_power[k][j + 1] += H_power[k-1][j];
                }
                for (int j = 0; j < static_cast<int>(H_power[k-2].size()); ++j) {
                    H_power[k][j] -= (k - 1) * H_power[k-2][j];
                }
            }
        }

        // Convert Hermite to power basis in s
        std::vector<double> power_s(n + 1, 0.0);
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k < static_cast<int>(H_power[j].size()); ++k) {
                power_s[k] += herm[j] * H_power[j][k];
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
