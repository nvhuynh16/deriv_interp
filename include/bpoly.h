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
//   scipy.interpolate.BPoly
//   https://github.com/scipy/scipy/blob/main/scipy/interpolate/_interpolate.py
//   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.html
//
// Algorithm references:
//   [1] De Casteljau's algorithm for Bernstein polynomial evaluation
//       https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
//
//   [2] Hermite interpolation in Bernstein basis (from_derivatives)
//       scipy/interpolate/_bpoly.pyx: _construct_from_derivatives()
//       Uses forward/backward difference formulas for endpoint constraints
//
//   [3] Bernstein polynomial derivative formula
//       d/dt[B_{j,n}(t)] = n * (B_{j-1,n-1}(t) - B_{j,n-1}(t))
//       https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties
//
//   [4] Degree elevation formula
//       c'[i] = (i/(n+1)) * c[i-1] + ((n+1-i)/(n+1)) * c[i]
//       https://en.wikipedia.org/wiki/Bernstein_polynomial#Degree_elevation
//
//   [5] Power-to-Bernstein basis conversion
//       b_j = sum_{k=0}^j C(j,k)/C(n,k) * a_k
//       https://en.wikipedia.org/wiki/Bernstein_polynomial#Conversion_to_B%C3%A9zier_form
// =============================================================================

/**
 * Bernstein polynomial implementation matching scipy.interpolate.BPoly EXACTLY
 *
 * Thread-safe: All methods are const after construction, no shared state.
 * KISS: Simple implementation prioritizing correctness over optimization.
 *
 * Mathematical foundation:
 * - Bernstein basis: B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
 * - Polynomial on [x_i, x_{i+1}] evaluated using normalized t = (x - x_i) / (x_{i+1} - x_i)
 * - Coefficients stored as [degree+1][num_intervals] matching scipy exactly
 * - Verified against scipy test data for numerical accuracy within 1e-12
 */
class BPoly : public PolyBase<BPoly> {
    friend class PolyBase<BPoly>;

public:
    /**
     * Construct from coefficients and breakpoints
     *
     * @param coefficients Polynomial coefficients [degree+1][num_intervals]
     * @param breakpoints Interval breakpoints (sorted, size = num_intervals + 1)
     * @param extrapolate Extrapolation mode (default: Extrapolate)
     * @param extrapolate_order_left Order of Taylor expansion for left extrapolation (-1 = full)
     * @param extrapolate_order_right Order of Taylor expansion for right extrapolation (-1 = full)
     */
    BPoly(std::vector<std::vector<double>> coefficients,
          std::vector<double> breakpoints,
          ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
          int extrapolate_order_left = -1,
          int extrapolate_order_right = -1);

    /**
     * Construct from array2d coefficients (zero-copy capable)
     */
    static BPoly from_array2d(ndarray::array2d<double> coefficients,
                              std::vector<double> breakpoints,
                              ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
                              int extrapolate_order_left = -1,
                              int extrapolate_order_right = -1);

    BPoly(BPoly&& other) noexcept = default;
    BPoly(const BPoly& other) = default;
    BPoly& operator=(const BPoly&) = delete;
    BPoly& operator=(BPoly&&) = delete;

    /**
     * Construct from derivatives (matches scipy.BPoly.from_derivatives EXACTLY)
     */
    static BPoly from_derivatives(std::vector<double> xi,
                                  std::vector<std::vector<double>> yi,
                                  std::vector<int> orders = {});

    /**
     * Construct BPoly from power basis (monomial) coefficients
     */
    static BPoly from_power_basis(std::vector<std::vector<double>> power_coeffs,
                                  std::vector<double> breakpoints,
                                  ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate);

    /**
     * Convert to power basis (monomial) coefficients
     */
    std::vector<std::vector<double>> to_power_basis() const;

private:
    // =========================================================================
    // CRTP hooks for PolyBase
    // =========================================================================

    double evaluate_basis(int interval_idx, double t) const;
    double map_to_local(double x, double a, double b) const;
    static constexpr double breakpoint_left_eval_param() { return 0.0; }
    static constexpr double breakpoint_right_eval_param() { return 1.0; }
    ndarray::array2d<double> compute_derivative_coefficients() const;
    ndarray::array2d<double> compute_antiderivative_coefficients(double& running_integral) const;
    static std::vector<double> elevate_degree_impl(const std::vector<double>& coeffs, int target_degree);
    BPoly make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                    ExtrapolateMode extrap, int ol, int or_) const;
    BPoly make_zero_poly() const;

    // =========================================================================
    // BPoly-specific helpers
    // =========================================================================

    /**
     * Evaluate Bernstein polynomial using de Casteljau's algorithm [1]
     */
    double evaluate_bernstein(int interval_idx, double t) const;

    static double binomial_coefficient(int n, int k);

    /**
     * Hermite interpolation in Bernstein basis [2]
     */
    static std::vector<double> hermite_interpolation(
        double x0, double x1,
        const std::vector<double>& y0,
        const std::vector<double>& y1);

    /**
     * Degree elevation for Bernstein coefficients [4]
     */
    static std::vector<double> elevate_degree(
        const std::vector<double>& coeffs,
        int target_degree);

    struct FromArray2DTag {};
    BPoly(FromArray2DTag,
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

inline BPoly::BPoly(std::vector<std::vector<double>> coefficients,
                   std::vector<double> breakpoints,
                   ExtrapolateMode extrapolate,
                   int extrapolate_order_left,
                   int extrapolate_order_right)
    : PolyBase(ndarray::array2d<double>(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right) {}

inline BPoly::BPoly(FromArray2DTag,
                   ndarray::array2d<double> coefficients,
                   std::vector<double> breakpoints,
                   ExtrapolateMode extrapolate,
                   int extrapolate_order_left,
                   int extrapolate_order_right)
    : PolyBase(std::move(coefficients), std::move(breakpoints),
               extrapolate, extrapolate_order_left, extrapolate_order_right) {}

inline BPoly BPoly::from_array2d(ndarray::array2d<double> coefficients,
                                 std::vector<double> breakpoints,
                                 ExtrapolateMode extrapolate,
                                 int extrapolate_order_left,
                                 int extrapolate_order_right) {
    return BPoly(FromArray2DTag{}, std::move(coefficients), std::move(breakpoints),
                 extrapolate, extrapolate_order_left, extrapolate_order_right);
}

// --- CRTP hooks ---

inline double BPoly::evaluate_basis(int interval_idx, double t) const {
    return evaluate_bernstein(interval_idx, t);
}

inline double BPoly::map_to_local(double x, double a, double b) const {
    return (x - a) / (b - a);
}

inline ndarray::array2d<double> BPoly::compute_derivative_coefficients() const {
    // Bernstein derivative formula [3]:
    // d/dx [sum(c_j * B_{j,n}(t))] = n/h * sum((c_{j+1} - c_j) * B_{j,n-1}(t))
    int new_degree = degree() - 1;
    ndarray::array2d<double> deriv_coeffs(new_degree + 1, num_intervals());

    for (int interval = 0; interval < num_intervals(); ++interval) {
        double h = breakpoints_[interval + 1] - breakpoints_[interval];
        double scale_factor = degree() / h;

        for (int j = 0; j <= new_degree; ++j) {
            deriv_coeffs(j, interval) = scale_factor *
                (coefficients_(j + 1, interval) - coefficients_(j, interval));
        }
    }

    return deriv_coeffs;
}

inline ndarray::array2d<double> BPoly::compute_antiderivative_coefficients(double& running_integral) const {
    // Bernstein antiderivative: integral increases degree by 1
    // C_j = h/(n+1) * cumsum(c_k) for k=0 to j-1, plus integration constant
    int new_degree = degree() + 1;
    ndarray::array2d<double> antideriv_coeffs(new_degree + 1, num_intervals());

    for (int interval = 0; interval < num_intervals(); ++interval) {
        double h = breakpoints_[interval + 1] - breakpoints_[interval];
        double scale_factor = h / new_degree;
        double cumsum = 0.0;

        antideriv_coeffs(0, interval) = running_integral;

        for (int j = 1; j <= new_degree; ++j) {
            if (j - 1 <= degree()) {
                cumsum += coefficients_(j - 1, interval);
            }
            antideriv_coeffs(j, interval) = running_integral + cumsum * scale_factor;
        }

        running_integral = antideriv_coeffs(new_degree, interval);
    }

    return antideriv_coeffs;
}

inline std::vector<double> BPoly::elevate_degree_impl(const std::vector<double>& coeffs, int target_degree) {
    return elevate_degree(coeffs, target_degree);
}

inline BPoly BPoly::make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
                              ExtrapolateMode extrap, int ol, int or_) const {
    return BPoly::from_array2d(std::move(coeffs), std::move(breaks), extrap, ol, or_);
}

inline BPoly BPoly::make_zero_poly() const {
    ndarray::array2d<double> zero_coeffs(1, num_intervals());
    for (int i = 0; i < num_intervals(); ++i) {
        zero_coeffs(0, i) = 0.0;
    }
    return BPoly::from_array2d(std::move(zero_coeffs), breakpoints_, extrapolate_,
                               extrapolate_order_left_, extrapolate_order_right_);
}

// --- BPoly-specific methods ---

inline double BPoly::evaluate_bernstein(int interval_idx, double t) const {
    const int n = degree();

    // De Casteljau's algorithm [1] for numerical stability
    constexpr int STACK_THRESHOLD = 32;
    double stack_buffer[STACK_THRESHOLD];
    std::vector<double> heap_buffer;

    double* b;
    if (n + 1 <= STACK_THRESHOLD) {
        b = stack_buffer;
    } else {
        heap_buffer.resize(n + 1);
        b = heap_buffer.data();
    }

    for (int i = 0; i <= n; ++i) {
        b[i] = coefficients_(i, interval_idx);
    }

    double one_minus_t = 1.0 - t;
    for (int r = 1; r <= n; ++r) {
        for (int i = 0; i <= n - r; ++i) {
            b[i] = one_minus_t * b[i] + t * b[i + 1];
        }
    }

    return b[0];
}

inline double BPoly::binomial_coefficient(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;

    if (k > n - k) {
        k = n - k;
    }

    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }

    return result;
}

inline std::vector<double> BPoly::hermite_interpolation(
    double x0, double x1,
    const std::vector<double>& y0,
    const std::vector<double>& y1) {
    // Hermite interpolation in Bernstein basis [2]
    // Reference: scipy/interpolate/_bpoly.pyx: _construct_from_derivatives()

    double h = x1 - x0;
    int n0 = static_cast<int>(y0.size());
    int n1 = static_cast<int>(y1.size());
    int n = n0 + n1 - 1;

    if (n0 == 0 || n1 == 0) {
        throw std::invalid_argument("Each point must have at least function value");
    }

    std::vector<double> c(n + 1, 0.0);

    // Left endpoint: forward differences
    for (int k = 0; k < n0; ++k) {
        double falling_fact = 1.0;
        for (int i = 0; i < k; ++i) {
            falling_fact *= (n - i);
        }

        double target = (k == 0) ? y0[k] : y0[k] * std::pow(h, k) / falling_fact;

        double sum = 0.0;
        for (int j = 0; j < k; ++j) {
            double sign = ((k - j) % 2 == 0) ? 1.0 : -1.0;
            sum += sign * binomial_coefficient(k, j) * c[j];
        }
        c[k] = target - sum;
    }

    // Right endpoint: backward differences
    for (int k = 0; k < n1; ++k) {
        double falling_fact = 1.0;
        for (int i = 0; i < k; ++i) {
            falling_fact *= (n - i);
        }

        double target = (k == 0) ? y1[k] : y1[k] * std::pow(h, k) / falling_fact;

        double sum = 0.0;
        for (int j = 1; j <= k; ++j) {
            double sign = ((k - j) % 2 == 0) ? 1.0 : -1.0;
            sum += sign * binomial_coefficient(k, j) * c[n - k + j];
        }
        double sign_k = (k % 2 == 0) ? 1.0 : -1.0;
        c[n - k] = sign_k * (target - sum);
    }

    return c;
}

inline std::vector<double> BPoly::elevate_degree(
    const std::vector<double>& coeffs,
    int target_degree) {
    // Degree elevation formula [4]
    int n = static_cast<int>(coeffs.size()) - 1;

    if (target_degree < n) {
        throw std::invalid_argument("Target degree must be >= current degree");
    }
    if (target_degree == n) {
        return coeffs;
    }

    std::vector<double> current = coeffs;

    while (static_cast<int>(current.size()) - 1 < target_degree) {
        int curr_n = static_cast<int>(current.size()) - 1;
        int new_n = curr_n + 1;
        std::vector<double> elevated(new_n + 1);

        for (int i = 0; i <= new_n; ++i) {
            double left_term = 0.0;
            double right_term = 0.0;

            if (i > 0 && i - 1 <= curr_n) {
                left_term = (static_cast<double>(i) / new_n) * current[i - 1];
            }
            if (i <= curr_n) {
                right_term = (static_cast<double>(new_n - i) / new_n) * current[i];
            }

            elevated[i] = left_term + right_term;
        }

        current = std::move(elevated);
    }

    return current;
}

// --- Static factories ---

inline BPoly BPoly::from_derivatives(std::vector<double> xi,
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

        std::vector<double> interval_coeffs = hermite_interpolation(
            xi[i], xi[i + 1], left_derivs, right_derivs);

        int interval_degree = static_cast<int>(interval_coeffs.size()) - 1;
        if (interval_degree < max_degree) {
            interval_coeffs = elevate_degree(interval_coeffs, max_degree);
        }

        for (int j = 0; j <= max_degree; ++j) {
            coefficients[j][i] = interval_coeffs[j];
        }
    }

    return BPoly(coefficients, xi);
}

// Power-to-Bernstein conversion [5]
inline BPoly BPoly::from_power_basis(std::vector<std::vector<double>> power_coeffs,
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

    std::vector<std::vector<double>> bern_coeffs(degree + 1, std::vector<double>(num_intervals));

    for (int interval = 0; interval < num_intervals; ++interval) {
        double a = breakpoints[interval];
        double b = breakpoints[interval + 1];
        double h = b - a;

        // Scale power coefficients: a_k = c_k * h^k
        std::vector<double> scaled_power(degree + 1);
        double h_power = 1.0;
        for (int k = 0; k <= degree; ++k) {
            scaled_power[k] = power_coeffs[k][interval] * h_power;
            h_power *= h;
        }

        // b_j = sum_{k=0}^j C(j,k)/C(n,k) * a_k
        for (int j = 0; j <= degree; ++j) {
            double sum = 0.0;
            for (int k = 0; k <= j; ++k) {
                double coeff = binomial_coefficient(j, k) / binomial_coefficient(degree, k);
                sum += coeff * scaled_power[k];
            }
            bern_coeffs[j][interval] = sum;
        }
    }

    return BPoly(bern_coeffs, breakpoints, extrapolate);
}

// Bernstein-to-power conversion
inline std::vector<std::vector<double>> BPoly::to_power_basis() const {
    int n = degree();
    int n_intervals = num_intervals();

    std::vector<std::vector<double>> power_coeffs(n + 1, std::vector<double>(n_intervals));

    for (int interval = 0; interval < n_intervals; ++interval) {
        double a = breakpoints_[interval];
        double b = breakpoints_[interval + 1];
        double h = b - a;

        std::vector<double> bern(n + 1);
        for (int j = 0; j <= n; ++j) {
            bern[j] = coefficients_(j, interval);
        }

        // Convert Bernstein to power basis (for normalized t in [0,1])
        std::vector<double> power_t(n + 1, 0.0);
        for (int k = 0; k <= n; ++k) {
            double sum = 0.0;
            for (int j = 0; j <= k; ++j) {
                double sign = ((k - j) % 2 == 0) ? 1.0 : -1.0;
                double coeff = binomial_coefficient(n, j) * binomial_coefficient(n - j, k - j);
                sum += sign * coeff * bern[j];
            }
            power_t[k] = sum;
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
