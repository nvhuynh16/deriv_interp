#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <optional>

#include "extrapolate_mode.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// REFERENCES
// =============================================================================
// Primary reference:
//   scipy.interpolate.BarycentricInterpolator
//   https://github.com/scipy/scipy/blob/main/scipy/interpolate/_polyint.py
//   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.html
//
// Algorithm references:
//   [1] Barycentric Lagrange interpolation formula
//       p(x) = [sum_j w_j * y_j / (x - x_j)] / [sum_k w_k / (x - x_k)]
//       Berrut & Trefethen, "Barycentric Lagrange Interpolation", SIAM Review 2004
//       https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
//
//   [2] Chebyshev nodes of the second kind
//       x_k = cos(k*pi/n) for k = 0, 1, ..., n
//       Weights: w_k = (-1)^k * delta_k where delta = 1/2 at endpoints
//
//   [3] Differentiation matrix for barycentric interpolation
//       Off-diagonal: D_ij = (w_j/w_i) / (x_i - x_j)
//       Diagonal: D_ii = -sum_{j!=i} D_ij
//       Used for derivative computation
//
//   [4] Hermite interpolation via confluent divided differences
//       Used in from_derivatives() for matching function and derivative values
// =============================================================================

/**
 * LagPoly - Piecewise Barycentric Lagrange Polynomial Interpolation
 *
 * Implements numerically stable barycentric Lagrange interpolation following
 * the same interface as BPoly/CPoly/LPoly/BsPoly.
 *
 * Mathematical Foundation [1]:
 * The barycentric form evaluates as:
 *   p(x) = [sum_j w_j * y_j / (x - x_j)] / [sum_k w_k / (x - x_k)]
 *
 * This is O(n) evaluation and numerically stable.
 *
 * For Chebyshev nodes of the second kind [2] (automatically detected):
 *   w_k = (-1)^k * delta_k where delta = 1/2 at endpoints, 1 otherwise
 *
 * For general nodes:
 *   w_k = 1 / prod_{j!=k}(x_k - x_j)
 *
 * Thread-safe: All public methods are const after construction.
 * Immutable: Assignment operators are deleted.
 */
class LagPoly {
public:
    //=========================================================================
    // Construction
    //=========================================================================

    /**
     * Construct from nodes, values, and breakpoints
     *
     * @param nodes Vector of node vectors, one per interval. nodes[i] contains
     *              the interpolation nodes for interval i in [breakpoints[i], breakpoints[i+1]]
     * @param values Vector of value vectors, same shape as nodes. values[i][j] is
     *               the function value at nodes[i][j]
     * @param breakpoints Interval boundaries (length = num_intervals + 1)
     * @param extrapolate Extrapolation mode
     * @param extrapolate_order_left Taylor order for left extrapolation (-1 = full degree)
     * @param extrapolate_order_right Taylor order for right extrapolation (-1 = full degree)
     */
    LagPoly(std::vector<std::vector<double>> nodes,
            std::vector<std::vector<double>> values,
            std::vector<double> breakpoints,
            ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate,
            int extrapolate_order_left = -1,
            int extrapolate_order_right = -1)
        : nodes_(std::move(nodes))
        , values_(std::move(values))
        , breakpoints_(std::move(breakpoints))
        , extrapolate_(extrapolate)
        , extrapolate_order_left_(extrapolate_order_left)
        , extrapolate_order_right_(extrapolate_order_right)
    {
        validate_and_setup();
    }

    /**
     * Construct Hermite interpolant from derivatives at specified points
     *
     * Uses confluent nodes approach for Hermite-Lagrange interpolation.
     *
     * @param xi Breakpoints/interpolation points
     * @param yi Values and derivatives at each point. yi[i] contains
     *           [f(xi[i]), f'(xi[i]), f''(xi[i]), ...] up to available derivatives
     * @param orders Optional limit on derivative orders used at each point
     *               If empty, uses all available derivatives
     * @return LagPoly interpolant
     */
    static LagPoly from_derivatives(
        std::vector<double> xi,
        std::vector<std::vector<double>> yi,
        std::vector<int> orders = {});

    /**
     * Convenience constructor for Chebyshev nodes of the second kind
     *
     * Automatically generates Chebyshev nodes for each interval.
     *
     * @param n Number of nodes per interval (degree + 1)
     * @param values Function values at Chebyshev nodes for each interval
     * @param breakpoints Interval boundaries
     * @param extrapolate Extrapolation mode
     * @return LagPoly with Chebyshev nodes
     */
    static LagPoly from_chebyshev_nodes(
        int n,
        std::vector<std::vector<double>> values,
        std::vector<double> breakpoints,
        ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate);

    /**
     * Convert from power basis coefficients
     *
     * @param power_coeffs Power basis coefficients [degree+1][num_intervals]
     *                     power_coeffs[k][i] is coefficient of (x-a)^k for interval i
     * @param breakpoints Interval boundaries
     * @param extrapolate Extrapolation mode
     * @return LagPoly representation
     */
    static LagPoly from_power_basis(
        std::vector<std::vector<double>> power_coeffs,
        std::vector<double> breakpoints,
        ExtrapolateMode extrapolate = ExtrapolateMode::Extrapolate);

    // Disable assignment (immutable after construction)
    LagPoly& operator=(const LagPoly&) = delete;
    LagPoly& operator=(LagPoly&&) = delete;

    // Enable copy and move construction
    LagPoly(const LagPoly&) = default;
    LagPoly(LagPoly&&) = default;

    //=========================================================================
    // Evaluation (thread-safe)
    //=========================================================================

    /**
     * Evaluate polynomial at x
     *
     * Uses numerically stable barycentric formula.
     * O(n) evaluation per point.
     */
    double operator()(double x) const;

    /**
     * Evaluate nu-th derivative at x
     *
     * Uses differentiation matrix approach for derivatives.
     *
     * @param x Evaluation point
     * @param nu Derivative order (0 = value, 1 = first derivative, etc.)
     */
    double operator()(double x, int nu) const;

    /**
     * Evaluate at multiple points
     */
    std::vector<double> operator()(const std::vector<double>& xs) const;

    //=========================================================================
    // Calculus
    //=========================================================================

    /**
     * Return derivative polynomial
     *
     * Creates new LagPoly representing the nu-th derivative.
     * Uses differentiation matrix to compute derivative values at nodes.
     *
     * @param order Derivative order (default 1)
     */
    LagPoly derivative(int order = 1) const;

    /**
     * Return antiderivative polynomial
     *
     * Creates new LagPoly representing an antiderivative.
     * Ensures C0 continuity at breakpoints.
     *
     * @param order Antiderivative order (default 1)
     */
    LagPoly antiderivative(int order = 1) const;

    /**
     * Compute definite integral from a to b
     *
     * @param a Lower integration bound
     * @param b Upper integration bound
     * @param extrapolate Optional override for extrapolation behavior.
     *                    If true, extrapolate beyond bounds.
     *                    If false, return NaN if bounds exceed domain.
     *                    If not specified, use the LagPoly's default extrapolation mode.
     * @return Integral value from a to b
     */
    double integrate(double a, double b, std::optional<bool> extrapolate = std::nullopt) const;

    //=========================================================================
    // Root Finding
    //=========================================================================

    /**
     * Find roots of the polynomial
     *
     * @param discontinuity If true, include roots at discontinuities
     * @param extrapolate If true, search in extrapolation regions
     * @return Vector of x values where p(x) = 0
     */
    std::vector<double> roots(bool discontinuity = true, bool extrapolate = true) const;

    //=========================================================================
    // Conversions
    //=========================================================================

    /**
     * Convert to power basis coefficients
     *
     * @return Power basis coefficients [degree+1][num_intervals]
     *         Result[k][i] is coefficient of (x-a)^k for interval i
     */
    std::vector<std::vector<double>> to_power_basis() const;

    //=========================================================================
    // Modification
    //=========================================================================

    /**
     * Extend polynomial with new intervals
     *
     * @param nodes New interval nodes
     * @param values New interval values
     * @param x New breakpoints (must be adjacent to existing)
     * @param right If true, extend to the right; otherwise extend left
     * @return New extended LagPoly
     */
    LagPoly extend(std::vector<std::vector<double>> nodes,
                   std::vector<std::vector<double>> values,
                   std::vector<double> x,
                   bool right = true) const;

    //=========================================================================
    // Accessors
    //=========================================================================

    /** Get interpolation nodes for each interval */
    const std::vector<std::vector<double>>& nodes() const { return nodes_; }

    /** Get function values at nodes for each interval */
    const std::vector<std::vector<double>>& values() const { return values_; }

    /** Get barycentric weights for each interval */
    const std::vector<std::vector<double>>& weights() const { return weights_; }

    /** Get breakpoints (scipy-compatible alias) */
    const std::vector<double>& x() const { return breakpoints_; }

    /** Get breakpoints */
    const std::vector<double>& breakpoints() const { return breakpoints_; }

    /** Get maximum polynomial degree across all intervals */
    int degree() const {
        int max_deg = 0;
        for (const auto& n : nodes_) {
            max_deg = std::max(max_deg, static_cast<int>(n.size()) - 1);
        }
        return max_deg;
    }

    /** Get number of intervals */
    int num_intervals() const {
        return static_cast<int>(breakpoints_.size()) - 1;
    }

    /** Check if breakpoints are in ascending order */
    bool is_ascending() const { return ascending_; }

    /** Get extrapolation mode */
    ExtrapolateMode extrapolate() const { return extrapolate_; }

private:
    //=========================================================================
    // Member Variables
    //=========================================================================

    std::vector<std::vector<double>> nodes_;      // [num_intervals][num_nodes_per_interval]
    std::vector<std::vector<double>> values_;     // [num_intervals][num_nodes_per_interval]
    std::vector<std::vector<double>> weights_;    // [num_intervals][num_nodes_per_interval]
    std::vector<double> breakpoints_;             // [num_intervals + 1]
    ExtrapolateMode extrapolate_;
    bool ascending_;
    int extrapolate_order_left_;
    int extrapolate_order_right_;

    //=========================================================================
    // Internal Methods
    //=========================================================================

    /** Validate construction parameters and compute weights */
    void validate_and_setup();

    /** Find interval index for x */
    int find_interval(double x) const;

    /** Compute barycentric weights for a set of nodes */
    static std::vector<double> compute_weights(const std::vector<double>& nodes,
                                               double a, double b);

    /** Check if nodes are Chebyshev of the second kind */
    static bool is_chebyshev_second_kind(const std::vector<double>& nodes,
                                         double a, double b, double tol = 1e-10);

    /** Evaluate using barycentric formula (internal) */
    double evaluate_barycentric(int interval, double x) const;

    /** Evaluate derivative using differentiation matrix (internal) */
    double evaluate_derivative(int interval, double x, int nu) const;

    /** Build differentiation matrix for interval */
    std::vector<std::vector<double>> build_diff_matrix(int interval) const;

    /** Taylor extrapolation with controlled order */
    double evaluate_taylor(double x, double x0, int order) const;

    /** Generate Chebyshev nodes of second kind on [a, b] */
    static std::vector<double> chebyshev_nodes(int n, double a, double b);
};

//=============================================================================
// Implementation
//=============================================================================

inline void LagPoly::validate_and_setup() {
    // Check breakpoints
    if (breakpoints_.size() < 2) {
        throw std::invalid_argument("LagPoly: need at least 2 breakpoints");
    }

    int num_intervals = static_cast<int>(breakpoints_.size()) - 1;

    // Check nodes and values sizes
    if (static_cast<int>(nodes_.size()) != num_intervals) {
        throw std::invalid_argument("LagPoly: nodes size must equal num_intervals");
    }
    if (static_cast<int>(values_.size()) != num_intervals) {
        throw std::invalid_argument("LagPoly: values size must equal num_intervals");
    }

    // Check each interval
    for (int i = 0; i < num_intervals; ++i) {
        if (nodes_[i].size() != values_[i].size()) {
            throw std::invalid_argument("LagPoly: nodes and values size mismatch at interval " + std::to_string(i));
        }
        if (nodes_[i].empty()) {
            throw std::invalid_argument("LagPoly: empty nodes at interval " + std::to_string(i));
        }
    }

    // Determine ascending/descending
    ascending_ = breakpoints_[1] > breakpoints_[0];

    // Validate monotonicity
    for (size_t i = 1; i < breakpoints_.size(); ++i) {
        bool this_ascending = breakpoints_[i] > breakpoints_[i - 1];
        if (this_ascending != ascending_) {
            throw std::invalid_argument("LagPoly: breakpoints must be monotonic");
        }
    }

    // Compute barycentric weights for each interval
    weights_.resize(num_intervals);
    for (int i = 0; i < num_intervals; ++i) {
        double a = breakpoints_[i];
        double b = breakpoints_[i + 1];
        weights_[i] = compute_weights(nodes_[i], a, b);
    }

    // Set default extrapolation orders
    if (extrapolate_order_left_ < 0) {
        extrapolate_order_left_ = degree();
    }
    if (extrapolate_order_right_ < 0) {
        extrapolate_order_right_ = degree();
    }
}

inline bool LagPoly::is_chebyshev_second_kind(const std::vector<double>& nodes,
                                              double a, double b, double tol) {
    int n = static_cast<int>(nodes.size()) - 1;
    if (n < 0) return false;

    // Chebyshev nodes of second kind on [a,b]:
    // x_k = (a+b)/2 + (b-a)/2 * cos(pi*k/n) for k = 0, 1, ..., n
    double mid = (a + b) / 2.0;
    double half = (b - a) / 2.0;

    for (int k = 0; k <= n; ++k) {
        double expected = mid + half * std::cos(M_PI * k / n);
        if (std::abs(nodes[k] - expected) > tol) {
            return false;
        }
    }
    return true;
}

inline std::vector<double> LagPoly::compute_weights(const std::vector<double>& nodes,
                                                    double a, double b) {
    int n = static_cast<int>(nodes.size()) - 1;
    std::vector<double> w(n + 1);

    // Check for Chebyshev nodes (use optimized O(n) formula)
    if (is_chebyshev_second_kind(nodes, a, b)) {
        for (int k = 0; k <= n; ++k) {
            double delta = (k == 0 || k == n) ? 0.5 : 1.0;
            w[k] = ((k % 2 == 0) ? 1.0 : -1.0) * delta;
        }
        return w;
    }

    // General O(n^2) weight computation
    // w_j = 1 / prod_{k != j}(x_j - x_k)
    for (int j = 0; j <= n; ++j) {
        double prod = 1.0;
        for (int k = 0; k <= n; ++k) {
            if (k != j) {
                double diff = nodes[j] - nodes[k];
                if (std::abs(diff) < 1e-15) {
                    throw std::invalid_argument("LagPoly: duplicate nodes detected");
                }
                prod *= diff;
            }
        }
        w[j] = 1.0 / prod;
    }
    return w;
}

inline std::vector<double> LagPoly::chebyshev_nodes(int n, double a, double b) {
    std::vector<double> nodes(n);
    double mid = (a + b) / 2.0;
    double half = (b - a) / 2.0;
    for (int k = 0; k < n; ++k) {
        nodes[k] = mid + half * std::cos(M_PI * k / (n - 1));
    }
    return nodes;
}

inline int LagPoly::find_interval(double x) const {
    int n = num_intervals();

    if (ascending_) {
        // Binary search for ascending breakpoints
        if (x < breakpoints_[0]) return -1;  // Left extrapolation
        if (x >= breakpoints_[n]) return n - 1;  // Right boundary or extrapolation

        auto it = std::upper_bound(breakpoints_.begin(), breakpoints_.end(), x);
        int idx = static_cast<int>(it - breakpoints_.begin()) - 1;
        return std::max(0, std::min(idx, n - 1));
    } else {
        // Descending breakpoints
        if (x > breakpoints_[0]) return -1;  // Left extrapolation
        if (x <= breakpoints_[n]) return n - 1;  // Right boundary or extrapolation

        auto it = std::lower_bound(breakpoints_.begin(), breakpoints_.end(), x,
                                   std::greater<double>());
        int idx = static_cast<int>(it - breakpoints_.begin()) - 1;
        return std::max(0, std::min(idx, n - 1));
    }
}

inline double LagPoly::evaluate_barycentric(int interval, double x) const {
    // Barycentric Lagrange interpolation [1]
    // Reference: Berrut & Trefethen, "Barycentric Lagrange Interpolation", SIAM Review 2004
    // Formula: p(x) = [sum_j w_j * y_j / (x - x_j)] / [sum_k w_k / (x - x_k)]

    const auto& xi = nodes_[interval];
    const auto& yi = values_[interval];
    const auto& wi = weights_[interval];
    int n = static_cast<int>(xi.size());

    // Check if x is exactly at a node (avoid division by zero)
    for (int j = 0; j < n; ++j) {
        if (std::abs(x - xi[j]) < 1e-15) {
            return yi[j];
        }
    }

    // Barycentric formula: sum(w_j * y_j / (x - x_j)) / sum(w_k / (x - x_k))
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        double term = wi[j] / (x - xi[j]);
        num += term * yi[j];
        den += term;
    }
    return num / den;
}

inline std::vector<std::vector<double>> LagPoly::build_diff_matrix(int interval) const {
    // Differentiation matrix for barycentric interpolation [3]
    // Reference: Berrut & Trefethen, SIAM Review 2004, Section 9
    // Off-diagonal: D_ij = (w_j / w_i) / (x_i - x_j)
    // Diagonal: D_ii = -sum_{j != i} D_ij

    const auto& xi = nodes_[interval];
    const auto& wi = weights_[interval];
    int n = static_cast<int>(xi.size());

    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        double diag_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                D[i][j] = (wi[j] / wi[i]) / (xi[i] - xi[j]);
                diag_sum += D[i][j];
            }
        }
        D[i][i] = -diag_sum;
    }
    return D;
}

inline double LagPoly::evaluate_derivative(int interval, double x, int nu) const {
    if (nu == 0) {
        return evaluate_barycentric(interval, x);
    }

    const auto& xi = nodes_[interval];
    const auto& yi = values_[interval];
    int n = static_cast<int>(xi.size());

    // Build differentiation matrix
    auto D = build_diff_matrix(interval);

    // Apply D^nu to values to get derivative values at nodes
    std::vector<double> v = yi;
    for (int k = 0; k < nu; ++k) {
        std::vector<double> v_new(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                v_new[i] += D[i][j] * v[j];
            }
        }
        v = std::move(v_new);
    }

    // Interpolate derivative values at x using barycentric formula
    // Check if x is exactly at a node
    for (int j = 0; j < n; ++j) {
        if (std::abs(x - xi[j]) < 1e-15) {
            return v[j];
        }
    }

    // Barycentric interpolation of derivative values
    const auto& wi = weights_[interval];
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        double term = wi[j] / (x - xi[j]);
        num += term * v[j];
        den += term;
    }
    return num / den;
}

inline double LagPoly::evaluate_taylor(double x, double x0, int order) const {
    double dx = x - x0;
    double result = 0.0;
    double dx_power = 1.0;
    double factorial = 1.0;

    for (int k = 0; k <= order; ++k) {
        double deriv = (*this)(x0, k);  // k-th derivative at boundary
        result += deriv * dx_power / factorial;
        dx_power *= dx;
        factorial *= (k + 1);
    }
    return result;
}

inline double LagPoly::operator()(double x) const {
    return (*this)(x, 0);
}

inline double LagPoly::operator()(double x, int nu) const {
    if (std::isnan(x)) return std::nan("");

    int n = num_intervals();
    double left = ascending_ ? breakpoints_[0] : breakpoints_[n];
    double right = ascending_ ? breakpoints_[n] : breakpoints_[0];

    // Handle periodic mode
    if (extrapolate_ == ExtrapolateMode::Periodic) {
        double period = right - left;
        x = left + std::fmod(x - left, period);
        if (x < left) x += period;
    }

    // Check bounds
    bool in_domain = (ascending_) ? (x >= breakpoints_[0] && x <= breakpoints_[n])
                                  : (x <= breakpoints_[0] && x >= breakpoints_[n]);

    if (!in_domain) {
        if (extrapolate_ == ExtrapolateMode::NoExtrapolate) {
            return std::nan("");
        }

        // Taylor extrapolation with controlled order
        if (nu > 0) {
            // For derivatives during extrapolation, use Taylor coefficients
            // d^nu/dx^nu of Taylor expansion at boundary
            double x0 = (x < left) ? left : right;
            int order = (x < left) ? extrapolate_order_left_ : extrapolate_order_right_;

            if (nu > order) return 0.0;  // Higher derivatives are zero in truncated Taylor

            // Compute nu-th derivative of Taylor expansion
            double dx = x - x0;
            double result = 0.0;
            for (int k = nu; k <= order; ++k) {
                // d^nu/dx^nu of (x-x0)^k / k! = (x-x0)^{k-nu} / (k-nu)!
                double deriv_at_x0 = (*this)(x0, k);
                double dk_factorial = 1.0;
                for (int j = 1; j <= k - nu; ++j) dk_factorial *= j;
                result += deriv_at_x0 * std::pow(dx, k - nu) / dk_factorial;
            }
            return result;
        }

        // Value extrapolation
        double x0 = (x < left) ? left : right;
        int order = (x < left) ? extrapolate_order_left_ : extrapolate_order_right_;
        return evaluate_taylor(x, x0, order);
    }

    // Find interval and evaluate
    int interval = find_interval(x);
    if (interval < 0) interval = 0;
    if (interval >= n) interval = n - 1;

    return (nu == 0) ? evaluate_barycentric(interval, x)
                     : evaluate_derivative(interval, x, nu);
}

inline std::vector<double> LagPoly::operator()(const std::vector<double>& xs) const {
    std::vector<double> result(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        result[i] = (*this)(xs[i]);
    }
    return result;
}

inline LagPoly LagPoly::derivative(int order) const {
    if (order < 0) {
        return antiderivative(-order);
    }
    if (order == 0) {
        return *this;
    }

    // Compute derivative values at nodes using differentiation matrix
    std::vector<std::vector<double>> new_values(nodes_.size());

    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& yi = values_[i];
        int n = static_cast<int>(nodes_[i].size());
        auto D = build_diff_matrix(static_cast<int>(i));

        // Apply D^order to values
        std::vector<double> v = yi;
        for (int k = 0; k < order; ++k) {
            std::vector<double> v_new(n, 0.0);
            for (int ii = 0; ii < n; ++ii) {
                for (int jj = 0; jj < n; ++jj) {
                    v_new[ii] += D[ii][jj] * v[jj];
                }
            }
            v = std::move(v_new);
        }
        new_values[i] = std::move(v);
    }

    // Adjust extrapolation orders
    int new_left = std::max(0, extrapolate_order_left_ - order);
    int new_right = std::max(0, extrapolate_order_right_ - order);

    return LagPoly(nodes_, new_values, breakpoints_, extrapolate_, new_left, new_right);
}

inline LagPoly LagPoly::antiderivative(int order) const {
    if (order < 0) {
        return derivative(-order);
    }
    if (order == 0) {
        return *this;
    }

    // For antiderivative, we need to integrate. Use quadrature.
    // The antiderivative at each node is the integral from the left endpoint.
    std::vector<std::vector<double>> new_values(nodes_.size());
    double cumulative = 0.0;  // For C0 continuity across intervals

    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& xi = nodes_[i];
        int n = static_cast<int>(xi.size());
        new_values[i].resize(n);

        double a = breakpoints_[i];

        for (int j = 0; j < n; ++j) {
            // Integrate from left endpoint of interval to node
            double integral = 0.0;
            double node_x = xi[j];

            // Use Gaussian quadrature or trapezoidal
            int num_quad = 100;
            double h = (node_x - a) / num_quad;
            for (int q = 0; q < num_quad; ++q) {
                double x0 = a + q * h;
                double x1 = a + (q + 1) * h;
                double f0 = evaluate_barycentric(static_cast<int>(i), x0);
                double f1 = evaluate_barycentric(static_cast<int>(i), x1);
                integral += 0.5 * (f0 + f1) * h;
            }
            new_values[i][j] = cumulative + integral;
        }

        // Update cumulative for next interval (integral over this interval)
        double b = breakpoints_[i + 1];
        double interval_integral = 0.0;
        int num_quad = 100;
        double h = (b - a) / num_quad;
        for (int q = 0; q < num_quad; ++q) {
            double x0 = a + q * h;
            double x1 = a + (q + 1) * h;
            double f0 = evaluate_barycentric(static_cast<int>(i), x0);
            double f1 = evaluate_barycentric(static_cast<int>(i), x1);
            interval_integral += 0.5 * (f0 + f1) * h;
        }
        cumulative += interval_integral;
    }

    // Adjust extrapolation orders
    int new_left = extrapolate_order_left_ + order;
    int new_right = extrapolate_order_right_ + order;

    LagPoly result(nodes_, new_values, breakpoints_, extrapolate_, new_left, new_right);

    if (order > 1) {
        return result.antiderivative(order - 1);
    }
    return result;
}

inline double LagPoly::integrate(double a, double b, std::optional<bool> extrapolate_opt) const {
    if (a == b) return 0.0;
    if (a > b) return -integrate(b, a, extrapolate_opt);

    // Determine effective extrapolation mode
    ExtrapolateMode effective_mode = extrapolate_;
    if (extrapolate_opt.has_value()) {
        effective_mode = extrapolate_opt.value() ? ExtrapolateMode::Extrapolate : ExtrapolateMode::NoExtrapolate;
    }

    double result = 0.0;
    int n = num_intervals();

    // Handle boundaries
    double left = ascending_ ? breakpoints_[0] : breakpoints_[n];
    double right = ascending_ ? breakpoints_[n] : breakpoints_[0];

    // Check if integration bounds are outside domain
    double a_eff = std::min(a, b);
    double b_eff = std::max(a, b);

    if (effective_mode == ExtrapolateMode::NoExtrapolate) {
        if (a_eff < left || b_eff > right) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Clamp to domain for now (could handle extrapolation regions too)
    double a_clamped = std::max(a, left);
    double b_clamped = std::min(b, right);

    if (a_clamped >= b_clamped) return 0.0;

    // Integrate over each interval
    for (int i = 0; i < n; ++i) {
        double i_left = ascending_ ? breakpoints_[i] : breakpoints_[i + 1];
        double i_right = ascending_ ? breakpoints_[i + 1] : breakpoints_[i];

        // Find overlap with [a_clamped, b_clamped]
        double overlap_left = std::max(a_clamped, i_left);
        double overlap_right = std::min(b_clamped, i_right);

        if (overlap_left < overlap_right) {
            // Integrate over this overlap using quadrature
            int num_quad = 100;
            double h = (overlap_right - overlap_left) / num_quad;
            for (int q = 0; q < num_quad; ++q) {
                double x0 = overlap_left + q * h;
                double x1 = overlap_left + (q + 1) * h;
                double f0 = evaluate_barycentric(i, x0);
                double f1 = evaluate_barycentric(i, x1);
                result += 0.5 * (f0 + f1) * h;
            }
        }
    }

    return result;
}

inline std::vector<double> LagPoly::roots(bool discontinuity, bool extrapolate) const {
    (void)discontinuity;  // Reserved for future use
    (void)extrapolate;    // Reserved for future use

    std::vector<double> all_roots;
    int n = num_intervals();

    for (int i = 0; i < n; ++i) {
        double a = breakpoints_[i];
        double b = breakpoints_[i + 1];
        if (!ascending_) std::swap(a, b);

        // Sample at many points to detect sign changes
        int num_samples = 100;
        double prev_val = evaluate_barycentric(i, a);
        double prev_x = a;

        for (int j = 1; j <= num_samples; ++j) {
            double x = a + (b - a) * j / num_samples;
            double val = evaluate_barycentric(i, x);

            // Check for sign change
            if (prev_val * val < 0) {
                // Bisection to find root
                double lo = prev_x, hi = x;
                double f_lo = prev_val;
                for (int iter = 0; iter < 60; ++iter) {
                    double mid = (lo + hi) / 2.0;
                    double f_mid = evaluate_barycentric(i, mid);
                    if (f_lo * f_mid < 0) {
                        hi = mid;
                    } else {
                        lo = mid;
                        f_lo = f_mid;
                    }
                }
                all_roots.push_back((lo + hi) / 2.0);
            }

            // Check if exactly zero
            if (std::abs(val) < 1e-14) {
                all_roots.push_back(x);
            }

            prev_val = val;
            prev_x = x;
        }
    }

    // Remove duplicates
    std::sort(all_roots.begin(), all_roots.end());
    auto it = std::unique(all_roots.begin(), all_roots.end(),
                          [](double a, double b) { return std::abs(a - b) < 1e-10; });
    all_roots.erase(it, all_roots.end());

    return all_roots;
}

inline std::vector<std::vector<double>> LagPoly::to_power_basis() const {
    // Convert each interval to power basis
    // For a polynomial of degree n, we need n+1 coefficients: a_0, a_1, ..., a_n
    // such that p(x) = sum_{k=0}^n a_k * (x - x_left)^k

    std::vector<std::vector<double>> result(nodes_.size());

    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& xi = nodes_[i];
        int n = static_cast<int>(xi.size());
        double x_left = breakpoints_[i];

        // Compute coefficients by evaluating derivatives at left endpoint
        // a_k = p^(k)(x_left) / k!
        result[i].resize(n);
        double factorial = 1.0;
        for (int k = 0; k < n; ++k) {
            double deriv = evaluate_derivative(static_cast<int>(i), x_left, k);
            result[i][k] = deriv / factorial;
            factorial *= (k + 1);
        }
    }

    return result;
}

inline LagPoly LagPoly::extend(std::vector<std::vector<double>> new_nodes,
                               std::vector<std::vector<double>> new_values,
                               std::vector<double> new_x,
                               bool right) const {
    if (new_nodes.size() != new_values.size()) {
        throw std::invalid_argument("LagPoly::extend: nodes and values size mismatch");
    }
    if (new_x.size() != new_nodes.size() + 1) {
        throw std::invalid_argument("LagPoly::extend: breakpoints size mismatch");
    }

    std::vector<std::vector<double>> combined_nodes;
    std::vector<std::vector<double>> combined_values;
    std::vector<double> combined_bp;

    if (right) {
        // Extend to the right
        combined_nodes = nodes_;
        combined_values = values_;
        combined_bp = breakpoints_;

        for (size_t i = 0; i < new_nodes.size(); ++i) {
            combined_nodes.push_back(new_nodes[i]);
            combined_values.push_back(new_values[i]);
        }
        for (size_t i = 1; i < new_x.size(); ++i) {
            combined_bp.push_back(new_x[i]);
        }
    } else {
        // Extend to the left
        combined_nodes = new_nodes;
        combined_values = new_values;
        combined_bp.assign(new_x.begin(), new_x.end() - 1);

        for (const auto& n : nodes_) combined_nodes.push_back(n);
        for (const auto& v : values_) combined_values.push_back(v);
        for (const auto& bp : breakpoints_) combined_bp.push_back(bp);
    }

    return LagPoly(combined_nodes, combined_values, combined_bp, extrapolate_,
                   extrapolate_order_left_, extrapolate_order_right_);
}

inline LagPoly LagPoly::from_chebyshev_nodes(
    int n,
    std::vector<std::vector<double>> values,
    std::vector<double> breakpoints,
    ExtrapolateMode extrapolate)
{
    if (values.size() != breakpoints.size() - 1) {
        throw std::invalid_argument("LagPoly::from_chebyshev_nodes: values/breakpoints size mismatch");
    }

    std::vector<std::vector<double>> nodes(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (static_cast<int>(values[i].size()) != n) {
            throw std::invalid_argument("LagPoly::from_chebyshev_nodes: values size != n at interval " + std::to_string(i));
        }
        nodes[i] = chebyshev_nodes(n, breakpoints[i], breakpoints[i + 1]);
    }

    return LagPoly(nodes, values, breakpoints, extrapolate);
}

inline LagPoly LagPoly::from_power_basis(
    std::vector<std::vector<double>> power_coeffs,
    std::vector<double> breakpoints,
    ExtrapolateMode extrapolate)
{
    if (power_coeffs.empty()) {
        throw std::invalid_argument("LagPoly::from_power_basis: empty coefficients");
    }
    if (power_coeffs.size() != breakpoints.size() - 1) {
        throw std::invalid_argument("LagPoly::from_power_basis: coefficients/breakpoints size mismatch");
    }

    // For each interval, evaluate the power polynomial at Chebyshev nodes
    std::vector<std::vector<double>> nodes(power_coeffs.size());
    std::vector<std::vector<double>> values(power_coeffs.size());

    for (size_t i = 0; i < power_coeffs.size(); ++i) {
        int n = static_cast<int>(power_coeffs[i].size());
        double a = breakpoints[i];
        double b = breakpoints[i + 1];

        nodes[i] = chebyshev_nodes(n, a, b);
        values[i].resize(n);

        for (int j = 0; j < n; ++j) {
            // Evaluate power polynomial at node
            // p(x) = sum_k c_k * (x - a)^k
            double x = nodes[i][j];
            double dx = x - a;
            double val = 0.0;
            double dx_power = 1.0;
            for (int k = 0; k < n; ++k) {
                val += power_coeffs[i][k] * dx_power;
                dx_power *= dx;
            }
            values[i][j] = val;
        }
    }

    return LagPoly(nodes, values, breakpoints, extrapolate);
}

inline LagPoly LagPoly::from_derivatives(
    std::vector<double> xi,
    std::vector<std::vector<double>> yi,
    std::vector<int> orders)
{
    if (xi.size() < 2) {
        throw std::invalid_argument("LagPoly::from_derivatives: need at least 2 points");
    }
    if (xi.size() != yi.size()) {
        throw std::invalid_argument("LagPoly::from_derivatives: xi and yi size mismatch");
    }

    int num_intervals = static_cast<int>(xi.size()) - 1;
    std::vector<std::vector<double>> nodes(num_intervals);
    std::vector<std::vector<double>> values(num_intervals);
    std::vector<double> breakpoints(xi);

    // Determine derivative orders to use at each point
    std::vector<int> actual_orders(yi.size());
    for (size_t i = 0; i < yi.size(); ++i) {
        int max_order = static_cast<int>(yi[i].size());
        if (!orders.empty() && i < orders.size()) {
            actual_orders[i] = std::min(max_order, orders[i]);
        } else {
            actual_orders[i] = max_order;
        }
    }

    // For each interval, set up Hermite interpolation
    for (int interval = 0; interval < num_intervals; ++interval) {
        int n_left = actual_orders[interval];
        int n_right = actual_orders[interval + 1];
        int total = n_left + n_right;  // Total number of constraints

        double a = xi[interval];
        double b = xi[interval + 1];

        // Use Chebyshev nodes of appropriate degree
        int n_nodes = total;
        nodes[interval] = chebyshev_nodes(n_nodes, a, b);

        // Build linear system to find values at nodes
        // We need to find y values such that the interpolant satisfies:
        // - f^(k)(a) = yi[interval][k] for k = 0..n_left-1
        // - f^(k)(b) = yi[interval+1][k] for k = 0..n_right-1

        // Build constraint matrix A where A[i][j] = L_j^(k)(x_constraint)
        // L_j is the j-th Lagrange basis polynomial

        // First compute weights for these nodes
        std::vector<double> w = compute_weights(nodes[interval], a, b);

        // Build differentiation matrix
        std::vector<std::vector<double>> D(n_nodes, std::vector<double>(n_nodes, 0.0));
        for (int i = 0; i < n_nodes; ++i) {
            double diag_sum = 0.0;
            for (int j = 0; j < n_nodes; ++j) {
                if (i != j) {
                    D[i][j] = (w[j] / w[i]) / (nodes[interval][i] - nodes[interval][j]);
                    diag_sum += D[i][j];
                }
            }
            D[i][i] = -diag_sum;
        }

        // Build constraint matrix A and RHS b
        // A[constraint][node] = derivative of L_node at constraint point
        std::vector<std::vector<double>> A(total, std::vector<double>(n_nodes, 0.0));
        std::vector<double> rhs(total);

        // For k-th derivative constraint at point x:
        // We need d^k/dx^k [ sum_j y_j * L_j(x) ] = sum_j y_j * L_j^(k)(x)
        // So A[constraint][j] = L_j^(k)(x_constraint)

        // Build matrices for L_j^(k) at each node
        // L_j(x_i) = delta_{ij}
        // D gives us: (d/dx L_j)(x_i) = D[i][j]
        // D^k gives us: (d^k/dx^k L_j)(x_i) = (D^k)[i][j]

        // Find indices of nodes closest to a and b
        int idx_a = 0, idx_b = 0;
        double min_dist_a = std::abs(nodes[interval][0] - a);
        double min_dist_b = std::abs(nodes[interval][0] - b);
        for (int j = 1; j < n_nodes; ++j) {
            if (std::abs(nodes[interval][j] - a) < min_dist_a) {
                min_dist_a = std::abs(nodes[interval][j] - a);
                idx_a = j;
            }
            if (std::abs(nodes[interval][j] - b) < min_dist_b) {
                min_dist_b = std::abs(nodes[interval][j] - b);
                idx_b = j;
            }
        }

        // Compute L_j^(k) matrices for needed derivative orders
        int max_deriv = std::max(n_left, n_right);
        std::vector<std::vector<std::vector<double>>> L_deriv(max_deriv);

        // L^(0) = Identity (L_j(x_i) = delta_{ij})
        L_deriv[0].resize(n_nodes, std::vector<double>(n_nodes, 0.0));
        for (int i = 0; i < n_nodes; ++i) L_deriv[0][i][i] = 1.0;

        // L^(k) = D^k
        for (int k = 1; k < max_deriv; ++k) {
            L_deriv[k].resize(n_nodes, std::vector<double>(n_nodes, 0.0));
            // L^(k) = D * L^(k-1)
            for (int i = 0; i < n_nodes; ++i) {
                for (int j = 0; j < n_nodes; ++j) {
                    for (int m = 0; m < n_nodes; ++m) {
                        L_deriv[k][i][j] += D[i][m] * L_deriv[k - 1][m][j];
                    }
                }
            }
        }

        // Fill constraint matrix
        int constraint = 0;

        // Constraints at left endpoint (a)
        for (int k = 0; k < n_left; ++k) {
            for (int j = 0; j < n_nodes; ++j) {
                // L_j^(k) at x closest to a
                A[constraint][j] = L_deriv[k][idx_a][j];
            }
            rhs[constraint] = yi[interval][k];
            ++constraint;
        }

        // Constraints at right endpoint (b)
        for (int k = 0; k < n_right; ++k) {
            for (int j = 0; j < n_nodes; ++j) {
                // L_j^(k) at x closest to b
                A[constraint][j] = L_deriv[k][idx_b][j];
            }
            rhs[constraint] = yi[interval + 1][k];
            ++constraint;
        }

        // Solve linear system A * values = rhs using Gaussian elimination
        values[interval].resize(n_nodes);

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n_nodes; ++col) {
            // Find pivot
            int pivot_row = col;
            double max_val = std::abs(A[col][col]);
            for (int row = col + 1; row < total; ++row) {
                if (std::abs(A[row][col]) > max_val) {
                    max_val = std::abs(A[row][col]);
                    pivot_row = row;
                }
            }

            // Swap rows
            if (pivot_row != col) {
                std::swap(A[col], A[pivot_row]);
                std::swap(rhs[col], rhs[pivot_row]);
            }

            // Eliminate
            for (int row = col + 1; row < total; ++row) {
                if (std::abs(A[col][col]) < 1e-15) continue;
                double factor = A[row][col] / A[col][col];
                for (int j = col; j < n_nodes; ++j) {
                    A[row][j] -= factor * A[col][j];
                }
                rhs[row] -= factor * rhs[col];
            }
        }

        // Back substitution
        for (int i = n_nodes - 1; i >= 0; --i) {
            double sum = rhs[i];
            for (int j = i + 1; j < n_nodes; ++j) {
                sum -= A[i][j] * values[interval][j];
            }
            values[interval][i] = (std::abs(A[i][i]) > 1e-15) ? sum / A[i][i] : 0.0;
        }
    }

    return LagPoly(nodes, values, breakpoints, ExtrapolateMode::Extrapolate);
}
