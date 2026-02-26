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

// =============================================================================
// CRTP base class for piecewise polynomial implementations
// =============================================================================
//
// Provides shared logic for BPoly, CPoly, LegPoly, HPoly, BsPoly.
// LagPoly is excluded (fundamentally different storage/evaluation model).
//
// Each Derived class must provide these private methods (friend access):
//
//   double evaluate_basis(int interval_idx, double local_param) const;
//   double map_to_local(double x, double a, double b) const;
//   static constexpr double breakpoint_left_eval_param();   // 0.0 or -1.0
//   static constexpr double breakpoint_right_eval_param();  // 1.0
//   ndarray::array2d<double> compute_derivative_coefficients() const;
//   ndarray::array2d<double> compute_antiderivative_coefficients(double& running_integral) const;
//   static std::vector<double> elevate_degree_impl(const std::vector<double>& coeffs, int target_degree);
//   Derived make_poly(ndarray::array2d<double> coeffs, std::vector<double> breaks,
//                     ExtrapolateMode extrap, int ol, int or_) const;
//   Derived make_zero_poly() const;
// =============================================================================

// =============================================================================
// Shared utilities for piecewise polynomial implementations
// =============================================================================
namespace poly_util {

// Gaussian elimination with partial pivoting.
// Solves A * x = b in-place. A is (n+1)x(n+1), b is (n+1).
// Returns solution vector x.
inline std::vector<double> solve_linear_system(
    std::vector<std::vector<double>>& A,
    std::vector<double>& b,
    int n) {

    std::vector<double> x(n + 1);

    // Forward elimination
    for (int col = 0; col <= n; ++col) {
        int pivot_row = col;
        double max_val = std::abs(A[col][col]);
        for (int row = col + 1; row <= n; ++row) {
            if (std::abs(A[row][col]) > max_val) {
                max_val = std::abs(A[row][col]);
                pivot_row = row;
            }
        }

        if (pivot_row != col) {
            std::swap(A[col], A[pivot_row]);
            std::swap(b[col], b[pivot_row]);
        }

        for (int row = col + 1; row <= n; ++row) {
            if (std::abs(A[col][col]) < 1e-15) continue;
            double factor = A[row][col] / A[col][col];
            for (int j = col; j <= n; ++j) {
                A[row][j] -= factor * A[col][j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    for (int row = n; row >= 0; --row) {
        double sum = b[row];
        for (int j = row + 1; j <= n; ++j) {
            sum -= A[row][j] * x[j];
        }
        x[row] = (std::abs(A[row][row]) < 1e-15) ? 0.0 : sum / A[row][row];
    }

    return x;
}

// Validates inputs to from_derivatives and computes setup information.
// Returns: get_num_derivs lambda via output parameter, max_degree, num_intervals.
struct FromDerivativesSetup {
    int num_intervals;
    int max_degree;
    std::function<int(size_t)> get_num_derivs;
};

inline FromDerivativesSetup validate_from_derivatives(
    const std::vector<double>& xi,
    const std::vector<std::vector<double>>& yi,
    const std::vector<int>& orders) {

    if (xi.size() != yi.size()) {
        throw std::invalid_argument("xi and yi must have same size");
    }
    if (xi.size() < 2) {
        throw std::invalid_argument("Need at least 2 points for interpolation");
    }
    for (size_t i = 1; i < xi.size(); ++i) {
        if (xi[i] <= xi[i-1]) {
            throw std::invalid_argument("Breakpoints must be strictly increasing");
        }
    }
    if (!orders.empty() && orders.size() != 1 && orders.size() != xi.size()) {
        throw std::invalid_argument("orders must be empty, single element, or same size as xi");
    }

    auto get_num_derivs = [&yi, orders](size_t i) -> int {
        int available = static_cast<int>(yi[i].size());
        if (orders.empty()) {
            return available;
        }
        int max_order = (orders.size() == 1) ? orders[0] : orders[i];
        return std::min(available, max_order + 1);
    };

    int num_intervals = static_cast<int>(xi.size()) - 1;

    int max_degree = 0;
    for (int i = 0; i < num_intervals; ++i) {
        int left_derivs = get_num_derivs(i);
        int right_derivs = get_num_derivs(i + 1);
        int interval_degree = left_derivs + right_derivs - 1;
        max_degree = std::max(max_degree, interval_degree);
    }

    return { num_intervals, max_degree, get_num_derivs };
}

} // namespace poly_util

template <typename Derived>
class PolyBase {
public:
    // =========================================================================
    // Evaluation
    // =========================================================================

    double operator()(double x) const {
        // Handle NaN and infinity input
        if (std::isnan(x) || std::isinf(x)) {
            if (std::isinf(x) && extrapolate_ == ExtrapolateMode::Extrapolate) {
                // Fall through to normal evaluation
            } else {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Get actual min/max regardless of breakpoint order
        double x_lo = ascending_ ? breakpoints_.front() : breakpoints_.back();
        double x_hi = ascending_ ? breakpoints_.back() : breakpoints_.front();

        // Handle out-of-bounds based on extrapolation mode
        if (x < x_lo || x > x_hi) {
            switch (extrapolate_) {
                case ExtrapolateMode::NoExtrapolate:
                    return std::numeric_limits<double>::quiet_NaN();

                case ExtrapolateMode::Periodic: {
                    double period = x_hi - x_lo;
                    double x_shifted = x - x_lo;
                    x_shifted = std::fmod(x_shifted, period);
                    if (x_shifted < 0) {
                        x_shifted += period;
                    }
                    x = x_lo + x_shifted;
                    if (x >= x_hi) {
                        x = x_lo;
                    }
                    break;
                }

                case ExtrapolateMode::Extrapolate:
                default: {
                    bool is_left = (x < x_lo);
                    int order = is_left ? extrapolate_order_left_ : extrapolate_order_right_;
                    if (order >= 0 && order <= degree()) {
                        double boundary = is_left ? x_lo : x_hi;
                        return evaluate_taylor(x, boundary, order);
                    }
                    break;
                }
            }
        }

        int interval = find_interval(x);

        double x0 = breakpoints_[interval];
        double x1 = breakpoints_[interval + 1];
        double local = self().map_to_local(x, x0, x1);

        return self().evaluate_basis(interval, local);
    }

    std::vector<double> operator()(const std::vector<double>& x_values) const {
        std::vector<double> results;
        results.reserve(x_values.size());
        for (double x : x_values) {
            results.push_back((*this)(x));
        }
        return results;
    }

    double operator()(double x, int nu) const {
        if (nu == 0) {
            return (*this)(x);
        }
        if (nu > 0) {
            return self().derivative(nu)(x);
        }
        return self().antiderivative(-nu)(x);
    }

    // =========================================================================
    // Calculus
    // =========================================================================

    Derived derivative(int order = 1) const {
        if (order < 0) {
            return self().antiderivative(-order);
        }
        if (order == 0) {
            return self_copy();
        }
        if (order > degree()) {
            return self().make_zero_poly();
        }

        ndarray::array2d<double> deriv_coeffs = self().compute_derivative_coefficients();

        int new_order_left = (extrapolate_order_left_ > 0) ? extrapolate_order_left_ - 1 : extrapolate_order_left_;
        int new_order_right = (extrapolate_order_right_ > 0) ? extrapolate_order_right_ - 1 : extrapolate_order_right_;

        Derived result = self().make_poly(std::move(deriv_coeffs), breakpoints_, extrapolate_,
                                          new_order_left, new_order_right);

        if (order > 1) {
            return result.derivative(order - 1);
        }
        return result;
    }

    Derived antiderivative(int order = 1) const {
        if (order <= 0) {
            return self_copy();
        }

        double running_integral = 0.0;
        ndarray::array2d<double> antideriv_coeffs = self().compute_antiderivative_coefficients(running_integral);

        int new_order_left = (extrapolate_order_left_ >= 0) ? extrapolate_order_left_ + 1 : extrapolate_order_left_;
        int new_order_right = (extrapolate_order_right_ >= 0) ? extrapolate_order_right_ + 1 : extrapolate_order_right_;

        Derived result = self().make_poly(std::move(antideriv_coeffs), breakpoints_, extrapolate_,
                                          new_order_left, new_order_right);

        if (order > 1) {
            return result.antiderivative(order - 1);
        }
        return result;
    }

    double integrate(double a, double b, std::optional<bool> extrapolate = std::nullopt) const {
        ExtrapolateMode effective_mode = extrapolate_;
        if (extrapolate.has_value()) {
            effective_mode = extrapolate.value() ? ExtrapolateMode::Extrapolate : ExtrapolateMode::NoExtrapolate;
        }

        double x_lo = ascending_ ? breakpoints_.front() : breakpoints_.back();
        double x_hi = ascending_ ? breakpoints_.back() : breakpoints_.front();
        double period = x_hi - x_lo;

        double a_eff = std::min(a, b);
        double b_eff = std::max(a, b);

        if (effective_mode == ExtrapolateMode::NoExtrapolate) {
            if (a_eff < x_lo || b_eff > x_hi) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        if (effective_mode == ExtrapolateMode::Periodic) {
            ndarray::array2d<double> coeffs_copy(coefficients_);
            Derived antideriv_extrap = self().make_poly(std::move(coeffs_copy), breakpoints_,
                                                        ExtrapolateMode::Extrapolate, -1, -1);
            Derived antideriv_period = antideriv_extrap.antiderivative();
            double integral_one_period = antideriv_period(x_hi) - antideriv_period(x_lo);

            auto wrap = [&](double x) -> double {
                double shifted = x - x_lo;
                shifted = std::fmod(shifted, period);
                if (shifted < 0) shifted += period;
                return x_lo + shifted;
            };

            double a_wrapped = wrap(a);
            double total_length = b - a;
            int sign = (total_length >= 0) ? 1 : -1;
            total_length = std::abs(total_length);

            int complete_periods = static_cast<int>(total_length / period);
            double remainder = std::fmod(total_length, period);

            double remainder_integral;
            if (sign > 0) {
                if (a_wrapped + remainder <= x_hi) {
                    remainder_integral = antideriv_period(a_wrapped + remainder) - antideriv_period(a_wrapped);
                } else {
                    double part1 = antideriv_period(x_hi) - antideriv_period(a_wrapped);
                    double part2 = antideriv_period(x_lo + (remainder - (x_hi - a_wrapped))) - antideriv_period(x_lo);
                    remainder_integral = part1 + part2;
                }
            } else {
                if (a_wrapped - remainder >= x_lo) {
                    remainder_integral = antideriv_period(a_wrapped - remainder) - antideriv_period(a_wrapped);
                } else {
                    double part1 = antideriv_period(x_lo) - antideriv_period(a_wrapped);
                    double part2 = antideriv_period(x_hi - (remainder - (a_wrapped - x_lo))) - antideriv_period(x_hi);
                    remainder_integral = part1 + part2;
                }
            }

            return sign * (complete_periods * integral_one_period + remainder_integral);
        }

        // Non-periodic: use fundamental theorem
        ndarray::array2d<double> coeffs_copy2(coefficients_);
        Derived bp_effective = self().make_poly(std::move(coeffs_copy2), breakpoints_, effective_mode, -1, -1);
        Derived antideriv = bp_effective.antiderivative();
        return antideriv(b) - antideriv(a);
    }

    // =========================================================================
    // Root finding
    // =========================================================================

    std::vector<double> roots(bool discontinuity = true, bool extrapolate_roots = true) const {
        std::vector<double> all_roots;
        const double tol = 1e-12;

        double x_lo = ascending_ ? breakpoints_.front() : breakpoints_.back();
        double x_hi = ascending_ ? breakpoints_.back() : breakpoints_.front();

        int n_intervals = num_intervals();

        auto find_roots_in_interval = [this, tol](int interval_idx, double x_start, double x_end,
                                                   std::vector<double>& roots_out,
                                                   bool include_start, bool include_end) {
            const int max_subdivisions = 100;
            const int max_bisect_iter = 100;

            auto eval_in_interval = [this, interval_idx](double x, double xs, double xe) -> double {
                double local = self().map_to_local(x, xs, xe);
                return self().evaluate_basis(interval_idx, local);
            };

            double h = (x_end - x_start) / max_subdivisions;
            double prev_x = x_start;
            double prev_val = eval_in_interval(prev_x, x_start, x_end);

            if (include_start && std::abs(prev_val) < tol) {
                roots_out.push_back(prev_x);
            }

            for (int i = 1; i <= max_subdivisions; ++i) {
                double curr_x = x_start + i * h;
                if (i == max_subdivisions) curr_x = x_end;
                double curr_val = eval_in_interval(curr_x, x_start, x_end);

                if (prev_val * curr_val < 0) {
                    double lo = prev_x, hi = curr_x;
                    double lo_val = prev_val;

                    for (int iter = 0; iter < max_bisect_iter; ++iter) {
                        double mid = (lo + hi) / 2.0;
                        double mid_val = eval_in_interval(mid, x_start, x_end);

                        if (std::abs(mid_val) < tol || (hi - lo) < tol) {
                            roots_out.push_back(mid);
                            break;
                        }

                        if (mid_val * lo_val < 0) {
                            hi = mid;
                        } else {
                            lo = mid;
                            lo_val = mid_val;
                        }
                    }
                } else if (std::abs(curr_val) < tol) {
                    bool should_add = (i < max_subdivisions) || include_end;
                    if (should_add) {
                        roots_out.push_back(curr_x);
                    }
                }

                prev_x = curr_x;
                prev_val = curr_val;
            }
        };

        // Find roots in each interval
        for (int i = 0; i < n_intervals; ++i) {
            double x_start = breakpoints_[i];
            double x_end = breakpoints_[i + 1];
            if (!ascending_) std::swap(x_start, x_end);

            bool include_start = (i == 0);
            bool include_end = (i == n_intervals - 1);

            find_roots_in_interval(i, x_start, x_end, all_roots, include_start, include_end);
        }

        // Check for roots at internal breakpoints
        constexpr double left_param = Derived::breakpoint_left_eval_param();
        constexpr double right_param = Derived::breakpoint_right_eval_param();

        for (int i = 1; i < n_intervals; ++i) {
            double bp = breakpoints_[i];
            double left_val = self().evaluate_basis(i - 1, right_param);
            double right_val = self().evaluate_basis(i, left_param);

            if (std::abs(left_val) < tol && std::abs(right_val) < tol) {
                all_roots.push_back(bp);
            } else if (discontinuity && left_val * right_val < 0) {
                all_roots.push_back(bp);
            }
        }

        // Extrapolation root finding
        if (extrapolate_roots && extrapolate_ == ExtrapolateMode::Extrapolate) {
            double domain_width = std::abs(x_hi - x_lo);
            double search_range = std::max(domain_width * 10.0, 100.0);

            auto find_extrap_roots = [this, tol](int interval_idx, double orig_x0, double orig_x1,
                                                  double search_start, double search_end,
                                                  std::vector<double>& roots_out) {
                const int max_subdivisions = 100;
                const int max_bisect_iter = 100;

                auto eval_extrap = [this, interval_idx, orig_x0, orig_x1](double x) -> double {
                    double local = self().map_to_local(x, orig_x0, orig_x1);
                    return self().evaluate_basis(interval_idx, local);
                };

                double h = (search_end - search_start) / max_subdivisions;
                double prev_x = search_start;
                double prev_val = eval_extrap(prev_x);

                for (int i = 1; i <= max_subdivisions; ++i) {
                    double curr_x = search_start + i * h;
                    if (i == max_subdivisions) curr_x = search_end;
                    double curr_val = eval_extrap(curr_x);

                    if (prev_val * curr_val < 0) {
                        double lo = prev_x, hi = curr_x;
                        double lo_val = prev_val;

                        for (int iter = 0; iter < max_bisect_iter; ++iter) {
                            double mid = (lo + hi) / 2.0;
                            double mid_val = eval_extrap(mid);

                            if (std::abs(mid_val) < tol || (hi - lo) < tol) {
                                roots_out.push_back(mid);
                                break;
                            }

                            if (mid_val * lo_val < 0) {
                                hi = mid;
                            } else {
                                lo = mid;
                                lo_val = mid_val;
                            }
                        }
                    } else if (std::abs(curr_val) < tol) {
                        roots_out.push_back(curr_x);
                    }

                    prev_x = curr_x;
                    prev_val = curr_val;
                }
            };

            // Left extrapolation
            int left_interval = ascending_ ? 0 : n_intervals - 1;
            double left_x0 = breakpoints_[left_interval];
            double left_x1 = breakpoints_[left_interval + 1];
            double left_search_start = x_lo - search_range;
            std::vector<double> left_roots;

            double val_at_lo = (*this)(x_lo);
            double val_at_far_left = (*this)(left_search_start);
            if (val_at_lo * val_at_far_left < 0 || std::abs(val_at_far_left) < tol) {
                find_extrap_roots(left_interval, left_x0, left_x1, left_search_start, x_lo, left_roots);
                for (double r : left_roots) {
                    if (r < x_lo - tol) {
                        all_roots.push_back(r);
                    }
                }
            }

            // Right extrapolation
            int right_interval = ascending_ ? n_intervals - 1 : 0;
            double right_x0 = breakpoints_[right_interval];
            double right_x1 = breakpoints_[right_interval + 1];
            double right_search_end = x_hi + search_range;
            std::vector<double> right_roots;

            double val_at_hi = (*this)(x_hi);
            double val_at_far_right = (*this)(right_search_end);
            if (val_at_hi * val_at_far_right < 0 || std::abs(val_at_far_right) < tol) {
                find_extrap_roots(right_interval, right_x0, right_x1, x_hi, right_search_end, right_roots);
                for (double r : right_roots) {
                    if (r > x_hi + tol) {
                        all_roots.push_back(r);
                    }
                }
            }
        }

        // Sort and remove duplicates
        std::sort(all_roots.begin(), all_roots.end());
        auto last = std::unique(all_roots.begin(), all_roots.end(),
                               [](double a, double b) { return std::abs(a - b) < 1e-10; });
        all_roots.erase(last, all_roots.end());

        return all_roots;
    }

    // =========================================================================
    // Extend
    // =========================================================================

    Derived extend(std::vector<std::vector<double>> c,
                   std::vector<double> x,
                   bool right = true) const {
        if (c.empty() || x.size() < 2) {
            throw std::invalid_argument("Need at least one interval to extend");
        }

        bool new_ascending = x[1] > x[0];
        if (new_ascending != ascending_) {
            throw std::invalid_argument("New breakpoints must have same order (ascending/descending) as existing");
        }

        for (size_t i = 1; i < x.size(); ++i) {
            if (ascending_) {
                if (x[i] <= x[i-1]) {
                    throw std::invalid_argument("New breakpoints must be strictly increasing");
                }
            } else {
                if (x[i] >= x[i-1]) {
                    throw std::invalid_argument("New breakpoints must be strictly decreasing");
                }
            }
        }

        int new_intervals = static_cast<int>(x.size()) - 1;
        if (c[0].size() != static_cast<size_t>(new_intervals)) {
            throw std::invalid_argument("Coefficient columns must match number of new intervals");
        }

        int this_degree = degree();
        int ext_degree = static_cast<int>(c.size()) - 1;
        int combined_degree = std::max(this_degree, ext_degree);

        std::vector<std::vector<double>> new_coeffs;
        std::vector<double> new_breaks;

        if (right) {
            if (std::abs(x[0] - breakpoints_.back()) > 1e-10) {
                throw std::invalid_argument("First new breakpoint must match last existing breakpoint");
            }

            new_breaks = breakpoints_;
            for (size_t i = 1; i < x.size(); ++i) {
                new_breaks.push_back(x[i]);
            }

            int total_intervals = num_intervals() + new_intervals;
            new_coeffs.resize(combined_degree + 1, std::vector<double>(total_intervals, 0.0));

            for (int i = 0; i < num_intervals(); ++i) {
                std::vector<double> interval_coeffs(this_degree + 1);
                for (int j = 0; j <= this_degree; ++j) {
                    interval_coeffs[j] = coefficients_(j, i);
                }
                std::vector<double> elevated = Derived::elevate_degree_impl(interval_coeffs, combined_degree);
                for (int j = 0; j <= combined_degree; ++j) {
                    new_coeffs[j][i] = elevated[j];
                }
            }

            for (int i = 0; i < new_intervals; ++i) {
                std::vector<double> interval_coeffs(ext_degree + 1);
                for (int j = 0; j <= ext_degree; ++j) {
                    interval_coeffs[j] = c[j][i];
                }
                std::vector<double> elevated = Derived::elevate_degree_impl(interval_coeffs, combined_degree);
                for (int j = 0; j <= combined_degree; ++j) {
                    new_coeffs[j][num_intervals() + i] = elevated[j];
                }
            }
        } else {
            if (std::abs(x.back() - breakpoints_.front()) > 1e-10) {
                throw std::invalid_argument("Last new breakpoint must match first existing breakpoint");
            }

            for (size_t i = 0; i < x.size() - 1; ++i) {
                new_breaks.push_back(x[i]);
            }
            for (double bp : breakpoints_) {
                new_breaks.push_back(bp);
            }

            int total_intervals = num_intervals() + new_intervals;
            new_coeffs.resize(combined_degree + 1, std::vector<double>(total_intervals, 0.0));

            for (int i = 0; i < new_intervals; ++i) {
                std::vector<double> interval_coeffs(ext_degree + 1);
                for (int j = 0; j <= ext_degree; ++j) {
                    interval_coeffs[j] = c[j][i];
                }
                std::vector<double> elevated = Derived::elevate_degree_impl(interval_coeffs, combined_degree);
                for (int j = 0; j <= combined_degree; ++j) {
                    new_coeffs[j][i] = elevated[j];
                }
            }

            for (int i = 0; i < num_intervals(); ++i) {
                std::vector<double> interval_coeffs(this_degree + 1);
                for (int j = 0; j <= this_degree; ++j) {
                    interval_coeffs[j] = coefficients_(j, i);
                }
                std::vector<double> elevated = Derived::elevate_degree_impl(interval_coeffs, combined_degree);
                for (int j = 0; j <= combined_degree; ++j) {
                    new_coeffs[j][new_intervals + i] = elevated[j];
                }
            }
        }

        return self().make_poly(ndarray::array2d<double>(new_coeffs), new_breaks, extrapolate_,
                                extrapolate_order_left_, extrapolate_order_right_);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    int degree() const { return static_cast<int>(coefficients_.rows) - 1; }
    int num_intervals() const { return static_cast<int>(breakpoints_.size()) - 1; }
    const ndarray::array2d<double>& coefficients_array() const { return coefficients_; }
    std::vector<std::vector<double>> coefficients() const { return coefficients_.to_vec2d(); }
    const std::vector<double>& breakpoints() const { return breakpoints_; }
    ExtrapolateMode extrapolate() const { return extrapolate_; }
    int extrapolate_order_left() const { return extrapolate_order_left_; }
    int extrapolate_order_right() const { return extrapolate_order_right_; }
    std::vector<std::vector<double>> c() const { return coefficients_.to_vec2d(); }
    const std::vector<double>& x() const { return breakpoints_; }
    bool is_ascending() const { return ascending_; }

protected:
    ndarray::array2d<double> coefficients_;
    std::vector<double> breakpoints_;
    ExtrapolateMode extrapolate_;
    bool ascending_;
    int extrapolate_order_left_;
    int extrapolate_order_right_;

    // Protected constructor for use by Derived classes
    PolyBase(ndarray::array2d<double> coefficients,
             std::vector<double> breakpoints,
             ExtrapolateMode extrapolate,
             int extrapolate_order_left,
             int extrapolate_order_right)
        : coefficients_(std::move(coefficients)),
          breakpoints_(std::move(breakpoints)),
          extrapolate_(extrapolate),
          ascending_(breakpoints_.size() >= 2 && breakpoints_[1] > breakpoints_[0]),
          extrapolate_order_left_(extrapolate_order_left),
          extrapolate_order_right_(extrapolate_order_right) {
        validate_construction(coefficients_, breakpoints_);
    }

    // Default/move/copy for derived classes
    PolyBase() = default;
    PolyBase(const PolyBase&) = default;
    PolyBase(PolyBase&&) noexcept = default;
    PolyBase& operator=(const PolyBase&) = delete;
    PolyBase& operator=(PolyBase&&) = delete;

    // =========================================================================
    // Shared internal methods
    // =========================================================================

    void validate_construction(const ndarray::array2d<double>& coefficients,
                               const std::vector<double>& breakpoints) {
        if (coefficients.empty()) {
            throw std::invalid_argument("Coefficients cannot be empty");
        }

        if (breakpoints.size() < 2) {
            throw std::invalid_argument("Need at least 2 breakpoints");
        }

        int num_intervals_expected = static_cast<int>(breakpoints.size()) - 1;
        if (coefficients.cols != static_cast<size_t>(num_intervals_expected)) {
            throw std::invalid_argument("Number of coefficient columns must equal number of intervals");
        }

        bool is_ascending = breakpoints[1] > breakpoints[0];
        for (size_t i = 1; i < breakpoints.size(); ++i) {
            if (is_ascending) {
                if (breakpoints[i] <= breakpoints[i-1]) {
                    throw std::invalid_argument("Breakpoints must be strictly monotonic (all increasing or all decreasing)");
                }
            } else {
                if (breakpoints[i] >= breakpoints[i-1]) {
                    throw std::invalid_argument("Breakpoints must be strictly monotonic (all increasing or all decreasing)");
                }
            }
        }
    }

    int find_interval(double x) const {
        if (ascending_) {
            if (x < breakpoints_[0]) {
                return 0;
            }
            if (x >= breakpoints_.back()) {
                return num_intervals() - 1;
            }
            auto it = std::upper_bound(breakpoints_.begin(), breakpoints_.end(), x);
            int interval = static_cast<int>(std::distance(breakpoints_.begin(), it)) - 1;
            return std::max(0, std::min(interval, num_intervals() - 1));
        } else {
            if (x > breakpoints_[0]) {
                return 0;
            }
            if (x <= breakpoints_.back()) {
                return num_intervals() - 1;
            }
            auto it = std::upper_bound(breakpoints_.begin(), breakpoints_.end(), x, std::greater<double>());
            int interval = static_cast<int>(std::distance(breakpoints_.begin(), it)) - 1;
            return std::max(0, std::min(interval, num_intervals() - 1));
        }
    }

    double evaluate_taylor(double x, double x0, int order) const {
        if (order < 0) {
            return (*this)(x0, 0);
        }

        double dx = x - x0;
        double result = 0.0;
        double dx_power = 1.0;
        double factorial = 1.0;

        for (int i = 0; i <= order; ++i) {
            double deriv_value = (*this)(x0, i);
            result += deriv_value * dx_power / factorial;
            dx_power *= dx;
            factorial *= (i + 1);
        }

        return result;
    }

private:
    const Derived& self() const { return static_cast<const Derived&>(*this); }
    Derived self_copy() const { return static_cast<const Derived&>(*this); }
};
