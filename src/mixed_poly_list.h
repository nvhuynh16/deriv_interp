#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <vector>
#include <variant>
#include <memory>
#include <cstddef>

#include "bpoly.h"
#include "cpoly.h"
#include "legpoly.h"
#include "hpoly.h"
#include "bspoly.h"
#include "lagpoly.h"

namespace nb = nanobind;

/**
 * PolyVariant - A variant type that can hold any of the polynomial types
 * Wrapped in shared_ptr to work around deleted assignment operators in polynomial classes
 */
using PolyVariant = std::variant<BPoly, CPoly, LegPoly, HPoly, BsPoly, LagPoly>;
using PolyVariantPtr = std::shared_ptr<PolyVariant>;

/**
 * Helper to create a PolyVariantPtr from a polynomial
 */
template<typename T>
PolyVariantPtr make_poly_variant(T&& poly) {
    return std::make_shared<PolyVariant>(std::forward<T>(poly));
}

/**
 * MixedPolyList - Batch evaluation container for mixed polynomial types
 *
 * Stores multiple polynomials of different types and evaluates them all at once,
 * returning an ndarray with the polynomial index as the leading dimension.
 *
 * Output shapes:
 *   f(scalar)    -> (N,)
 *   f(1D array)  -> (N, M)
 *   f(ND array)  -> (N, *input_shape)
 *
 * where N = number of polynomials
 */
class MixedPolyList {
public:
    /**
     * Construct from a C++ vector of PolyVariantPtr
     */
    explicit MixedPolyList(std::vector<PolyVariantPtr> polys)
        : polys_(std::move(polys)) {}

    /**
     * Default constructor - empty list
     */
    MixedPolyList() = default;

    /**
     * Append a polynomial variant to the list
     */
    void append(PolyVariantPtr poly) {
        polys_.push_back(std::move(poly));
    }

    /**
     * Clear all polynomials from the list
     */
    void clear() {
        polys_.clear();
    }

    /**
     * Reserve capacity for n polynomials
     */
    void reserve(size_t n) {
        polys_.reserve(n);
    }

    /**
     * Evaluate all polynomials at x
     *
     * @param x Scalar or ndarray of evaluation points
     * @param nu Derivative order (0 = value, 1 = first derivative, etc.)
     * @return ndarray of shape (N, *input_shape)
     */
    nb::object call(nb::object x, int nu = 0) const {
        if (polys_.empty()) {
            // Return empty array
            double* data = new double[0];
            nb::capsule owner(data, [](void* p) noexcept { delete[] (double*)p; });
            size_t shape[] = {0};
            return nb::cast(nb::ndarray<nb::numpy, double>(data, 1, shape, owner));
        }

        // Check if x is an ndarray
        if (nb::isinstance<nb::ndarray<>>(x)) {
            return call_array(x, nu);
        } else {
            // Scalar input
            double val = nb::cast<double>(x);
            return nb::cast(call_scalar(val, nu));
        }
    }

    /**
     * Get number of polynomials
     */
    size_t size() const { return polys_.size(); }

    /**
     * Get individual polynomial variant by index
     */
    const PolyVariant& get(size_t i) const {
        if (i >= polys_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return *polys_[i];
    }

    /**
     * Replace polynomial at index
     */
    void set(size_t i, PolyVariantPtr poly) {
        if (i >= polys_.size()) {
            throw std::out_of_range("Index out of range");
        }
        polys_[i] = std::move(poly);
    }

    /**
     * Integrate all polynomials from a to b
     * @return 1D ndarray of shape (N,) containing integral values
     */
    nb::ndarray<nb::numpy, double> integrate(double a, double b) const {
        size_t n = polys_.size();
        double* data = new double[n];
        nb::capsule owner(data, [](void* p) noexcept { delete[] (double*)p; });

        for (size_t i = 0; i < n; ++i) {
            data[i] = std::visit([a, b](const auto& p) {
                return p.integrate(a, b);
            }, *polys_[i]);
        }

        size_t shape[] = {n};
        return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
    }

    /**
     * Create a new MixedPolyList containing derivatives of all polynomials
     */
    MixedPolyList derivative(int order = 1) const {
        std::vector<PolyVariantPtr> derivs;
        derivs.reserve(polys_.size());
        for (const auto& p : polys_) {
            derivs.push_back(std::visit([order](const auto& poly) -> PolyVariantPtr {
                return make_poly_variant(poly.derivative(order));
            }, *p));
        }
        return MixedPolyList(std::move(derivs));
    }

    /**
     * Create a new MixedPolyList containing antiderivatives of all polynomials
     */
    MixedPolyList antiderivative(int order = 1) const {
        std::vector<PolyVariantPtr> antiderivs;
        antiderivs.reserve(polys_.size());
        for (const auto& p : polys_) {
            antiderivs.push_back(std::visit([order](const auto& poly) -> PolyVariantPtr {
                return make_poly_variant(poly.antiderivative(order));
            }, *p));
        }
        return MixedPolyList(std::move(antiderivs));
    }

    /**
     * Access internal vector
     */
    const std::vector<PolyVariantPtr>& polys() const { return polys_; }

private:
    std::vector<PolyVariantPtr> polys_;

    /**
     * Helper to evaluate a single polynomial variant at a point
     */
    static double eval_variant(const PolyVariant& poly, double x, int nu) {
        return std::visit([x, nu](const auto& p) {
            return nu == 0 ? p(x) : p(x, nu);
        }, poly);
    }

    /**
     * Evaluate at a single scalar point
     * @return 1D ndarray of shape (N,)
     */
    nb::ndarray<nb::numpy, double> call_scalar(double x, int nu) const {
        size_t n = polys_.size();
        double* data = new double[n];
        nb::capsule owner(data, [](void* p) noexcept { delete[] (double*)p; });

        for (size_t i = 0; i < n; ++i) {
            data[i] = eval_variant(*polys_[i], x, nu);
        }

        size_t shape[] = {n};
        return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
    }

    /**
     * Evaluate at array of points
     * @return ndarray of shape (N, *input_shape)
     */
    nb::object call_array(nb::object x, int nu) const {
        // Cast to a generic ndarray to get shape info
        auto arr = nb::cast<nb::ndarray<nb::c_contig, nb::device::cpu>>(x);

        // Get input shape
        size_t ndim_in = arr.ndim();
        std::vector<size_t> input_shape(ndim_in);
        size_t total_points = 1;
        for (size_t i = 0; i < ndim_in; ++i) {
            input_shape[i] = arr.shape(i);
            total_points *= arr.shape(i);
        }

        // Build output shape: (N, *input_shape)
        size_t n_polys = polys_.size();
        std::vector<size_t> output_shape;
        output_shape.reserve(1 + ndim_in);
        output_shape.push_back(n_polys);
        for (size_t i = 0; i < ndim_in; ++i) {
            output_shape.push_back(input_shape[i]);
        }

        // Allocate output
        size_t total_output = n_polys * total_points;
        double* out_data = new double[total_output];
        nb::capsule owner(out_data, [](void* p) noexcept { delete[] (double*)p; });

        // Get input data pointer - try double first, then float32
        try {
            auto arr_double = nb::cast<nb::ndarray<double, nb::c_contig, nb::device::cpu>>(x);
            const double* in_data = arr_double.data();

            // Evaluate each polynomial at all points
            for (size_t p = 0; p < n_polys; ++p) {
                for (size_t j = 0; j < total_points; ++j) {
                    out_data[p * total_points + j] = eval_variant(*polys_[p], in_data[j], nu);
                }
            }
        } catch (...) {
            // Try float32
            try {
                auto arr_float = nb::cast<nb::ndarray<float, nb::c_contig, nb::device::cpu>>(x);
                const float* in_data = arr_float.data();

                for (size_t p = 0; p < n_polys; ++p) {
                    for (size_t j = 0; j < total_points; ++j) {
                        double xval = static_cast<double>(in_data[j]);
                        out_data[p * total_points + j] = eval_variant(*polys_[p], xval, nu);
                    }
                }
            } catch (...) {
                delete[] out_data;
                throw std::runtime_error("Unsupported array dtype. Use float32 or float64.");
            }
        }

        return nb::cast(nb::ndarray<nb::numpy, double>(
            out_data,
            output_shape.size(),
            output_shape.data(),
            owner
        ));
    }
};
