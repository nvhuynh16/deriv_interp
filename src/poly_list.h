#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <vector>
#include <cstddef>

namespace nb = nanobind;

/**
 * PolyList - Batch evaluation container for polynomial objects
 *
 * Stores multiple polynomials and evaluates them all at once, returning
 * an ndarray with the polynomial index as the leading dimension.
 *
 * Output shapes:
 *   f(scalar)    -> (N,)
 *   f(1D array)  -> (N, M)
 *   f(ND array)  -> (N, *input_shape)
 *
 * where N = number of polynomials
 */
template<typename PolyType>
class PolyList {
public:
    /**
     * Construct from a Python list of polynomials
     */
    explicit PolyList(nb::list py_polys) {
        polys_.reserve(nb::len(py_polys));
        for (auto item : py_polys) {
            polys_.push_back(nb::cast<PolyType>(item));
        }
    }

    /**
     * Construct from a C++ vector of polynomials
     */
    explicit PolyList(std::vector<PolyType> polys)
        : polys_(std::move(polys)) {}

    /**
     * Default constructor - empty list
     */
    PolyList() = default;

    /**
     * Append a single polynomial to the list
     */
    void append(const PolyType& poly) {
        polys_.push_back(poly);
    }

    /**
     * Append a single polynomial (move version)
     */
    void append(PolyType&& poly) {
        polys_.push_back(std::move(poly));
    }

    /**
     * Extend the list with polynomials from another list
     */
    void extend(const PolyList<PolyType>& other) {
        polys_.reserve(polys_.size() + other.polys_.size());
        for (const auto& p : other.polys_) {
            polys_.push_back(p);
        }
    }

    /**
     * Extend the list with polynomials from a Python list
     */
    void extend_from_pylist(nb::list py_polys) {
        polys_.reserve(polys_.size() + nb::len(py_polys));
        for (auto item : py_polys) {
            polys_.push_back(nb::cast<PolyType>(item));
        }
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
     * Get individual polynomial by index
     */
    const PolyType& get(size_t i) const {
        if (i >= polys_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return polys_[i];
    }

    /**
     * Replace polynomial at index
     * Rebuilds vector since polynomial classes have deleted assignment operators
     */
    void set(size_t i, const PolyType& poly) {
        if (i >= polys_.size()) {
            throw std::out_of_range("Index out of range");
        }
        // Rebuild vector using only constructors (no assignment)
        std::vector<PolyType> new_polys;
        new_polys.reserve(polys_.size());
        for (size_t j = 0; j < polys_.size(); ++j) {
            if (j == i) {
                new_polys.push_back(poly);  // Copy construct new element
            } else {
                new_polys.push_back(std::move(polys_[j]));  // Move construct existing
            }
        }
        polys_ = std::move(new_polys);  // Vector move assignment (not element assignment)
    }

    /**
     * Create a new PolyList containing derivatives of all polynomials
     */
    PolyList<PolyType> derivative(int order = 1) const {
        std::vector<PolyType> derivs;
        derivs.reserve(polys_.size());
        for (const auto& p : polys_) {
            derivs.push_back(p.derivative(order));
        }
        return PolyList<PolyType>(std::move(derivs));
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
            data[i] = polys_[i].integrate(a, b);
        }

        size_t shape[] = {n};
        return nb::ndarray<nb::numpy, double>(data, 1, shape, owner);
    }

    /**
     * Access internal vector (for derivative PolyList creation)
     */
    const std::vector<PolyType>& polys() const { return polys_; }

private:
    std::vector<PolyType> polys_;

    /**
     * Evaluate at a single scalar point
     * @return 1D ndarray of shape (N,)
     */
    nb::ndarray<nb::numpy, double> call_scalar(double x, int nu) const {
        size_t n = polys_.size();
        double* data = new double[n];
        nb::capsule owner(data, [](void* p) noexcept { delete[] (double*)p; });

        for (size_t i = 0; i < n; ++i) {
            if (nu == 0) {
                data[i] = polys_[i](x);
            } else {
                data[i] = polys_[i](x, nu);
            }
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

        // Get input data pointer - need to handle different dtypes
        // Try to cast to double array first
        try {
            auto arr_double = nb::cast<nb::ndarray<double, nb::c_contig, nb::device::cpu>>(x);
            const double* in_data = arr_double.data();

            // Evaluate each polynomial at all points
            for (size_t p = 0; p < n_polys; ++p) {
                for (size_t j = 0; j < total_points; ++j) {
                    double xval = in_data[j];
                    if (nu == 0) {
                        out_data[p * total_points + j] = polys_[p](xval);
                    } else {
                        out_data[p * total_points + j] = polys_[p](xval, nu);
                    }
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
                        if (nu == 0) {
                            out_data[p * total_points + j] = polys_[p](xval);
                        } else {
                            out_data[p * total_points + j] = polys_[p](xval, nu);
                        }
                    }
                }
            } catch (...) {
                // Fall back to item-by-item access via Python
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
