/**
 * deriv_poly - Python bindings for piecewise polynomial interpolation
 *
 * Provides: BPoly, CPoly, LegPoly, HPoly, BsPoly, LagPoly
 * Plus list classes: BPolyList, CPolyList, LegPolyList, HPolyList, BsPolyList, LagPolyList
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/ndarray.h>

#include "bpoly.h"
#include "cpoly.h"
#include "legpoly.h"
#include "hpoly.h"
#include "bspoly.h"
#include "lagpoly.h"
#include "poly_list.h"
#include "mixed_poly_list.h"
#include "ndarray.h"

namespace nb = nanobind;

// Helper to evaluate a polynomial at an array or scalar
template<typename PolyType>
nb::object poly_call(const PolyType& self, nb::object x, int nu) {
    // Check if x is an ndarray
    if (nb::isinstance<nb::ndarray<>>(x)) {
        // Try double array first
        try {
            auto arr = nb::cast<nb::ndarray<double, nb::c_contig, nb::device::cpu>>(x);
            size_t ndim = arr.ndim();
            std::vector<size_t> shape(ndim);
            size_t total = 1;
            for (size_t i = 0; i < ndim; ++i) {
                shape[i] = arr.shape(i);
                total *= arr.shape(i);
            }

            double* out_data = new double[total];
            nb::capsule owner(out_data, [](void* p) noexcept { delete[] (double*)p; });

            const double* in_data = arr.data();
            for (size_t i = 0; i < total; ++i) {
                if (nu == 0) {
                    out_data[i] = self(in_data[i]);
                } else {
                    out_data[i] = self(in_data[i], nu);
                }
            }

            return nb::cast(nb::ndarray<nb::numpy, double>(out_data, ndim, shape.data(), owner));
        } catch (...) {
            // Try float32
            auto arr = nb::cast<nb::ndarray<float, nb::c_contig, nb::device::cpu>>(x);
            size_t ndim = arr.ndim();
            std::vector<size_t> shape(ndim);
            size_t total = 1;
            for (size_t i = 0; i < ndim; ++i) {
                shape[i] = arr.shape(i);
                total *= arr.shape(i);
            }

            double* out_data = new double[total];
            nb::capsule owner(out_data, [](void* p) noexcept { delete[] (double*)p; });

            const float* in_data = arr.data();
            for (size_t i = 0; i < total; ++i) {
                double val = static_cast<double>(in_data[i]);
                if (nu == 0) {
                    out_data[i] = self(val);
                } else {
                    out_data[i] = self(val, nu);
                }
            }

            return nb::cast(nb::ndarray<nb::numpy, double>(out_data, ndim, shape.data(), owner));
        }
    } else {
        // Scalar input
        double val = nb::cast<double>(x);
        if (nu == 0) {
            return nb::cast(self(val));
        } else {
            return nb::cast(self(val, nu));
        }
    }
}

// Flexible 1D conversion - handles numpy arrays, lists, tuples, and other sequences
std::vector<double> to_vec1d(nb::object obj) {
    std::vector<double> result;

    // Fast path: numpy array
    if (nb::isinstance<nb::ndarray<>>(obj)) {
        try {
            // Try double array first
            auto arr = nb::cast<nb::ndarray<double, nb::c_contig, nb::device::cpu>>(obj);
            const double* data = arr.data();
            size_t n = 1;
            for (size_t i = 0; i < arr.ndim(); ++i) n *= arr.shape(i);
            result.assign(data, data + n);
            return result;
        } catch (...) {
            try {
                // Try float32
                auto arr = nb::cast<nb::ndarray<float, nb::c_contig, nb::device::cpu>>(obj);
                const float* data = arr.data();
                size_t n = 1;
                for (size_t i = 0; i < arr.ndim(); ++i) n *= arr.shape(i);
                result.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    result.push_back(static_cast<double>(data[i]));
                }
                return result;
            } catch (...) {
                // Fall through to generic sequence handling
            }
        }
    }

    // Generic sequence path (list, tuple, etc.)
    for (auto item : obj) {
        result.push_back(nb::cast<double>(item));
    }
    return result;
}

// Flexible 2D conversion - handles numpy arrays, lists of lists, tuples, etc.
std::vector<std::vector<double>> to_vec2d(nb::object obj) {
    std::vector<std::vector<double>> result;

    // Fast path: 2D numpy array
    if (nb::isinstance<nb::ndarray<>>(obj)) {
        try {
            auto arr = nb::cast<nb::ndarray<double, nb::c_contig, nb::device::cpu>>(obj);
            if (arr.ndim() == 2) {
                size_t rows = arr.shape(0);
                size_t cols = arr.shape(1);
                result.reserve(rows);
                const double* data = arr.data();
                for (size_t i = 0; i < rows; ++i) {
                    result.emplace_back(data + i * cols, data + (i + 1) * cols);
                }
                return result;
            }
        } catch (...) {
            try {
                // Try float32
                auto arr = nb::cast<nb::ndarray<float, nb::c_contig, nb::device::cpu>>(obj);
                if (arr.ndim() == 2) {
                    size_t rows = arr.shape(0);
                    size_t cols = arr.shape(1);
                    result.reserve(rows);
                    const float* data = arr.data();
                    for (size_t i = 0; i < rows; ++i) {
                        std::vector<double> row;
                        row.reserve(cols);
                        for (size_t j = 0; j < cols; ++j) {
                            row.push_back(static_cast<double>(data[i * cols + j]));
                        }
                        result.push_back(std::move(row));
                    }
                    return result;
                }
            } catch (...) {
                // Fall through to sequence handling
            }
        }
    }

    // Generic sequence of sequences
    for (auto row : obj) {
        result.push_back(to_vec1d(nb::borrow<nb::object>(row)));
    }
    return result;
}

// Flexible int conversion - handles numpy arrays, lists, tuples
std::vector<int> to_vec_int(nb::object obj) {
    std::vector<int> result;

    if (nb::isinstance<nb::ndarray<>>(obj)) {
        try {
            // Try int64 first (numpy default)
            auto arr = nb::cast<nb::ndarray<int64_t, nb::c_contig, nb::device::cpu>>(obj);
            const int64_t* data = arr.data();
            size_t n = 1;
            for (size_t i = 0; i < arr.ndim(); ++i) n *= arr.shape(i);
            result.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                result.push_back(static_cast<int>(data[i]));
            }
            return result;
        } catch (...) {
            try {
                // Try int32
                auto arr = nb::cast<nb::ndarray<int32_t, nb::c_contig, nb::device::cpu>>(obj);
                const int32_t* data = arr.data();
                size_t n = 1;
                for (size_t i = 0; i < arr.ndim(); ++i) n *= arr.shape(i);
                result.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    result.push_back(static_cast<int>(data[i]));
                }
                return result;
            } catch (...) {
                // Fall through to generic sequence handling
            }
        }
    }

    // Generic sequence path
    for (auto item : obj) {
        result.push_back(nb::cast<int>(item));
    }
    return result;
}

// Helper to convert vector<vector<double>> to Python list
nb::list from_vec2d(const std::vector<std::vector<double>>& vec) {
    nb::list result;
    for (const auto& row : vec) {
        nb::list inner;
        for (double val : row) {
            inner.append(val);
        }
        result.append(inner);
    }
    return result;
}

// Helper to convert vector<double> to Python list
nb::list from_vec1d(const std::vector<double>& vec) {
    nb::list result;
    for (double val : vec) {
        result.append(val);
    }
    return result;
}

// Zero-copy helper: checks if numpy array is C-contiguous double and creates array2d
// Returns nullopt if not suitable for zero-copy
std::optional<ndarray::array2d<double>> try_zerocopy_array2d(nb::object obj) {
    if (!nb::isinstance<nb::ndarray<>>(obj)) {
        return std::nullopt;
    }

    try {
        auto arr = nb::cast<nb::ndarray<double, nb::c_contig, nb::device::cpu>>(obj);
        if (arr.ndim() != 2) {
            return std::nullopt;
        }

        size_t rows = arr.shape(0);
        size_t cols = arr.shape(1);
        double* ptr = const_cast<double*>(arr.data());

        // Use shared_ptr aliasing constructor: the shared_ptr holds a reference
        // to the Python object (via inc_ref), preventing deallocation while we use it.
        // When the shared_ptr is destroyed, the Python object refcount is decremented.
        // We borrow from the original object (obj), not the casted array.
        nb::object* holder = new nb::object(nb::borrow(obj));
        auto owner = std::shared_ptr<void>(holder, [](void* p) {
            nb::object* obj_ptr = static_cast<nb::object*>(p);
            nb::gil_scoped_acquire guard;  // Ensure GIL is held for dec_ref
            delete obj_ptr;
        });

        return ndarray::array2d<double>(ptr, rows, cols, owner);
    } catch (...) {
        return std::nullopt;
    }
}

// =============================================================================
// Template factory: construct a coefficient-based polynomial with zero-copy support
// Works for BPoly, CPoly, LegPoly, BsPoly (all share same constructor signature)
// =============================================================================
template <typename PolyType>
PolyType make_coeff_poly(nb::object coefficients, nb::object breakpoints,
                         ExtrapolateMode extrapolate,
                         int extrapolate_order_left, int extrapolate_order_right) {
    auto arr2d = try_zerocopy_array2d(coefficients);
    if (arr2d.has_value()) {
        return PolyType::from_array2d(std::move(*arr2d), to_vec1d(breakpoints),
                                      extrapolate, extrapolate_order_left, extrapolate_order_right);
    }
    return PolyType(to_vec2d(coefficients), to_vec1d(breakpoints),
                    extrapolate, extrapolate_order_left, extrapolate_order_right);
}

// =============================================================================
// Register common methods shared by all coefficient-based polynomial types
// (BPoly, CPoly, LegPoly, HPoly, BsPoly)
// =============================================================================
template <typename PolyType, typename ClassBinding>
void register_common_poly_methods(ClassBinding& cls) {
    cls
        .def("__call__", [](const PolyType& self, nb::object x, int nu) {
            return poly_call(self, x, nu);
        }, nb::arg("x"), nb::arg("nu") = 0, "Evaluate polynomial (nu = derivative order)")
        .def("derivative", &PolyType::derivative, nb::arg("order") = 1,
            "Return derivative polynomial")
        .def("antiderivative", &PolyType::antiderivative, nb::arg("order") = 1,
            "Return antiderivative polynomial")
        .def("integrate", &PolyType::integrate, nb::arg("a"), nb::arg("b"),
            nb::arg("extrapolate") = nb::none(), "Compute definite integral from a to b")
        .def("roots", &PolyType::roots, nb::arg("discontinuity") = true,
            nb::arg("extrapolate") = true, "Find all real roots")
        .def("extend", [](const PolyType& self, nb::object c, nb::object x, bool right) {
            return self.extend(to_vec2d(c), to_vec1d(x), right);
        }, nb::arg("c"), nb::arg("x"), nb::arg("right") = true,
            "Extend polynomial with new intervals")
        .def("to_power_basis", [](const PolyType& self) {
            return from_vec2d(self.to_power_basis());
        }, "Convert to power basis coefficients")
        .def_prop_ro("c", [](const PolyType& self) { return from_vec2d(self.c()); },
            "Coefficients (scipy compatibility)")
        .def_prop_ro("c_array", [](const PolyType& self) {
            const auto& arr = self.coefficients_array();
            size_t shape[2] = {arr.rows, arr.cols};
            auto data_holder = new std::shared_ptr<double[]>(arr.data);
            nb::capsule owner(data_holder, [](void* p) noexcept {
                delete static_cast<std::shared_ptr<double[]>*>(p);
            });
            return nb::ndarray<nb::numpy, const double>(
                arr.data.get(), 2, shape, owner);
        }, nb::rv_policy::automatic, "Coefficients as numpy array (zero-copy view)")
        .def_prop_ro("x", [](const PolyType& self) { return from_vec1d(self.x()); },
            "Breakpoints (scipy compatibility)")
        .def_prop_ro("degree", &PolyType::degree, "Polynomial degree")
        .def_prop_ro("num_intervals", &PolyType::num_intervals, "Number of intervals")
        .def_prop_ro("extrapolate", &PolyType::extrapolate, "Extrapolation mode")
        .def_prop_ro("is_ascending", &PolyType::is_ascending, "Whether breakpoints are ascending");
}

// =============================================================================
// Register standard constructor + from_derivatives + from_power_basis
// for BPoly, CPoly, LegPoly, BsPoly (all share same signatures)
// =============================================================================
template <typename PolyType, typename ClassBinding>
void register_standard_constructors(ClassBinding& cls) {
    cls
        .def("__init__", [](PolyType* self, nb::object coefficients, nb::object breakpoints,
                            ExtrapolateMode extrapolate,
                            int extrapolate_order_left, int extrapolate_order_right) {
            new (self) PolyType(make_coeff_poly<PolyType>(coefficients, breakpoints, extrapolate,
                                                           extrapolate_order_left, extrapolate_order_right));
        },
            nb::arg("coefficients"), nb::arg("breakpoints"),
            nb::arg("extrapolate") = ExtrapolateMode::Extrapolate,
            nb::arg("extrapolate_order_left") = -1,
            nb::arg("extrapolate_order_right") = -1,
            "Construct from coefficients and breakpoints")
        .def_static("from_derivatives", [](nb::object xi, nb::object yi, nb::object orders) {
            auto xi_vec = to_vec1d(xi);
            auto yi_vec = to_vec2d(yi);
            std::vector<int> orders_vec;
            if (!orders.is_none()) {
                orders_vec = to_vec_int(orders);
            }
            return PolyType::from_derivatives(xi_vec, yi_vec, orders_vec);
        },
            nb::arg("xi"), nb::arg("yi"), nb::arg("orders") = nb::none(),
            "Construct from derivatives at breakpoints")
        .def_static("from_power_basis", [](nb::object power_coeffs, nb::object breakpoints,
                                           ExtrapolateMode extrapolate) {
            return PolyType::from_power_basis(to_vec2d(power_coeffs), to_vec1d(breakpoints), extrapolate);
        },
            nb::arg("power_coeffs"), nb::arg("breakpoints"),
            nb::arg("extrapolate") = ExtrapolateMode::Extrapolate,
            "Construct from power basis coefficients");
}

// =============================================================================
// Register a PolyList<T> class
// =============================================================================
template <typename PolyType>
void register_poly_list(nb::module_& m, const char* name, const char* doc) {
    nb::class_<PolyList<PolyType>>(m, name, doc)
        .def(nb::init<>(), "Create empty list")
        .def(nb::init<nb::list>(), nb::arg("polys"), "Create from Python list")
        .def("__call__", &PolyList<PolyType>::call, nb::arg("x"), nb::arg("nu") = 0,
            "Evaluate all polynomials. Returns shape (N, *input_shape)")
        .def("__len__", &PolyList<PolyType>::size)
        .def("__getitem__", &PolyList<PolyType>::get, nb::rv_policy::reference)
        .def("__setitem__", &PolyList<PolyType>::set, nb::arg("i"), nb::arg("poly"),
            "Replace polynomial at index")
        .def("append", nb::overload_cast<const PolyType&>(&PolyList<PolyType>::append),
            nb::arg("poly"), "Append a polynomial")
        .def("extend", &PolyList<PolyType>::extend_from_pylist, nb::arg("polys"),
            "Extend with polynomials from a list")
        .def("clear", &PolyList<PolyType>::clear, "Remove all polynomials")
        .def("reserve", &PolyList<PolyType>::reserve, nb::arg("n"), "Reserve capacity")
        .def("derivative", &PolyList<PolyType>::derivative, nb::arg("order") = 1,
            "Return new list of derivative polynomials")
        .def("integrate", &PolyList<PolyType>::integrate, nb::arg("a"), nb::arg("b"),
            "Integrate all polynomials from a to b. Returns 1D array");
}

NB_MODULE(deriv_poly, m) {
    m.doc() = "Piecewise polynomial interpolation: BPoly, CPoly, LegPoly, BsPoly, LagPoly";

    //=========================================================================
    // Enums
    //=========================================================================

    nb::enum_<ExtrapolateMode>(m, "ExtrapolateMode",
        "Extrapolation mode for polynomial evaluation (shared by all polynomial classes)")
        .value("Extrapolate", ExtrapolateMode::Extrapolate,
               "Use polynomial from nearest segment (default)")
        .value("NoExtrapolate", ExtrapolateMode::NoExtrapolate,
               "Return NaN for out-of-bounds")
        .value("Periodic", ExtrapolateMode::Periodic,
               "Wrap around periodically");

    nb::enum_<HermiteKind>(m, "HermiteKind",
        "Hermite polynomial variant for HPoly")
        .value("Physicist", HermiteKind::Physicist,
               "Physicist's Hermite polynomials H_n (numpy.polynomial.hermite)")
        .value("Probabilist", HermiteKind::Probabilist,
               "Probabilist's Hermite polynomials He_n (numpy.polynomial.hermite_e)");

    //=========================================================================
    // BPoly, CPoly, LegPoly, BsPoly — standard coefficient-based polynomials
    //=========================================================================

    auto bpoly_cls = nb::class_<BPoly>(m, "BPoly",
        "Bernstein polynomial piecewise interpolant (matches scipy.interpolate.BPoly)");
    register_standard_constructors<BPoly>(bpoly_cls);
    register_common_poly_methods<BPoly>(bpoly_cls);

    auto cpoly_cls = nb::class_<CPoly>(m, "CPoly",
        "Chebyshev polynomial piecewise interpolant (matches numpy.polynomial.chebyshev)");
    register_standard_constructors<CPoly>(cpoly_cls);
    register_common_poly_methods<CPoly>(cpoly_cls);

    auto legpoly_cls = nb::class_<LegPoly>(m, "LegPoly",
        "Legendre polynomial piecewise interpolant (matches numpy.polynomial.legendre)");
    register_standard_constructors<LegPoly>(legpoly_cls);
    register_common_poly_methods<LegPoly>(legpoly_cls);

    auto bspoly_cls = nb::class_<BsPoly>(m, "BsPoly",
        "B-Spline with Bernstein storage (conceptually B-spline, internally Bernstein)");
    register_standard_constructors<BsPoly>(bspoly_cls);
    register_common_poly_methods<BsPoly>(bspoly_cls);

    //=========================================================================
    // HPoly — special: extra HermiteKind parameter
    //=========================================================================

    auto hpoly_cls = nb::class_<HPoly>(m, "HPoly",
        "Hermite polynomial piecewise interpolant (Physicist's H_n or Probabilist's He_n)");
    hpoly_cls
        .def("__init__", [](HPoly* self, nb::object coefficients, nb::object breakpoints,
                            HermiteKind kind,
                            ExtrapolateMode extrapolate,
                            int extrapolate_order_left, int extrapolate_order_right) {
            auto arr2d = try_zerocopy_array2d(coefficients);
            if (arr2d.has_value()) {
                new (self) HPoly(HPoly::from_array2d(std::move(*arr2d), to_vec1d(breakpoints),
                                                     kind, extrapolate, extrapolate_order_left, extrapolate_order_right));
            } else {
                new (self) HPoly(to_vec2d(coefficients), to_vec1d(breakpoints),
                                 kind, extrapolate, extrapolate_order_left, extrapolate_order_right);
            }
        },
            nb::arg("coefficients"), nb::arg("breakpoints"),
            nb::arg("kind") = HermiteKind::Physicist,
            nb::arg("extrapolate") = ExtrapolateMode::Extrapolate,
            nb::arg("extrapolate_order_left") = -1,
            nb::arg("extrapolate_order_right") = -1,
            "Construct from coefficients and breakpoints")
        .def_static("from_derivatives", [](nb::object xi, nb::object yi, nb::object orders,
                                           HermiteKind kind) {
            auto xi_vec = to_vec1d(xi);
            auto yi_vec = to_vec2d(yi);
            std::vector<int> orders_vec;
            if (!orders.is_none()) {
                orders_vec = to_vec_int(orders);
            }
            return HPoly::from_derivatives(xi_vec, yi_vec, orders_vec, kind);
        }, nb::arg("xi"), nb::arg("yi"), nb::arg("orders") = nb::none(),
           nb::arg("kind") = HermiteKind::Physicist,
           "Construct from derivatives at breakpoints")
        .def_static("from_power_basis", [](nb::object power_coeffs, nb::object breakpoints,
                                           HermiteKind kind,
                                           ExtrapolateMode extrapolate) {
            return HPoly::from_power_basis(to_vec2d(power_coeffs), to_vec1d(breakpoints), kind, extrapolate);
        }, nb::arg("power_coeffs"), nb::arg("breakpoints"),
           nb::arg("kind") = HermiteKind::Physicist,
           nb::arg("extrapolate") = ExtrapolateMode::Extrapolate,
           "Construct from power basis coefficients");
    register_common_poly_methods<HPoly>(hpoly_cls);
    hpoly_cls.def_prop_ro("kind", &HPoly::kind,
        "Hermite polynomial variant (Physicist or Probabilist)");

    //=========================================================================
    // LagPoly — special: different constructor and properties
    //=========================================================================

    auto lagpoly_cls = nb::class_<LagPoly>(m, "LagPoly",
        "Barycentric Lagrange polynomial interpolant (matches scipy.interpolate.BarycentricInterpolator)");
    lagpoly_cls
        .def("__init__", [](LagPoly* self, nb::object nodes, nb::object values, nb::object breakpoints,
                            ExtrapolateMode extrapolate,
                            int extrapolate_order_left, int extrapolate_order_right) {
            new (self) LagPoly(to_vec2d(nodes), to_vec2d(values), to_vec1d(breakpoints),
                              extrapolate, extrapolate_order_left, extrapolate_order_right);
        },
            nb::arg("nodes"), nb::arg("values"), nb::arg("breakpoints"),
            nb::arg("extrapolate") = ExtrapolateMode::Extrapolate,
            nb::arg("extrapolate_order_left") = -1,
            nb::arg("extrapolate_order_right") = -1,
            "Construct from nodes, values, and breakpoints")
        .def_static("from_derivatives", [](nb::object xi, nb::object yi, nb::object orders) {
            auto xi_vec = to_vec1d(xi);
            auto yi_vec = to_vec2d(yi);
            std::vector<int> orders_vec;
            if (!orders.is_none()) {
                orders_vec = to_vec_int(orders);
            }
            return LagPoly::from_derivatives(xi_vec, yi_vec, orders_vec);
        }, nb::arg("xi"), nb::arg("yi"), nb::arg("orders") = nb::none())
        .def_static("from_power_basis", [](nb::object power_coeffs, nb::object breakpoints,
                                           ExtrapolateMode extrapolate) {
            return LagPoly::from_power_basis(to_vec2d(power_coeffs), to_vec1d(breakpoints), extrapolate);
        }, nb::arg("power_coeffs"), nb::arg("breakpoints"),
           nb::arg("extrapolate") = ExtrapolateMode::Extrapolate)
        .def_static("from_chebyshev_nodes", [](int n, nb::object values, nb::object breakpoints,
                                               ExtrapolateMode extrapolate) {
            return LagPoly::from_chebyshev_nodes(n, to_vec2d(values), to_vec1d(breakpoints), extrapolate);
        }, nb::arg("n"), nb::arg("values"), nb::arg("breakpoints"),
           nb::arg("extrapolate") = ExtrapolateMode::Extrapolate)
        .def("__call__", [](const LagPoly& self, nb::object x, int nu) {
            return poly_call(self, x, nu);
        }, nb::arg("x"), nb::arg("nu") = 0)
        .def("derivative", &LagPoly::derivative, nb::arg("order") = 1)
        .def("antiderivative", &LagPoly::antiderivative, nb::arg("order") = 1)
        .def("integrate", &LagPoly::integrate, nb::arg("a"), nb::arg("b"),
            nb::arg("extrapolate") = nb::none())
        .def("roots", &LagPoly::roots, nb::arg("discontinuity") = true,
            nb::arg("extrapolate") = true)
        .def("extend", [](const LagPoly& self, nb::object nodes, nb::object values,
                          nb::object x, bool right) {
            return self.extend(to_vec2d(nodes), to_vec2d(values), to_vec1d(x), right);
        }, nb::arg("nodes"), nb::arg("values"), nb::arg("x"), nb::arg("right") = true)
        .def("to_power_basis", [](const LagPoly& self) {
            return from_vec2d(self.to_power_basis());
        })
        .def_prop_ro("nodes", [](const LagPoly& self) { return from_vec2d(self.nodes()); })
        .def_prop_ro("values", [](const LagPoly& self) { return from_vec2d(self.values()); })
        .def_prop_ro("weights", [](const LagPoly& self) { return from_vec2d(self.weights()); })
        .def_prop_ro("x", [](const LagPoly& self) { return from_vec1d(self.x()); })
        .def_prop_ro("degree", &LagPoly::degree)
        .def_prop_ro("num_intervals", &LagPoly::num_intervals)
        .def_prop_ro("extrapolate", &LagPoly::extrapolate)
        .def_prop_ro("is_ascending", &LagPoly::is_ascending);

    //=========================================================================
    // PolyList classes
    //=========================================================================

    register_poly_list<BPoly>(m, "BPolyList", "Batch evaluation container for BPoly polynomials");
    register_poly_list<CPoly>(m, "CPolyList", "Batch evaluation container for CPoly polynomials");
    register_poly_list<LegPoly>(m, "LegPolyList", "Batch evaluation container for LegPoly polynomials");
    register_poly_list<HPoly>(m, "HPolyList", "Batch evaluation container for HPoly polynomials");
    register_poly_list<BsPoly>(m, "BsPolyList", "Batch evaluation container for BsPoly polynomials");
    register_poly_list<LagPoly>(m, "LagPolyList", "Batch evaluation container for LagPoly polynomials");

    //=========================================================================
    // MixedPolyList - heterogeneous polynomial container
    //=========================================================================

    // Helper lambda to convert a Python object to PolyVariantPtr
    auto to_poly_variant_ptr = [](nb::object item) -> PolyVariantPtr {
        if (nb::isinstance<BPoly>(item)) {
            return make_poly_variant(nb::cast<BPoly>(item));
        } else if (nb::isinstance<CPoly>(item)) {
            return make_poly_variant(nb::cast<CPoly>(item));
        } else if (nb::isinstance<LegPoly>(item)) {
            return make_poly_variant(nb::cast<LegPoly>(item));
        } else if (nb::isinstance<HPoly>(item)) {
            return make_poly_variant(nb::cast<HPoly>(item));
        } else if (nb::isinstance<BsPoly>(item)) {
            return make_poly_variant(nb::cast<BsPoly>(item));
        } else if (nb::isinstance<LagPoly>(item)) {
            return make_poly_variant(nb::cast<LagPoly>(item));
        } else {
            throw std::runtime_error(
                "MixedPolyList only accepts BPoly, CPoly, LegPoly, HPoly, BsPoly, or LagPoly");
        }
    };

    nb::class_<MixedPolyList>(m, "MixedPolyList",
        "Batch evaluation container for mixed polynomial types (BPoly, CPoly, LegPoly, HPoly, BsPoly, LagPoly)")
        .def("__init__", [to_poly_variant_ptr](MixedPolyList* self, nb::list py_polys) {
            std::vector<PolyVariantPtr> polys;
            polys.reserve(nb::len(py_polys));
            for (auto item : py_polys) {
                polys.push_back(to_poly_variant_ptr(nb::borrow<nb::object>(item)));
            }
            new (self) MixedPolyList(std::move(polys));
        }, nb::arg("polys"), "Construct from a list of mixed polynomial types")
        .def(nb::init<>(), "Create empty list")
        .def("__call__", &MixedPolyList::call, nb::arg("x"), nb::arg("nu") = 0,
            "Evaluate all polynomials. Returns shape (N,) for scalar, (N, M) for 1D array")
        .def("__len__", &MixedPolyList::size)
        .def("__getitem__", [](const MixedPolyList& self, size_t i) -> nb::object {
            const PolyVariant& v = self.get(i);
            return std::visit([](const auto& p) -> nb::object {
                return nb::cast(p);
            }, v);
        }, nb::arg("i"), "Get polynomial at index (returns the specific polynomial type)")
        .def("__setitem__", [to_poly_variant_ptr](MixedPolyList& self, size_t i, nb::object item) {
            self.set(i, to_poly_variant_ptr(item));
        }, nb::arg("i"), nb::arg("poly"), "Replace polynomial at index")
        .def("append", [to_poly_variant_ptr](MixedPolyList& self, nb::object item) {
            self.append(to_poly_variant_ptr(item));
        }, nb::arg("poly"), "Append a polynomial")
        .def("extend", [to_poly_variant_ptr](MixedPolyList& self, nb::list py_polys) {
            self.reserve(self.size() + nb::len(py_polys));
            for (auto item : py_polys) {
                self.append(to_poly_variant_ptr(nb::borrow<nb::object>(item)));
            }
        }, nb::arg("polys"), "Extend with polynomials from a list")
        .def("clear", &MixedPolyList::clear, "Remove all polynomials")
        .def("reserve", &MixedPolyList::reserve, nb::arg("n"), "Reserve capacity")
        .def("integrate", &MixedPolyList::integrate, nb::arg("a"), nb::arg("b"),
            "Integrate all polynomials from a to b. Returns 1D array")
        .def("derivative", &MixedPolyList::derivative, nb::arg("order") = 1,
            "Return new MixedPolyList of derivative polynomials")
        .def("antiderivative", &MixedPolyList::antiderivative, nb::arg("order") = 1,
            "Return new MixedPolyList of antiderivative polynomials");
}
