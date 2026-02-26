# BPoly - Piecewise Polynomial C++ Library

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Header-only](https://img.shields.io/badge/Header--only-yes-brightgreen.svg)
![Thread-safe](https://img.shields.io/badge/Thread--safe-yes-brightgreen.svg)

Thread-safe, header-only C++17 library for piecewise polynomial interpolation with Python bindings. All implementations verified against scipy/numpy within 1e-10 relative error.

## Features

- **Six polynomial representations**: Bernstein, Chebyshev, Legendre, Hermite, B-Spline, and Lagrange
- **Header-only**: Just `#include` and go - no libraries to link
- **Thread-safe**: Immutable after construction; all methods are const
- **scipy/numpy compatible**: API mirrors scipy.interpolate and numpy.polynomial
- **Zero-copy Python bindings**: Efficient NumPy interoperability via nanobind
- **Full calculus support**: Derivatives, antiderivatives, integration, root finding

## Quick Start

### C++

```cpp
#include "bpoly.h"

BPoly p = BPoly::from_derivatives({0.0, 1.0}, {{0.0, 1.0}, {1.0, -1.0}});
double y = p(0.5);        // evaluate
double dy = p(0.5, 1);    // first derivative
```

### Python

```python
import deriv_poly as dp

p = dp.BPoly.from_derivatives([0, 1], [[0, 1], [1, -1]])
y = p(0.5)        # evaluate
dy = p(0.5, nu=1) # first derivative
```

## Installation

### C++ (Header-only)

Copy the `include/` directory to your project, or use CMake FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(
    BPoly
    GIT_REPOSITORY https://github.com/nvhuynh16/deriv_interp.git
    GIT_TAG main
)
FetchContent_MakeAvailable(BPoly)
target_include_directories(your_target PRIVATE ${bpoly_SOURCE_DIR}/include)
```

**Dependency**: Eigen (header-only) - fetched automatically if using CMake.

### Python

```bash
pip install deriv_poly
```

Or build from source:

```bash
git clone https://github.com/nvhuynh16/deriv_interp.git
cd BPoly
pip install -e .
```

## When to Use Each Polynomial Type

| Use Case | Recommended | Why |
|----------|-------------|-----|
| General interpolation | BPoly | Stable de Casteljau algorithm |
| Function approximation | CPoly | Chebyshev near-optimal approximation |
| Numerical integration | LegPoly | Natural for Gauss-Legendre quadrature |
| Physics / quantum mechanics | HPoly | Hermite polynomials arise naturally in QM and probability |
| B-spline interface | BsPoly | scipy.interpolate.BSpline compatible |
| Arbitrary node data | LagPoly | Barycentric Lagrange for any nodes |

## Polynomial Types

### BPoly (Bernstein Basis)

Piecewise Bernstein polynomials using de Casteljau evaluation. Mirrors `scipy.interpolate.BPoly`.

```cpp
#include "bpoly.h"

// From Hermite interpolation data (function values and derivatives)
std::vector<double> xi = {0.0, 1.0, 2.0};
std::vector<std::vector<double>> yi = {
    {0.0, 1.0},   // f(0)=0, f'(0)=1
    {1.0, 0.0},   // f(1)=1, f'(1)=0
    {0.0, -1.0}   // f(2)=0, f'(2)=-1
};
BPoly p = BPoly::from_derivatives(xi, yi);

// Or from coefficients directly
std::vector<std::vector<double>> coeffs = {{0.0, 1.0}, {1.0, 0.0}};
std::vector<double> breaks = {0.0, 1.0, 2.0};
BPoly p2(coeffs, breaks);
```

### CPoly (Chebyshev Basis)

Piecewise Chebyshev polynomials using Clenshaw evaluation. Mirrors `numpy.polynomial.chebyshev`.

```cpp
#include "cpoly.h"

CPoly cp = CPoly::from_derivatives({0.0, 1.0}, {{0.0, 1.0}, {1.0, -1.0}});
double y = cp(0.5);
```

### LegPoly (Legendre Basis)

Piecewise Legendre polynomials. Mirrors `numpy.polynomial.legendre`.

```cpp
#include "legpoly.h"

LegPoly lp = LegPoly::from_derivatives({0.0, 1.0}, {{0.0, 1.0}, {1.0, -1.0}});
double y = lp(0.5);
```

### HPoly (Hermite Basis)

Piecewise Hermite polynomials using Clenshaw evaluation. Supports both Physicist's (H_n) and Probabilist's (He_n) variants. Mirrors `numpy.polynomial.hermite` and `numpy.polynomial.hermite_e`.

```cpp
#include "hpoly.h"

// Physicist's Hermite (default)
HPoly hp = HPoly::from_derivatives({0.0, 1.0}, {{0.0, 1.0}, {1.0, -1.0}});
double y = hp(0.5);

// Probabilist's Hermite
HPoly hpe = HPoly::from_derivatives({0.0, 1.0}, {{0.0, 1.0}, {1.0, -1.0}},
                                     {}, HermiteKind::Probabilist);
```

### BsPoly (B-Spline)

B-spline conceptual wrapper using Bernstein storage internally. scipy compatible.

```cpp
#include "bspoly.h"

BsPoly bs = BsPoly::from_derivatives({0.0, 1.0}, {{0.0, 1.0}, {1.0, -1.0}});
double y = bs(0.5);
```

### LagPoly (Barycentric Lagrange)

Barycentric Lagrange interpolation. Mirrors `scipy.interpolate.BarycentricInterpolator`.

```cpp
#include "lagpoly.h"

LagPoly lag = LagPoly::from_derivatives({0.0, 0.5, 1.0}, {{0.0}, {0.25}, {1.0}});
double y = lag(0.3);
```

## API Reference

### Common Interface

All polynomial classes share the same interface:

```cpp
// Construction
Poly(coefficients, breakpoints, extrapolate);
Poly::from_derivatives(xi, yi, orders);
Poly::from_power_basis(power_coeffs, breakpoints, extrapolate);

// Evaluation (thread-safe)
double y = poly(x);              // value at x
double dy = poly(x, 1);          // first derivative
std::vector<double> ys = poly(xs); // vectorized

// Calculus
Poly dp = poly.derivative(2);     // second derivative polynomial
Poly ap = poly.antiderivative();  // antiderivative polynomial
double area = poly.integrate(a, b);

// Root finding
std::vector<double> r = poly.roots();

// Accessors
poly.c();              // coefficients (copy)
poly.x();              // breakpoints
poly.degree();         // polynomial degree
poly.num_intervals();  // number of piecewise intervals
```

### ExtrapolateMode

Controls behavior outside the defined interval:

```cpp
enum class ExtrapolateMode {
    Extrapolate,    // Extend polynomial beyond boundaries (default)
    NoExtrapolate,  // Return NaN outside boundaries
    Periodic        // Wrap around periodically
};

BPoly p(coeffs, breaks, ExtrapolateMode::Periodic);
```

### Zero-Copy NumPy Integration

Python bindings support zero-copy data sharing with NumPy:

```python
import deriv_poly as dp
import numpy as np

# Zero-copy construction from C-contiguous arrays
coeffs = np.array([[0.0], [1.0]], dtype=np.float64, order='C')
p = dp.BPoly(coeffs, [0.0, 1.0])  # shares numpy buffer

# Zero-copy coefficient access
c_view = p.c_array  # numpy view (no copy)
c_copy = p.c        # Python list (copy)
```

### PolyList (Batch Evaluation)

Efficiently evaluate multiple polynomials at the same points:

```python
# Homogeneous (same type)
pl = dp.BPolyList([p1, p2, p3])
y = pl(0.5)                        # shape: (3,)
y = pl(np.linspace(0, 1, 100))     # shape: (3, 100)

# Heterogeneous (mixed types)
ml = dp.MixedPolyList([bp, cp, hp])
y = ml(0.5)                        # shape: (3,)
```

## Building from Source

### Requirements

- C++17 compiler (MSVC 2022, GCC 9+, or Clang 10+)
- CMake 3.20+
- Python 3.8+ with nanobind (for Python bindings)

### Build Commands

```bash
# Configure and build
cmake -B build
cmake --build build --config Release

# Run C++ tests (2,640 total)
build/Release/bpoly_test.exe       # BPoly (711 tests)
build/Release/cpoly_test.exe       # CPoly (463 tests)
build/Release/legpoly_test.exe     # LegPoly (368 tests)
build/Release/hpoly_test.exe       # HPoly (375 tests)
build/Release/bspoly_test.exe      # BsPoly (368 tests)
build/Release/lagpoly_test.exe     # LagPoly (355 tests)

# Run Python tests
pip install -e .
python -m pytest tests/test_bindings.py -v  # 158 tests
```

### Verification Against scipy/numpy

```bash
python scripts/verify_cpp_values.py       # BPoly
python scripts/verify_cpoly_values.py     # CPoly
python scripts/verify_legpoly_values.py   # LegPoly
python scripts/verify_hpoly_values.py     # HPoly
python scripts/verify_bspoly_values.py    # BsPoly
python scripts/verify_lagpoly_values.py   # LagPoly
```

## Project Structure

```
BPoly/
├── include/
│   ├── poly_base.h      # CRTP base class + shared utilities
│   ├── bpoly.h          # Bernstein polynomial
│   ├── cpoly.h          # Chebyshev polynomial
│   ├── legpoly.h        # Legendre polynomial
│   ├── hpoly.h          # Hermite polynomial
│   ├── bspoly.h         # B-Spline (Bernstein storage)
│   ├── lagpoly.h        # Barycentric Lagrange
│   ├── ndarray.h        # Zero-copy array template
│   └── extrapolate_mode.h
├── src/
│   ├── bindings.cpp     # nanobind Python bindings
│   ├── poly_list.h      # PolyList template
│   └── mixed_poly_list.h # MixedPolyList (heterogeneous)
├── bpoly_test.cpp       # BPoly tests
├── cpoly_test.cpp       # CPoly tests
├── legpoly_test.cpp     # LegPoly tests
├── hpoly_test.cpp       # HPoly tests
├── bspoly_test.cpp      # BsPoly tests
├── lagpoly_test.cpp     # LagPoly tests
├── scripts/             # scipy/numpy verification scripts
└── tests/               # Python binding tests
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
