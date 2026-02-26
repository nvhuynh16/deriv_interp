# Contributing to BPoly

Thank you for your interest in contributing to BPoly! This document provides guidelines for contributing to the project.

## Development Setup

### Requirements

- C++17 compiler (MSVC 2022, GCC 9+, or Clang 10+)
- CMake 3.20+
- Eigen (header-only, fetched automatically via CMake)
- Python 3.8+ with nanobind (for Python bindings)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/nvhuynh16/deriv_interp.git
cd BPoly

# Configure and build
cmake -B build
cmake --build build --config Release

# Run tests
build/Release/bpoly_test.exe       # BPoly (711 tests)
build/Release/cpoly_test.exe       # CPoly (463 tests)
build/Release/legpoly_test.exe     # LegPoly (368 tests)
build/Release/bspoly_test.exe      # BsPoly (368 tests)
build/Release/lagpoly_test.exe     # LagPoly (355 tests)
```

### Python Development

```bash
# Install in development mode
pip install -e .

# Run Python tests
python -m pytest tests/test_bindings.py -v
```

## Code Style

### C++ Guidelines

- **Standard**: C++17
- **Architecture**: Header-only libraries
- **Principle**: KISS (Keep It Simple, Stupid) - prefer clarity over cleverness
- **Thread Safety**: All classes must be immutable after construction; all methods const
- **No Static State**: Avoid mutable static variables

### Formatting

- Use consistent indentation (4 spaces recommended)
- Keep functions reasonably sized
- Add comments for non-obvious algorithms, with citations where applicable

### Documentation

Each header file should include a REFERENCES section with citations to:
- Primary reference (scipy/numpy documentation and source)
- Algorithm references with Wikipedia/paper links
- Inline citations at key algorithm implementations

## Testing Requirements

### Numerical Verification

All polynomial implementations must be verified against scipy/numpy reference implementations:

- **Tolerance**: 1e-10 relative error
- **Coverage**: Evaluation, derivatives, integration, root finding
- **Edge Cases**: Boundary conditions, extrapolation modes, single-interval cases

### Running Verification Scripts

```bash
python scripts/verify_cpp_values.py       # BPoly
python scripts/verify_cpoly_values.py     # CPoly
python scripts/verify_legpoly_values.py   # LegPoly
python scripts/verify_bspoly_values.py    # BsPoly
python scripts/verify_lagpoly_values.py   # LagPoly
```

### Adding New Tests

- Add C++ tests to the appropriate `*_test.cpp` file
- Follow the existing test pattern using the `TEST()` macro
- Verify against scipy/numpy when adding new functionality

## Pull Request Process

1. **Fork** the repository and create a feature branch from `master`
2. **Write tests** for any new functionality
3. **Verify** all existing tests pass
4. **Update documentation** if adding new features
5. **Submit** a pull request with a clear description of changes

### PR Checklist

- [ ] All C++ tests pass (2,265 total tests)
- [ ] All Python tests pass (62 tests)
- [ ] New features include tests
- [ ] Code follows existing style conventions
- [ ] Documentation updated if applicable
- [ ] Commit messages are clear and descriptive

## Reporting Issues

When reporting bugs, please include:

- Operating system and compiler version
- Steps to reproduce the issue
- Expected vs actual behavior
- Minimal code example if possible

## Feature Requests

Feature requests are welcome! Please:

- Check existing issues first to avoid duplicates
- Describe the use case and expected behavior
- Consider if the feature fits the project scope (scipy/numpy compatible piecewise polynomials)

## License

By contributing to BPoly, you agree that your contributions will be licensed under the MIT License.
