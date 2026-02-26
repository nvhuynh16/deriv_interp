#ifndef _NDARRAY_H_
#define _NDARRAY_H_

#include <tuple>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>
#include <initializer_list>
#include <unordered_map>
#include <stdexcept>

namespace ndarray {

// =============================================================================
// array2d<T>: Optimized 2D array for zero-copy NumPy data sharing
// =============================================================================
//
// Key features:
// - Row-major (C-contiguous) storage: element (i,j) at index i*cols + j
// - Uses shared_ptr with aliasing constructor for zero-copy NumPy integration
// - Backward compatible with vector<vector<T>> via conversion constructors
// - Immutable dimensions after construction (fits polynomial classes' design)
//
// Memory layout matches coefficient storage [degree+1][num_intervals]:
//   rows = degree + 1 (coefficient index j)
//   cols = num_intervals (interval index i)
//   coefficients_(j, i) = coefficient for basis B_{j,n} in interval i
// =============================================================================

template <typename T>
struct array2d {
    std::shared_ptr<T[]> data;
    size_t rows;
    size_t cols;

    // Default constructor: empty array
    array2d() : rows(0), cols(0) {}

    // Allocating constructor: creates owned memory
    array2d(size_t r, size_t c) : rows(r), cols(c) {
        if (r > 0 && c > 0) {
            data = std::shared_ptr<T[]>(new T[r * c]());
        }
    }

    // External memory constructor (for numpy zero-copy)
    // Uses shared_ptr aliasing constructor: data points to ptr, but lifetime
    // is managed by owner. When owner's refcount hits zero, the original
    // shared_ptr destructor runs (which can release the Python object).
    //
    // @param ptr Pointer to external data (e.g., numpy array's data buffer)
    // @param r Number of rows
    // @param c Number of columns
    // @param owner Shared pointer that owns the external memory lifetime
    array2d(T* ptr, size_t r, size_t c, std::shared_ptr<void> owner)
        : rows(r), cols(c) {
        // Aliasing constructor: stores ptr but shares ownership with owner
        // The deleter from owner will be called when all references are gone
        data = std::shared_ptr<T[]>(owner, ptr);
    }

    // Construct from vector<vector<T>> (backward compatibility)
    // Copies data into owned contiguous memory
    explicit array2d(const std::vector<std::vector<T>>& vv) {
        if (vv.empty()) {
            rows = 0;
            cols = 0;
            return;
        }
        rows = vv.size();
        cols = vv[0].size();

        if (rows > 0 && cols > 0) {
            data = std::shared_ptr<T[]>(new T[rows * cols]);
            for (size_t i = 0; i < rows; ++i) {
                if (vv[i].size() != cols) {
                    throw std::invalid_argument("All rows must have same size");
                }
                for (size_t j = 0; j < cols; ++j) {
                    data[i * cols + j] = vv[i][j];
                }
            }
        }
    }

    // Move constructor
    array2d(array2d&& other) noexcept
        : data(std::move(other.data)), rows(other.rows), cols(other.cols) {
        other.rows = 0;
        other.cols = 0;
    }

    // Copy constructor
    array2d(const array2d& other)
        : rows(other.rows), cols(other.cols) {
        if (rows > 0 && cols > 0) {
            data = std::shared_ptr<T[]>(new T[rows * cols]);
            std::copy(other.data.get(), other.data.get() + rows * cols, data.get());
        }
    }

    // Move assignment
    array2d& operator=(array2d&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            rows = other.rows;
            cols = other.cols;
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    // Copy assignment
    array2d& operator=(const array2d& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            if (rows > 0 && cols > 0) {
                data = std::shared_ptr<T[]>(new T[rows * cols]);
                std::copy(other.data.get(), other.data.get() + rows * cols, data.get());
            } else {
                data.reset();
            }
        }
        return *this;
    }

    // Convert to vector<vector<T>> (for backward compatibility with existing APIs)
    std::vector<std::vector<T>> to_vec2d() const {
        std::vector<std::vector<T>> result(rows, std::vector<T>(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = data[i * cols + j];
            }
        }
        return result;
    }

    // 2D element access: arr(row, col)
    T& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }

    // Total number of elements
    size_t size() const { return rows * cols; }

    // Check if empty
    bool empty() const { return rows == 0 || cols == 0; }

    // Raw pointer access (for algorithms that need it)
    T* get() { return data.get(); }
    const T* get() const { return data.get(); }
};

}
#endif // _NDARRAY_H_