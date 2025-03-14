//
// Created by XiaoWu on 2024/11/23.
//

/**
 * This implementation is based on the following references:
 * See https://github.com/facebookresearch/faiss and
 * https://github.com/JieFengWang/mini_rnn for more details.
 */

#ifndef MYANNS_DTYPE_H
#define MYANNS_DTYPE_H

#include <malloc.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "logger.h"
#include "metric.h"

#ifdef __GNUC__
#ifdef __AVX__
#define ALIGNMENT 32
#else
#ifdef __SSE2__
#define ALIGNMENT 16
#else
#define ALIGNMENT 4
#endif
#endif
#endif

#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

namespace graph {
template <typename T>
class Matrix {
    unsigned col{};
    unsigned row{};
    size_t stride{};
    char* data;

    void
    reset(unsigned r, unsigned c) {
        row = r;
        col = c;
        stride = (sizeof(T) * c + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        if (data)
            free(data);
        data = (char*)memalign(ALIGNMENT, row * stride);
    }

public:
    Matrix() : col(0), row(0), stride(0), data(nullptr) {
    }

    Matrix(unsigned r, unsigned c) {
        data = nullptr;
        reset(r, c);
    }

    Matrix(const Matrix& m) {
        col = m.col, row = m.row, stride = m.stride, data = nullptr;
        data = (char*)memalign(ALIGNMENT, row * stride);
        memcpy(data, m.data, row * stride);
    }

    explicit Matrix(std::vector<std::shared_ptr<Matrix>>& matrices) {
        auto& base = matrices[0];
        col = base->col;
        stride = base->stride;
        row = 0;
        for (const auto& matrix : matrices) {
            row += matrix->row;
        }
        data = (char*)memalign(32, row * stride);
        if (data == nullptr) {
            throw std::runtime_error("Cannot allocate memory for matrix.");
        }
        size_t offset = 0;
        for (const auto& matrix : matrices) {
            memcpy(data + offset, matrix->data, matrix->row * stride);
            offset += matrix->row * stride;
        }
    }

    ~Matrix() {
        if (data)
            free(data);
    }

    [[nodiscard]] bool
    empty() const {
        return data == nullptr;
    }

    [[nodiscard]] unsigned
    size() const {
        return row;
    }

    [[nodiscard]] unsigned
    dim() const {
        return col;
    }

    [[nodiscard]] size_t
    step() const {
        return stride;
    }

    void
    resize(unsigned r, unsigned c) {
        reset(r, c);
    }

    T*
    operator[](unsigned i) {
        return reinterpret_cast<T*>(&data[stride * i]);
    }

    T const*
    operator[](unsigned i) const {
        return reinterpret_cast<T const*>(&data[stride * i]);
    }

    T&
    operator()(unsigned i, unsigned j) {
        return reinterpret_cast<T*>(&data[stride * i])[j];
    }

    Matrix&
    operator=(const Matrix& m) {
        if (this == &m) {
            return *this;
        }
        if (row * col != m.row * m.col) {
            delete[] data;
            data = (char*)memalign(ALIGNMENT, m.row * m.stride);
        }
        memcpy(data, m.data, m.row * m.stride);
        row = m.row;
        col = m.col;
        stride = m.stride;
        return *this;
    }

    void
    zero() {
        memset(data, 0, row * stride);
    }

    void
    load(const std::string& path, unsigned int skip = 0, unsigned int gap = 4) {
        logger << "Loading data from " << path << std::endl;
        std::ifstream is(path.c_str(), std::ios::binary);
        if (!is) {
            throw std::runtime_error("Cannot open file " + path);
        }
        is.seekg(0, std::ios::end);
        size_t size = is.tellg();
        size -= skip;
        is.seekg(0, std::ios::beg);
        unsigned dim;
        is.read((char*)&dim, sizeof(unsigned int));
        logger << "Vector dimension: " << dim << std::endl;
        unsigned line = sizeof(T) * dim + gap;
        unsigned N = size / line;
        reset(N, dim);
        zero();
        is.seekg(skip, std::ios::beg);
        for (unsigned i = 0; i < N; ++i) {
            is.seekg(gap, std::ios::cur);
            is.read(&data[stride * i], sizeof(T) * dim);
        }
    }

    /**
         * Append a matrix to the current matrix.
         * @param matrix
         */
    void
    append(const Matrix& matrix) {
        size_t new_rows = row + matrix.row;
        size_t new_columns = col;
        size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
        char* new_data = (char*)memalign(32, new_rows * new_stride);
        memcpy(new_data, data, row * stride);
        memcpy(new_data + row * stride, matrix.data, matrix.row * matrix.step());
        free(data);
        data = new_data;
        row = new_rows;
        col = new_columns;
        stride = new_stride;
    }

    /**
         * Append a list of matrices to the current matrix.
         * @param matrices std::vector<Matrix>
         */
    void
    append(const std::vector<Matrix>& matrices) {
        size_t new_rows = row;
        for (const auto& matrix : matrices) {
            new_rows += matrix.row;
        }
        char* new_data = (char*)memalign(32, new_rows * stride);
        memcpy(new_data, data, row * stride);
        size_t offset = row * stride;
        for (const auto& matrix : matrices) {
            memcpy(new_data + offset, matrix.data, matrix.row * matrix.step());
            offset += matrix.row * matrix.step();
        }
        free(data);
        data = new_data;
        row = new_rows;
    }

    /**
         * Append a list of matrices to the current matrix.
         * @param matrices std::vector<std::shared_ptr<Matrix>>
         */
    void
    append(const std::vector<std::shared_ptr<Matrix>>& matrices) {
        size_t new_rows = row;
        for (const auto& matrix : matrices) {
            new_rows += matrix->row;
        }
        char* new_data = (char*)memalign(32, new_rows * stride);
        memcpy(new_data, data, row * stride);
        size_t offset = row * stride;
        for (const auto& matrix : matrices) {
            memcpy(new_data + offset, matrix->data, matrix->row * matrix->step());
            offset += matrix->row * matrix->step();
        }
        free(data);
        data = new_data;
        row = new_rows;
    }

    /**
         * Split the matrix into num parts. Note that the original matrix will be resized but not freed.
         * @param num The number of parts to split.
         * @return
         */
    std::vector<Matrix>
    split(size_t num) {
        size_t new_rows = row / num;
        size_t remaining = row % num;
        size_t new_columns = col;
        size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
        std::vector<Matrix> matrices;
        for (size_t i = 1; i < num - 1; ++i) {
            Matrix part(new_rows, new_columns);
            memcpy(part.data, data + i * new_rows * new_stride, new_rows * new_stride);
            matrices.push_back(std::move(part));
        }

        Matrix last_part(new_rows + remaining, new_columns);
        memcpy(last_part.data,
               data + (num - 1) * new_rows * new_stride,
               (new_rows + remaining) * new_stride);
        matrices.push_back(std::move(last_part));

        row = new_rows;
        return matrices;
    }

    void
    halve(Matrix& other) {
        size_t total = row;
        size_t new_rows = row / 2;
        row = new_rows;

        char* tmp = (char*)memalign(32, new_rows * stride);
        memcpy(tmp, data, new_rows * stride);
        char* new_data = (char*)memalign(32, (total - new_rows) * stride);
        memcpy(new_data, data + new_rows * stride, (total - new_rows) * stride);
        free(data);
        data = tmp;
        other.data = new_data;
        other.row = total - new_rows;
        other.col = col;
        other.stride = stride;
    }
};

template <typename T>
using MatrixPtr = std::shared_ptr<Matrix<T>>;

template <typename T>
void
mergeMatrix(const Matrix<T>& m1, const Matrix<T>& m2, Matrix<T>& merged) {
    size_t r1 = m1.size();
    size_t r2 = m2.size();
    size_t c = m1.dim();
    if (c != m2.dim()) {
        throw std::runtime_error("Dimension mismatch");
    }

    if (&m1 == &merged) {
        Matrix temp(m1);
        merged.resize(r1 + r2, c);
        for (size_t i = 0; i < r1; ++i) {
            std::copy(temp[i], temp[i] + c, merged[i]);
        }
    } else {
        merged.resize(r1 + r2, c);
        for (size_t i = 0; i < r1; ++i) {
            std::copy(m1[i], m1[i] + c, merged[i]);
        }
    }
    for (size_t i = 0; i < r2; ++i) {
        std::copy(m2[i], m2[i] + c, merged[i + r1]);
    }
}

template <typename T>
class MatrixProxy {
    unsigned rows{0};
    unsigned cols{0};
    size_t stride{0};
    uint8_t const* data{nullptr};

public:
    explicit MatrixProxy(Matrix<T> const& m) {
        reset(m);
    }

    void
    reset(Matrix<T> const& m) {
        rows = m.size();
        cols = m.dim();
        stride = m.step();
        data = reinterpret_cast<uint8_t const*>(m[0]);
    }

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
    /// Construct from FLANN matrix.
    MatrixProxy(flann::Matrix<float> const& m_)
        : rows(m_.rows), cols(m_.cols), stride(m_.stride), data(m_.data) {
        if (stride % ALIGNMENT)
            throw invalid_argument("bad alignment");
    }
#endif
#ifdef CV_MAJOR_VERSION
    /// Construct from OpenCV matrix.
    MatrixProxy(cv::Mat const& m_) : rows(m_.rows), cols(m_.cols), stride(m_.step), data(m_.data) {
        if (stride % ALIGNMENT)
            throw invalid_argument("bad alignment");
    }
#endif
#ifdef NPY_NDARRAYOBJECT_H
    /// Construct from NumPy matrix.
    MatrixProxy(PyArrayObject* obj) {
        if (!obj || (obj->nd != 2))
            throw invalid_argument("bad array shape");
        rows = obj->dimensions[0];
        cols = obj->dimensions[1];
        stride = obj->strides[0];
        data = reinterpret_cast<uint8_t const*>(obj->data);
        if (obj->descr->elsize != sizeof(float))
            throw invalid_argument("bad data type size");
        if (stride % ALIGNMENT)
            throw invalid_argument("bad alignment");
        if (!(stride >= cols * sizeof(float)))
            throw invalid_argument("bad stride");
    }
#endif
#endif

    [[nodiscard]] unsigned
    size() const {
        return rows;
    }

    [[nodiscard]] unsigned
    dim() const {
        return cols;
    }

    T const*
    operator[](unsigned i) const {
#ifdef USE_SSE
        _mm_prefetch(data + stride * i, _MM_HINT_T0);
#endif
        return reinterpret_cast<T const*>(data + stride * i);
    }

    T*
    operator[](unsigned i) {
#ifdef USE_SSE
        _mm_prefetch(data + stride * i, _MM_HINT_T0);
#endif
        return const_cast<float*>(reinterpret_cast<T const*>(data + stride * i));
    }
};

template <typename T>
class IndexOracle {
public:
    [[nodiscard]] virtual unsigned
    size() const = 0;

    [[nodiscard]] virtual unsigned
    dim() const = 0;

    virtual void
    reset(const Matrix<T>& m) = 0;

    virtual T
    operator()(unsigned i, unsigned j) const = 0;

    virtual T
    operator()(unsigned, const T*) const = 0;

    virtual T*
    operator[](unsigned i) const = 0;
};

using OraclePtr = std::shared_ptr<IndexOracle<float>>;

template <typename T, typename DIST_TYPE>
class MatrixOracle : public IndexOracle<T> {
public:
    MatrixProxy<T> proxy;

    explicit MatrixOracle(const Matrix<T>& m) : proxy(m) {
    }

    void
    reset(const Matrix<T>& m) {
        proxy.reset(m);
    }

    [[nodiscard]] unsigned
    size() const override {
        return proxy.size();
    }

    [[nodiscard]] unsigned
    dim() const override {
        return proxy.dim();
    }

    T
    operator()(unsigned i, unsigned j) const override {
        return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
    }

    T
    operator()(unsigned i, const T* vec) const override {
        return DIST_TYPE::apply((T*)proxy[i], vec, proxy.dim());
    }

    T*
    operator[](unsigned i) const override {
        return const_cast<T*>(proxy[i]);
    }

    static std::shared_ptr<IndexOracle<T>>
    getInstance(const Matrix<T>& m) {
        return std::make_shared<MatrixOracle<T, DIST_TYPE>>(m);
    }
};
}  // namespace graph

#endif  // MYANNS_DTYPE_H
