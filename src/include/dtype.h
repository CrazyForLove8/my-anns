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
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "file.h"
#include "logger.h"

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

using IdType = uint32_t;

using Value = std::variant<uint64_t, double_t, std::string>;
using ParamMap = std::unordered_map<std::string, Value>;

template <typename T>
class Matrix {
    unsigned col_{};
    unsigned row_{};
    size_t stride_{};
    size_t offset_bytes_{};

    std::shared_ptr<char> data;

    bool use_disk_{false};
    FileReaderParams fin_params_;

    void
    reset(const unsigned r, const unsigned c) {
        row_ = r;
        col_ = c;
        stride_ = (sizeof(T) * c + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        if (data) {
            data.reset();
            data = nullptr;
        }
        if (use_disk_) {
            return;
        }
        auto deleter = [](char* p) { free(p); };
        std::unique_ptr<char, decltype(deleter)> uptr(
            static_cast<char*>(memalign(ALIGNMENT, row_ * stride_)), deleter);
        if (!uptr) {
            throw std::bad_alloc();
        }
        data = std::shared_ptr<char>(std::move(uptr));
    }

    void
    load_vecs_data(const std::string& path) {
        std::ifstream is(path.c_str(), std::ios::binary);
        if (!is) {
            throw std::runtime_error("Cannot open file " + path);
        }
        is.seekg(0, std::ios::end);
        size_t size = is.tellg();
        is.seekg(0, std::ios::beg);
        unsigned dim;
        is.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
        is.seekg(0, std::ios::beg);
        unsigned line = sizeof(T) * dim + 4;
        unsigned N = size / line;
        logger << "Vector size: " << N << std::endl;
        logger << "Vector dimension: " << dim << std::endl;
        reset(N, dim);
        if (!use_disk_) {
            zero();
            for (unsigned i = 0; i < N; ++i) {
                is.seekg(sizeof(unsigned int), std::ios::cur);
                is.read(data.get() + stride_ * i, sizeof(T) * dim);
            }
            is.close();
        } else {
            fin_params_.data_path = path;
            fin_params_.each_offset = sizeof(unsigned int);
        }
    }

    void
    load_bin_data(const std::string& path) {
        std::ifstream is(path.c_str(), std::ios::binary);
        if (!is) {
            throw std::runtime_error("Cannot open file " + path);
        }

        unsigned size, dim;
        is.read(reinterpret_cast<char*>(&size), sizeof(unsigned int));
        is.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
        logger << "Vector size: " << size << std::endl;
        logger << "Vector dimension: " << dim << std::endl;
        reset(size, dim);
        if (!use_disk_) {
            zero();
            for (unsigned i = 0; i < size; ++i) {
                is.read(data.get() + stride_ * i, sizeof(T) * dim);
            }
        } else {
            fin_params_.data_path = path;
            fin_params_.begin_offset = 2 * sizeof(unsigned int);
        }
        is.close();
    }

    void
    load_hdf5_data(const std::string& path) {
        std::ifstream is(path.c_str(), std::ios::binary);
        if (!is) {
            throw std::runtime_error("Cannot open file " + path);
        }

        if (is.is_open()) {
            logger << "Loading data from HDF5 file is not implemented." << std::endl;
        }
        is.close();
    }

public:
    Matrix() : data(nullptr) {
    }

    Matrix(const unsigned r, const unsigned c) {
        data = nullptr;
        reset(r, c);
    }

    Matrix(const Matrix& m) {
        col_ = m.col_, row_ = m.row_, stride_ = m.stride_, data = m.data;
    }

    Matrix(const Matrix& original_matrix, unsigned start_row, unsigned num_rows)
        : use_disk_(original_matrix.use_disk_),
          col_(original_matrix.col_),
          row_(num_rows),
          stride_(original_matrix.stride_),
          data(original_matrix.data) {
        offset_bytes_ = original_matrix.offset_bytes_ + start_row * original_matrix.stride_;
        fin_params_ = original_matrix.fin_params_;
        fin_params_.begin_offset += offset_bytes_;
        if (start_row + num_rows > original_matrix.row_) {
            throw std::out_of_range("Sub-matrix dimensions out of bounds of original matrix.");
        }
    }

    explicit Matrix(std::vector<std::shared_ptr<Matrix> >& matrices) {
        auto& base = matrices[0];
        col_ = base->col_;
        stride_ = base->stride_;
        row_ = 0;
        for (const auto& matrix : matrices) {
            row_ += matrix->row_;
        }
        data = std::shared_ptr<char>(static_cast<char*>(memalign(ALIGNMENT, row_ * stride_)),
                                     [](char* p) { free(p); });
        if (data == nullptr) {
            throw std::runtime_error("Cannot allocate memory for matrix.");
        }
        size_t offset = 0;
        for (const auto& matrix : matrices) {
            memcpy(data.get() + offset, matrix->data.get(), matrix->row_ * stride_);
            offset += matrix->row_ * stride_;
        }
    }

    ~Matrix() {
        if (data)
            data.reset();
    }

    [[nodiscard]] bool
    empty() const {
        return row_ == 0 || col_ == 0;
    }

    [[nodiscard]] unsigned
    size() const {
        return row_;
    }

    [[nodiscard]] unsigned
    dim() const {
        return col_;
    }

    [[nodiscard]] size_t
    step() const {
        return stride_;
    }

    [[nodiscard]] size_t
    offset() const {
        return offset_bytes_;
    }

    [[nodiscard]] bool
    is_use_disk() const {
        return use_disk_;
    }

    void
    resize(unsigned r, unsigned c) {
        reset(r, c);
    }

    bool
    belong(const Matrix& m) const {
        return data.get() == m.data.get();
    }

    std::shared_ptr<T>
    operator[](unsigned i) {
        if (use_disk_) {
            thread_local std::unique_ptr<FileReader<T> > thread_fin;
            if (!thread_fin) {
                thread_fin = std::make_unique<FileReader<T> >(fin_params_, col_);
            }
            return thread_fin->read(i);
        }
        return std::shared_ptr<T>(data,
                                  reinterpret_cast<T*>(data.get() + stride_ * i + offset_bytes_));
    }

    T
    operator()(unsigned i, unsigned j) const {
        if (use_disk_) {
            thread_local std::unique_ptr<FileReader<T> > thread_fin;
            if (!thread_fin) {
                thread_fin = std::make_unique<FileReader<T> >(fin_params_, col_);
            }
            return thread_fin->read(i).get()[j];
        }
        return reinterpret_cast<T*>(data.get() + stride_ * i + offset_bytes_)[j];
    }

    Matrix&
    operator=(const Matrix& m) {
        if (this == &m) {
            return *this;
        }
        if (row_ * col_ != m.row_ * m.col_) {
            if (data) {
                data.reset();
            }
            data =
                std::shared_ptr<char>(static_cast<char*>(memalign(ALIGNMENT, m.row_ * m.stride_)),
                                      [](char* p) { free(p); });
        }
        memcpy(data.get(), m.data.get(), m.row_ * m.stride_);
        row_ = m.row_;
        col_ = m.col_;
        stride_ = m.stride_;
        return *this;
    }

    void
    zero() const {
        memset(data.get(), 0, row_ * stride_);
    }

    void
    load(const std::string& path, bool use_disk = false) {
        use_disk_ = use_disk;
        logger << "Loading data from " << path << std::endl;

        if (path.find(".fbin") != std::string::npos || path.find(".ibin") != std::string::npos) {
            load_bin_data(path);
        } else if (path.find(".fvecs") != std::string::npos ||
                   path.find(".ivecs") != std::string::npos) {
            load_vecs_data(path);
        } else if (path.find(".hdf5") != std::string::npos) {
            load_hdf5_data(path);
        } else {
            throw std::runtime_error("Unsupported file format: " + path);
        }
    }

    /**
     * Append a matrix to the current matrix.
     * @param matrix
     */
    void
    append(const Matrix& matrix) {
        size_t new_rows = row_ + matrix.row_;
        size_t new_columns = col_;
        size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
        auto new_data = std::shared_ptr<char>(
            static_cast<char*>(memalign(32, new_rows * new_stride)), [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get(), row_ * stride_);
        memcpy(new_data.get() + row_ * stride_, matrix.data.get(), matrix.row_ * matrix.step());
        data.reset();
        data = new_data;
        row_ = new_rows;
        col_ = new_columns;
        stride_ = new_stride;
    }

    /**
     * Append a list of matrices to the current matrix.
     * @param matrices std::vector<Matrix>
     */
    void
    append(const std::vector<Matrix>& matrices) {
        size_t new_rows = row_;
        for (const auto& matrix : matrices) {
            new_rows += matrix.row_;
        }
        auto new_data = std::make_shared<char>(static_cast<char*>(memalign(32, new_rows * stride_)),
                                               [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get(), row_ * stride_);
        size_t offset = row_ * stride_;
        for (const auto& matrix : matrices) {
            memcpy(new_data.get() + offset, matrix.data.get(), matrix.row_ * matrix.stride_);
            offset += matrix.row_ * matrix.stride_;
        }
        data.reset();
        data = new_data;
        row_ = new_rows;
    }

    /**
     * Append a list of matrices to the current matrix.
     * @param matrices std::vector<std::shared_ptr<Matrix>>
     */
    void
    append(const std::vector<std::shared_ptr<Matrix> >& matrices) {
        size_t new_rows = row_;
        for (const auto& matrix : matrices) {
            new_rows += matrix->row_;
        }
        auto new_data = std::shared_ptr<char>(static_cast<char*>(memalign(32, new_rows * stride_)),
                                              [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get(), row_ * stride_);
        size_t offset = row_ * stride_;
        for (const auto& matrix : matrices) {
            memcpy(new_data.get() + offset, matrix->data.get(), matrix->row_ * matrix->stride_);
            offset += matrix->row_ * matrix->step();
        }
        data.reset();
        data = new_data;
        row_ = new_rows;
    }

    /**
     * Split the matrix into num parts. Note that the original matrix will be resized but not freed.
     * @param num The number of parts to split.
     * @return
     */
    std::vector<Matrix>
    split(const size_t num) {
        size_t new_rows = row_ / num;
        size_t remaining = row_ % num;
        size_t new_columns = col_;
        size_t new_stride = (sizeof(T) * new_columns + 31) / 32 * 32;
        std::vector<Matrix> matrices;
        for (size_t i = 1; i < num - 1; ++i) {
            Matrix part(new_rows, new_columns);
            memcpy(part.data.get(), data.get() + i * new_rows * new_stride, new_rows * new_stride);
            matrices.push_back(std::move(part));
        }

        Matrix last_part(new_rows + remaining, new_columns);
        memcpy(last_part.data.get(),
               data.get() + (num - 1) * new_rows * new_stride,
               (new_rows + remaining) * new_stride);
        matrices.push_back(std::move(last_part));

        row_ = new_rows;
        return matrices;
    }

    void
    halve(Matrix& other) {
        size_t total = row_;
        size_t new_rows = row_ / 2;
        row_ = new_rows;
        auto tmp = std::shared_ptr<char>(static_cast<char*>(memalign(32, new_rows * stride_)),
                                         [](char* p) { free(p); });
        if (!tmp) {
            throw std::bad_alloc();
        }
        memcpy(tmp.get(), data.get(), new_rows * stride_);
        auto new_data =
            std::shared_ptr<char>(static_cast<char*>(memalign(32, (total - new_rows) * stride_)),
                                  [](char* p) { free(p); });
        if (!new_data) {
            throw std::bad_alloc();
        }
        memcpy(new_data.get(), data.get() + new_rows * stride_, (total - new_rows) * stride_);
        data.reset();
        data = tmp;
        other.data = new_data;
        other.row_ = total - new_rows;
        other.col_ = col_;
        other.stride_ = stride_;
    }
};

template <typename T>
using MatrixPtr = std::shared_ptr<Matrix<T> >;

template <typename T>
class IndexOracle {
public:
    [[nodiscard]] virtual unsigned
    size() const = 0;

    [[nodiscard]] virtual unsigned
    dim() const = 0;

    virtual void
    reset(const MatrixPtr<T>& ptr) = 0;

    virtual T
    operator()(unsigned i, unsigned j) const = 0;

    virtual T
    operator()(unsigned, const T*) const = 0;

    virtual T
    operator()(const T*, const T*) const = 0;

    virtual std::shared_ptr<T>
    operator[](unsigned i) const = 0;

    virtual ~IndexOracle() = default;
};

using OraclePtr = std::shared_ptr<IndexOracle<float> >;

template <typename T, typename DIST_TYPE>
class MatrixOracle : public IndexOracle<T> {
public:
    MatrixPtr<T> matrix_;

    explicit MatrixOracle(const MatrixPtr<T> ptr) : matrix_(ptr) {
        if (!matrix_) {
            throw std::runtime_error("Matrix pointer is null");
        }
    }

    void
    reset(const MatrixPtr<T>& ptr) override {
        matrix_ = ptr;
    }

    [[nodiscard]] unsigned
    size() const override {
        return matrix_->size();
    }

    [[nodiscard]] unsigned
    dim() const override {
        return matrix_->dim();
    }

    T
    operator()(unsigned i, unsigned j) const override {
        return DIST_TYPE::apply((*matrix_)[i].get(), (*matrix_)[j].get(), matrix_->dim());
    }

    T
    operator()(unsigned i, const T* vec) const override {
        return DIST_TYPE::apply(vec, (*matrix_)[i].get(), matrix_->dim());
    }

    T
    operator()(const T* vec1, const T* vec2) const override {
        return DIST_TYPE::apply(vec1, vec2, matrix_->dim());
    }

    std::shared_ptr<T>
    operator[](unsigned i) const override {
        return (*matrix_)[i];
    }

    static std::shared_ptr<IndexOracle<T> >
    getInstance(const MatrixPtr<T> ptr) {
        return std::make_shared<MatrixOracle<T, DIST_TYPE> >(ptr);
    }
};
}  // namespace graph

#endif  // MYANNS_DTYPE_H
