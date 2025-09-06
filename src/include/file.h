//
// Created by XiaoWu on 2025/9/5.
//

#ifndef MYANNS_FILE_H
#define MYANNS_FILE_H

#include <csignal>

#include "fstream"
#include "list"
#include "memory"
#include "stdexcept"
#include "string"
#include "typing.h"
#include "unordered_map"
#include "vector"

namespace graph {

struct FileReaderParams {
    size_t begin_offset{0};  // for bin file, the first 8 bytes are size and dim
    size_t each_offset{0};   // for vecs file, the first 4 bytes is dimension
    size_t capacity{128};    // cache capacity
    std::string data_path;   // for disk-based matrix
};

template <typename T>
class FileReader {
    std::ifstream fin_;
    size_t stride_;
    size_t begin_offset_;  // for bin file, the first 8 bytes are size and dim
    size_t each_offset_;   // for vecs file, the first 4 bytes is dimension
    int dim_;

    size_t capacity_;

    std::list<IdType> lru_;
    struct CacheEntry {
        std::shared_ptr<T> data;
        typename std::list<IdType>::iterator lru_it;
    };
    std::unordered_map<IdType, CacheEntry> cache_;

public:
    explicit FileReader(const std::string& filename,
                        int dim,
                        size_t begin_offset = 0,
                        size_t each_offset = 0,
                        size_t capacity = 128)
        : stride_(sizeof(T) * dim),
          begin_offset_(begin_offset),
          each_offset_(each_offset),
          dim_(dim),
          capacity_(capacity) {
        fin_.open(filename, std::ios::binary);
        fin_.seekg(0, std::ios::beg);
        cache_.reserve(capacity_);
        if (!fin_) {
            throw std::runtime_error("Cannot open file " + filename);
        }
    }

    FileReader(const FileReaderParams& params, int dim)
        : stride_(sizeof(T) * dim),
          begin_offset_(params.begin_offset),
          each_offset_(params.each_offset),
          dim_(dim),
          capacity_(params.capacity) {
        fin_.open(params.data_path, std::ios::binary);
        fin_.seekg(0, std::ios::beg);
        cache_.reserve(capacity_);
        if (!fin_) {
            throw std::runtime_error("Cannot open file " + params.data_path);
        }
    }

    ~FileReader() {
        if (fin_.is_open()) {
            fin_.close();
        }
    }

    std::shared_ptr<T>
    read(IdType idx) {
        auto it = cache_.find(idx);
        if (it != cache_.end()) {
            lru_.erase(it->second.lru_it);
            lru_.push_front(idx);
            it->second.lru_it = lru_.begin();
            return it->second.data;
        }
        auto buf = std::shared_ptr<T>(new T[dim_], std::default_delete<T[]>());
        fin_.seekg(begin_offset_ + idx * (stride_ + each_offset_) + each_offset_, std::ios::beg);
        fin_.read(reinterpret_cast<char*>(buf.get()), stride_);
        if (!fin_) {
            throw std::runtime_error("Failed to read row " + std::to_string(idx));
        }

        if (cache_.size() >= capacity_) {
            IdType old_idx = lru_.back();
            lru_.pop_back();
            cache_.erase(old_idx);
        }

        lru_.push_front(idx);
        cache_[idx] = {std::move(buf), lru_.begin()};

        return cache_[idx].data;
    }
};

}  // namespace graph

#endif  //MYANNS_FILE_H
