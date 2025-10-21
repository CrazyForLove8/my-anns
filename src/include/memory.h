//
// Created by root on 25-7-10.
//

#ifndef MEMORY_H
#define MEMORY_H

#include <cstring>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>

#include "io.h"
#include "logger.h"

namespace graph {

void
print_memory_usage();

class MemoryIO final : public IO {
    uint8_t* data_{nullptr};
    size_t stride_{0};  // byte of each row
    mutable std::shared_mutex rw_mutex_;
    static constexpr size_t alignment_ = 64;

public:
    explicit MemoryIO(const size_t stride) {
        logger << "Using MemoryIO with stride " << stride << " bytes." << std::endl;
        stride_ = (stride + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }

    DataPtr
    read(uint64_t stride, IdType offset) override;

    void
    write(const DataPtr& src, uint64_t stride, IdType offset) override;

    ~MemoryIO() override;
};

}  // namespace graph

#endif  //MEMORY_H
