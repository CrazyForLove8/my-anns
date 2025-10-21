//
// Created by XiaoWu on 2025/9/28.
//

#ifndef MYANNS_IO_H
#define MYANNS_IO_H

#include <memory>

#include "typing.h"

namespace graph {

enum class IOType { FILE_IO = 0, MEMORY_IO = 1, MMAP_IO = 2 };

class IO {
protected:
    uint64_t size_{0};

public:
    /**
     *
     * @param stride Number of bytes to be read
     * @param offset Read at the given inner id
     * @return
     */
    virtual DataPtr
    read(uint64_t stride, IdType offset) = 0;

    virtual void
    write(const DataPtr& data, uint64_t stride, IdType offset) = 0;

    virtual ~IO();
};

using IOPtr = std::shared_ptr<IO>;

}  // namespace graph

#endif  //MYANNS_IO_H
