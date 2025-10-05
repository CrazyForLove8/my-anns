#include "memory.h"

void
graph::print_memory_usage() {
    // TODO Support for Windows, macOS
    std::ifstream file("/proc/self/status");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open /proc/self/status" << std::endl;
        return;
    }

    std::string line;
    long vm_rss = 0;
    long vm_size = 0;

    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            std::string value_str;
            iss >> key >> value_str;
            vm_rss = std::stol(value_str);
        } else if (line.rfind("VmSize:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            std::string value_str;
            iss >> key >> value_str;
            vm_size = std::stol(value_str);
        }
    }
    file.close();

    std::cout << "--- Linux Process Memory Usage ---" << std::endl;
    std::cout << "Resident Set Size (RSS): " << vm_rss << " KB ("
              << (double)vm_rss / 1024.0 / 1024.0 << " GB)" << std::endl;
    std::cout << "Virtual Memory Size (VmSize): " << vm_size << " KB ("
              << (double)vm_size / 1024.0 / 1024.0 << " GB)" << std::endl;
    std::cout << "----------------------------------" << std::endl;
}

graph::DataPtr
graph::MemoryIO::read(const uint64_t stride, const uint64_t offset) {
    // std::shared_lock lock(rw_mutex_);
    if (data_ == nullptr || stride + offset > size_) {
        throw std::runtime_error("Read out of bounds in MemoryIO");
    }
    return data_ + offset;
}

void
graph::MemoryIO::write(const DataPtr& src, const uint64_t stride, const uint64_t offset) {
    const uint64_t new_size = stride + offset;

    {
        std::unique_lock lock(rw_mutex_);
        if (data_ == nullptr || new_size > size_) {
            uint8_t* new_data = nullptr;
            if (posix_memalign(reinterpret_cast<void**>(&new_data), alignment_, new_size) != 0) {
                throw std::bad_alloc();
            }

            if (data_ != nullptr) {
                std::memcpy(new_data, data_, size_);
                free(data_);
            }

            data_ = new_data;
            size_ = new_size;
        }
    }
    {
        std::unique_lock lock(rw_mutex_);
        std::memcpy(data_ + offset, src, stride);
    }
}

graph::MemoryIO::~MemoryIO() {
    std::unique_lock lock(rw_mutex_);
    if (data_ != nullptr) {
        free(data_);
        data_ = nullptr;
    }
}
