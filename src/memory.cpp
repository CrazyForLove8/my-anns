#include "memory.h"

void
print_memory_usage() {
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