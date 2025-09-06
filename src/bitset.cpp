#include "bitset.h"

using namespace graph;

Bitset::Bitset(size_t n) : size_(n) {
    data_.reserve((n + 63) / 64);
    data_.resize((n + 63) / 64, 0ULL);
}

void
Bitset::clear() {
    std::fill(data_.begin(), data_.end(), 0ULL);
}

size_t
Bitset::size() const {
    return size_;
}
