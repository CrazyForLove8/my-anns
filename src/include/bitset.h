//
// Created by XiaoWu on 2025/9/6.
//

#ifndef MYANNS_BITSET_H
#define MYANNS_BITSET_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace graph {

class Bitset {
public:
    explicit Bitset(size_t n);

    /**
     * Set the bit at position pos to 1
     * @param pos position
     */
    inline void
    set(size_t pos) {
        data_[pos >> 6] |= (1ULL << (pos & 63));
    }

    /**
     * Set the bit at position pos to 0
     * @param pos position
     */
    inline void
    reset(size_t pos) {
        data_[pos >> 6] &= ~(1ULL << (pos & 63));
    }

    /**
     * Test if the bit at position pos is 1
     * @param pos position
     * @return true if the bit is 1, false otherwise
     */
    [[nodiscard]] inline bool
    test(size_t pos) const {
        return (data_[pos >> 6] >> (pos & 63)) & 1ULL;
    }

    /**
     * Atomically set the bit at position pos to 1 and return the old value
     * @param pos position
     * @return true if the old value is 1, false otherwise
     */
    inline bool
    test_and_set(size_t pos) {
        size_t idx = pos >> 6;
        uint64_t mask = 1ULL << (pos & 63);
        uint64_t old = __atomic_fetch_or(&data_[idx], mask, __ATOMIC_SEQ_CST);
        return old & mask;
    }

    void
    clear();

    [[nodiscard]] size_t
    size() const;

private:
    size_t size_;
    std::vector<uint64_t> data_;
};

}  // namespace graph

#endif  //MYANNS_BITSET_H
