//
// Created by XiaoWu on 2024/12/14.
//

#ifndef MYANNS_NSG_H
#define MYANNS_NSG_H

#include <omp.h>

#include <random>

#include "index.h"

using namespace graph;

namespace nsg {
    class NSG : public Index {
    private:
        int root{};

        /**
                   * search pool size
                   */
        unsigned L_;

        /**
                   * maximum number of neighbors
                   */
        unsigned m_;

        void
        build_internal() override;

    public:
        NSG(DatasetPtr &dataset,
            unsigned L,
            unsigned m);

        ~NSG() override = default;

        void
        set_L(unsigned L) {
            this->L_ = L;
        }

        void
        set_m(unsigned m) {
            this->m_ = m;
        }

        Neighbors
        prune(std::vector<Neighbor> &candidates);

        void
        tree();
    };

}  // namespace nsg

#endif  // MYANNS_NSG_H
