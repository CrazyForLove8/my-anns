//
// Created by XiaoWu on 2024/12/14.
//

#ifndef MYANNS_NSG_H
#define MYANNS_NSG_H

#include <omp.h>

#include <random>

#include "index.h"
#include "nndescent.h"

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

    /**
         * k used in nn-descent
         */
    unsigned K_;

    void
    build_internal() override;

public:
    /**
        * @brief Build an NSG. Note that the graph is destroyed after the build since std::move is called.
        * @param oracle
        * @param graph
        * @param L search pool size
        * @param m maximum number of neighbors
        */
    NSG(DatasetPtr& dataset, unsigned K, unsigned L, unsigned m);

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
    prune(std::vector<Neighbor>& candidates);

    void
    tree();

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;
};

}  // namespace nsg

#endif  // MYANNS_NSG_H
