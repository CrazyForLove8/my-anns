//
// Created by XiaoWu on 2024/12/14.
//

#ifndef MYANNS_NSG_H
#define MYANNS_NSG_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace graph;

namespace nsg {
    class NSG {
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
    public:
        NSG(unsigned m,
            unsigned L) : m_(m), L_(L) {}

        void set_L(unsigned L) {
            this->L_ = L;
        }

        void set_m(unsigned m) {
            this->m_ = m;
        }

        void build(Graph &graph,
                   IndexOracle &oracle);

        std::vector<Neighbor> prune(IndexOracle &oracle,
                                    std::vector<Neighbor> &candidates);

        void tree(Graph &graph,
                  IndexOracle &oracle,
                  int root);
    };


}

#endif //MYANNS_NSG_H
