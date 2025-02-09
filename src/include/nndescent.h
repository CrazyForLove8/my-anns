//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_NNDESCENT_H
#define MYANNS_NNDESCENT_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace graph;

namespace nndescent {

class NNDescent {
    private:
        unsigned K_{64};

        float rho_{0.5};

        float delta_{0.001};

        unsigned iteration_{100};

        void initializeGraph(Graph &graph,
                             IndexOracle &oracle);

        void generateUpdate(Graph &graph);

        int applyUpdate(unsigned sample,
                        Graph &graph,
                        IndexOracle &oracle);

        void clearGraph(Graph &graph);

    public:
        NNDescent() = default;

        explicit NNDescent(int K, float rho=0.5, float delta=0.001, int iteration=20)
                : K_(K), rho_(rho), delta_(delta), iteration_(iteration) {}

        ~NNDescent() = default;

        Graph build(IndexOracle &oracle);
    };
}

#endif //MYANNS_NNDESCENT_H
