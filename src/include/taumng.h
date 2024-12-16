//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_TAUMNG_H
#define MYANNS_TAUMNG_H

#include <random>
#include <omp.h>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace graph;

namespace taumng {

class TauMNG {
    private:
        /**
         * tau
         */
        float t_;

        /**
         * same as k in knn HNSW_search
         */
        int h_;

        /**
         * HNSW_search pool size
         */
        int b_;

    public:
        TauMNG(float t,
               int h,
               int b);

        void set_b(int b);

        void set_h(int h);

        void build(Graph &graph,
                   IndexOracle &oracle);

    };
}

#endif //MYANNS_TAUMNG_H
