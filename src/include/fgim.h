//
// Created by XiaoWu on 2024/11/24.
//

#ifndef MYANNS_MERGE_H
#define MYANNS_MERGE_H

#include "graph.h"
#include "dtype.h"
#include "logger.h"
#include "timer.h"

namespace graph {

    class FGIM {
    private:
        unsigned M0_{20};

        unsigned M_{80};

        unsigned L_{20};

        void Sampling(Graph &graph,
                      const Graph &g1,
                      const Graph &g2,
                      IndexOracle &oracle1,
                      IndexOracle &oracle2,
                      IndexOracle &oracle);

        void Refinement(Graph &graph,
                        IndexOracle &oracle);

    public:
        static constexpr unsigned ITER_MAX = 100;

        static constexpr unsigned SAMPLES = 100;

        static constexpr float THRESHOLD = 0.001;

        FGIM() = default;

        explicit FGIM(unsigned M0, unsigned M, unsigned L);

        ~FGIM() = default;

        /**
         * @brief FGIM two PGs
         * @param g1  The first graph
         * @param oracle1 The distance oracle of the first graph
         * @param g2  The second nearest neighbor graph
         * @param oracle2 The distance oracle of the second graph
         * @param oracle The distance oracle
         * @return The merged graph
         */
        Graph merge(const Graph &g1,
                    IndexOracle &oracle1,
                    const Graph &g2,
                    IndexOracle &oracle2,
                    IndexOracle &oracle);

    };

}

#endif //MYANNS_MERGE_H
