//
// Created by XiaoWu on 2024/12/10.
//

#ifndef MYANNS_HNSW_H
#define MYANNS_HNSW_H

#include <random>
#include <omp.h>
#include <unordered_set>
#include "graph.h"
#include "dtype.h"
#include "metric.h"
#include "logger.h"
#include "timer.h"

using namespace graph;

namespace hnsw {

    using HNSWGraph = std::vector<Graph>;

    class HNSW {
    protected:
        int max_neighbors_;

        int ef_construction_;

        unsigned enter_point_;

        double reverse_;

        std::unordered_set<int> visited_table_;

        std::default_random_engine random_engine_;

        virtual void addPoint(HNSWGraph &hnsw_graph,
                              IndexOracle &oracle,
                              unsigned index);

        /**
         * This implementation follows the original paper.
         * @param graph
         * @param oracle
         * @param query
         * @param enter_point
         * @param ef
         * @return
         */
        Neighbors searchLayer(Graph &graph,
                              IndexOracle &oracle,
                              float *query,
                              int enter_point,
                              int ef);

        int seekPos(const Neighbors &vec);


        Neighbors prune(IndexOracle &oracle,
                        Neighbors &candidates);

    public:
        HNSW(int max_neighbors,
             int ef_construction);

        void set_max_neighbors(int max_neighbors) {
            this->max_neighbors_ = max_neighbors;
        }

        void set_ef_construction(int ef_construction) {
            this->ef_construction_ = ef_construction;
        }

        void reset();

        virtual HNSWGraph build(IndexOracle &oracle);

        virtual Neighbors HNSW_search(HNSWGraph &hnsw_graph,
                                      IndexOracle &oracle,
                                      float *query,
                                      int topk,
                                      int ef_search) const;
    };
}

#endif //MYANNS_HNSW_H
