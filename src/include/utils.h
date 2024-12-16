//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_UTILS_H
#define MYANNS_UTILS_H

#include "graph.h"
#include "timer.h"
#include <unordered_set>

namespace graph{

    std::vector<std::vector<unsigned int>> loadGroundTruth(const std::string &filename,
                                                           unsigned int qsize,
                                                           unsigned int K = 100);

    /**
     * Evaluate the HNSW_search performance of the index
     * @param graph
     * @param K
     * @param query
     * @param groundTruth
     * @param oracle
     * @param search_L
     */
    void evaluate(const Graph &graph,
                  unsigned int K,
                  const Matrix &query,
                  const std::vector<std::vector<unsigned int>> &groundTruth,
                  IndexOracle &oracle,
                  unsigned search_L = -1);
}

#endif //MYANNS_UTILS_H
