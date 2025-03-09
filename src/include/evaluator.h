//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_EVALUATOR_H
#define MYANNS_EVALUATOR_H

#include <unordered_set>
#include <variant>

#include "dataset.h"
#include "graph.h"
#include "index.h"
#include "timer.h"

using namespace graph;

namespace graph {

///**
// * Evaluate the search performance of the index
// * @param graph
// * @param K
// * @param query
// * @param groundTruth
// * @param oracle
// * @param search_L
// */
//void
//evaluate(const Graph& graph,
//         unsigned int K,
//         const Matrix<float>& query,
//         const std::vector<std::vector<unsigned int>>& groundTruth,
//         IndexOracle<float>* oracle,
//         unsigned search_L = -1);

/**
 * Evaluate the search performance of the index
 * @param index
 * @param dataset
 * @param K
 * @param search_L
 * @param runs
 */
void
eval(std::variant<std::reference_wrapper<Index>, IndexPtr> index,
     DatasetPtr& dataset,
     unsigned search_L = -1,
     unsigned K = 10,
     unsigned runs = 5);
}  // namespace graph

#endif  // MYANNS_EVALUATOR_H
