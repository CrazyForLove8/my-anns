#include "evaluator.h"

#ifndef MULTITHREAD
#define MULTITHREAD 0
#endif

void
evaluate(Index& index, DatasetPtr& dataset, unsigned qsize, unsigned L, unsigned K, unsigned runs) {
    auto& query = dataset->getQuery();
    auto& groundTruth = dataset->getGroundTruth();
    float recall = 0;
    double qps = 0;
    for (int x = 0; x < runs; ++x) {
        Timer timer;
        timer.start();
        float local_recall = 0;
#if MULTITHREAD
#pragma omp parallel for reduction(+ : local_recall)
#endif
        for (size_t i = 0; i < qsize; ++i) {
            auto result = index.search(query[i], K, L);
            std::unordered_set<unsigned> gt(groundTruth[i], groundTruth[i] + K);
            size_t correct = 0;
            for (const auto& res : result) {
                if (gt.find(res.id) != gt.end()) {
                    correct++;
                }
            }
            local_recall += static_cast<float>(correct);
        }
        timer.end();
        qps = std::max(qps, (double)qsize / timer.elapsed());
        recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
    }
    std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
}

void
evaluate(const IndexPtr& index,
         DatasetPtr& dataset,
         unsigned qsize,
         unsigned L,
         unsigned K,
         unsigned runs) {
    auto& query = dataset->getQuery();
    auto& groundTruth = dataset->getGroundTruth();
    float recall = 0;
    double qps = 0;
    for (int x = 0; x < runs; ++x) {
        Timer timer;
        timer.start();
        float local_recall = 0;
#if MULTITHREAD
#pragma omp parallel for reduction(+ : local_recall)
#endif
        for (size_t i = 0; i < qsize; ++i) {
            auto result = index->search(query[i], K, L);
            std::unordered_set<unsigned> gt(groundTruth[i], groundTruth[i] + K);
            size_t correct = 0;
            for (const auto& res : result) {
                if (gt.find(res.id) != gt.end()) {
                    correct++;
                }
            }
            local_recall += static_cast<float>(correct);
        }
        timer.end();
        qps = std::max(qps, (double)qsize / timer.elapsed());
        recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
    }
    std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
}

void
graph::eval(std::variant<std::reference_wrapper<Index>, IndexPtr> index,
            DatasetPtr& dataset,
            unsigned search_L,
            unsigned K,
            unsigned runs) {
    std::vector<unsigned> search_Ls;
    if (search_L == -1) {
        search_Ls = {20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800};
    } else {
        search_Ls = {search_L};
    }
    size_t qsize = dataset->getQuery().size();

    for (auto L : search_Ls) {
        if (std::holds_alternative<IndexPtr>(index)) {
            evaluate(std::get<IndexPtr>(index), dataset, qsize, L, K, runs);
        } else {
            evaluate(std::get<std::reference_wrapper<Index>>(index), dataset, qsize, L, K, runs);
        }
    }
}

//void
//graph::evaluate(const Graph& graph,
//                unsigned int K,
//                const Matrix<float>& query,
//                const std::vector<std::vector<unsigned int>>& groundTruth,
//                IndexOracle<float>* oracle,
//                unsigned search_L) {
//    std::vector<unsigned> search_Ls;
//    if (search_L == -1) {
//        search_Ls = {
//            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
//    } else {
//        search_Ls = {search_L};
//    }
//    size_t qsize = query.size();
//    size_t total = graph.size();
//
//    FlattenGraph fg(graph);
//
//    for (auto L : search_Ls) {
//        float recall = 0;
//        double qps = 0;
//        auto runs = 5;
//        for (int x = 0; x < runs; ++x) {
//            Timer timer;
//            timer.start();
//            float local_recall = 0;
//            //#pragma omp parallel for reduction(+:local_recall)
//            for (size_t i = 0; i < qsize; ++i) {
//                auto result = search(oracle, fg, query[i], K, L);
//                std::unordered_set<unsigned> gt(groundTruth[i].begin(), groundTruth[i].begin() + K);
//                size_t correct = 0;
//                for (const auto& res : result) {
//                    if (gt.find(res.id) != gt.end()) {
//                        correct++;
//                    }
//                }
//                local_recall += static_cast<float>(correct);
//            }
//            timer.end();
//            qps = std::max(qps, (double)qsize / timer.elapsed());
//            recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
//        }
//        std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
//    }
//}