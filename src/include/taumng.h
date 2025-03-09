//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_TAUMNG_H
#define MYANNS_TAUMNG_H

#include <omp.h>

#include <random>

#include "index.h"

namespace taumng {

class TauMNG : public Index {
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

    void
    build_internal() override;

public:
    /**
             * @brief Build a TauMNG graph. Note that the graph is destroyed after the build since std::move is called.
             * @param oracle
             * @param graph
             * @param t
             * @param h
             * @param b
             */
    TauMNG(DatasetPtr& dataset, Graph& graph, float t, int h, int b);

    void
    set_b(int b);

    void
    set_h(int h);
};
}  // namespace taumng

#endif  // MYANNS_TAUMNG_H
