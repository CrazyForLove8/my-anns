//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_VAMANA_H
#define MYANNS_VAMANA_H

#include <omp.h>

#include <random>

#include "index.h"

namespace diskann {
    class Vamana : public Index {
    private:
        /**
                   * alpha
                   */
        float alpha_;

        /*
                   * HNSW_search pool size
                   */
        int L_;

        /**
                   * maximum number of neighbors
                   */
        int R_;

        void
        RobustPrune(float alpha,
                    int point,
                    Neighbors &candidates);

        void
        build_internal() override;

    public:
        Vamana(DatasetPtr &dataset,
               float alpha,
               int L,
               int R);

        ~Vamana() override = default;

        void
        set_alpha(float alpha);

        void
        set_L(int L);

        void
        set_R(int R);
    };

}  // namespace diskann

#endif  // MYANNS_VAMANA_H
