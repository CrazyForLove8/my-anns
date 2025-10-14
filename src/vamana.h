//
// Created by XiaoWu on 2024/11/23.
//

#ifndef MYANNS_VAMANA_H
#define MYANNS_VAMANA_H

#include <omp.h>

#include <random>

#include "index.h"
#include "kmeans.h"

namespace diskann {
class Vamana : public Index {
protected:
    /**
       * alpha
       */
    float alpha_;

    /*
       * search pool size
       */
    int L_;

    /**
       * maximum number of neighbors
       */
    int R_;

    IdType root{};

    void
    RobustPrune(float alpha, IdType point, Neighbors& candidates);

    void
    build_internal(DatasetPtr& dataset) override;

    void
    find_root();

    void
    resize(graph::IdType new_size) override;

public:
    /**
     *
     * @param alpha
     * @param L
     * @param R
     */
    Vamana(const IndexParam& param, float alpha, int L, int R);

    ~Vamana() override = default;

    void
    set_alpha(float alpha);

    void
    set_L(int L);

    void
    set_R(int R);

    void
    add(graph::DatasetPtr& dataset) override;

    void
    partial_build(graph::IdType start, graph::IdType end) override;

    void
    partial_build(graph::IdType num) override;

    void
    partial_build(std::vector<IdType>& permutation);

    void
    print_info() const override;

    ParamMap
    extract_params() override;
};

class ParlayVamana : public Vamana {
    int theta_;

    Graph reverse_graph_;

    void
    batch_insert(IdType start, IdType end);

    void
    build_internal(graph::DatasetPtr& dataset) override;

    void
    resize(graph::IdType new_size) override;

public:
    ParlayVamana(const IndexParam& param, float alpha, int L, int R, int theta = -1);

    void
    print_info() const override;
};

class DiskANN : public Index {
private:
    float alpha_;

    int L_;

    int R_;

    int k_;

    int ell_;

    void
    build_internal(DatasetPtr& dataset) override;

public:
    DiskANN(const IndexParam& param, float alpha, int L, int R, int k, int ell);

    ~DiskANN() override = default;
};

}  // namespace diskann

#endif  // MYANNS_VAMANA_H
