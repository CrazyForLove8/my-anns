//
// Created by XiaoWu on 2024/12/10.
//

#ifndef MYANNS_HNSW_H
#define MYANNS_HNSW_H

#include <omp.h>

#include <random>
#include <unordered_set>

#include "index.h"

namespace hnsw {

class HNSW : public Index {
protected:
    HGraph graph_;

    FlattenHGraph flatten_graph_;

    uint32_t max_neighbors_{32};

    uint32_t max_base_neighbors_{64};

    uint8_t max_level_{};

    uint8_t cur_max_level_{};

    uint32_t ef_construction_{};

    std::vector<uint8_t> levels_;

    double reverse_{};

    std::unordered_set<int> visited_table_;

    std::default_random_engine random_engine_;

    virtual void
    addPoint(IdType index);

    void
    prune(Neighbors& candidates, IdType max_neighbors);

    void
    build_internal(DatasetPtr& dataset) override;

    void
    partial_build(IdType start, IdType end) override;

    void
    resize(IdType new_size) override;

public:
    uint32_t enter_point_{};

    HNSW(const IndexParam& param, int max_neighbors, int ef_construction);

    HNSW(const IndexParam& param,
         HGraph& graph,
         bool partial = false,
         int max_neighbors = 32,
         int ef_construction = 200);

    ~HNSW() override = default;

    void
    set_max_neighbors(int max_neighbors);

    void
    set_ef_construction(int ef_construction);

    void
    build(DatasetPtr& dataset) override;

    void
    partial_build(IdType num) override;

    Graph&
    extract_graph() override;

    HGraph&
    extract_hgraph();

    ParamMap
    extract_params() override;

    void
    add(DatasetPtr& dataset) override;

    void
    remove(graph::IdType id) override;

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;

    void
    load_params(const graph::ParamMap& params) override;

    void
    print_info() const override;
};

class ParlayHNSW : public HNSW {
    int theta_;

    Graph reverse_graph_;

    void
    batch_insert(IdType start, IdType end);

    void
    build_internal(graph::DatasetPtr& dataset) override;

    void
    resize(graph::IdType new_size) override;

public:
    ParlayHNSW(const IndexParam& param, int M, int ef_construction, int theta = -1);

    void
    print_info() const override;
};

}  // namespace hnsw

#endif  // MYANNS_HNSW_H
