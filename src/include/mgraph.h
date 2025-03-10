//
// Created by XiaoWu on 2025/3/2.
//

#ifndef MYANNS_MGRAPH_H
#define MYANNS_MGRAPH_H

#include "fgim.h"

using namespace graph;

class MGraph : public FGIM {
private:
    HGraph graph_;

    FlattenHGraph flatten_graph_;

    std::default_random_engine random_engine_;

    double reverse_;

    uint32_t enter_point_;

    uint32_t ef_construction_;

    void
    CrossQuery(std::vector<IndexPtr>& indexes) override;

    void
    Refinement() override;

    void
    ReconstructHGraph();

    void
    heuristic(Neighbors& candidates, unsigned max_degree);

public:
    MGraph();

    explicit MGraph(unsigned int max_degree, unsigned int ef_construction, float sample_rate = 0.3);

    void
    Combine(std::vector<IndexPtr>& indexes) override;

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;
};

#endif  //MYANNS_MGRAPH_H
