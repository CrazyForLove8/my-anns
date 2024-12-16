//
// Created by XiaoWu on 2024/12/16.
//

#ifndef MYANNS_SHNSW_H
#define MYANNS_SHNSW_H

#include "hnsw.h"

using namespace hnsw;

namespace shnsw {

    struct Family {
        Neighbors children_;
    };

    using FamilyMap = std::vector<Family>;

    class SHNSW : public HNSW {
    private:
        float radius_;

        FamilyMap families_;

        HNSWGraph graph_;

    protected:
        void addPoint(HNSWGraph &hnsw_graph,
                      IndexOracle &oracle,
                      unsigned index) override;

    public:
        SHNSW(int max_neighbors,
              int ef_construction,
              float radius);

        HNSWGraph build(IndexOracle &oracle) override;

        Neighbors HNSW_search(HNSWGraph &hnsw_graph,
                              IndexOracle &oracle,
                              float *query,
                              int topk,
                              int ef_search) const override;
    };


}

#endif //MYANNS_SHNSW_H
