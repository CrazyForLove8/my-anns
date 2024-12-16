//
// Created by XiaoWu on 2024/12/15.
//

#ifndef MYANNS_DHNSW_H
#define MYANNS_DHNSW_H

#include "hnsw.h"

using namespace hnsw;

namespace dhnsw {

    class DHNSW : public HNSW {

    public:
        DHNSW(int max_neighbors,
              int ef_construction);

        HNSWGraph build(IndexOracle &oracle) override;

        Neighbors HNSW_search(HNSWGraph &hnsw_graph,
                              IndexOracle &oracle,
                              float *query,
                              int topk,
                              int ef_search) const override;

    };


}

#endif //MYANNS_DHNSW_H
