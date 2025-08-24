//
// Created by XiaoWu on 2024/12/15.
//

#ifndef MYANNS_DHNSW_H
#define MYANNS_DHNSW_H

#include "hnsw.h"

using namespace hnsw;

namespace dhnsw {

class DHNSW : public HNSW {
private:
    float lambda_;

public:
    DHNSW(DatasetPtr& dataset, int max_neighbors, int ef_construction, float lambda);

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;
};

}  // namespace dhnsw

#endif  // MYANNS_DHNSW_H
