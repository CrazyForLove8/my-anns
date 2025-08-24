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

    void
    build_internal() override;

protected:
    void
    addPoint(unsigned index) override;

public:
    SHNSW(DatasetPtr& dataset, int max_neighbors, int ef_construction, float radius);

    Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const override;
};

}  // namespace shnsw

#endif  // MYANNS_SHNSW_H
