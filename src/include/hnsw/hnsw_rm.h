//
// Created by XiaoWu on 2025/8/24.
//

#ifndef MYANNS_HNSW_RM_H
#define MYANNS_HNSW_RM_H

#include "hnsw.h"

using namespace hnsw;

namespace hnsw {

class HNSW_RM : public HNSW {
private:
    std::vector<int8_t> version_;

public:
    HNSW_RM(DatasetPtr& dataset, int max_neighbors, int ef_construction);

    void
    remove(IdType id) override;
};

}  // namespace hnsw

#endif  //MYANNS_HNSW_RM_H
