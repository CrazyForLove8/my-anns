//
// Created by XiaoWu on 2025/8/24.
//

#ifndef MYANNS_HNSW_RM_H
#define MYANNS_HNSW_RM_H

#include "hnsw.h"

namespace hnsw {

class HNSW_RM : public HNSW {
public:
    void
    remove(IdType id) override;
};

}  // namespace hnsw

#endif  //MYANNS_HNSW_RM_H
