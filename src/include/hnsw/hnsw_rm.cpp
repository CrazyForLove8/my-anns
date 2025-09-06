#include "hnsw/hnsw_rm.h"

HNSW_RM::HNSW_RM(DatasetPtr& dataset, int max_neighbors, int ef_construction)
    : HNSW(dataset, max_neighbors, ef_construction) {
    version_.resize(oracle_->size(), 0);
}

void
hnsw::HNSW_RM::remove(IdType id) {
    if (version_[id] + 1 == 0) {
        throw std::runtime_error("Version overflow for id: " + std::to_string(id));
    }
    version_[id] += 1;
}
