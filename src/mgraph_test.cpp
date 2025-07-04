#include "annslib.h"

void
test_filter_same_index(DatasetPtr& dataset) {
    auto datasets = dataset->subsets(2);

    auto index1 = std::make_shared<hnsw::HNSW>(datasets[0], 32, 200);
    index1->build();

    auto index2 = std::make_shared<hnsw::HNSW>(datasets[1], 32, 200);
    index2->build();

    std::vector<IndexPtr> indexes = {index1, index2};
    MGraph mGraph(dataset, 20, 200);
    mGraph.Combine(indexes);

    recall(mGraph, dataset);
}

int
main() {
    Log::setVerbose(true);
    Log::redirect("with_filter");

    auto dataset = Dataset::getInstance("sift", "1m");
    test_filter_same_index(dataset);
    return 0;
}