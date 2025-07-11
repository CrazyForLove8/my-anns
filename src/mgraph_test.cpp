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
    mGraph.combine(indexes);

    recall(mGraph, dataset);
}

void
test_enable_save_help(DatasetPtr& dataset) {
    auto datasets = dataset->subsets(2);

    auto index1 = std::make_shared<hnsw::HNSW>(datasets[0], 32, 200);
    index1->build();

    auto index2 = std::make_shared<hnsw::HNSW>(datasets[1], 32, 200);
    index2->build();

    std::vector<IndexPtr> indexes = {index1, index2};
    MGraph mGraph(dataset, 20, 200);

    omp_set_num_threads(1);
    mGraph.set_save_helper({3, "mgraph_checkpoint.bin"});
    mGraph.combine(indexes);
}

void
test_load_checkpoint(DatasetPtr& dataset) {
    auto datasets = dataset->subsets(2);

    auto index1 = std::make_shared<hnsw::HNSW>(datasets[0], 32, 200);
    index1->build();

    auto index2 = std::make_shared<hnsw::HNSW>(datasets[1], 32, 200);
    index2->build();

    std::vector<IndexPtr> indexes = {index1, index2};
    MGraph mGraph(dataset, "./graph_output/mgraph_checkpoint.bin");

    mGraph.set_save_helper({3, "mgraph_checkpoint.bin"});
    mGraph.combine(indexes);
}

int
main() {
    Log::setVerbose(true);

    auto dataset = Dataset::getInstance("sift", "1m");
    test_load_checkpoint(dataset);
    return 0;
}