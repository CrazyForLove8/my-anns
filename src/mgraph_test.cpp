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

void
test_cache_friendly_update() {
    omp_set_num_threads(20);
    auto name = "deep";
    std::vector<IndexPtr> indexes;
    {
        auto dataset = Dataset::getInstance(name, "1m");
        Log::redirect("our_disk_" + dataset->getName());
        auto subsets = dataset->subsets(2);
        for (auto& subset : subsets) {
            auto idx = std::make_shared<diskann::Vamana>(subset, 1.2, 200, 32);
            idx->build();
            indexes.emplace_back(idx);
        }
    }

    auto dataset = Dataset::getInstance(name, "1m", true);
    auto merge = std::make_shared<MGraph>(dataset, 16, 200);
    merge->combine(indexes);
    recall(merge, dataset, 200);
}

int
main() {
    Log::setVerbose(true);

    test_cache_friendly_update();
    return 0;
}