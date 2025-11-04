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
test_multi_thread() {
    int num = 12;
    auto name = "sift";
    std::vector<IndexPtr> indexes;
    {
        auto dataset = Dataset::getInstance(name, "1m");
        Log::redirect("multi" + dataset->getName());
        auto subsets = dataset->subsets(2);
        for (auto& subset : subsets) {
            auto idx = std::make_shared<hnsw::HNSW>(subset, 16, 200);
            idx->build();
            indexes.emplace_back(idx);
        }
    }

    auto dataset = Dataset::getInstance(name, "1m");
    omp_set_num_threads(num);
    auto merge = std::make_shared<MGraph>(dataset, 16, 200);
    merge->combine(indexes);
    recall(merge, dataset, 200);
}

void
test_merge(){
    auto dataset = Dataset::getInstance("msong", "10k");
    auto subsets = dataset->subsets(2);
    std::vector<IndexPtr> indexes;
    for (auto& subset : subsets) {
        auto idx = std::make_shared<hnsw::HNSW>(subset, 16, 200);
        idx->build();
        indexes.emplace_back(idx);
    }

    auto merge = std::make_shared<MGraph>(dataset, 16, 200);
    merge->combine(indexes);
    recall(merge, dataset, 200);
}

void
testMergeNSG() {
    auto ds = {"glove", "gist", "crawl"};
    for (auto d : ds) {
        auto dataset = Dataset::getInstance(d, "1m");
        Log::redirect("mgraph_nsg_" + dataset->getName());
        auto datasets = dataset->subsets(2);

        std::vector<int> ks;
        if (dataset->getName() == "deep" || dataset->getName() == "msong" || dataset->getName() == "sift") {
            ks = {8, 12, 16, 20, 24, 28, 32};
        } else {
            ks = {8, 16, 24, 32, 40, 48, 56};
        }

        for (auto k :ks) {
            omp_set_num_threads(16);
            auto index1 = std::make_shared<nsg::NSG>(datasets[0], k * 2, 200, k * 2);
            index1->build();

            auto index2 = std::make_shared<nsg::NSG>(datasets[1], k * 2, 200, k * 2);
            index2->build();

            std::vector<IndexPtr> indexes = {index1, index2};
            auto mgraph = std::make_shared<MGraph>(dataset, k, 200);
            omp_set_num_threads(1);
            mgraph->combine(indexes);
            recall(mgraph, dataset, 200);
        }
    }
}


int
main() {
    Log::setVerbose(true);

    testMergeNSG();
    // int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    // if (ret != 0) {
    //     std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    // }
    return 0;
}