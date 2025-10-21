#include "annslib.h"

void
testMergeHNSW() {
    auto dataset = Dataset::getInstance("msong", "1m");
    Log::redirect("mgraph_" + dataset->getName());
    auto datasets = dataset->subsets(2);

    auto index1 = std::make_shared<hnsw::HNSW>(getParam(datasets[0]), 16, 200);
    index1->build(datasets[0]);

    auto index2 = std::make_shared<hnsw::HNSW>(getParam(datasets[1]), 16, 200);
    index2->build(datasets[1]);

    std::vector<IndexPtr> indexes = {index1, index2};
    auto mgraph = std::make_shared<MGraph>(getParam(dataset), 16, 200);
    omp_set_num_threads(1);
    mgraph->combine(indexes);
    recall(mgraph, dataset);
}

void
testMergeNSW() {
    auto ds = {"sift", "deep", "msong", "glove", "gist", "crawl"};
    for (auto d : ds) {
        auto dataset = Dataset::getInstance(d, "1m");
        Log::redirect("mgraph_nsw_" + dataset->getName());
        auto datasets = dataset->subsets(2);

        auto m = std::string(d) == "sift" or std::string(d) == "msong" or std::string(d) == "deep" ? 16 : 32;

        omp_set_num_threads(16);
        auto index1 = std::make_shared<nsw::NSW>(getParam(datasets[0]), m * 2, 200);
        index1->build(datasets[0]);

        auto index2 = std::make_shared<nsw::NSW>(getParam(datasets[1]), m * 2, 200);
        index2->build(datasets[1]);

        std::vector<IndexPtr> indexes = {index1, index2};
        auto mgraph = std::make_shared<MGraph>(getParam(dataset), m, 200);
        omp_set_num_threads(1);
        mgraph->combine(indexes);
        recall(mgraph, dataset);

        if (m == 32) {
            Log::redirect("vamana_incremental_" + dataset->getName());
            auto index = std::make_shared<diskann::Vamana>(getParam(dataset), 1.2, 200, m * 2);
            omp_set_num_threads(1);
            index->build(dataset);
            recall(index, dataset);
            recall(index, dataset, -1, 100);
            dist(index, dataset);
        }
    }
}

void
test_multi_thread() {
    auto name = "gist";
    std::vector<IndexPtr> indexes;
    auto dataset = Dataset::getInstance(name, "1m");
    Log::redirect("multi_vamana_" + dataset->getName());
    logger << std::endl;
    auto subsets = dataset->subsets(2);
    for (auto& subset : subsets) {
        auto idx = std::make_shared<diskann::Vamana>(getParam(dataset), 1.2, 64, 200);
        idx->build(subset);
        indexes.emplace_back(idx);
    }

    auto nums = {1};
    for (auto num : nums) {
        omp_set_num_threads(num);
        auto merge = std::make_shared<MGraph>(getParam(dataset),32, 200);
        merge->combine(indexes);
        recall(merge, dataset, 200);
    }
}

int
main() {
    Log::setVerbose(true);

    testMergeNSW();

     // int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
     // if (ret != 0) {
     //     std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
     // }
    return 0;
}