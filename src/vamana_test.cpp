#include <set>

#include "annslib.h"

void
testAdd() {
    auto dataset = Dataset::getInstance("sift", "10k");
    auto datasets = dataset->subsets(2);

    auto index = std::make_shared<diskann::Vamana>(getParam(dataset), 1.2, 200, 32);
    index->build(datasets[0]);

    omp_set_num_threads(1);
    index->add(datasets[1]);

    recall(index, dataset);
}

void
testParlayBuild() {
    auto dataset = Dataset::getInstance("sift", "10k");
    auto index = std::make_shared<diskann::ParlayVamana>(getParam(dataset), 1.2, 200, 32);
    index->build(dataset);
    recall(index, dataset, std::vector<int>{20, 50, 80, 100, 200});
}

void
test_multi_thread_parlay() {
    auto name = "sift";
    std::vector<IndexPtr> indexes;
    auto dataset = Dataset::getInstance(name, "1m");
    Log::redirect("multi_parlay_vamana_" + dataset->getName());
    logger << std::endl;

    auto nums = {10, 8, 6, 4, 2};
    for (auto num : nums) {
        omp_set_num_threads(num);
        auto index = std::make_shared<diskann::ParlayVamana>(getParam(dataset),1.2, 200, 32);
        index->build(dataset);
        recall(index, dataset, 200);
    }
}

void
testBuild() {
    // TODO 还没跑
    auto dataset = Dataset::getInstance("gist", "1m");
    Log::redirect("vamana_incremental_" + dataset->getName());
    auto index = std::make_shared<diskann::Vamana>(getParam(dataset), 1.2, 200, 64);
    omp_set_num_threads(1);
    index->build(dataset);
    recall(index, dataset);
    recall(index, dataset, -1, 100);
    dist(index, dataset);
}

int
main() {
    Log::setVerbose(true);

    testBuild();
    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
    return 0;
}