#include "annslib.h"

void
testMerge() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto datasets = dataset->subsets(2);

    auto index1 = std::make_shared<hnsw::HNSW>(getParam(datasets[0]), 32, 200);
    index1->build(datasets[0]);

    auto index2 = std::make_shared<hnsw::HNSW>(getParam(datasets[1]), 32, 200);
    index2->build(datasets[1]);

    std::vector<IndexPtr> indexes = {index1, index2};
    auto mgraph = std::make_shared<MGraph>(getParam(dataset), 20, 200, 0.5);
    mgraph->combine(indexes);
    recall(mgraph, dataset);
}

int
main() {
    Log::setVerbose(true);

    testMerge();

    // int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    // if (ret != 0) {
    //     std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    // }
    return 0;
}