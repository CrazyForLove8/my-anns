#include "annslib.h"

void
testBuild() {
    auto dataset = Dataset::getInstance("crawl", "10k");

    auto index = std::make_shared<nsg::NSG>(dataset, 32, 200, 32);
    // omp_set_num_threads(1);
    index->build();

    recall(index, dataset);
}

void
testBuildNSG() {
    auto ds = {"crawl"};
    for (auto d : ds) {
        auto dataset = Dataset::getInstance(d, "1m");
        Log::redirect("nsg_build_" + dataset->getName());
        omp_set_num_threads(1);
        std::vector<int> ks;
        if (dataset->getName() == "deep" || dataset->getName() == "msong" || dataset->getName() == "sift") {
            ks = {8, 16, 20, 24, 28, 32, 36, 40};
        } else {
            ks = {8};
        }
        for (auto k : ks) {
            auto index = std::make_shared<nsg::NSG>(dataset, k, 200, k);
            index->build();
            recall(index, dataset, 200);
        }
    }
}

int
main() {
    Log::setVerbose(true);
    testBuildNSG();
    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
}