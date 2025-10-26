#include "annslib.h"

void
testBuildNSG() {
    auto ds = {"deep", "msong", "glove", "gist", "crawl"};
    for (auto d : ds) {
        auto dataset = Dataset::getInstance(d, "1m");
        Log::redirect("nsg_build_" + dataset->getName());
        omp_set_num_threads(1);
        std::vector<int> ks;
        if (dataset->getName() == "deep" || dataset->getName() == "msong") {
            ks = {8, 16, 20, 24, 28, 32, 36, 40};
        } else {
            ks = {32, 40, 48, 56, 64, 72, 80};
        }
        for (auto k : ks) {
            auto index = std::make_shared<nsg::NSG>(getParam(dataset), k, 200, k);
            index->build(dataset);
            recall(index, dataset, 200);
        }
    }
}

void
testMergeNSG() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto datasets = dataset->subsets(2);

    auto index = std::make_shared<nsg::NSG>(getParam(dataset), 32, 200, 32);
    index->build(datasets[0]);
    auto index1 = std::make_shared<nsg::NSG>(getParam(dataset), 32, 200, 32);
    index1->build(datasets[1]);

    std::vector<IndexPtr> indexes = {index, index1};
    auto mgraph = std::make_shared<MGraph>(getParam(dataset), 16, 200);
    omp_set_num_threads(1);
    mgraph->combine(indexes);
    recall(mgraph, dataset);
}

void
mergeNSG() {
    auto ds = {"sift", "deep", "msong", "glove", "gist", "crawl"};
    for (auto d : ds) {
        auto dataset = Dataset::getInstance(d, "1m");
        auto m = std::string(d) == "sift" or std::string(d) == "msong" or std::string(d) == "deep" ? 16 : 32;
        Log::redirect("mgraph_nsg_" + dataset->getName());
        auto datasets = dataset->subsets(2);

        omp_set_num_threads(16);
        auto index1 = std::make_shared<nsg::NSG>(getParam(datasets[0]), m * 2, 200, m * 2);
        index1->build(datasets[0]);

        auto index2 = std::make_shared<nsg::NSG>(getParam(datasets[1]), m * 2, 200, m * 2);
        index2->build(datasets[1]);

        std::vector<IndexPtr> indexes = {index1, index2};
        auto mgraph = std::make_shared<MGraph>(getParam(dataset), m, 200);
        omp_set_num_threads(1);
        mgraph->combine(indexes);
        recall(mgraph, dataset);

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
