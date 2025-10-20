#include <set>

#include "annslib.h"

using namespace hnsw;

void
testBuild() {
    auto dataset = Dataset::getInstance("msong", "1m");
    auto index = std::make_shared<HNSW>(getParam(dataset), 16, 200);
    index->build(dataset);
    recall(index, dataset, 200);
}

void
testAdd() {
    auto dataset = Dataset::getInstance("sift", "10k");
    auto datasets = dataset->subsets(2);

    auto index = std::make_shared<HNSW>(getParam(dataset), 32, 200);
    index->build(datasets[0]);

    index->add(datasets[1]);
    recall(index, dataset);
}

void
testParlayBuild() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto index = std::make_shared<ParlayHNSW>(getParam(dataset), 32, 200);
    index->build(dataset);
    recall(index, dataset, std::vector<int>{20, 50, 80, 100, 200});
}

void
testESBlog(){
    auto dataset = Dataset::getInstance("sift", "1m");
//    Log::redirect("esblog_" + dataset->getName());
    auto datasets = dataset->subsets(2);

    auto index1 = std::make_shared<LuceneHNSW>(getParam(dataset), 16, 200, 100);
    index1->build(datasets[0]);

    auto index2 = std::make_shared<HNSW>(getParam(dataset), 16, 200);
    index2->build(datasets[1]);

    omp_set_num_threads(1);
    index1->combine(index2);

    recall(index1, dataset);
    saveHGraph(index1->extract_hgraph(), "esblog_" + dataset->getName());
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