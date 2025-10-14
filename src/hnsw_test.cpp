#include <set>

#include "annslib.h"

using namespace hnsw;

void
testBuild() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto index = std::make_shared<HNSW>(getParam(dataset), 32, 200);
    index->build(dataset);
    recall(index, dataset);
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

int
main() {
    Log::setVerbose(true);

    testBuild();
    return 0;
}