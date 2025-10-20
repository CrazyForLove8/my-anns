#include <set>

#include "annslib.h"

void
testBuild() {
    auto dataset = Dataset::getInstance("msong", "100k");
    auto index = std::make_shared<diskann::Vamana>(getParam(dataset), 1.2, 200, 32);
    index->build(dataset);
    recall(index, dataset);
}

void
testAdd() {
    auto dataset = Dataset::getInstance("gist", "1m");
    auto datasets = dataset->subsets(2);

    auto index = std::make_shared<diskann::Vamana>(getParam(dataset), 1.2, 200, 64);
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

int
main() {
    Log::setVerbose(true);

    testBuild();

    return 0;
}