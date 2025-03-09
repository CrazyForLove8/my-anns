#include "annslib.h"

void
testMerge() {
    omp_set_num_threads(1);
    auto dataset = Dataset::getInstance("sift", "1m");
    auto datasets = std::vector<DatasetPtr>();

    dataset->split(datasets, 2);
    datasets.insert(datasets.begin(), dataset);

    auto hnsw1 = std::make_shared<hnsw::HNSW>(datasets[0], 20, 200);
    auto hnsw2 = std::make_shared<hnsw::HNSW>(datasets[1], 20, 200);

    hnsw1->build();
    hnsw2->build();

    std::vector<IndexPtr> vec = {hnsw1, hnsw2};
    MGraph mgraph(40, 200);

    mgraph.Combine(vec);

    eval(mgraph, mgraph.extractDataset(), 200);
}

int
main() {
    Log::setVerbose(true);

    testMerge();

    return 0;
}