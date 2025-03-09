#include "annslib.h"

using namespace hnsw;

void testBuild() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto index = std::make_shared<HNSW>(dataset, 20, 200);
    index->build();
    eval(index, dataset, 200);

    // 100k
    // L: 200 recall: 0.9995 qps: 4609.67

    // L: 200 recall: 0.9995 qps: 4948.73

    // L: 200 recall: 0.9995 qps: 5183.3
}

void testAdd() {
    DatasetPtr dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 4);
    auto index = std::make_shared<HNSW>(dataset, 20, 200);
    index->build();

    for (auto &data: datasets) {
        index->add(data);
        std::cout << "After adding, the dataset has " << dataset->getBase().size() << " points" << std::endl;
    }

    eval(index, dataset, 200);
}

int main() {
    Log::setVerbose(true);

    testBuild();
}