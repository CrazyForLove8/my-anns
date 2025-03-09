//
// Created by XiaoWu on 2025/3/4.
//

#include "annslib.h"

void testDataset() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 3);
    auto another_dataset = datasets.front();

    std::cout << "The first dataset has " << dataset->getBase().size() << " points" << std::endl;
    std::cout << "Oracle size: " << dataset->getOracle()->size() << std::endl;

    std::cout << "The second dataset has " << another_dataset->getBase().size() << " points" << std::endl;
    std::cout << "Oracle size: " << another_dataset->getOracle()->size() << std::endl;

    std::cout << "The last dataset has " << datasets.back()->getBase().size() << " points" << std::endl;
    std::cout << "Oracle size: " << datasets.back()->getOracle()->size() << std::endl;

    dataset->merge(datasets);
    std::cout << "After merging, the dataset has " << dataset->getBase().size() << " points" << std::endl;
    std::cout << "Oracle size: " << dataset->getOracle()->size() << std::endl;
}

void testMerge() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 3);

    std::cout << "The original dataset has " << dataset->getBase().size() << " points" << std::endl;

    for (int x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size() << " points" << std::endl;
    }

    dataset->merge(datasets);

    std::cout << "After merging, the dataset has " << dataset->getBase().size() << " points" << std::endl;
    std::cout << "Oracle size: " << dataset->getOracle()->size() << std::endl;

    for (int x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size() << " points" << std::endl;
        std::cout << "Oracle size: " << datasets[x - 1]->getOracle()->size() << std::endl;
    }
}

void testAggregate() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 3);

    std::cout << "The original dataset has " << dataset->getBase().size() << " points" << std::endl;

    for (int x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size() << " points" << std::endl;
    }

    datasets.insert(datasets.begin(), dataset);
    auto res = Dataset::aggregate(datasets);

    std::cout << "After merging, the dataset has " << res->getBase().size() << " points" << std::endl;
    std::cout << "Oracle size: " << res->getOracle()->size() << std::endl;

    for (int x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size() << " points" << std::endl;
        std::cout << "Oracle size: " << datasets[x - 1]->getOracle()->size() << std::endl;
    }
}

int main() {

    testMerge();

    testAggregate();

}
