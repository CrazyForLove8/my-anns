//
// Created by XiaoWu on 2025/3/4.
//

#include "annslib.h"

void
testDataset() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 3);
    auto another_dataset = datasets.front();

    std::cout << "The first dataset has " << dataset->getBase().size() << " points" << std::endl;
    std::cout << "Oracle size: " << dataset->getOracle()->size() << std::endl;

    std::cout << "The second dataset has " << another_dataset->getBase().size() << " points"
              << std::endl;
    std::cout << "Oracle size: " << another_dataset->getOracle()->size() << std::endl;

    std::cout << "The last dataset has " << datasets.back()->getBase().size() << " points"
              << std::endl;
    std::cout << "Oracle size: " << datasets.back()->getOracle()->size() << std::endl;

    dataset->merge(datasets);
    std::cout << "After merging, the dataset has " << dataset->getBase().size() << " points"
              << std::endl;
    std::cout << "Oracle size: " << dataset->getOracle()->size() << std::endl;
}

void
testMerge() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 3);

    std::cout << "The original dataset has " << dataset->getBase().size() << " points" << std::endl;

    for (size_t x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size()
                  << " points" << std::endl;
    }

    dataset->merge(datasets);

    std::cout << "After merging, the dataset has " << dataset->getBase().size() << " points"
              << std::endl;
    std::cout << "Oracle size: " << dataset->getOracle()->size() << std::endl;

    for (size_t x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size()
                  << " points" << std::endl;
        std::cout << "Oracle size: " << datasets[x - 1]->getOracle()->size() << std::endl;
    }
}

void
testAggregate() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 3);

    std::cout << "The original dataset has " << dataset->getBase().size() << " points" << std::endl;

    for (size_t x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size()
                  << " points" << std::endl;
    }

    datasets.insert(datasets.begin(), dataset);
    auto res = Dataset::aggregate(datasets);

    std::cout << "After merging, the dataset has " << res->getBase().size() << " points"
              << std::endl;
    std::cout << "Oracle size: " << res->getOracle()->size() << std::endl;

    for (size_t x = 1; x <= datasets.size(); ++x) {
        std::cout << "The " << x << " dataset has " << datasets[x - 1]->getBase().size()
                  << " points" << std::endl;
        std::cout << "Oracle size: " << datasets[x - 1]->getOracle()->size() << std::endl;
    }
}

void
testSubset() {
    auto dataset =
        Dataset::getInstance("/root/mount/dataset/siftsmall/siftsmall_base.fvecs", DISTANCE::L2);
    std::cout << "The original dataset has " << dataset->getBase().size() << " points" << std::endl;

    for (auto& v : {2, 3, 4, 5, 6, 7}) {
        auto datasets = dataset->subsets(v);
        for (size_t x = 1; x <= datasets.size(); ++x) {
            std::cout << "The " << x << " / " << v << " dataset has "
                      << datasets[x - 1]->getBase().size() << " points" << std::endl;
            std::cout << "Oracle size: " << datasets[x - 1]->getOracle()->size() << std::endl;
            auto last = datasets[x - 1]->getBase().size() - 1;
            auto dist = (*datasets[x - 1]->getOracle())(0, last);
            std::cout << "Distance between first and last point in dataset " << x << "/" << v
                      << " is: " << dist << std::endl;
        }

        std::cout << "The original dataset has " << dataset->getBase().size() << " points"
                  << std::endl;
        std::cout << "Oracle size: " << dataset->getOracle()->size() << std::endl;
    }
}

void
testSSD() {
    {
        print_memory_usage();
        auto dataset = Dataset::getInstance("sift", "1m");

        std::cout << "In-memory dataset && direct access" << std::endl;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 10; ++j) {
                std::cout << (*dataset->getBasePtr())(i, j) << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "In-memory dataset && oracle access" << std::endl;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 10; ++j) {
                std::cout << (*dataset->getOracle())(i, j) << " ";
            }
            std::cout << std::endl;
        }
        print_memory_usage();
    }

    {
        print_memory_usage();
        std::cout << "SSD dataset && direct access" << std::endl;
        auto dataset = Dataset::getInstance("sift", "1m", true);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 10; ++j) {
                std::cout << (*dataset->getBasePtr())(i, j) << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "SSD dataset && oracle access" << std::endl;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 10; ++j) {
                std::cout << (*dataset->getOracle())(i, j) << " ";
            }
            std::cout << std::endl;
        }
        print_memory_usage();
    }
}

int
main() {
    Log::setVerbose(true);

    testSSD();
}
