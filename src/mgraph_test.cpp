#include "annslib.h"

void
mergeExp1_1() {
    auto dataset = Dataset::getInstance("gist", "1m");
    std::ofstream out("mgraph_" + dataset->getName() + ".log", std::ios::app);
    std::cout.rdbuf(out.rdbuf());
    std::cout << "Exp1: Merge 2 HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);

    std::vector<IndexPtr> vec(datasets.size() + 1);
    vec[0] = std::make_shared<hnsw::HNSW>(dataset, 20, 200);
    vec[0]->build();
    for (size_t i = 1; i < vec.size(); i++) {
        vec[i] = std::make_shared<hnsw::HNSW>(datasets[i - 1], 20, 200);
        vec[i]->build();
    }

    {
        std::vector<int> params = {12, 16, 20, 24, 28, 32, 36, 40};
        for (auto param : params) {
            std::cout << "Parameter: Max degree: " << param << std::endl;
            MGraph mgraph(param, 200);
            mgraph.Combine(vec);
            eval(mgraph, mgraph.extractDataset(), 200);
        }
    }
}

void
mergeExp1_2() {
    auto dataset = Dataset::getInstance("sift", "1m");
    std::ofstream out("mgraph_" + dataset->getName() + ".log", std::ios::app);
    std::cout.rdbuf(out.rdbuf());
    std::cout << "Exp1: Merge 2 HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    std::vector<int> params = {20, 40, 60, 80, 100, 120, 140, 160};
    for (auto param : params) {
        auto datasets = std::vector<DatasetPtr>();
        int num_splits = 2;
        std::cout << "Number of splits: " << num_splits << std::endl;
        dataset->split(datasets, num_splits);

        std::vector<IndexPtr> vec(datasets.size() + 1);
        std::cout << "Parameter: ef_construction: " << param << std::endl;
        vec[0] = std::make_shared<hnsw::HNSW>(dataset, 20, param);
        vec[0]->build();
        {
            Timer timer;
            timer.start();
            for (auto& data : datasets) {
                vec[0]->add(data);
            }
            timer.end();
            std::cout << "Total adding time: " << timer.elapsed() << "s" << std::endl;
            eval(vec[0], vec[0]->extractDataset(), 200);
        }
    }
}

void
testMerge() {
    auto dataset = Dataset::getInstance("gist", "1m");
    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    dataset->split(datasets, num_splits);

    std::vector<IndexPtr> vec(datasets.size() + 1);
    vec[0] = std::make_shared<hnsw::HNSW>(dataset, 20, 200);
    vec[0]->build();
    for (size_t i = 1; i < vec.size(); i++) {
        vec[i] = std::make_shared<hnsw::HNSW>(datasets[i - 1], 20, 200);
        vec[i]->build();
    }

    MGraph mgraph(32, 200);

    mgraph.Combine(vec);

    eval(mgraph, mgraph.extractDataset(), 200);

    //    Timer timer;
    //    timer.start();
    //    for (auto &data: datasets) {
    //        vec[0]->add(data);
    //    }
    //    timer.end();
    //    std::cout << "Total adding time: " << timer.elapsed() << "s" << std::endl;
    //    eval(vec[0], vec[0]->extractDataset(), 200);
}

int
main() {
    Log::setVerbose(true);
    //    omp_set_num_threads(1);

    testMerge();

    return 0;
}