//
// Created by XiaoWu on 2025/3/4.
//

#include "annslib.h"

#define MULTI_THREAD 1

void testCombine() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto datasets = std::vector<DatasetPtr>();
//    auto nnd = std::make_shared<nndescent::NNDescent>(dataset, 20);
//    nnd->build();
//    eval(nnd, dataset, 200);

    dataset->split(datasets, 2);
    datasets.insert(datasets.begin(), dataset);

    auto nnd1 = std::make_shared<nndescent::NNDescent>(datasets[0], 20);
    auto nnd2 = std::make_shared<nndescent::NNDescent>(datasets[1], 20);
//    auto nnd3 = std::make_shared<nndescent::NNDescent>(datasets[2], 40);
//    auto nnd4 = std::make_shared<nndescent::NNDescent>(datasets[3], 40);
//    auto nnd5 = std::make_shared<nndescent::NNDescent>(datasets[4], 40);

    nnd1->build();
    nnd2->build();
//    nnd3->build();
//    nnd4->build();
//    nnd5->build();

    std::vector<IndexPtr> indexes = {nnd1, nnd2};
//    std::vector<IndexPtr> indexes = {nnd1, nnd2, nnd3, nnd4, nnd5};

    FGIM m(20);

    m.Combine(indexes);

    eval(m, dataset, 200);
}

void testMerge() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 3);
    datasets.insert(datasets.begin(), dataset);

    nndescent::NNDescent nnd(dataset, 20);
    nndescent::NNDescent nnd1(datasets[1], 20);
    nndescent::NNDescent nnd2(datasets[2], 20);

    nnd.build();
    nnd1.build();
    nnd2.build();

    std::vector<Index> indexes;

    indexes.push_back(nnd);
    indexes.push_back(nnd1);
    indexes.push_back(nnd2);

    Graph merged;
    FGIM m(20);
}

int main() {
#if not MULTI_THREAD
    omp_set_num_threads(1);
#endif
    Log::setVerbose(true);

    testCombine();

}