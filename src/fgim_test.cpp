//
// Created by XiaoWu on 2025/3/4.
//

#include "annslib.h"

#define MULTI_THREAD 1

void
testCombine() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto datasets = std::vector<DatasetPtr>();
    //    auto nnd = std::make_shared<nndescent::NNDescent>(dataset, 20);
    //    nnd->build();
    //    recall(nnd, dataset, 200);

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

    recall(m, dataset, 200);
}

void
testMerge() {
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

void
exp3_nsw(DatasetPtr& dataset) {
    //    auto datasets = std::vector<DatasetPtr>();
    //    dataset->split(datasets, 2);
    //    datasets.pushHeap(datasets.begin(), dataset);

    auto nsw = std::make_shared<nsw::NSW>(dataset, 32, 200);

    nsw->build();

    recall(nsw, dataset);
}

void
exp3_taumng(DatasetPtr& dataset) {
    auto nsg = std::make_shared<nsg::NSG>(dataset, 32, 200, 32);

    nsg->build();

    recall(nsg, dataset, 200);

    auto mng = std::make_shared<taumng::TauMNG>(dataset, nsg->extractGraph(), 10, 32, 200);
    mng->build();

    recall(mng, dataset, 200);
}

int
main() {
#if not MULTI_THREAD
    omp_set_num_threads(1);
#endif
    Log::setVerbose(true);

    auto dataset = Dataset::getInstance("sift", "1m");

    exp3_taumng(dataset);

    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
}