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

    m.combine(indexes);

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

    auto mng = std::make_shared<taumng::TauMNG>(dataset, nsg->extract_graph(), 10, 32, 200);
    mng->build();

    recall(mng, dataset, 200);
}

void
test_cache_friendly_update() {
    omp_set_num_threads(20);
    std::vector<IndexPtr> indexes;
    {
        auto dataset = Dataset::getInstance("sift", "1m");
        Log::redirect("fgim_cache_friendly_update_" + dataset->getName());
        auto subsets = dataset->subsets(2);
        for (auto& subset : subsets) {
            auto idx = std::make_shared<hnsw::HNSW>(subset, 20, 200);
            idx->build();
            indexes.emplace_back(idx);
        }
    }
    auto dataset = Dataset::getInstance("sift", "1m", true);

    omp_set_num_threads(1);
    auto merge = std::make_shared<MGraph>(dataset, 16, 200);
    merge->combine(indexes);
    recall(merge, dataset, 200);
}

int
main() {
    Log::setVerbose(true);

    test_cache_friendly_update();

    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
}