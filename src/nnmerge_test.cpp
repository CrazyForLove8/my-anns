#include "annslib.h"

void
testMerge(DatasetPtr &dataset) {
    Log::redirect("12_nnm_" + dataset->getName());
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 2);
    datasets.insert(datasets.begin(), dataset);

    int param = 64;

    auto nnd1 = std::make_shared<nndescent::NNDescent>(datasets[0], param);
    nnd1->build();

    auto nnd2 = std::make_shared<nndescent::NNDescent>(datasets[1], param);
    nnd2->build();

    auto merged_dataset = Dataset::aggregate(datasets);

    {
        std::cout << "Max degree: " << param << std::endl;
        auto nnm = std::make_shared<nnmerge::NNMerge>(merged_dataset, param);
        nnm->Combine(nnd1, nnd2);
        recall(nnm, dataset);
    }
}

void
testSaveGraph(DatasetPtr &dataset) {
    auto nnd = std::make_shared<nndescent::NNDescent>(dataset, 32);
    nnd->build();

    auto &graph = nnd->extractGraph();
    saveGraph(graph, "nnd_" + dataset->getName());
}

int
main() {
    Log::setVerbose(true);

    auto dataset = Dataset::getInstance("crawl", "1m");

    testMerge(dataset);

    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
    return 0;
}