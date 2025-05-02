#include "annslib.h"

void
testBuild(DatasetPtr& dataset) {
    auto nsw = std::make_shared<nsw::NSW>(dataset, 32, 200);
    nsw->build();

    recall(nsw, dataset);
}

void
testAdd(DatasetPtr& dataset) {
    std::vector<DatasetPtr> datasets;
    dataset->split(datasets, 2);

    auto nsw = std::make_shared<nsw::NSW>(dataset, 32, 200);

    nsw->build();

    nsw->add(datasets[0]);

    recall(nsw, dataset);
}

int
main() {
    Log::setVerbose(true);

    auto dataset = Dataset::getInstance("sift", "1m");

    testBuild(dataset);

    return 0;
}