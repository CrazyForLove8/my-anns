#include <set>

#include "annslib.h"

using namespace hnsw;

IndexParam
getParam(const DatasetPtr& dataset) {
    const auto dim = dataset->getBase().dim();
    const auto metric = dataset->getDistance();

    IndexParam param;
    param.dim_ = dim;
    param.metric_ = metric;
    param.io_type_ = IOType::MEMORY_IO;
    return param;
}

void
testBuild() {
    auto dataset = Dataset::getInstance("sift", "10k");
    auto index = std::make_shared<HNSW>(getParam(dataset), 32, 200);
    index->build(dataset);
    recall(index, dataset);
}

int
main() {
    Log::setVerbose(true);

    testBuild();
    return 0;
}