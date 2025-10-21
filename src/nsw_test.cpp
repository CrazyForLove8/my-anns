#include "annslib.h"

void
testBuild() {
    auto dataset = Dataset::getInstance("sift", "10k");
    auto index = std::make_shared<nsw::NSW>(getParam(dataset), 32, 200);
    index->build(dataset);
    recall(index, dataset, 200);
}

int
main() {
    Log::setVerbose(true);

    testBuild();

    return 0;
}