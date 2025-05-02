#include "annslib.h"

void
testBuild() {
    auto dataset = Dataset::getInstance("sift", "100k");

    auto index = std::make_shared<nsg::NSG>(dataset, 20, 200, 64);
    index->build();

    recall(index, dataset);
}

int
main() {
    Log::setVerbose(true);
    testBuild();
}