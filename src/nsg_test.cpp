#include "annslib.h"

void
testBuildNSG() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto index = std::make_shared<nsg::NSG>(getParam(dataset), 32, 200, 32);
    index->build(dataset);
    recall(index, dataset);
}

int
main() {
    Log::setVerbose(true);
}