#include "annslib.h"

int
main() {
    Log::setVerbose(true);

    auto dataset = Dataset::getInstance("sift", "1m");

    return 0;
}