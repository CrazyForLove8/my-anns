#include "annslib.h"

void
testKMeans() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto kmeans = std::make_shared<Kmeans>(dataset, 4);
    kmeans->Run();

    std::cout << "1 Center of 0: " << kmeans->NearestCenter(0, 1)[0] << std::endl;
    std::cout << "2 Center of 0: " << kmeans->NearestCenter(0, 2)[1] << std::endl;
    std::cout << "3 Center of 0: " << kmeans->NearestCenter(0, 3)[2] << std::endl;

    std::cout << "1 Center of 1: " << kmeans->NearestCenter(1, 1)[0] << std::endl;
    std::cout << "2 Center of 1: " << kmeans->NearestCenter(1, 2)[1] << std::endl;
    std::cout << "3 Center of 1: " << kmeans->NearestCenter(1, 3)[2] << std::endl;
}

int
main() {
    Log::setVerbose(true);

    testKMeans();

    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
    return 0;
}