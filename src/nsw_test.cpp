#include "annslib.h"

void
testBuild() {
    auto dataset = Dataset::getInstance("sift", "1m");
    auto index = std::make_shared<nsw::NSW>(getParam(dataset), 32, 200);
    index->build(dataset);
    recall(index, dataset);
}

void
testAdd() {
    auto ds = {"sift", "deep", "msong", "glove", "gist", "crawl"};
    for (auto d : ds) {
        auto dataset = Dataset::getInstance(d, "1m");
        auto m = std::string(d) == "sift" or std::string(d) == "msong" or std::string(d) == "deep" ? 16 : 32;
        Log::redirect("nsw_baseline_" + dataset->getName());
        auto datasets = dataset->subsets(2);

        omp_set_num_threads(1);
        auto index = std::make_shared<nsw::NSW>(getParam(datasets[0]), m * 2, 200);
        Timer timer;
        timer.start();
        index->build(datasets[0]);
        index->add(datasets[1]);
        timer.end();
        logger << "Total time (build + add): " << timer.elapsed() << "s" << std::endl;
    }
}

int
main() {
    Log::setVerbose(true);

    testBuild();

    return 0;
}