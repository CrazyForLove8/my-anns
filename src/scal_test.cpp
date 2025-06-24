#include "annslib.h"

void buildSubIndexes(const std::string& base_file,
                const std::string& metric,
                const std::string& output_path,
                const int split_number) {
    DISTANCE metric_type;
    if (metric == "l2") {
        metric_type = DISTANCE::L2;
    } else if (metric == "cosine") {
        metric_type = DISTANCE::COSINE;
    } else if (metric == "hamming") {
        metric_type = DISTANCE::HAMMING;
    } else {
        std::cerr << "Unsupported metric: " << metric << std::endl;
        return;
    }

    auto dataset = Dataset::getInstance(base_file, metric_type);

    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, split_number);
    datasets.insert(datasets.begin(), dataset);

    int part_id = 0;
    std::string dataset_name = base_file.substr(base_file.find_last_of("/\\") + 1);
    for (auto& data : datasets) {
        auto index = std::make_shared<hnsw::HNSW>(data, 20, 200);
        index->build();
        std::string output_file = output_path + "/index_part_";
        output_file += dataset_name + "_" + std::to_string(part_id) + ".index";
        part_id++;
        saveHGraph(index->extractHGraph(),output_file);
    }
}

int main() {
    Log::setVerbose(true);

    buildSubIndexes("/root/mount/dataset/sift/learn.fvecs", "l2", );
}

// int
// main(int argc, char *argv[]) {
//     Log::setVerbose(true);
//     if (argc < 4) {
//         std::cerr << "Usage: " << argv[0] << " <base_file> <metric> <output_path> <split_number>" << std::endl;
//         return 1;
//     }
//
//     std::string base_file = argv[1];
//     std::string metric = argv[2];
//     std::string output_path = argv[3];
//     int split_number = 2;
//     if (argc > 4) {
//         split_number = std::stoi(argv[4]);
//     }
//
//     buildSubIndexes(base_file, metric, output_path, split_number);
//
//     return 0;
// }