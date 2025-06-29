#include "annslib.h"

DatasetPtr get_dataset(const std::string& base_file,
                       const std::string& metric,
                       const std::string& query_file = "",
                        const std::string& gt_file = "") {
    DISTANCE metric_type;
    if (metric == "l2") {
        metric_type = DISTANCE::L2;
    } else if (metric == "cosine") {
        metric_type = DISTANCE::COSINE;
    } else if (metric == "hamming") {
        metric_type = DISTANCE::HAMMING;
    } else {
        throw std::invalid_argument("Unsupported metric type");
    }

    if (query_file.empty() && gt_file.empty()) {
        return Dataset::getInstance(base_file, metric_type);
    }
    return Dataset::getInstance(base_file, query_file, gt_file, metric_type);
}

void restore_distance(HGraph& graph, DatasetPtr& dataset) {
    auto& oracle = dataset->getOracle();
    for (int i = 0 ; i < static_cast<int>(graph.size()); ++i) {
        for (int j = 0 ; j < static_cast<int>(graph[i].size()); ++j) {
            for (auto& c : graph[i][j].candidates_) {
                c.distance = (*oracle)(j, c.id);
            }
        }
    }
}

void buildSubIndexes(const std::string& base_file,
                    const std::string& metric,
                    const std::string& output_path,
                    const int split_number) {
    Log::setVerbose(true);
    Log::redirect("/root/mount/my-anns/output/log");
    auto dataset = get_dataset(base_file, metric);
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, split_number);
    datasets.insert(datasets.begin(), dataset);

    int part_id = 0;
    std::string dataset_name = base_file.substr(base_file.find_last_of("/\\") + 1);
    for (auto& data : datasets) {
        auto index = std::make_shared<hnsw::HNSW>(data, 32, 200);
        index->build();
        std::string output_file = output_path + "/";
        output_file += dataset_name + "_" + std::to_string(split_number) + "_" + std::to_string(part_id);
        part_id++;
        saveHGraph(index->extractHGraph(),output_file);
    }
}

void mergeSubindexes(const std::string& base_file,
                    const std::string& metric,
                    const std::vector<std::string>& subindex_files,
                     const std::string& output_path) {
    auto dataset = get_dataset(base_file, metric);
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, subindex_files.size());

    HGraph hgraph;
    loadHGraph(hgraph, subindex_files[0]);

    restore_distance(hgraph, dataset);
    auto hnsw = std::make_shared<hnsw::HNSW>(dataset, hgraph);
    hnsw->set_max_neighbors(32);
    hnsw->set_ef_construction(200);

    omp_set_num_threads(1);
    for (auto& data : datasets) {
        hnsw->add(data);
    }

    std::string output_file = output_path + "/";
    std::string dataset_name = base_file.substr(base_file.find_last_of("/\\") + 1);
    output_file += dataset_name + "_" + std::to_string(subindex_files.size()) + "_hnsw_add_1";
    saveHGraph(hnsw->extractHGraph(), output_file);
}

void hnswAdd(const std::string& base_file, const std::string& query_file, const std::string& gt_file,
                    const std::string& metric, const int split_number,
                     const std::string& output_path) {
    Log::redirect("hnsw_add");
    auto dataset = get_dataset(base_file, metric, query_file, gt_file);
    int size = (int)dataset->getOracle()->size() / split_number;
    int remainder = (int)dataset->getOracle()->size() % split_number;

    auto hnsw = std::make_shared<hnsw::HNSW>(dataset, 32, 200);
    for (int i = 0 ; i < split_number; ++i) {
        if (i == 1) {
            omp_set_num_threads(1);
        }
        if (i == split_number - 1) {
            size += remainder;
        }
        hnsw->partial_build(size);
    }

    recall(hnsw, dataset);

    std::string output_file = output_path + "/";
    std::string dataset_name = dataset->getName();
    output_file += dataset_name + "_" + std::to_string(split_number) + "_hnsw_add_from_scratch";
    saveHGraph(hnsw->extractHGraph(), output_file);
}

void mergeSubindexesOurs(const std::string& base_file,
                    const std::string& metric,
                    const std::vector<std::string>& subindex_files,
                     const std::string& output_path) {
    auto dataset = get_dataset(base_file, metric);
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, subindex_files.size());
    datasets.insert( datasets.begin(), dataset);

    std::vector<IndexPtr> vec(subindex_files.size());
    for (size_t i = 0; i < subindex_files.size(); ++i) {
        HGraph g;
        loadHGraph(g, subindex_files[i]);
        restore_distance(g, datasets[i]);
        vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], g);
    }
    MGraph mgraph(20, 200);
    omp_set_num_threads(1);
    mgraph.Combine(vec);

    std::string output_file = output_path + "/";
    std::string dataset_name = base_file.substr(base_file.find_last_of("/\\") + 1);
    output_file += dataset_name + "_" + std::to_string(subindex_files.size()) + "_ours_1";
    saveHGraph(mgraph.extractHGraph(), output_file);
}

void loadAndTest(const std::string& base_file, const std::string& metric, const std::string& index_file, const std::string& query_file, const std::string& gt_file) {
    logger << "Testing index: " << index_file << std::endl;
    auto dataset = get_dataset(base_file, metric, query_file, gt_file);
    HGraph hgraph;
    loadHGraph(hgraph, index_file);
    auto index = std::make_shared<hnsw::HNSW>(dataset,hgraph);
    recall(index, dataset);
}

int main() {
    Log::setVerbose(true);

    // mergeSubindexes("/root/mount/dataset/siftsmall/siftsmall_base.fvecs",
    //                 "l2",
    //                 {"/root/mount/my-anns/output/siftsmall/index_part_siftsmall_base.fvecs_0.bin",
    //                  "/root/mount/my-anns/output/siftsmall/index_part_siftsmall_base.fvecs_1.bin"},
    //                 "/root/mount/my-anns/output/siftsmall");
    //
    // mergeSubindexesOurs("/root/mount/dataset/siftsmall/siftsmall_base.fvecs",
    //                 "l2",
    //                 {"/root/mount/my-anns/output/siftsmall/index_part_siftsmall_base.fvecs_0.bin",
    //                  "/root/mount/my-anns/output/siftsmall/index_part_siftsmall_base.fvecs_1.bin"},
    //                 "/root/mount/my-anns/output/siftsmall");

    // mergeSubindexes("/root/mount/dataset/internet_search/internet_search_train.fbin",
    //                 "l2",
    //                 {"/root/mount/my-anns/output/internet/index_part_internet_search_train.fbin_0.bin",
    //                  "/root/mount/my-anns/output/internet/index_part_internet_search_train.fbin_1.bin"},
    //                 "/root/mount/my-anns/output/internet");
    //
    // mergeSubindexesOurs("/root/mount/dataset/internet_search/internet_search_train.fbin",
    //                 "l2",
    //                 {"/root/mount/my-anns/output/internet/index_part_internet_search_train.fbin_0.bin",
    //                  "/root/mount/my-anns/output/internet/index_part_internet_search_train.fbin_1.bin"},
    //                 "/root/mount/my-anns/output/internet");

    // loadAndTest("/root/mount/dataset/internet_search/internet_search_train.fbin",
    //     "l2", "/root/mount/my-anns/output/internet/internet_search_train.fbin_2_hnsw_add_1.bin",
    //     "/root/mount/dataset/internet_search/internet_search_test.fbin",
    //     "/root/mount/dataset/internet_search/internet_search_neighbors.fbin");
    //
    // loadAndTest("/root/mount/dataset/internet_search/internet_search_train.fbin",
    //     "l2", "/root/mount/my-anns/output/internet/internet_search_train.fbin_2_ours_1.bin",
    //     "/root/mount/dataset/internet_search/internet_search_test.fbin",
    //     "/root/mount/dataset/internet_search/internet_search_neighbors.fbin");

    // hnswAdd("/root/mount/dataset/siftsmall/siftsmall_base.fvecs",
    //     "/root/mount/dataset/siftsmall/siftsmall_query.fvecs",
    //     "/root/mount/dataset/siftsmall/siftsmall_gt.ivecs", "l2", 2,
    //     "/root/mount/my-anns/output/siftsmall");

    hnswAdd("/root/mount/dataset/internet_search/internet_search_train.fbin",
        "/root/mount/dataset/internet_search/internet_search_test.fbin",
        "/root/mount/dataset/internet_search/internet_search_neighbors.fbin", "l2", 2,
        "/root/mount/my-anns/output/internet");
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