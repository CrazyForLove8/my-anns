#include "annslib.h"

DatasetPtr
get_dataset(const std::string& base_file,
            const std::string& metric,
            const std::string& query_file = "",
            const std::string& gt_file = "") {
    DISTANCE metric_type;
    if (metric == "l2") {
        metric_type = DISTANCE::L2;
    } else if (metric == "cosine") {
        metric_type = DISTANCE::COSINE;
    } else {
        throw std::invalid_argument("Unsupported metric type");
    }

    if (query_file.empty() && gt_file.empty()) {
        return Dataset::getInstance(base_file, metric_type);
    }
    return Dataset::getInstance(base_file, query_file, gt_file, metric_type);
}

std::vector<std::string>
build_sub_indexes(const DatasetPtr& dataset,
                  const std::string& output_path,
                  const int split_number,
                  const int index_type = 0,
                  const int max_neighbors = 32,
                  const int ef_construction = 200,
                  const float alpha = 1.2) {
    auto datasets = dataset->subsets(split_number);

    std::vector<std::string> output_files;
    int part_id = 0;
    for (auto& data : datasets) {
        std::string output_file = output_path + "/";
        output_file += data->getName() + "_" + std::to_string(split_number) + "_" +
                       std::to_string(index_type) + "_" + std::to_string(part_id) + "_" +
                       std::to_string(max_neighbors) + "_" + std::to_string(ef_construction) + "_" +
                       std::to_string(alpha).substr(0, 3) + ".bin";
        std::cout << "Checking if " << output_file << " exists." << std::endl;
        if (std::filesystem::exists(output_file)) {
            std::cout << "Skip " << output_file << ", already exists." << std::endl;
            part_id++;
            output_files.emplace_back(output_file);
            continue;
        }
        part_id++;
        if (index_type == 0) {
            auto index = std::make_shared<hnsw::HNSW>(data, max_neighbors, ef_construction);
            index->build();
            output_files.emplace_back(saveHGraph(index->extractHGraph(), output_file));
        } else if (index_type == 1) {
            auto index =
                std::make_shared<diskann::Vamana>(data, alpha, ef_construction, max_neighbors);
            index->build();
            output_files.emplace_back(saveGraph(index->extractGraph(), output_file));
        } else {
            throw std::invalid_argument("Unsupported index type");
        }
    }
    return output_files;
}

void
hnsw_add(DatasetPtr& dataset,
         const std::string& index_file,
         const int split_number,
         const std::string& output_path,
         const int num_threads = 48,
         const int max_neighbors = 32,
         const int ef_construction = 200) {
    std::string output_file = output_path + "/";
    output_file += dataset->getName() + "_" + std::to_string(split_number) + "_baseline_hnsw_add" +
                   "_" + std::to_string(max_neighbors) + "_" + std::to_string(ef_construction) +
                   ".bin";
    if (std::filesystem::exists(output_file)) {
        std::cout << "Skip " << output_file << ", already exists." << std::endl;
        return;
    }

    HGraph hgraph;
    loadHGraph(hgraph, index_file, dataset->getOracle());
    auto hnsw = std::make_shared<hnsw::HNSW>(dataset, hgraph, true, max_neighbors, ef_construction);
    omp_set_num_threads(num_threads);
    hnsw->partial_build();

    recall(hnsw, dataset);

    saveHGraph(hnsw->extractHGraph(), output_file);
}

void
vamana_build(DatasetPtr& dataset,
             const std::string& output_path,
             const int num_threads = 48,
             const int ef_construction = 200,
             const int max_neighbors = 32,
             const float alpha = 1.2) {
    std::string output_file = output_path + "/";
    output_file += dataset->getName() + "_vamana_build" + "_" + std::to_string(ef_construction) +
                   "_" + std::to_string(max_neighbors) + "_" + std::to_string(alpha) + ".bin";
    if (std::filesystem::exists(output_file)) {
        std::cout << "Skip " << output_file << ", already exists." << std::endl;
        return;
    }

    auto vamana = std::make_shared<diskann::Vamana>(dataset, alpha, ef_construction, max_neighbors);
    omp_set_num_threads(num_threads);
    vamana->build();

    recall(vamana, dataset);

    saveGraph(vamana->extractGraph(), output_file);
}

void
mgraph_merge(DatasetPtr& dataset,
             const std::string& output_path,
             const std::vector<std::string>& subindex_files,
             const int num_threads = 48,
             const int index_type = 0,
             const int k = 20,
             const int ef_construction = 200) {
    std::string output_file = output_path + "/";
    output_file += dataset->getName() + "_" + std::to_string(subindex_files.size()) + "_ours" +
                   "_" + std::to_string(k) + "_index_type_" + std::to_string(index_type) + "_k_" +
                   std::to_string(k) + "_ef_" + std::to_string(ef_construction) + ".bin";
    if (std::filesystem::exists(output_file)) {
        std::cout << "Skip creating" << output_file << ", already exists." << std::endl;
        return;
    }

    auto datasets = dataset->subsets(subindex_files.size());

    std::vector<IndexPtr> vec(subindex_files.size());
    for (size_t i = 0; i < subindex_files.size(); ++i) {
        if (index_type == 0) {
            HGraph hgraph;
            loadHGraph(hgraph, subindex_files[i], datasets[i]->getOracle());
            vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], hgraph);
        } else if (index_type == 1) {
            Graph graph;
            loadGraph(graph, subindex_files[i], datasets[i]->getOracle());
            vec[i] = std::make_shared<IndexWrapper>(datasets[i], graph);
        } else {
            throw std::invalid_argument("Unsupported index type");
        }
    }
    MGraph mgraph(dataset, k, ef_construction);
    omp_set_num_threads(num_threads);
    mgraph.Combine(vec);

    recall(mgraph, dataset);

    saveHGraph(mgraph.extractHGraph(), output_file);
}

int
main(int argc, char* argv[]) {
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <base_file> <query_file> <gt_file> <metric> <output_path> "
                     "<split_number> <num_threads> <k> <max_neighbors> <ef_construction> <alpha>"
                  << std::endl;
        return 1;
    }

    std::string base_file = argv[1];
    std::string query_file = argv[2];
    std::string gt_file = argv[3];
    std::string metric = argv[4];
    std::string output_path = argv[5];

    int split_number = std::stoi(argv[6]);
    int num_threads = std::stoi(argv[7]);

    int k = std::stoi(argv[8]);
    int max_neighbors = std::stoi(argv[9]);
    int ef_construction = std::stoi(argv[10]);
    float alpha = std::stof(argv[11]);

    // std::filesystem::path originalPath(output_path);
    // if (std::filesystem::exists(originalPath)) {
    //     if (!std::filesystem::is_directory(originalPath)) {
    //         std::filesystem::path newDirectoryName = originalPath.filename().string() + "_" + Log::getTimestamp();
    //         std::filesystem::path newDirectoryPath = originalPath.parent_path() / newDirectoryName;
    //         std::filesystem::create_directories(newDirectoryPath);
    //         output_path = newDirectoryPath.string();
    //     } else {
    //         bool empty = true;
    //         for (const auto& entry : std::filesystem::directory_iterator(originalPath)) {
    //             if (entry.is_regular_file() || entry.is_directory()) {
    //                     empty = false;
    //                     break;
    //             }
    //         }
    //         if (!empty) {
    //             std::filesystem::path newDirectoryName = originalPath.filename().string() + "_" + Log::getTimestamp();
    //             std::filesystem::path newDirectoryPath = originalPath.parent_path() / newDirectoryName;
    //             std::filesystem::create_directories(newDirectoryPath);
    //             output_path = newDirectoryPath.string();
    //         }
    //     }
    // } else {
    //     std::filesystem::create_directories(originalPath);
    // }

    Log::setVerbose(true);
    Log::redirect(output_path);

    auto dataset = get_dataset(base_file, metric, query_file, gt_file);

    std::cout << "Step 1, build HNSW sub-indexes." << std::endl;
    auto hnsw_indexes =
        build_sub_indexes(dataset, output_path, split_number, 0, max_neighbors, ef_construction);
    std::cout << "------------------------------------" << std::endl;

    std::cout << "Step 2, use HNSW to insert smaller datasets into larger index." << std::endl;
    hnsw_add(dataset,
             hnsw_indexes[0],
             split_number,
             output_path,
             num_threads,
             max_neighbors,
             ef_construction);
    std::cout << "------------------------------------" << std::endl;

    std::cout << "Step 3, merge sub-indexes using our method." << std::endl;
    mgraph_merge(dataset, output_path, hnsw_indexes, num_threads, 0, k, ef_construction);
    std::cout << "------------------------------------" << std::endl;

    std::cout << "Step 4, build Vamana sub-indexes." << std::endl;
    auto vamana_indexes = build_sub_indexes(
        dataset, output_path, split_number, 1, max_neighbors, ef_construction, alpha);
    std::cout << "------------------------------------" << std::endl;

    std::cout << "Step 5, merge sub-indexes using our method." << std::endl;
    mgraph_merge(dataset, output_path, vamana_indexes, num_threads, 1, k, ef_construction);

    if (split_number == 2) {
        std::cout << "Step 6, reconstruct Vamana index on the full dataset." << std::endl;
        vamana_build(dataset, output_path, num_threads, ef_construction, max_neighbors, alpha);
    }

    std::cout << "Finished!" << std::endl;

    return 0;
}

// int
// main1() {
//     Log::setVerbose(true);
//
//     hnsw_add("/root/mount/dataset/siftsmall/siftsmall_base.fvecs",
//              "/root/mount/dataset/siftsmall/siftsmall_query.fvecs",
//              "/root/mount/dataset/siftsmall/siftsmall_gt.ivecs",
//              "l2",
//              "/root/mount/my-anns/output/siftsmall/index_part_siftsmall_base.fvecs_0.bin",
//              2,
//              "/root/mount/my-anns/output/siftsmall",
//              48);
//
//     // mgraphMerge("/root/mount/dataset/internet_search/internet_search_train.fbin",
//     //                 "l2",
//     //                 2,
//     //                 "/root/mount/dataset/internet_search/internet_search_test.fbin",
//     //                 "/root/mount/dataset/internet_search/internet_search_neighbors.fbin",
//     //                 "/root/mount/my-anns/output/internet",
//     //                 48,
//     //                 0);
//
//     // for (const auto& v : {3, 4, 5, 6, 7}) {
//     //     mgraphMerge("/root/mount/dataset/internet_search/internet_search_train.fbin",
//     //                 "l2",
//     //                 v,
//     //                 "/root/mount/dataset/internet_search/internet_search_test.fbin",
//     //                 "/root/mount/dataset/internet_search/internet_search_neighbors.fbin",
//     //                 "/root/mount/my-anns/output/internet",
//     //                 48,
//     //                 1);
//     // }
//     return 0;
// }
