#include <set>

#include "annslib.h"

void
testBuild() {
//    Log::redirect("4.1_vamana_baseline");
    auto dataset = Dataset::getInstance("deep", "1m");

    auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, 32);
    index->build();

    recall(index, dataset);
}

void
buildExp2_4() {
    auto dataset = Dataset::getInstance("sift", "200k");
    Log::redirect("2.4_" + dataset->getName() + "_baseline");

    {
        omp_set_num_threads(1);
        auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, 32);
        index->build();
    }
}

void
buildExp3_2() {
    auto dataset = Dataset::getInstance("gist", "1m");
    Log::redirect("3.2_vamana_" + dataset->getName() + "_baseline");

    {
        omp_set_num_threads(1);
        auto params = {32, 36, 40, 44, 48, 52, 56, 60, 64};
        for (auto param : params) {
            std::cout << "Parameter: Max degree: " << param << std::endl;
            auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, param);
            index->build();

            recall(index, dataset, 200);
        }
    }
}

void
vamana_exp_observation() {
    //    Log::redirect("11.vamana_exp_observation");
    DatasetPtr dataset = Dataset::getInstance("sift", "10k");
    auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, 80);
    index->build();

    recall(index, dataset, 20);

    auto& ori_graph = index->extract_graph();

    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 2);
    datasets.insert(datasets.begin(), dataset);

    auto idx_1 = std::make_shared<diskann::Vamana>(datasets[0], 1.2, 200, 80);
    idx_1->build();
    auto& graph_1 = idx_1->extract_graph();

    auto idx_2 = std::make_shared<diskann::Vamana>(datasets[1], 1.2, 200, 80);
    idx_2->build();
    auto& graph_2 = idx_2->extract_graph();

    double hits = 0;
    double new_local = 0, new_cross = 0;
    double total = 0;
    for (size_t i = 0; i < datasets[0]->getOracle()->size(); i++) {
        auto& neighbors = ori_graph[i].candidates_;
        std::set<int> neighbor_ids;
        for (auto& neighbor : neighbors) {
            neighbor_ids.insert(neighbor.id);
        }
        total += neighbor_ids.size();
        std::set<int> visited;
        auto& neighbors_1 = graph_1[i].candidates_;
        for (auto& neighbor : neighbors_1) {
            if (neighbor_ids.find(neighbor.id) != neighbor_ids.end()) {
                hits++;
                visited.insert(neighbor.id);
            }
        }

        for (auto& neighbor : neighbors) {
            if (visited.find(neighbor.id) == visited.end()) {
                if ((size_t)neighbor.id < datasets[0]->getOracle()->size()) {
                    new_local++;
                } else {
                    new_cross++;
                }
            }
        }
    }

    std::cout << std::fixed;
    std::cout << "Hits: " << hits << " Total: " << total << std::endl;
    std::cout << "Hits / Total: " << (double)hits / total << std::endl;
    std::cout << "Misses / Total: " << 1 - (double)hits / total << std::endl;
    std::cout << "New local: " << new_local << " New cross: " << new_cross << std::endl;
    std::cout << "New cross: " << new_cross << " Total: " << total << std::endl;
    std::cout << "New local / Total: " << new_local / total << std::endl;
    std::cout << "New cross / Total: " << new_cross / total << std::endl;

    for (size_t i = 0; i < datasets[1]->getOracle()->size(); i++) {
        auto& neighbors = ori_graph[i + datasets[0]->getOracle()->size()].candidates_;
        std::set<int> neighbor_ids;
        for (auto& neighbor : neighbors) {
            neighbor_ids.insert(neighbor.id);
        }
        total += neighbor_ids.size();
        std::set<int> visited;
        auto& neighbors_2 = graph_2[i].candidates_;
        for (auto& neighbor : neighbors_2) {
            if (neighbor_ids.find(neighbor.id + datasets[0]->getOracle()->size()) !=
                neighbor_ids.end()) {
                hits++;
                visited.insert(neighbor.id + datasets[0]->getOracle()->size());
            }
        }

        for (auto& neighbor : neighbors) {
            if (visited.find(neighbor.id) == visited.end()) {
                if ((size_t)neighbor.id < datasets[0]->getOracle()->size()) {
                    new_cross++;
                } else {
                    new_local++;
                }
            }
        }
    }

    std::cout << std::fixed;
    std::cout << "Hits: " << hits << " Total: " << total << std::endl;
    std::cout << "Hits / Total: " << (double)hits / total << std::endl;
    std::cout << "Misses / Total: " << 1 - (double)hits / total << std::endl;
    std::cout << "New local: " << new_local << " New cross: " << new_cross << std::endl;
    std::cout << "New cross: " << new_cross << " Total: " << total << std::endl;
    std::cout << "New local / Total: " << new_local / total << std::endl;
    std::cout << "New cross / Total: " << new_cross / total << std::endl;
}

void
testPartialBuild() {
    auto dataset = Dataset::getInstance("sift", "10k");

    auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, 64);
    auto permutation = std::vector<uint32_t>{1, 2, 3, 4};
    index->partial_build(permutation);

    auto& graph = index->extract_graph();
    std::cout << graph.size() << std::endl;
    recall(index, dataset);
}

void
testMerge() {
    auto dataset = Dataset::getInstance("sift", "1m");
    Log::redirect("13_" + dataset->getName() + "_merge");

    {
        omp_set_num_threads(1);
        std::cout << "Parameter: Max degree: " << 64 << std::endl;
        auto index = std::make_shared<diskann::DiskANN>(dataset, 1.2, 200, 64, 4, 2);
        index->build();
        recall(index, dataset);
        auto& graph = index->extract_graph();
        saveGraph(graph, dataset->getName() + "_diskann");
    }
}

void
testDiskANNMerge() {
    auto dataset = Dataset::getInstance("crawl", "1m");
    Log::redirect("14_" + dataset->getName() + "_merge");
    {
        int param = 64;

        auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, param);

        index->build();

        auto& graph = index->extract_graph();
        for (auto& u : graph) {
            u.candidates_.insert(u.candidates_.end(), u.candidates_.begin(), u.candidates_.end());
            std::sort(u.candidates_.begin(), u.candidates_.end());
        }
        auto& flatten = index->extract_flatten_graph();
        flatten = FlattenGraph(graph);

        recall(index, dataset);
    }
}

void
vamana_exp_extend_observation() {
    Log::redirect("11.vamana_exp_observation");
    DatasetPtr dataset = Dataset::getInstance("deep", "1m");
    auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, 40);
    index->build();

    auto total_size = dataset->getOracle()->size();

    auto& ori_graph = index->extract_graph();

    size_t idx_size = 5;
    std::cout << "Number of splits: " << idx_size << std::endl;

    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, idx_size);
    datasets.insert(datasets.begin(), dataset);

    std::vector<std::shared_ptr<diskann::Vamana> > indexes;
    std::vector<std::reference_wrapper<Graph> > graphs;
    std::vector<int> offsets;
    offsets.emplace_back(0);
    for (size_t i = 0; i < idx_size; i++) {
        auto idx = std::make_shared<diskann::Vamana>(datasets[i], 1.2, 200, 40);
        idx->build();
        indexes.emplace_back(idx);
        graphs.emplace_back(idx->extract_graph());
        offsets.emplace_back(datasets[i]->getOracle()->size() + offsets[i]);
    }

    double hits = 0;
    double new_local = 0, new_cross = 0;
    double total = 0;
    for (int i = 0; i < (int)total_size; i++) {
        auto& neighbors = ori_graph[i].candidates_;
        std::set<int> neighbor_ids;
        for (auto& neighbor : neighbors) {
            neighbor_ids.insert(neighbor.id);
        }
        total += neighbor_ids.size();
        std::set<int> visited;

        auto graph_idx = std::upper_bound(offsets.begin(), offsets.end(), i) - offsets.begin();
        graph_idx = graph_idx == 0 ? 0 : graph_idx - 1;
        auto& sub_neighbors = graphs[graph_idx].get()[i - offsets[graph_idx]].candidates_;
        for (auto& neighbor : sub_neighbors) {
            if (neighbor_ids.find(neighbor.id + offsets[graph_idx]) != neighbor_ids.end()) {
                hits++;
                visited.insert(neighbor.id + offsets[graph_idx]);
            }
        }

        for (auto& neighbor : neighbors) {
            if (visited.find(neighbor.id) == visited.end()) {
                size_t upper_bound = offsets[graph_idx + 1];
                size_t lower_bound = offsets[graph_idx];
                if ((size_t)neighbor.id >= lower_bound && (size_t)neighbor.id < upper_bound) {
                    new_local++;
                } else {
                    new_cross++;
                }
            }
        }
    }

    std::cout << std::fixed;
    std::cout << "Hits: " << hits << " Total: " << total << std::endl;
    std::cout << "Hits / Total: " << (double)hits / total << std::endl;
    std::cout << "Misses / Total: " << 1 - (double)hits / total << std::endl;
    std::cout << "New local: " << new_local << " New cross: " << new_cross << std::endl;
    std::cout << "New cross: " << new_cross << " Total: " << total << std::endl;
    std::cout << "New local / Total: " << new_local / total << std::endl;
    std::cout << "New cross / Total: " << new_cross / total << std::endl;
}

void
testSimpleMerge() {
    Log::redirect("17_simple_merge_added");
    auto dataset = Dataset::getInstance("crawl", "1m");

    std::vector<DatasetPtr> datasets;
    dataset->split(datasets, 2);
    datasets.insert(datasets.begin(), dataset);

    {
        int param = 80;

        std::shared_ptr<Index> index =
            std::make_shared<diskann::Vamana>(datasets[0], 1.2, 200, param);
        index->build();

        std::shared_ptr<Index> index2 =
            std::make_shared<diskann::Vamana>(datasets[1], 1.2, 200, param);
        index2->build();

        auto index_wrapper = std::make_shared<IndexWrapper>(index);
        std::vector<IndexPtr> indexes = {index2};
        index_wrapper->append(indexes);

        auto& graph = index_wrapper->extract_graph();
        auto& oracle = *(index_wrapper->extract_dataset()->getOracle());
        std::mt19937 rng(2024);
        for (int i = 0; i < (int)graph.size(); i++) {
            if (i < (int)datasets[0]->getOracle()->size()) {
                while (graph[i].candidates_.size() < 40) {
                    int cross =
                        rng() % datasets[1]->getOracle()->size() + datasets[0]->getOracle()->size();
                    graph[i].candidates_.emplace_back(cross, oracle(i, cross), false);
                }
                std::sort(graph[i].candidates_.begin(), graph[i].candidates_.end());
            } else {
                while (graph[i].candidates_.size() < 40) {
                    int local = rng() % datasets[0]->getOracle()->size();
                    graph[i].candidates_.emplace_back(local, oracle(i, local), false);
                }
                std::sort(graph[i].candidates_.begin(), graph[i].candidates_.end());
            }
        }

        recall(index_wrapper, datasets[0]);
        saveGraph(index_wrapper->extract_graph(), "vamana-merge-add-" + datasets[0]->getName());
    }
}

void
test_save() {
    DatasetPtr dataset = Dataset::getInstance("/root/mount/dataset/siftsmall/siftsmall_base.fvecs",
                                              "/root/mount/dataset/siftsmall/siftsmall_query.fvecs",
                                              "/root/mount/dataset/siftsmall/siftsmall_gt.ivecs",
                                              DISTANCE::L2);
    auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, 40);
    index->build();

    recall(index, dataset);
    saveGraph(index->extract_graph(), "/root/mount/my-anns/output/siftsmall/vamana");

    Graph graph;
    loadGraph(graph, "/root/mount/my-anns/output/siftsmall/vamana");
    auto index_ = std::make_shared<Index>(dataset, graph);
    recall(index, dataset);
}

void
test_vamana_partial_build_disk() {
    auto dataset = Dataset::getInstance("msong", "1m", true);
    Log::redirect("freshvamana_disk" + dataset->getName());

    auto size = dataset->getOracle()->size();
    auto index = std::make_shared<diskann::Vamana>(dataset, 1.2, 200, 32);
    index->partial_build(size / 2);
    index->partial_build(size - size / 2);

    recall(index, dataset);
}

void
test_diskann_build_in_memory() {
    print_memory_usage();
    auto dataset = Dataset::getInstance("sift", "1m");
    auto index = std::make_shared<diskann::DiskANN>(dataset, 1.2, 200, 32, 40, 2);
    index->build();
    recall(index, dataset);
    print_memory_usage();
}

void
test_diskann_build_with_ssd() {
    print_memory_usage();
    auto dataset = Dataset::getInstance("sift", "1m", true);
    auto index = std::make_shared<diskann::DiskANN>(dataset, 1.2, 200, 32, 40, 2);
    index->build();
    recall(index, dataset);
    print_memory_usage();
}

int
main() {
    Log::setVerbose(true);

    test_vamana_partial_build_disk();
    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
    return 0;
}