#include <set>

#include "annslib.h"

using namespace hnsw;

void
testBuild() {
    //    Log::redirect("16.hnsw_test");

    auto dataset = Dataset::getInstance("sift", "10k");
    auto index = std::make_shared<HNSW>(dataset, 32, 200);
    index->build();
    recall(index, dataset);

    //    saveHGraph(index->extract_hgraph(), "hnsw_" + dataset->getName());
}

void
testAdd() {
    DatasetPtr dataset = Dataset::getInstance("sift", "100k");
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 4);
    auto index = std::make_shared<HNSW>(dataset, 20, 200);
    index->build();

    for (auto& data : datasets) {
        index->add(data);
        std::cout << "After adding, the dataset has " << dataset->getBase().size() << " points"
                  << std::endl;
    }

    recall(index, dataset, 200);
}

void
hnsw_exp_observation() {
    Log::redirect("11.hnsw_exp_observation");
    DatasetPtr dataset = Dataset::getInstance("gist", "1m");
    auto index = std::make_shared<HNSW>(dataset, 20, 200);
    index->build();

    auto& ori_graph = index->extract_hgraph()[0];

    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 2);
    datasets.insert(datasets.begin(), dataset);

    auto idx_1 = std::make_shared<HNSW>(datasets[0], 20, 200);
    idx_1->build();
    auto& graph_1 = idx_1->extract_hgraph()[0];

    auto idx_2 = std::make_shared<HNSW>(datasets[1], 20, 200);
    idx_2->build();
    auto& graph_2 = idx_2->extract_hgraph()[0];

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
hnsw_exp_extend_observation() {
    Log::redirect("11.hnsw_exp_observation");
    DatasetPtr dataset = Dataset::getInstance("deep", "1m");
    auto index = std::make_shared<HNSW>(dataset, 20, 200);
    index->build();

    auto total_size = dataset->getOracle()->size();

    auto& ori_graph = index->extract_hgraph()[0];

    size_t idx_size = 5;
    std::cout << "Number of splits: " << idx_size << std::endl;

    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, idx_size);
    datasets.insert(datasets.begin(), dataset);

    std::vector<std::shared_ptr<HNSW> > indexes;
    std::vector<std::reference_wrapper<Graph> > graphs;
    std::vector<int> offsets;
    offsets.emplace_back(0);
    for (size_t i = 0; i < idx_size; i++) {
        auto idx = std::make_shared<HNSW>(datasets[i], 20, 200);
        idx->build();
        indexes.emplace_back(idx);
        graphs.emplace_back(idx->extract_hgraph()[0]);
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
test_save_and_load() {
    Log::redirect("hnsw_build_internet_search");
    DatasetPtr dataset =
        Dataset::getInstance("/root/mount/dataset/internet_search/internet_search_train.fbin",
                             "/root/mount/dataset/internet_search/internet_search_test.fbin",
                             "/root/mount/dataset/internet_search/internet_search_neighbors.fbin",
                             DISTANCE::L2);
    auto index = std::make_shared<HNSW>(dataset, 32, 200);
    index->build();

    logger << "HNSW index built successfully" << std::endl;
    recall(index, dataset);
    logger << "Saving HNSW index to disk" << std::endl;
    saveHGraph(index->extract_hgraph(), "/root/mount/my-anns/internet/hnsw_internet_search");
    logger << "HNSW index saved successfully" << std::endl;

    HGraph hgraph;
    logger << "Loading HNSW index from disk" << std::endl;
    loadHGraph(hgraph, "/root/mount/my-anns/internet/hnsw_internet_search");
    logger << "HNSW index loaded successfully" << std::endl;
    auto index_ = std::make_shared<HNSW>(dataset, hgraph);
    logger << "HNSW index created from loaded graph" << std::endl;
    recall(index, dataset);
}

void
test_partial_build() {
    DatasetPtr dataset = Dataset::getInstance("/root/mount/dataset/siftsmall/siftsmall_base.fvecs",
                                              "/root/mount/dataset/siftsmall/siftsmall_query.fvecs",
                                              "/root/mount/dataset/siftsmall/siftsmall_gt.ivecs",
                                              DISTANCE::L2);
    auto index = std::make_shared<HNSW>(dataset, 32, 200);
    int split = 3;
    int size = dataset->getOracle()->size() / split;
    int remainder = dataset->getOracle()->size() % split;
    for (int i = 0; i < split; ++i) {
        if (i == split - 1) {
            size += remainder;
        }
        index->partial_build(size);
    }
    logger << "HNSW index built successfully" << std::endl;
    recall(index, dataset);
}

void
test_enable_save_help() {
    auto dataset = Dataset::getInstance("sift", "100k");
    auto datasets = dataset->subsets(2);

    auto index1 = std::make_shared<hnsw::HNSW>(datasets[0], 32, 200);
    index1->build();

    auto path = saveHGraph(index1->extract_hgraph(), "index1.bin", index1->extract_params());
    auto hnsw = std::make_shared<hnsw::HNSW>(dataset, path);
    hnsw->set_save_helper({3, "hnsw_checkpoint.bin"});
    hnsw->partial_build();
}

void
test_load_checkpoint() {
    auto dataset = Dataset::getInstance("sift", "100k");

    auto path = "./graph_output/hnsw_checkpoint.bin";
    auto hnsw = std::make_shared<hnsw::HNSW>(dataset, path);
    hnsw->set_save_helper({4, "hnsw_checkpoint.bin"});
    hnsw->partial_build();
}

void
test_multi_thread() {
    int num = 12;
    DatasetPtr dataset = Dataset::getInstance("sift", "1m");
    Log::redirect("hnsw_multi_" + dataset->getName());
    auto index = std::make_shared<HNSW>(dataset, 20, 200);
    auto size = dataset->getOracle()->size();
    index->partial_build(size / 2);
    omp_set_num_threads(num);
    index->partial_build(size - size / 2);
    recall(index, dataset, 200);
}

int
main() {
    Log::setVerbose(true);

    test_multi_thread();
    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
    return 0;
}