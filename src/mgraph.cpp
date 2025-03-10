#include "mgraph.h"

MGraph::MGraph()
    : FGIM(),
      random_engine_(2024),
      enter_point_(0),
      reverse_(1 / log(1.0 * 20)),
      ef_construction_(200) {
}

MGraph::MGraph(unsigned int max_degree, unsigned int ef_construction, float sample_rate)
    : FGIM(max_degree, sample_rate),
      ef_construction_(ef_construction),
      random_engine_(2024),
      enter_point_(0),
      reverse_(1 / log(1.0 * max_degree)) {
}

void
MGraph::CrossQuery(std::vector<IndexPtr>& indexes) {
    Timer timer;
    timer.start();

    std::vector<std::reference_wrapper<HGraph>> graphs;
    for (auto& index : indexes) {
        auto hnsw_index = std::static_pointer_cast<hnsw::HNSW>(index);
        if (hnsw_index == nullptr) {
            throw std::runtime_error("Index is not HNSW, cannot extract HNSW graph");
        }
        graphs.emplace_back(hnsw_index->extractHGraph());
    }

    size_t offset = 0;
    std::vector<size_t> offsets;
    for (auto& g : graphs) {
        auto& graph_ref = g.get();
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < graph_ref[0].size(); ++i) {
            auto& neighbors = graph_ref[0][i].candidates_;
            graph_[0][i + offset].candidates_.reserve(max_degree_);
            for (size_t j = 0; j < neighbors.size() && j < max_degree_; ++j) {
                auto& neighbor = neighbors[j];
                graph_[0][i + offset].candidates_.emplace_back(
                    neighbor.id + offset, neighbor.distance, false);
            }
            std::make_heap(graph_[0][i + offset].candidates_.begin(),
                           graph_[0][i + offset].candidates_.end());
        }
        offset += graph_ref[0].size();
        offsets.emplace_back(offset);
    }

    unsigned L = max_degree_ * 2 / indexes.size();
#pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < oracle_->size(); ++u) {
        auto cur_graph_idx = std::lower_bound(offsets.begin(), offsets.end(), u) - offsets.begin();
        auto data = (*oracle_)[u];

        for (size_t graph_idx = 0; graph_idx < graphs.size(); graph_idx++) {
            if (graph_idx == cur_graph_idx) {
                continue;
            }
            auto _offset = graph_idx == 0 ? 0 : offsets[graph_idx - 1];
            auto& index = indexes[graph_idx];
            auto result = index->search(data, L, L);
            for (auto&& res : result) {
                graph_[0][u].insert(res.id + _offset, res.distance);
            }
        }
    }

    timer.end();
    logger << "Cross query time: " << timer.elapsed() << "s" << std::endl;
}

void
MGraph::Refinement() {
    Timer timer;

    timer.start();
    update_neighbors(graph_[0]);
    timer.end();
    logger << "Iterative update time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    connect_no_indegree(graph_[0]);
    timer.end();
    logger << "Connecting no indegree time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    prune(graph_[0]);
    timer.end();
    logger << "Pruning time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    add_reverse_edge(graph_[0]);
    timer.end();
    logger << "Adding reverse edge time: " << timer.elapsed() << "s" << std::endl;
}

void
MGraph::heuristic(Neighbors& candidates, unsigned max_degree) {
    if (candidates.size() <= max_degree) {
        return;
    }
    Neighbors ret_set;
    for (auto& v : candidates) {
        if (v.distance <= std::numeric_limits<float>::epsilon()) {
            continue;
        }
        bool prune = false;
        for (auto& w : ret_set) {
            if ((v.id == w.id) || (*oracle_)(v.id, w.id) < v.distance) {
                prune = true;
                break;
            }
        }
        if (!prune) {
            ret_set.emplace_back(v);
        }
        if (ret_set.size() >= max_degree_) {
            break;
        }
    }
    ret_set.erase(std::unique(ret_set.begin(), ret_set.end()), ret_set.end());
    candidates.swap(ret_set);
}

void
MGraph::ReconstructHGraph() {
    Timer timer;
    timer.start();

    for (int u = 0; u < oracle_->size(); u++) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        auto level = (int)(-log(distribution(random_engine_)) * reverse_);
        if (level == 0) {
            continue;
        }

        int cur_max_level_ = graph_.size() - 1;
        unsigned cur_node_ = enter_point_;
        for (auto i = cur_max_level_; i > level; --i) {
            auto res = knn_search(
                oracle_.get(), visited_list_pool_.get(), graph_[i], (*oracle_)[u], 1, 1, cur_node_);
            cur_node_ = res[0].id;
        }

        for (auto i = std::min(level, cur_max_level_); i > 0; --i) {
            Graph& graph = graph_[i];
            auto& candidates = graph[u].candidates_;
            auto res = knn_search(oracle_.get(),
                                  visited_list_pool_.get(),
                                  graph,
                                  (*oracle_)[u],
                                  ef_construction_,
                                  ef_construction_,
                                  cur_node_);

            candidates.swap(res);
            heuristic(candidates, max_degree_);

            for (auto& e : candidates) {
                graph[e.id].addNeighbor(Neighbor(u, e.distance, false));
                heuristic(graph[e.id].candidates_, max_degree_);
            }
            cur_node_ = candidates[0].id;
        }

        while (level > cur_max_level_) {
            graph_.emplace_back(oracle_->size());
            enter_point_ = u;
            ++cur_max_level_;
        }
    }

    timer.end();
    logger << "Reconstruct time: " << timer.elapsed() << "s" << std::endl;
}

void
MGraph::Combine(std::vector<IndexPtr>& indexes) {
    std::vector<DatasetPtr> datasets;
    for (auto& index : indexes) {
        datasets.emplace_back(index->extractDataset());
    }

    dataset_ = Dataset::aggregate(datasets);
    oracle_ = dataset_->getOracle();
    visited_list_pool_ = dataset_->getVisitedListPool();
    base_ = dataset_->getBasePtr();
    graph_.emplace_back(oracle_->size());

    Timer timer;
    timer.start();

    CrossQuery(indexes);

    Refinement();

    ReconstructHGraph();

    timer.end();
    logger << "Merging time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenHGraph(graph_);
    built_ = true;
}

Neighbors
MGraph::search(const float* query, unsigned int topk, unsigned int L) const {
    unsigned cur_node_ = enter_point_;
    for (int i = flatten_graph_.size() - 1; i > 0; --i) {
        auto res = graph::search(
            oracle_.get(), visited_list_pool_.get(), flatten_graph_[i], query, 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }
    auto res = graph::search(
        oracle_.get(), visited_list_pool_.get(), flatten_graph_[0], query, topk, L, cur_node_);
    return res;
}
