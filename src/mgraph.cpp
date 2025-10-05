#include "mgraph.h"

MGraph::MGraph(const IndexParam& param,
               unsigned int max_degree,
               unsigned int ef_construction,
               float sample_rate)
    : FGIM(param, max_degree, sample_rate, false),
      ef_construction_(ef_construction),
      random_engine_(2024),
      enter_point_(0),
      max_level_(0),
      cur_max_level_(0),
      reverse_(1 / log(1.0 * max_degree)) {
}

Graph&
MGraph::extract_graph() {
    throw std::runtime_error(
        "MGraph does not support extract_graph, please use extract_hgraph instead");
}

HGraph&
MGraph::extract_hgraph() {
    return graph_;
}

void
MGraph::CrossQuery(std::vector<IndexPtr>& indexes) {
    Timer timer;
    timer.start();

    std::vector<std::reference_wrapper<Graph> > graphs;
    std::vector<std::reference_wrapper<HGraph> > hgraphs;
    bool isHGraph = true;
    for (auto& index : indexes) {
        auto hnsw_index = std::dynamic_pointer_cast<hnsw::HNSW>(index);
        if (hnsw_index == nullptr) {
            isHGraph = false;
            graphs.emplace_back(index->extract_graph());
        } else {
            hgraphs.emplace_back(hnsw_index->extract_hgraph());
        }
    }

    size_t offset = 0;
    offsets_.clear();
    if (isHGraph) {
        for (auto& g : hgraphs) {
            auto& graph_ref = g.get();
            auto graph_size = graph_ref[0].size();
            if (start_id_ == 0) {
#pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < graph_size; ++i) {
                    auto& neighbors = graph_ref[0][i].candidates_;
                    for (size_t j = 0; j < neighbors.size() && j < max_base_degree_; ++j) {
                        auto& neighbor = neighbors[j];
                        graph_[0][i + offset].candidates_.emplace_back(
                            neighbor.id + offset, neighbor.distance, true);
                    }
                    std::make_heap(graph_[0][i + offset].candidates_.begin(),
                                   graph_[0][i + offset].candidates_.end());
                }
            }
            offset += graph_size;
            max_index_size_ = std::max(max_index_size_, graph_size);
            offsets_.emplace_back(offset);
        }
    } else {
        for (auto& g : graphs) {
            auto& graph_ref = g.get();
            auto graph_size = graph_ref.size();
            if (start_id_ == 0) {
#pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < graph_size; ++i) {
                    auto& neighbors = graph_ref[i].candidates_;
                    for (size_t j = 0; j < neighbors.size() && j < max_base_degree_; ++j) {
                        auto& neighbor = neighbors[j];
                        graph_[0][i + offset].candidates_.emplace_back(
                            neighbor.id + offset, neighbor.distance, true);
                    }
                    std::make_heap(graph_[0][i + offset].candidates_.begin(),
                                   graph_[0][i + offset].candidates_.end());
                }
            }
            offset += graph_size;
            max_index_size_ = std::max(max_index_size_, graph_size);
            offsets_.emplace_back(offset);
        }
    }

    //    logger << "Performing Random Sampling" << std::endl;
    //    std::mt19937_64 rng(2024);
    //#pragma omp parallel for schedule(dynamic)
    //    for (int u = 0; u < oracle_->size(); ++u) {
    //        int cur = 0;
    //        while (cur < max_base_degree_) {
    //            int id = rng() % oracle_->size();
    //            if (id == u) {
    //                continue;
    //            }
    //            graph_[0][u].pushHeap(id, (*oracle_)(u, id));
    //            ++cur;
    //        }
    //    }

    logger << "Performing Cross Query" << std::endl;
    unsigned L = max_degree_ / (indexes.size() - 1);
    logger << "ef_construction: " << L << std::endl;
#pragma omp parallel for schedule(dynamic)
    for (auto u = start_id_; u < oracle_->size(); ++u) {
        const auto cur_graph_idx =
            std::upper_bound(offsets_.begin(), offsets_.end(), u) - offsets_.begin();
        const auto data = (*oracle_)[u];

        for (size_t graph_idx = 0; graph_idx < indexes.size(); graph_idx++) {
            if (graph_idx == cur_graph_idx) {
                continue;
            }
            const auto _offset = graph_idx == 0 ? 0 : offsets_[graph_idx - 1];
            const auto& index = indexes[graph_idx];
            auto result = index->search(data, L, L);
            for (auto&& res : result) {
                graph_[0][u].pushHeap(res.id + _offset, res.distance);
            }
        }

        if (save_helper_.should_save(u)) {
            logger << "Saving temporary index at point " << u << std::endl;
            logger << "Last save point: " << save_helper_.last_save_point << std::endl;
            logger << "Saving temporary index to " << save_helper_.save_path << std::endl;
            auto params = extract_params();
            params["phase"] = cur_phase_;
            params["save_point"] = u;
            saveGraph(graph_[0], save_helper_.save_path, params);
            logger << "Saved index at point " << u << std::endl;
        }
    }

    timer.end();
    logger << "Cross query time: " << timer.elapsed() << "s" << std::endl;
    logger << "Cross query finished" << std::endl;
}

void
MGraph::Refinement() {
    cur_phase_ = "2";
    Timer timer;

    timer.start();
    update_neighbors(graph_[0]);
    timer.end();
    logger << "Iterative update time: " << timer.elapsed() << "s" << std::endl;

    // timer.start();
    // connect_no_indegree(graph_[0]);
    // timer.end();
    // logger << "Connecting no indegree time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    prune(graph_[0]);
    timer.end();
    logger << "Pruning time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    add_reverse_edge(graph_[0]);
    timer.end();
    logger << "Adding reverse edge time: " << timer.elapsed() << "s" << std::endl;
    logger << "Refinement finished" << std::endl;
}

void
MGraph::heuristic(Neighbors& candidates, unsigned max_degree) {
    if (candidates.size() <= max_degree) {
        return;
    }
    Neighbors ret_set;
    for (auto& v : candidates) {
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
        if (ret_set.size() >= max_degree) {
            break;
        }
    }
    candidates.swap(ret_set);
}

void
MGraph::resize(IdType new_size) {
    visited_list_pool_ = VisitedListPool::getInstance(new_size);

    levels_.reserve(oracle_->size());
    levels_.resize(oracle_->size(), 0);

    auto total = oracle_->size();
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (auto i = cur_size_; i < total; i++) {
        levels_[i] = (int)(-log(distribution(random_engine_)) * reverse_);
        max_level_ = std::max(max_level_, levels_[i]);
    }

    graph_.reserve(max_level_ + 1);
    for (auto i = graph_.size(); i <= max_level_; ++i) {
        graph_.emplace_back(total);
    }

    auto& base_layer = graph_[0];
    for (auto i = cur_size_; i < total; ++i) {
        base_layer[i].candidates_.reserve(max_base_degree_);
        for (int level = 1; level <= levels_[i]; ++level) {
            graph_[level][i].candidates_.reserve(max_degree_);
        }
    }
}

void
MGraph::ReconstructHGraph() {
    Timer timer;
    timer.start();

#pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < oracle_->size(); u++) {
        int level = levels_[u];
        if (level == 0) {
            continue;
        }

        std::lock_guard<std::mutex> guard(graph_[0][u].lock_);
        std::unique_lock<std::mutex> graph_lock(graph_lock_);
        int max_level_copy = cur_max_level_;
        if (level <= max_level_copy) {
            graph_lock.unlock();
        }

        uint32_t cur_node_ = enter_point_;
        for (auto i = max_level_copy; i > level; --i) {
            auto res = search_hgraph_layer(
                oracle_.get(), visited_list_pool_.get(), graph_, i, (*oracle_)[u], 1, 1, cur_node_);
            cur_node_ = res[0].id;
        }

        for (auto i = std::min(level, max_level_copy); i > 0; --i) {
            auto res = search_hgraph_layer(oracle_.get(),
                                           visited_list_pool_.get(),
                                           graph_,
                                           i,
                                           (*oracle_)[u],
                                           ef_construction_,
                                           ef_construction_,
                                           cur_node_);

            res.erase(std::remove_if(
                          res.begin(), res.end(), [u](const Neighbor& n) { return n.id == u; }),
                      res.end());
            res.erase(std::unique(res.begin(), res.end()), res.end());
            heuristic(res, max_degree_);

            auto& graph = graph_[i];
            auto& candidates = graph[u].candidates_;
            candidates.swap(res);
            for (auto& e : candidates) {
                std::lock_guard<std::mutex> lock(graph_[0][e.id].lock_);
                graph[e.id].addNeighbor(Neighbor(u, e.distance, false));
                heuristic(graph[e.id].candidates_, max_degree_);
            }
            cur_node_ = candidates[0].id;
        }

        if (level > max_level_copy) {
            enter_point_ = u;
            cur_max_level_ = level;
        }
    }

    timer.end();
    logger << "Reconstruct time: " << timer.elapsed() << "s" << std::endl;
}

void
MGraph::combine(std::vector<IndexPtr>& indexes) {
    IdType total_size = 0;
    for (const auto& index : indexes) {
        const auto vec_ptr = index->extract_vectors();
        oracle_->insert(vec_ptr, total_size);
        total_size += vec_ptr->size();
    }
    this->resize(total_size);

    print_info();

    Timer timer;
    timer.start();

    CrossQuery(indexes);
    print_memory_usage();

    logger << "Max Index Size " << max_index_size_ << std::endl;
    logger << "Offsets:";
    for (auto& v : offsets_) {
        logger << " " << v;
    }
    logger << std::endl;
    Refinement();
    print_memory_usage();

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
        auto res = search_flatten_graph(
            oracle_.get(), visited_list_pool_.get(), flatten_graph_[i], query, 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }
    auto res = search_flatten_graph(
        oracle_.get(), visited_list_pool_.get(), flatten_graph_[0], query, topk, L, cur_node_);
    return res;
}
void
MGraph::print_info() const {
    FGIM::print_info();
    logger << "MGraph Info:" << std::endl;
    logger << "Ef Construction: " << ef_construction_ << std::endl;
}

ParamMap
MGraph::extract_params() {
    auto params = FGIM::extract_params();
    params["index_type"] = "MGraph";
    params["ef_construction"] = (uint64_t)ef_construction_;
    return params;
}
void
MGraph::load_params(const ParamMap& params) {
    if (params.find("ef_construction") != params.end()) {
        ef_construction_ = std::get<uint64_t>(params.at("ef_construction"));
    }
    if (params.find("max_degree") != params.end()) {
        max_degree_ = std::get<uint64_t>(params.at("max_degree"));
        max_base_degree_ = max_degree_ * 2;
        reverse_ = 1 / log(1.0 * max_degree_);
    }
    if (params.find("sample_rate") != params.end()) {
        sample_rate_ = std::get<double_t>(params.at("sample_rate"));
    }
}
