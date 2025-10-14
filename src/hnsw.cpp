#include "hnsw.h"

using namespace graph;

hnsw::HNSW::HNSW(const IndexParam& param, const int max_neighbors, const int ef_construction)
    : max_neighbors_(max_neighbors),
      max_base_neighbors_(max_neighbors * 2),
      ef_construction_(ef_construction),
      Index(param, false) {
    visited_table_ = std::unordered_set<int>();
    random_engine_.seed(2024);
    reverse_ = 1 / log(1.0 * max_neighbors_);
}

hnsw::HNSW::HNSW(
    const IndexParam& param, HGraph& graph, bool partial, int max_neighbors, int ef_construction)
    : Index(param, false),
      graph_(std::move(graph)),
      max_neighbors_(max_neighbors),
      max_base_neighbors_(max_neighbors * 2),
      ef_construction_(ef_construction) {
    random_engine_.seed(2024);
    reverse_ = 1 / log(1.0 * max_neighbors_);
    for (int i = graph_.size() - 1; i >= 0; --i) {
        bool found = false;
        for (int j = 0; j < graph_[i].size(); ++j) {
            if (!graph_[i][j].candidates_.empty()) {
                enter_point_ = j;
                found = true;
                break;
            }
        }
        if (found) {
            cur_max_level_ = max_level_ = i;
            break;
        }
    }

    if (!partial) {
        flatten_graph_ = FlattenHGraph(graph_);
        built_ = true;
    } else {
        levels_.reserve(oracle_->size());
        levels_.resize(oracle_->size(), 0);
        cur_size_ = graph_[0].size();

        int total = oracle_->size();

        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int i = cur_size_; i < total; i++) {
            levels_[i] = (int)(-log(distribution(random_engine_)) * reverse_);
            max_level_ = std::max(max_level_, levels_[i]);
        }

        graph_.reserve(max_level_ + 1);
        graph_.resize(max_level_ + 1);
        for (int i = 0; i <= max_level_; ++i) {
            graph_[i].resize(total);
        }
    }
}

void
hnsw::HNSW::addPoint(IdType index) {
    std::lock_guard<std::mutex> guard(graph_[0][index].lock_);

    int level = levels_[index];
    std::unique_lock<std::mutex> graph_lock(graph_lock_);
    int max_level_copy = cur_max_level_;
    if (level <= max_level_copy) {
        graph_lock.unlock();
    }

    uint32_t cur_node_ = enter_point_;
    for (auto i = max_level_copy; i > level; --i) {
        auto res = search_hgraph_layer(
            oracle_.get(), visited_list_pool_.get(), graph_, i, (*oracle_)[index], 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }

    for (auto i = std::min(level, max_level_copy); i >= 0; --i) {
        auto res = search_hgraph_layer(oracle_.get(),
                                       visited_list_pool_.get(),
                                       graph_,
                                       i,
                                       (*oracle_)[index],
                                       ef_construction_,
                                       ef_construction_,
                                       cur_node_);

        res.erase(std::remove_if(
                      res.begin(), res.end(), [index](const Neighbor& n) { return n.id == index; }),
                  res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        auto cur_max_cnt = i ? max_neighbors_ : max_base_neighbors_;
        prune(res, cur_max_cnt);

        auto& graph = graph_[i];
        auto& candidates = graph[index].candidates_;
        candidates.swap(res);
        for (auto& e : candidates) {
            std::lock_guard<std::mutex> lock(graph_[0][e.id].lock_);
            graph[e.id].addNeighbor(Neighbor(index, e.distance, false));
            prune(graph[e.id].candidates_, cur_max_cnt);
        }
        cur_node_ = candidates[0].id;
    }

    if (level > max_level_copy) {
        enter_point_ = index;
        cur_max_level_ = level;
    }
}

void
hnsw::HNSW::prune(Neighbors& candidates, IdType max_neighbors) {
    if (candidates.size() <= max_neighbors) {
        return;
    }
    Neighbors ret_set;
    for (auto& v : candidates) {
        bool prune = false;
        for (auto& w : ret_set) {
            if ((*oracle_)(v.id, w.id) < v.distance) {
                prune = true;
                break;
            }
        }
        if (!prune) {
            ret_set.emplace_back(v);
        }
        if (ret_set.size() >= max_neighbors) {
            break;
        }
    }
    candidates.swap(ret_set);
}

Neighbors
hnsw::HNSW::search(const float* query, unsigned int topk, unsigned int L) const {
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
hnsw::HNSW::print_info() const {
    Index::print_info();
    logger << "HNSW Index Info:" << std::endl;
    logger << "Max Neighbors: " << max_neighbors_ << std::endl;
    logger << "Max Base Neighbors: " << max_base_neighbors_ << std::endl;
    logger << "EF Construction: " << ef_construction_ << std::endl;
    logger << "Max Level: " << max_level_ << std::endl;
    logger << "Current Max Level: " << cur_max_level_ << std::endl;
    logger << "Enter Centroid: " << enter_point_ << std::endl;

    if (max_neighbors_ == 0 || max_base_neighbors_ == 0 || ef_construction_ == 0) {
        logger << "Warning: max_neighbors, max_base_neighbors, and ef_construction should be "
                  "greater than 0."
               << std::endl;
    }
}

Graph&
hnsw::HNSW::extract_graph() {
    throw std::runtime_error(
        "HNSW does not support extract_graph, please use extract_hgraph instead");
}

HGraph&
hnsw::HNSW::extract_hgraph() {
    if (!built_) {
        throw std::runtime_error("Index is not built yet");
    }
    return graph_;
}

void
hnsw::HNSW::build_internal(DatasetPtr& dataset) {
    const auto start = cur_size_ == 0 ? 1 : cur_size_;
    this->partial_build(start, oracle_->size());
}

void
hnsw::HNSW::partial_build(IdType start, IdType end) {
    logger << "Adding from " << start << " to " << end << std::endl;
    Timer timer;
    timer.start();
    {
#pragma omp parallel for schedule(dynamic)
        for (auto i = start; i < end; ++i) {
            if (save_helper_.should_save(i)) {
                logger << "Saving temporary index to " << save_helper_.save_path << std::endl;
                auto params = extract_params();
                params["save_point"] = i;
                saveHGraph(graph_, save_helper_.save_path, params);
                logger << "Saved index at point " << i << std::endl;
            }

            if (i % 100000 == 0) {
                logger << "Adding " << i << " / " << end << std::endl;
            }
            addPoint(i);
        }
    }
    timer.end();
    logger << "Adding time: " << timer.elapsed() << "s" << std::endl;
}

void
hnsw::HNSW::resize(const IdType new_size) {
    visited_list_pool_ = VisitedListPool::getInstance(new_size);

    levels_.resize(oracle_->size(), 0);

    auto total = oracle_->size();
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (auto i = cur_size_; i < total; i++) {
        levels_[i] = (int)(-log(distribution(random_engine_)) * reverse_);
        max_level_ = std::max(max_level_, levels_[i]);
    }

    if (graph_.size() <= max_level_) {
        graph_.resize(max_level_ + 1);
    }
    for (auto& g : graph_) {
        g.resize(total);
    }
}

void
hnsw::HNSW::partial_build(IdType num) {
    print_info();
    auto start = cur_size_ == 0 ? 1 : cur_size_;
    auto end = cur_size_ + num;
    if (num == 0) {
        end = oracle_->size();
        num = end - start;
        if (num == 0) {
            logger << "No points to add, skipping build." << std::endl;
            return;
        }
    }
    if (end > oracle_->size()) {
        num = oracle_->size() - start;
        end = oracle_->size();
    }

    this->partial_build(start, end);

    cur_size_ += num;
    if (cur_size_ == oracle_->size()) {
        logger << "Partial build completed, total size: " << cur_size_ << std::endl;
        flatten_graph_ = FlattenHGraph(graph_);
        built_ = true;
        logger << "Index built successfully." << std::endl;
    } else {
        logger << "Partial build completed, total size: " << cur_size_
               << ", but not all points are added yet." << std::endl;
        built_ = false;
    }
}

void
hnsw::HNSW::build(DatasetPtr& dataset) {
    print_info();
    if (built_) {
        logger << "Index is already built, skipping build." << std::endl;
        return;
    }

    oracle_->insert(dataset->getBasePtr());
    this->resize(oracle_->size());

    Timer timer;
    timer.start();

    build_internal(dataset);

    timer.end();
    logger << "Indexing time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenHGraph(graph_);
    built_ = true;
    cur_size_ += oracle_->size();
    logger << "HNSW Index built successfully." << std::endl;
}

void
hnsw::HNSW::add(DatasetPtr& dataset) {
    if (!built_) {
        throw std::runtime_error("Index is not built yet");
    }
    built_ = false;

    Timer timer;
    timer.start();

    auto cur_size = cur_size_;
    oracle_->insert(dataset->getBasePtr());
    auto total = oracle_->size();
    this->resize(total);

#pragma omp parallel for schedule(dynamic)
    for (auto i = cur_size; i < total; ++i) {
        if (i % (oracle_->size() / 10) == 0) {
            logger << "Adding " << i << " / " << total << std::endl;
        }
        addPoint(i);
    }

    timer.end();
    logger << "Adding time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenHGraph(graph_);
    built_ = true;
}

void
hnsw::HNSW::set_max_neighbors(int max_neighbors) {
    max_neighbors_ = max_neighbors;
    max_base_neighbors_ = max_neighbors * 2;
    reverse_ = 1 / log(1.0 * max_neighbors_);
}

void
hnsw::HNSW::set_ef_construction(int ef_construction) {
    this->ef_construction_ = ef_construction;
}

ParamMap
hnsw::HNSW::extract_params() {
    auto params = Index::extract_params();
    params["index_type"] = "HNSW";
    params["max_neighbors"] = (uint64_t)max_neighbors_;
    params["ef_construction"] = (uint64_t)ef_construction_;
    params["enter_point"] = (uint64_t)enter_point_;
    params["max_level"] = (uint64_t)max_level_;
    params["cur_max_level"] = (uint64_t)cur_max_level_;
    return params;
}
void
hnsw::HNSW::load_params(const ParamMap& params) {
    if (params.find("max_neighbors") != params.end()) {
        max_neighbors_ = std::get<uint64_t>(params.at("max_neighbors"));
        max_base_neighbors_ = max_neighbors_ * 2;
        reverse_ = 1 / log(1.0 * max_neighbors_);
    }
    if (params.find("ef_construction") != params.end()) {
        ef_construction_ = std::get<uint64_t>(params.at("ef_construction"));
    }
    if (params.find("enter_point") != params.end()) {
        enter_point_ = std::get<uint64_t>(params.at("enter_point"));
    }
    if (params.find("max_level") != params.end()) {
        max_level_ = std::get<uint64_t>(params.at("max_level"));
    }
    if (params.find("cur_max_level") != params.end()) {
        cur_max_level_ = std::get<uint64_t>(params.at("cur_max_level"));
    }
}

void
hnsw::HNSW::remove(IdType id) {
}

hnsw::ParlayHNSW::ParlayHNSW(const IndexParam& param, int M, int ef_construction, int theta)
    : HNSW(param, M, ef_construction), theta_(theta) {
}

void
hnsw::ParlayHNSW::batch_insert(IdType start, IdType end) {
#pragma omp parallel for schedule(dynamic)
    for (auto i = start; i < end; ++i) {
        int level = levels_[i];
        std::unique_lock<std::mutex> graph_lock(graph_lock_);
        int max_level_copy = cur_max_level_;
        if (level <= max_level_copy) {
            graph_lock.unlock();
        }

        uint32_t cur_node_ = enter_point_;
        for (auto l = max_level_copy; l > level; --l) {
            auto res = search_one_graph<true>(
                oracle_.get(), visited_list_pool_.get(), graph_[l], (*oracle_)[i], 1, 1, cur_node_);
            cur_node_ = res[0].id;
        }

        for (auto l = std::min(level, max_level_copy); l >= 0; --l) {
            auto res = search_one_graph<true>(oracle_.get(),
                                              visited_list_pool_.get(),
                                              graph_[l],
                                              (*oracle_)[i],
                                              ef_construction_,
                                              ef_construction_,
                                              cur_node_);

            auto cur_max_cnt = l ? max_neighbors_ : max_base_neighbors_;
            prune(res, cur_max_cnt);

            graph_[l][i].candidates_.swap(res);
            cur_node_ = graph_[l][i].candidates_[0].id;
        }

        if (level > max_level_copy) {
            enter_point_ = i;
            cur_max_level_ = level;
        }
    }

    for (int l = 0; l <= cur_max_level_; ++l) {
#pragma omp parallel for schedule(dynamic)
        for (auto u = start; u < end; ++u) {
            if (graph_[l][u].candidates_.empty()) {
                continue;
            }
            for (auto& v : graph_[l][u].candidates_) {
                std::lock_guard<std::mutex> guard(reverse_graph_[v.id].lock_);
                reverse_graph_[v.id].candidates_.emplace_back(u, v.distance, false);
            }
        }
#pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < reverse_graph_.size(); ++u) {
            if (reverse_graph_[u].candidates_.empty()) {
                continue;
            }
            graph_[l][u].candidates_.insert(graph_[l][u].candidates_.end(),
                                            reverse_graph_[u].candidates_.begin(),
                                            reverse_graph_[u].candidates_.end());
            reverse_graph_[u].candidates_.clear();
            std::sort(graph_[l][u].candidates_.begin(), graph_[l][u].candidates_.end());
            auto max_cnt = l ? max_neighbors_ : max_base_neighbors_;
            prune(graph_[l][u].candidates_, max_cnt);
        }
    }
}

void
hnsw::ParlayHNSW::print_info() const {
    HNSW::print_info();
    logger << "ParlayHNSW parameters:" << std::endl;
}

void
hnsw::ParlayHNSW::build_internal(DatasetPtr& dataset) {
    if (theta_ <= 0) {
        logger << "Theta is not set, using 2% of data size : " << (int)(0.02 * oracle_->size())
               << " as default." << std::endl;
        theta_ = (int)(0.02 * oracle_->size());
    }
    IdType start = 0;
    while (start < oracle_->size()) {
        auto end = std::min(start * 2, start + theta_);
        end = std::max(end, start + 1);
        end = std::min(end, oracle_->size());
        logger << "Inserting from " << start << " to " << end << std::endl;
        batch_insert(start, end);
        start = end + 1;
    }
}

void
hnsw::ParlayHNSW::resize(graph::IdType new_size) {
    HNSW::resize(new_size);
    reverse_graph_.resize(new_size);
}
