#include "include/fgim.h"

using namespace graph;

FGIM::FGIM() : max_degree_(20), sample_rate_(0.3) {
}

FGIM::FGIM(unsigned int max_degree,
           float sample_rate) : max_degree_(max_degree), sample_rate_(sample_rate) {
}

//Graph
//FGIM::merge(
//        const Graph &g1,
//        OraclePtr &oracle1,
//        const Graph &g2,
//        OraclePtr &oracle2,
//        OraclePtr &oracle) {
//    Timer timer;
//    timer.start();
//
//    Graph graph(g1.size() + g2.size());
//
//    Sampling(graph, g1, g2, oracle1, oracle2, oracle);
//
//    Refinement(graph, oracle);
//
//    timer.end();
//    logger << "Merging time: " << timer.elapsed() << "s" << std::endl;
//    return graph;
//}
//
//void
//FGIM::Sampling(Graph &graph,
//               const Graph &g1,
//               const Graph &g2,
//               OraclePtr &oracle1,
//               OraclePtr &oracle2,
//               OraclePtr &oracle) {
//    auto total = graph.size();
//    auto _gs1 = g1.size();
//    auto _gs2 = g2.size();
//    FlattenGraph fg1(g1), fg2(g2);
//
//#pragma omp parallel for schedule(dynamic, 256)
//    for (size_t u = 0; u < total; ++u) {
//        if (u % 10000 == 0) {
//            logger << "Processing " << u << "/" << total << std::endl;
//        }
//        auto data = (*oracle)[u];
//        graph[u].M_ = max_degree_;
//        graph[u].candidates_.reserve(max_degree_);
//        if (u < _gs1) {
//            auto &neighbors = g1[u].candidates_;
//            std::copy(neighbors.begin(), neighbors.end(), std::back_inserter(graph[u].candidates_));
//            if (graph[u].candidates_.size() > max_degree_) {
//                graph[u].candidates_.resize(max_degree_);
//            }
//            std::make_heap(graph[u].candidates_.begin(), graph[u].candidates_.end());
//            // TODO search
////            auto result = search(oracle2.get(), fg2, data, L_, _gs2, L_);
////            for (auto &&res: result) {
////                graph[u].insert(res.id + _gs1, res.distance);
////            }
//        } else {
//            auto &neighbors = g2[u - _gs1].candidates_;
//            for (const auto &neighbor: neighbors) {
//                graph[u].candidates_.emplace_back(neighbor.id + _gs1, neighbor.distance, false);
//            }
//            if (graph[u].candidates_.size() > max_degree_) {
//                graph[u].candidates_.resize(max_degree_);
//            }
//            std::make_heap(graph[u].candidates_.begin(), graph[u].candidates_.end());
////            auto result = search(oracle1.get(), fg1, data, L_, _gs1, L_);
////            for (auto&& res : result) {
////                graph[u].insert(res.id, res.distance);
////            }
//        }
//    }
//}

void FGIM::update_neighbors(Graph &graph) {
    size_t it = 0;
    unsigned samples = sample_rate_ * max_degree_;
#pragma omp parallel for
    for (auto &u: graph) {
        for (int v = (int) (u.candidates_.size() / 2); v < u.candidates_.size(); ++v) {
            u.new_.emplace_back(u.candidates_[v].id);
            u.candidates_[v].flag = false;
        }
    }
    while (++it && it <= ITER_MAX) {
        int cnt = 0;
#pragma omp parallel
        {
            std::mt19937 rng(2024 + omp_get_thread_num());
#pragma omp for reduction(+ : cnt) schedule(dynamic, 256)
            for (int vv = 0; vv < graph.size(); ++vv) {
                auto &v = graph[vv];
                auto &_old = v.old_;
                auto &_new = v.new_;

                {
                    std::lock_guard<std::mutex> guard(v.lock_);
                    auto &_r_old = v.reverse_old_;
                    auto &_r_new = v.reverse_new_;
                    shuffle(_r_new.begin(), _r_new.end(), rng);
//                    if (_r_new.size() > SAMPLES) {
//                        _r_new.resize(SAMPLES);
//                    }
                    shuffle(_r_old.begin(), _r_old.end(), rng);
//                    if (_r_old.size() > SAMPLES) {
//                        _r_old.resize(SAMPLES);
//                    }
                    if (!_r_old.empty()) {
                        _old.insert(_old.end(), _r_old.begin(), _r_old.end());
                        _r_old.clear();
                    }
                    if (!_r_new.empty()) {
                        _new.insert(_new.end(), _r_new.begin(), _r_new.end());
                        _r_new.clear();
                    }
                }

                std::sort(_old.begin(), _old.end());
                std::sort(_new.begin(), _new.end());
                _old.erase(std::unique(_old.begin(), _old.end()), _old.end());
                _new.erase(std::unique(_new.begin(), _new.end()), _new.end());
                if (_old.size() > samples) {
                    _old.resize(samples);
                }
                if (_new.size() > samples) {
                    _new.resize(samples);
                }

                auto _new_size = _new.size();
                auto _old_size = _old.size();
                for (size_t i = 0; i < _new_size; ++i) {
                    for (size_t j = i + 1; j < _new_size; ++j) {
                        if (_new[i] == _new[j]) {
                            continue;
                        }
                        auto dist = (*oracle_)(_new[i], _new[j]);
                        if (dist < graph[_new[i]].candidates_.front().distance ||
                            dist < graph[_new[j]].candidates_.front().distance) {
                            cnt += graph[_new[i]].insert(_new[j], dist);
                            cnt += graph[_new[j]].insert(_new[i], dist);
                        }
                    }
                    for (size_t j = 0; j < _old_size; ++j) {
                        if (_new[i] == _old[j]) {
                            continue;
                        }
                        auto dist = (*oracle_)(_new[i], _old[j]);
                        if (dist < graph[_new[i]].candidates_.front().distance ||
                            dist < graph[_old[j]].candidates_.front().distance) {
                            cnt += graph[_new[i]].insert(_old[j], dist);
                            cnt += graph[_old[j]].insert(_new[i], dist);
                        }
                    }
                }
                _old.clear();
                _new.clear();

                for (auto &u: v.candidates_) {
                    if (u.flag) {
                        _new.emplace_back(u.id);
                        {
                            std::lock_guard<std::mutex> guard(graph[u.id].lock_);
                            graph[u.id].reverse_new_.emplace_back(vv);
                        }
                        u.flag = false;
                    } else {
                        _old.emplace_back(u.id);
                        {
                            std::lock_guard<std::mutex> guard(graph[u.id].lock_);
                            graph[u.id].reverse_old_.emplace_back(vv);
                        }
                    }
                }
            }
        }
//        connect_no_indegree(graph);
        logger << "Iteration " << it << " with " << cnt << " new edges" << std::endl;
        unsigned convergence =
                std::lround(THRESHOLD * static_cast<float>(graph.size()) * static_cast<float>(max_degree_));
        if (cnt <= convergence) {
            break;
        }
    }
}

void FGIM::prune(Graph &graph) {
#pragma omp parallel for schedule(dynamic)
    for (auto &u: graph) {
        auto &neighbors = u.candidates_;
        Neighbors candidates, _new_neighbors;
        {
            std::lock_guard<std::mutex> guard(u.lock_);
            candidates = neighbors;
            neighbors.clear();
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(
                std::unique(candidates.begin(),
                            candidates.end(),
                            [](const Neighbor &a,
                               const Neighbor &b) { return a.id == b.id; }),
                candidates.end());
        for (auto &&v: candidates) {
            bool reserve = true;
            for (auto &nn: _new_neighbors) {
                if (nn.distance <= std::numeric_limits<float>::epsilon()) {
                    continue;
                }
                auto dist = (*oracle_)(v.id, nn.id);
                if (dist < v.distance) {
                    {
                        std::lock_guard<std::mutex> guard(graph[nn.id].lock_);
                        graph[nn.id].candidates_.emplace_back(v.id, dist, true);
                    }
                    {
                        std::lock_guard<std::mutex> guard(graph[v.id].lock_);
                        graph[v.id].candidates_.emplace_back(nn.id, dist, true);
                    }
                    reserve = false;
                    break;
                }
            }
            if (reserve) {
                _new_neighbors.emplace_back(v);
            }
        }
        {
            std::lock_guard<std::mutex> guard(u.lock_);
            neighbors.insert(neighbors.end(), _new_neighbors.begin(), _new_neighbors.end());
        }
    }
#pragma omp parallel for
    for (auto &u: graph) {
        auto &candidates = u.candidates_;
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(
                std::unique(candidates.begin(),
                            candidates.end(),
                            [](const Neighbor &a,
                               const Neighbor &b) { return a.id == b.id; }),
                candidates.end());
        if (candidates.size() > max_degree_ * 2) {
            candidates.resize(max_degree_ * 2);
        }
    }
}

void FGIM::add_reverse_edge(Graph &graph) {
    Graph reverse_graph(graph.size());
#pragma omp parallel for
    for (int u = 0; u < graph.size(); ++u) {
        for (auto &v: graph[u].candidates_) {
            std::lock_guard<std::mutex> guard(reverse_graph[v.id].lock_);
            reverse_graph[v.id].candidates_.emplace_back(u, v.distance, true);
        }
    }
#pragma omp parallel for
    for (int u = 0; u < graph.size(); ++u) {
        graph[u].candidates_.insert(graph[u].candidates_.end(),
                                    reverse_graph[u].candidates_.begin(),
                                    reverse_graph[u].candidates_.end());
        std::sort(graph[u].candidates_.begin(), graph[u].candidates_.end());
        graph[u].candidates_.erase(
                std::unique(graph[u].candidates_.begin(),
                            graph[u].candidates_.end()),
                graph[u].candidates_.end());
        if (graph[u].candidates_.size() > max_degree_ * 2) {
            graph[u].candidates_.resize(max_degree_ * 2);
        }
    }
}

void FGIM::connect_no_indegree(Graph &graph) {
    std::vector<int> indegree(graph.size(), 0);
    for (auto &u: graph) {
        for (auto &v: u.candidates_) {
            indegree[v.id]++;
        }
    }

    std::vector<int> replace_pos(graph.size(), std::min((unsigned) graph.size(), max_degree_) - 1);
    for (int u = 0; u < graph.size(); ++u) {
        auto &neighbors = graph[u].candidates_;
        int need_replace = 0;
        while (indegree[u] < 1 && need_replace < max_degree_) {
            int need_replace_id = neighbors[need_replace].id;
            bool has_connect = false;
            for (auto &neighbor: graph[need_replace_id].candidates_) {
                if (neighbor.id == u) {
                    has_connect = true;
                    break;
                }
            }
            if (replace_pos[need_replace_id] > 0 && !has_connect) {
                std::lock_guard<std::mutex> guard(graph[need_replace_id].lock_);
                auto &replace_node = graph[need_replace_id].candidates_[replace_pos[need_replace_id]];
                auto replace_id = replace_node.id;
                if (indegree[replace_id] > 1) {
                    indegree[replace_id]--;
                    replace_node.id = u;
                    replace_node.distance = neighbors[need_replace].distance;
                    indegree[u]++;
                }
                replace_pos[need_replace_id]--;
            }
            need_replace++;
        }
    }
//#pragma omp parallel for schedule(dynamic)
//    for (int x = 0; x < graph_.size(); ++x) {
//        if (indegree[x])
//            continue;
//        auto &u = graph_[x];
//        auto nearest = u.candidates_.front();
//        graph_[nearest.id].candidates_.emplace_back(x, nearest.distance, true);
//    }
}

void FGIM::CrossQuery(std::vector<IndexPtr> &indexes) {
    Timer timer;
    timer.start();

    std::vector<std::reference_wrapper<Graph>> graphs;
    for (auto &index: indexes) {
        graphs.emplace_back(index->extractGraph());
    }

    size_t offset = 0;
    std::vector<size_t> offsets;
    for (auto &g: graphs) {
        auto &graph_ref = g.get();
#pragma omp parallel for schedule(dynamic)
        for (size_t j = 0; j < graph_ref.size(); ++j) {
            auto &neighbors = graph_ref[j].candidates_;
            graph_[j + offset].candidates_.reserve(max_degree_);
            for (auto &&neighbor: neighbors) {
                graph_[j + offset].candidates_.emplace_back(neighbor.id + offset, neighbor.distance, false);
            }
            std::make_heap(graph_[j + offset].candidates_.begin(), graph_[j + offset].candidates_.end());
        }
        offset += graph_ref.size();
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
            auto &index = indexes[graph_idx];
            auto result = index->search(data, L, L);
            for (auto &&res: result) {
                graph_[u].insert(res.id + _offset, res.distance);
            }
        }
    }

    timer.end();
    logger << "Cross query time: " << timer.elapsed() << "s" << std::endl;
}

void
FGIM::Refinement() {
    Timer timer;

    timer.start();
    update_neighbors(graph_);
    timer.end();
    logger << "Iterative update time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    prune(graph_);
    timer.end();
    logger << "Pruning time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    add_reverse_edge(graph_);
    timer.end();
    logger << "Adding reverse edge time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    prune(graph_);
    timer.end();
    logger << "Pruning time: " << timer.elapsed() << "s" << std::endl;

    timer.start();
    connect_no_indegree(graph_);
    timer.end();
    logger << "Connecting no indegree time: " << timer.elapsed() << "s" << std::endl;
}

void FGIM::Combine(std::vector<IndexPtr> &indexes) {
    for (auto &index: indexes) {
        if (typeid(*index) == typeid(hnsw::HNSW)) {
            throw std::runtime_error("FGIM does not support HNSW, please use MGraph instead");
        }
    }
    std::vector<DatasetPtr> datasets;
    for (auto &index: indexes) {
        datasets.emplace_back(index->extractDataset());
    }
    dataset_ = Dataset::aggregate(datasets);
    oracle_ = dataset_->getOracle();
    visited_list_pool_ = dataset_->getVisitedListPool();
    base_ = dataset_->getBasePtr();
    Graph(oracle_->size()).swap(graph_);

    Timer timer;
    timer.start();

    CrossQuery(indexes);

    Refinement();

    timer.end();
    logger << "Merging time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}
