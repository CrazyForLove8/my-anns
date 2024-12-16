#include "hnsw.h"

using namespace graph;

int cur_max_level = 0;

hnsw::HNSW::HNSW(int max_neighbors,
                 int ef_construction) : max_neighbors_(max_neighbors),
                                        ef_construction_(ef_construction) {
    visited_table_ = std::unordered_set<int>();
    reset();
}

void hnsw::HNSW::reset() {
    random_engine_.seed(2024);
    enter_point_ = 0;
    reverse_ = 1 / log(1.0 * max_neighbors_);
}

int hnsw::HNSW::seekPos(const Neighbors &vec) {
    int left = 0, right = vec.size() - 1;
    if (vec[right].id > 0) {
        return right;
    }
    int result = right;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (vec[mid].id == -1) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

void hnsw::HNSW::addPoint(HNSWGraph &hnsw_graph,
                          IndexOracle &oracle,
                          unsigned int index) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto level = (int) (-log(distribution(random_engine_)) * reverse_);
    int cur_max_level_ = hnsw_graph.size() - 1;

    unsigned cur_node_ = enter_point_;
    for (auto i = cur_max_level_; i > level; --i) {
//        auto res = searchLayer(hnsw_graph[i], oracle, oracle[index], cur_node_, 1);
        auto res = knn_search(oracle, hnsw_graph[i], oracle[index], 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }

    for (auto i = std::min(level, cur_max_level_); i >= 0; --i) {
        Graph &graph = hnsw_graph[i];
        auto &candidates = graph[index].candidates_;
//        auto res = searchLayer(graph, oracle, oracle[index], cur_node_, ef_construction_);
        auto res = knn_search(oracle, graph, oracle[index], ef_construction_, ef_construction_, cur_node_);
        auto pos = seekPos(res);

        candidates.reserve(candidates.size() + res.size());
        std::merge(candidates.begin(), candidates.end(), res.begin(), res.begin() + pos,
                   std::back_inserter(candidates));

        if (candidates.size() > max_neighbors_) {
            candidates = prune(oracle, candidates);
        }

        for (auto &e: candidates) {
            graph[e.id].addNeighbor(Neighbor(index, e.distance, false));
            if (graph[e.id].candidates_.size() > max_neighbors_) {
                graph[e.id].candidates_ = prune(oracle, graph[e.id].candidates_);
            }
        }
        cur_node_ = candidates[0].id;
    }

    while (level > cur_max_level_) {
        Graph higher_level(oracle.size());
        hnsw_graph.emplace_back(std::move(higher_level));
        enter_point_ = index;
        cur_max_level_ = hnsw_graph.size() - 1;
    }

    if (level > cur_max_level) {
        cur_max_level = level;
    }

}

struct CompareByCloser {
    bool operator()(const Node &a,
                    const Node &b) {
        return a.distance > b.distance;
    }
};

Neighbors hnsw::HNSW::searchLayer(Graph &graph,
                                  IndexOracle &oracle,
                                  float *query,
                                  int enter_point,
                                  int ef) {
    visited_table_.clear();
    visited_table_.insert(enter_point);
    std::priority_queue<Node, std::vector<Node>, CompareByCloser> candidates;
    std::priority_queue<Node> result;
    auto dist = oracle(enter_point, query);
    candidates.emplace(enter_point, dist);
    result.emplace(enter_point, dist);

    auto farthest = result.top().distance;

    while (!candidates.empty()) {
        auto c = candidates.top();
        if (c.distance > farthest && result.size() == ef) {
            break;
        }
        candidates.pop();
        for (auto &n: graph[c.id].candidates_) {
            if (visited_table_.find(n.id) == visited_table_.end()) {
                visited_table_.insert(n.id);
                auto d = oracle(n.id, query);
                if (result.size() < ef || d < farthest) {
                    candidates.emplace(n.id, d);
                    result.emplace(n.id, d);
                    if (result.size() > ef) {
                        result.pop();
                    }
                    if (!result.empty()) {
                        farthest = result.top().distance;
                    }
                }
            }
        }
    }
    Neighbors ret;
    while (!result.empty()) {
        auto r = result.top();
        ret.emplace_back(r.id, r.distance, false);
        result.pop();
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

Neighbors hnsw::HNSW::prune(IndexOracle &oracle,
                            Neighbors &candidates) {
    Neighbors ret_set;
    for (auto &v: candidates) {
        bool prune = false;
        for (auto &w: ret_set) {
            if (oracle(v.id, w.id) < v.distance) {
                prune = true;
                break;
            }
        }
        if (!prune) {
            ret_set.emplace_back(v);
        }
        if (ret_set.size() >= max_neighbors_) {
            break;
        }
    }
    return ret_set;
}

hnsw::HNSWGraph hnsw::HNSW::build(IndexOracle &oracle) {
    Timer timer;
    timer.start();

    HNSWGraph graph;
    int total = oracle.size();
    Graph base(total);
    graph.emplace_back(base);

    for (int i = 1; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Adding " << i << " / " << total << std::endl;
        }
        addPoint(graph, oracle, i);
    }

    timer.end();
    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;

    logger << "Constructed HNSW with enter_point: " << enter_point_ << std::endl;

    return graph;
}

Neighbors hnsw::HNSW::HNSW_search(HNSWGraph &hnsw_graph,
                                  IndexOracle &oracle,
                                  float *query,
                                  int topk,
                                  int ef_search) const {
    unsigned cur_node_ = enter_point_;
    for (int i = hnsw_graph.size() - 1; i > 0; --i) {
        auto res = knn_search(oracle, hnsw_graph[i], query, 1, 1, cur_node_);
//        auto res = searchLayer(hnsw_graph[i], oracle, query, cur_node_, 1);
        cur_node_ = res[0].id;
    }
//    logger << "Starting search from node: " << cur_node_ << std::endl;
    auto res = knn_search(oracle, hnsw_graph[0], query, topk, ef_search, cur_node_);
//    auto res = searchLayer(hnsw_graph[0], oracle, query, cur_node_, ef_search);
//    while (res.size() > topk) {
//        res.pop_back();
//    }
    return res;
}
