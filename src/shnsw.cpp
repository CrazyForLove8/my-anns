#include "shnsw.h"

shnsw::SHNSW::SHNSW(int max_neighbors,
                    int ef_construction,
                    float radius) : HNSW(max_neighbors, ef_construction), radius_(radius) {}

Neighbors searchLayer(IndexOracle &oracle,
                      Graph &graph,
                      const float *query,
                      int topk,
                      int L,
                      int entry_id) {
    int graph_sz = graph.size();
    std::vector<bool> visited(graph_sz, false);
    Neighbors retset(L + 1, Neighbor(-1, std::numeric_limits<float>::max(), false));
    auto dist = oracle(entry_id, query);
    retset[0] = Neighbor(entry_id, dist, true);
    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            int n = retset[k].id;
            for (const auto &candidate: graph[n].candidates_) {
                int id = candidate.id;
                if (visited[id]) continue;
                visited[id] = true;
                dist = oracle(id, query);
                if (dist >= retset[L - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L, nn);
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    retset.resize(topk);
    return retset;
}

void shnsw::SHNSW::addPoint(HNSWGraph &hnsw_graph,
                            IndexOracle &oracle,
                            unsigned int index) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto level = (int) (-log(distribution(random_engine_)) * reverse_);
    int cur_max_level_ = hnsw_graph.size() - 1;

    unsigned cur_node_ = enter_point_;
    for (auto i = cur_max_level_; i > level; --i) {
        auto res = ::searchLayer(oracle, hnsw_graph[i], oracle[index], 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }

    std::vector<Neighbors> resset;
    float min_dist_ = std::numeric_limits<float>::max();

    for (auto i = std::min(level, cur_max_level_); i >= 0; --i) {
        Graph &graph = hnsw_graph[i];
        auto res = ::searchLayer(oracle, graph, oracle[index], ef_construction_, ef_construction_, cur_node_);
        auto pos = seekPos(res);
        resset.emplace_back(res.begin(), res.begin() + pos);
        cur_node_ = res[0].id;
        min_dist_ = res[0].distance;
    }

    /*
     check if the point is under the radius of its nearest neighbor
     if so, we just add the point to the nearest neighbor's children list
     if not, then none of the neighbors' radius cover the point, we add the point to the graph
    */
    if (radius_ > 0 && min_dist_ < radius_) {
        families_[cur_node_].children_.emplace_back(index, min_dist_, false);
        return;
    }

    int idx = 0;
    for (auto i = std::min(level, cur_max_level_); i >= 0; --i) {
        Graph &graph = hnsw_graph[i];

        auto &candidates = graph[index].candidates_;
        auto res = resset[idx++];
        candidates.reserve(candidates.size() + res.size());
        std::merge(candidates.begin(), candidates.end(), res.begin(), res.end(),
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
    }

    while (level > cur_max_level_) {
        Graph higher_level(oracle.size());
        hnsw_graph.emplace_back(std::move(higher_level));
        enter_point_ = index;
        cur_max_level_ = hnsw_graph.size() - 1;
    }
}

HNSWGraph shnsw::SHNSW::build(IndexOracle &oracle) {
    Timer timer;
    timer.start();

    graph_.clear();
    int total = oracle.size();
    families_.resize(total);
    Graph base(total);
    graph_.emplace_back(base);

    for (int i = 1; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << total << std::endl;
        }
        addPoint(graph_, oracle, i);
    }

    timer.end();
    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;

    logger << "Constructed SHNSW with enter_point: " << enter_point_ << std::endl;

    return graph_;
}

Neighbors shnsw::SHNSW::HNSW_search(HNSWGraph &hnsw_graph,
                                    IndexOracle &oracle,
                                    float *query,
                                    int topk,
                                    int ef_search) const {
    unsigned cur_node_ = enter_point_;
    for (int i = hnsw_graph.size() - 1; i > 0; --i) {
        auto res = ::searchLayer(oracle, hnsw_graph[i], query, 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }
    auto res = ::searchLayer(oracle, hnsw_graph[0], query, topk, ef_search, cur_node_);

    return res;
}
