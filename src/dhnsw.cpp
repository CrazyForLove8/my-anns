#include "dhnsw.h"

dhnsw::DHNSW::DHNSW(int max_neighbors,
                    int ef_construction) : HNSW(max_neighbors, ef_construction) {}

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

HNSWGraph dhnsw::DHNSW::build(IndexOracle &oracle) {
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

    logger << "Constructed DHNSW with enter_point: " << enter_point_ << std::endl;

    return graph;
}

Neighbors naive(Neighbors res,
                IndexOracle &oracle,
                int topk) {
    if (res.size() <= topk) {
        return res;
    }
    Neighbors ret;
    ret.emplace_back(res[0]);
    // lambda controls the pruning threshold
    auto lambda = 1.0f;
    int idx = 1;
    while (idx < res.size() && ret.size() < topk) {
        bool add = true;
        for (auto &r: ret) {
            if (oracle(r.id, res[idx].id) < lambda) {
                add = false;
                break;
            }
        }
        if (add) {
            ret.emplace_back(res[idx]);
        }
        idx++;
    }
    return ret;
}

Neighbors MMR(Neighbors res,
              IndexOracle &oracle,
              int topk) {
    if (res.size() <= topk) {
        return res;
    }
    Neighbors ret;
    ret.emplace_back(res[0]);
    // lambda controls the trade-off between similarity and diversity, ranging from 0 to 1
    auto lambda = 0.8f;

    for (auto &r: res) {
        r.flag = false;
    }
    while (ret.size() < topk) {
        std::vector<std::pair<int, float>> scores;
        for (int idx = 1; idx < res.size(); ++idx) {
            if (res[idx].flag) continue;
            float min_dist = std::numeric_limits<float>::max();
            for (auto &r: ret) {
                min_dist = std::min(min_dist, oracle(r.id, res[idx].id));
            }
            float score = -lambda * res[idx].distance + (1 - lambda) * min_dist;
            scores.emplace_back(idx, score);
        }
        auto max_ = std::max_element(scores.begin(), scores.end(), [](const auto &a,
                                                                      const auto &b) {
            return a.second < b.second;
        });
        ret.emplace_back(res[max_->first]);
        res[max_->first].flag = true;
    }
    std::sort(ret.begin(), ret.end());
    return ret;
}

Neighbors dhnsw::DHNSW::HNSW_search(HNSWGraph &hnsw_graph,
                                    IndexOracle &oracle,
                                    float *query,
                                    int topk,
                                    int ef_search) const {
    unsigned cur_node_ = enter_point_;
    for (int i = hnsw_graph.size() - 1; i > 0; --i) {
        auto res = ::searchLayer(oracle, hnsw_graph[i], query, 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }
    auto res = ::searchLayer(oracle, hnsw_graph[0], query, ef_search, ef_search, cur_node_);

    //naive method
    return naive(res, oracle, topk);

    //MMR method
//    return MMR(res, oracle, topk);

}
