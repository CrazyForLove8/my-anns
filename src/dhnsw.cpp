#include "dhnsw.h"

dhnsw::DHNSW::DHNSW(DatasetPtr &dataset,
                    int max_neighbors,
                    int ef_construction,
                    float lambda)
        : HNSW(dataset, max_neighbors, ef_construction), lambda_(lambda) {
}

Neighbors
naive(Neighbors res,
      IndexOracle<float> *oracle,
      int topk,
      float lambda) {
    if (res.size() <= topk) {
        return res;
    }
    Neighbors ret;
    ret.emplace_back(res[0]);
    // lambda controls the pruning threshold
    int idx = 1;
    while (idx < res.size() && ret.size() < topk) {
        if (res[idx].id == -1) {
            break;
        }
        bool add = true;
        for (auto &r: ret) {
            if ((*oracle)(r.id, res[idx].id) < lambda) {
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

Neighbors
MMR(Neighbors res,
    IndexOracle<float> *oracle,
    int topk,
    float lambda) {
    if (res.size() <= topk) {
        return res;
    }
    Neighbors ret;
    ret.emplace_back(res[0]);
    // lambda controls the trade-off between similarity and diversity, ranging
    // from 0 to 1
    for (auto &r: res) {
        r.flag = false;
    }
    while (ret.size() < topk) {
        std::vector <std::pair<int, float>> scores;
        for (int idx = 1; idx < res.size(); ++idx) {
            if (res[idx].flag)
                continue;
            float min_dist = std::numeric_limits<float>::max();
            for (auto &r: ret) {
                min_dist = std::min(min_dist, (*oracle)(r.id, res[idx].id));
            }
            float score = -lambda * res[idx].distance + (1 - lambda) * min_dist;
            scores.emplace_back(idx, score);
        }
        auto max_ =
                std::max_element(scores.begin(), scores.end(), [](const auto &a,
                                                                  const auto &b) {
                    return a.second < b.second;
                });
        ret.emplace_back(res[max_->first]);
        res[max_->first].flag = true;
    }
    std::sort(ret.begin(), ret.end());
    return ret;
}

Neighbors
dhnsw::DHNSW::search(const float *query,
                     unsigned int topk,
                     unsigned int L) const {
    unsigned cur_node_ = enter_point_;
    for (auto i = graph_.size() - 1; i > 0; --i) {
        auto res = searchLayer(graph_[i], query, 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }
    auto res = searchLayer(graph_[0], query, L, L, cur_node_);

    // naive method
    return naive(res, oracle_.get(), topk, lambda_);

    // MMR method
    //    return MMR(res, oracle, topk, lambda_);
}

//Neighbors
//dhnsw::DHNSW::HNSW_search(HGraph &hnsw_graph,
//                          IndexOracle<float> &oracle,
//                          float *query,
//                          int topk,
//                          int ef_search) const {
//    unsigned cur_node_ = enter_point_;
//    for (int i = hnsw_graph.size() - 1; i > 0; --i) {
//        auto res = ::searchLayer(oracle, hnsw_graph[i], query, 1, 1, cur_node_);
//        cur_node_ = res[0].id;
//    }
//    auto res = ::searchLayer(oracle, hnsw_graph[0], query, ef_search, ef_search, cur_node_);
//
//    // naive method
//    return naive(res, oracle, topk, lambda_);
//
//    // MMR method
//    //    return MMR(res, oracle, topk, lambda_);
//}
