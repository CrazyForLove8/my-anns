#include "hnsw/shnsw.h"

shnsw::SHNSW::SHNSW(DatasetPtr& dataset, int max_neighbors, int ef_construction, float radius)
    : HNSW(dataset, max_neighbors, ef_construction), radius_(radius) {
}

void
shnsw::SHNSW::addPoint(unsigned int index) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    auto level = (int)(-log(distribution(random_engine_)) * reverse_);
    int cur_max_level_ = graph_.size() - 1;

    unsigned cur_node_ = enter_point_;
    for (auto i = cur_max_level_; i > level; --i) {
        auto res = searchLayer(graph_[i], (*oracle_)[index], 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }

    std::vector<Neighbors> resset;
    float min_dist_ = std::numeric_limits<float>::max();

    for (auto i = std::min(level, cur_max_level_); i >= 0; --i) {
        Graph& graph = graph_[i];
        auto res =
            searchLayer(graph, (*oracle_)[index], ef_construction_, ef_construction_, cur_node_);
        auto pos = seekPos(res);
        resset.emplace_back(res.begin(), res.begin() + pos);
        cur_node_ = res[0].id;
        min_dist_ = res[0].distance;
    }

    /*
   check if the point is under the radius of its nearest neighbor
   if so, we just add the point to the nearest neighbor's children list
   if not, then none of the neighbors' radius cover the point, we add the point
   to the graph
  */
    if (radius_ > 0 && min_dist_ < radius_) {
        families_[cur_node_].children_.emplace_back(index, min_dist_, false);
        return;
    }

    int idx = 0;
    for (auto i = std::min(level, cur_max_level_); i >= 0; --i) {
        Graph& graph = graph_[i];

        auto& candidates = graph[index].candidates_;
        auto res = resset[idx++];

        auto cur_max_cnt = level ? max_neighbors_ : max_base_neighbors_;
        if (candidates.size() + res.size() > cur_max_cnt) {
            res.reserve(candidates.size() + res.size());
            std::merge(candidates.begin(),
                       candidates.end(),
                       res.begin(),
                       res.end(),
                       std::back_inserter(res));
            prune(res, cur_max_cnt);
            candidates.swap(res);
        } else {
            candidates.reserve(candidates.size() + res.size());
            std::merge(candidates.begin(),
                       candidates.end(),
                       res.begin(),
                       res.end(),
                       std::back_inserter(candidates));
        }
        std::vector<Neighbor>().swap(res);

        for (auto& e : candidates) {
            if (graph[e.id].candidates_.size() < max_neighbors_) {
                graph[e.id].addNeighbor(Neighbor(index, e.distance, false));
            } else {
                auto pool = graph[e.id];
                pool.addNeighbor(Neighbor(index, e.distance, false));
                prune(pool.candidates_, cur_max_cnt);
                graph[e.id].move(pool);
            }
        }
        cur_node_ = candidates[0].id;
    }

    while (level > cur_max_level_) {
        Graph higher_level(oracle_->size());
        graph_.emplace_back(std::move(higher_level));
        enter_point_ = index;
        cur_max_level_ = graph_.size() - 1;
    }
}

//void
//shnsw::SHNSW::build() {
//    Timer timer;
//    timer.start();
//
//    graph_.clear();
//    int total = oracle_->size();
//    families_.resize(total);
//    Graph base(total);
//    graph_.emplace_back(base);
//
//    for (int i = 1; i < total; ++i) {
//        if (i % 10000 == 0) {
//            logger << "Processing " << i << " / " << total << std::endl;
//        }
//        addPoint(i);
//    }
//
//    timer.end();
//    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;
//
//    logger << "Constructed SHNSW with enter_point: " << enter_point_ << std::endl;
//}

Neighbors
shnsw::SHNSW::search(const float* query, unsigned int topk, unsigned int L) const {
    unsigned cur_node_ = enter_point_;
    for (auto i = graph_.size() - 1; i > 0; --i) {
        auto res = searchLayer(graph_[i], query, 1, 1, cur_node_);
        cur_node_ = res[0].id;
    }
    auto res = searchLayer(graph_[0], query, topk, L, cur_node_);

    return res;
}

void
shnsw::SHNSW::build_internal() {
    HGraph().swap(graph_);

    int total = oracle_->size();
    families_.resize(total);
    graph_.emplace_back(total);

    Graph& base = graph_.back();
    for (auto& u : base) {
        u.candidates_.reserve(max_base_neighbors_);
    }

    for (int i = 1; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << total << std::endl;
        }
        addPoint(i);
    }
}
