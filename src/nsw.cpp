#include "nsw.h"

nsw::NSW::NSW(DatasetPtr& dataset, int max_neighbors, int ef_construction)
    : Index(dataset), max_neighbors_(max_neighbors), ef_construction_(ef_construction) {
}

//void
//nsw::NSW::build() {
//    Timer timer;
//    timer.start();
//
//    int total = oracle_->size();
//    graph_.emplace_back(max_neighbors_);
//
//    for (int i = 1; i < total; ++i) {
//        if (i % 10000 == 0) {
//            logger << "Processing " << i << " / " << graph_.size() << std::endl;
//        }
//        addPoint(i);
//    }
//
//    timer.end();
//    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;
//}

void
nsw::NSW::addPoint(unsigned int index) {
    auto res = knn_search(oracle_.get(),
                          visited_list_pool_.get(),
                          graph_,
                          (*oracle_)[index],
                          max_neighbors_,
                          ef_construction_,
                          -1,
                          index);
    graph_.emplace_back(max_neighbors_);
    for (int x = 0; x < max_neighbors_; ++x) {
        if (res[x].id == index || res[x].id == -1) {
            continue;
        }
        graph_[index].addNeighbor(res[x]);
        graph_[res[x].id].addNeighbor(Neighbor(index, res[x].distance, false));
    }
}

Neighbors
nsw::NSW::multisearch(const Graph& graph_,
                      const IndexOracle<float>& oracle,
                      unsigned int query,
                      int attempts,
                      int k) {
    std::priority_queue<Neighbor> candidates;
    Neighbors results;
    std::vector<bool> visited(graph_.size(), false);
    std::mt19937 rng(2024);
    for (int it = 0; it < attempts; ++it) {
        Neighbors temp_results;

        int entry_point = rng() % graph_.size();
        while (visited[entry_point]) {
            entry_point = rng() % graph_.size();
        }
        auto dist = oracle(query, entry_point);
        candidates.emplace(entry_point, dist, false);
        while (!candidates.empty()) {
            auto& c = candidates.top();
            candidates.pop();

            if (results.size() >= k && c.distance > results[k - 1].distance) {
                break;
            }

            for (auto& v : graph_[c.id].candidates_) {
                if (!visited[v.id]) {
                    visited[v.id] = true;
                    dist = oracle(query, v.id);
                    temp_results.emplace_back(v.id, dist, false);
                    candidates.push(Neighbor(v.id, dist, false));
                }
            }
        }
        results.insert(results.end(), temp_results.begin(), temp_results.end());
        std::sort(results.begin(), results.end());
        results.erase(
            std::unique(results.begin(),
                        results.end(),
                        [](const Neighbor& a, const Neighbor& b) { return a.id == b.id; }),
            results.end());
    }
    results.resize(k);
    return results;
}

void
nsw::NSW::build_internal() {
    int total = oracle_->size();
    graph_.emplace_back(max_neighbors_);

    for (int i = 1; i < total; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << graph_.size() << std::endl;
        }
        addPoint(i);
    }
}
