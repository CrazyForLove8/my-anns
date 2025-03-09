#include "nsg.h"

nsg::NSG::NSG(DatasetPtr& dataset, unsigned int L, unsigned int m) : Index(dataset), L_(L), m_(m) {
}

//void
//nsg::NSG::build() {
//    auto* center = new float[oracle_->dim()];
//    for (unsigned i = 0; i < oracle_->size(); ++i) {
//        auto pt = (*oracle_)[i];
//        for (unsigned j = 0; j < oracle_->dim(); ++j) {
//            center[j] += pt[j];
//        }
//    }
//    for (unsigned i = 0; i < oracle_->dim(); ++i) {
//        center[i] /= oracle_->size();
//    }
//    root = knn_search(oracle_.get(), graph_, center, 1, L_)[0].id;
//    delete center;
//    logger << "Root: " << root << std::endl;
//
//    for (int u = 0; u < graph_.size(); ++u) {
//        if (u % 10000 == 0) {
//            logger << "Adding " << u << " / " << graph_.size() << std::endl;
//        }
//        std::vector<Neighbor> candidates =
//            track_search(oracle_.get(), graph_, (*oracle_)[u], root, L_);
//        graph_[u].candidates_ = prune(candidates);
//    }
//
//    tree();
//}

std::vector<Neighbor>
nsg::NSG::prune(std::vector<Neighbor>& candidates) {
    std::vector<Neighbor> prunedNeighbors;
    for (auto&& v : candidates) {
        auto flag = false;
        for (auto&& w : prunedNeighbors) {
            if ((*oracle_)(w.id, v.id) < v.distance) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            prunedNeighbors.push_back(v);
        }
        if (prunedNeighbors.size() >= m_) {
            break;
        }
    }
    return prunedNeighbors;
}

void
nsg::NSG::tree() {
    /**
   * TODO Here needs to be fixed
   * Too many recursion
   */
    auto dfs = [&](int start, const Graph& g, std::vector<bool>& visited) {
        std::stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int node = s.top();
            s.pop();
            for (const auto& neighbor : g[node].candidates_) {
                if (!visited[neighbor.id]) {
                    visited[neighbor.id] = true;
                    s.push(neighbor.id);
                }
            }
        }
    };
    std::vector<bool> visited(graph_.size(), false);
    bool built = false;
    while (!built) {
        dfs(root, graph_, visited);
        built = true;
        for (int i = 0; i < graph_.size(); ++i) {
            if (!visited[i]) {
                built = false;
                auto candidates = track_search(oracle_.get(), graph_, (*oracle_)[i], root, L_);
                bool added = false;
                int idx = 0;
                for (auto&& candidate : candidates) {
                    if (!visited[candidate.id]) {
                        continue;
                    }
                    graph_[candidate.id].addNeighbor(Neighbor(i, candidate.distance, true));
                    added = true;
                    logger << candidate.id << " " << i << std::endl;
                    break;
                }

                if (!added) {
                    std::mt19937 rng(2024);
                    do {
                        idx = rng() % graph_.size();
                        if (visited[idx]) {
                            graph_[idx].addNeighbor(Neighbor(i, (*oracle_)(idx, i), true));
                            added = true;
                            logger << idx << " " << i << std::endl;
                        }
                    } while (!added);
                }
                break;
            }
        }
        std::fill(visited.begin(), visited.end(), false);
    }
}

void
nsg::NSG::build_internal() {
    auto* center = new float[oracle_->dim()];
    for (unsigned i = 0; i < oracle_->size(); ++i) {
        auto pt = (*oracle_)[i];
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center[j] += pt[j];
        }
    }
    for (unsigned i = 0; i < oracle_->dim(); ++i) {
        center[i] /= oracle_->size();
    }
    root = knn_search(oracle_.get(), visited_list_pool_.get(), graph_, center, 1, L_)[0].id;
    delete center;
    logger << "Root: " << root << std::endl;

    for (int u = 0; u < graph_.size(); ++u) {
        if (u % 10000 == 0) {
            logger << "Adding " << u << " / " << graph_.size() << std::endl;
        }
        std::vector<Neighbor> candidates =
            track_search(oracle_.get(), graph_, (*oracle_)[u], root, L_);
        graph_[u].candidates_ = prune(candidates);
    }

    tree();
}
