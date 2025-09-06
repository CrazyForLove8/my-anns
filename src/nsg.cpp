#include "nsg.h"

nsg::NSG::NSG(DatasetPtr& dataset, unsigned int K, unsigned int L, unsigned int m)
    : Index(dataset), L_(L), m_(m), K_(K) {
}

//TODO Extract pruning strategy to a separate class like metrics
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
    auto dfs = [](int start, const Graph& g, std::vector<bool>& visited) {
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

    std::vector<bool> visited(oracle_->size(), false);
    bool built = false;
    while (!built) {
        std::fill(visited.begin(), visited.end(), false);
        dfs(root, graph_, visited);
        built = true;
        for (int i = 0; i < oracle_->size(); ++i) {
            if (visited[i]) {
                continue;
            }
            built = false;
            auto candidates = track_search(
                oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[i].get(), L_, root);
            bool added = false;
            int idx = 0;
            for (auto& candidate : candidates) {
                if (graph_[candidate.id].candidates_.size() >= m_) {
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
}

void
nsg::NSG::build_internal() {
    {
        nndescent::NNDescent nnd(dataset_, K_);
        nnd.build();
        graph_ = std::move(nnd.extract_graph());
    }

    {
        auto* center = new float[oracle_->dim()];
        for (unsigned i = 0; i < oracle_->size(); ++i) {
            auto pt = (*oracle_)[i];
            for (unsigned j = 0; j < oracle_->dim(); ++j) {
                center[j] += pt.get()[j];
            }
        }
        for (unsigned i = 0; i < oracle_->dim(); ++i) {
            center[i] /= oracle_->size();
        }
        root = knn_search(oracle_.get(), visited_list_pool_.get(), graph_, center, 1, L_)[0].id;
        delete[] center;
    }

    logger << "Root: " << root << std::endl;

#pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < graph_.size(); ++u) {
        if (u % 10000 == 0) {
            logger << "Adding " << u << " / " << graph_.size() << std::endl;
        }
        std::vector<Neighbor> candidates = track_search(
            oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[u].get(), L_, root);
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        candidates.erase(
            std::remove_if(
                candidates.begin(), candidates.end(), [u](const Neighbor& n) { return n.id == u; }),
            candidates.end());
        {
            std::lock_guard<std::mutex> guard(graph_[u].lock_);
            graph_[u].candidates_ = prune(candidates);
        }
    }

    //    tree();
}

Neighbors
nsg::NSG::search(const float* query, unsigned int topk, unsigned int L) const {
    return graph::search(
        oracle_.get(), visited_list_pool_.get(), flatten_graph_, query, topk, L, root);
}
void
nsg::NSG::print_info() const {
    Index::print_info();
    logger << "NSG index: " << std::endl;
    logger << "  L: " << L_ << std::endl;
    logger << "  m: " << m_ << std::endl;
    logger << "  K: " << K_ << std::endl;
}
