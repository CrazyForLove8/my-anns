#include "nsg.h"

nsg::NSG::NSG(const IndexParam& param, unsigned int K, unsigned int L, unsigned int m)
    : Index(param), L_(L), m_(m), K_(K) {
}

//TODO Extract pruning strategy to a separate class like metrics
void
nsg::NSG::prune(Neighbors& candidates) {
    if (candidates.size() <= m_) {
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
        if (ret_set.size() >= m_) {
            break;
        }
    }
    candidates.swap(ret_set);
}

void
nsg::NSG::tree() {
    /**
   * TODO Here needs to be fixed
   * Too many recursion
   */
    std::vector<IdType> stack;
    stack.reserve(4 * m_);
    auto dfs = [&](IdType start, const Graph& g, Bitset& visited) {
        stack.emplace_back(start);
        visited.set(start);
        IdType visited_cnt = 1;

        while (!stack.empty() && visited_cnt < oracle_->size()) {
            auto node = stack.back();
            stack.pop_back();
            for (const auto& neighbor : g[node].candidates_) {
                if (!visited.test_and_set(neighbor.id)) {
                    stack.emplace_back(neighbor.id);
                    ++visited_cnt;
                }
            }
        }
    };

    Timer timer;
    timer.start();

    // Bitset visited(oracle_->size());
    // bool built = false;
    // while (!built) {
    //     visited.clear();
    //     dfs(root, graph_, visited);
    //     built = true;
    //     for (int i = 0; i < oracle_->size(); ++i) {
    //         if (visited.test(i)) {
    //             continue;
    //         }
    //         built = false;
    //         auto candidates = search_one_graph_track(
    //             oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[i], L_, root);
    //         for (auto& candidate : candidates) {
    //             if (graph_[candidate.id].candidates_.size() >= m_) {
    //                 continue;
    //             }
    //             graph_[candidate.id].addNeighbor(Neighbor(i, candidate.distance, true));
    //             break;
    //         }
    //         break;
    //     }
    // }

    Bitset visited(oracle_->size());
    IdType next_unvisited = 0, next_root = root;
    while (true) {
        dfs(next_root, graph_, visited);
        while (next_unvisited < oracle_->size() && visited.test(next_unvisited)) {
            ++next_unvisited;
        }
        if (next_unvisited == oracle_->size()) break;
        const auto i = next_unvisited;
        next_root = i;

        auto candidates = search_one_graph(
                oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[i], L_, L_, root);
        for (auto& candidate : candidates) {
            if (graph_[candidate.id].candidates_.size() >= m_) {
                continue;
            }
            graph_[candidate.id].addNeighbor(Neighbor(i, candidate.distance, true));
            break;
        }
    }

    // Bitset visited(oracle_->size());
    // IdType next_unvisited = 0;
    // dfs(root, graph_, visited);
    // while (true) {
    //     while (next_unvisited < oracle_->size() && visited.test(next_unvisited)) {
    //         ++next_unvisited;
    //     }
    //     if (next_unvisited == oracle_->size()) break;
    //     const auto i = next_unvisited;
    //
    //     auto candidates = search_one_graph(
    //             oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[i], L_, L_, root);
    //     for (auto& candidate : candidates) {
    //         if (graph_[candidate.id].candidates_.size() >= m_) {
    //             continue;
    //         }
    //         graph_[candidate.id].addNeighbor(Neighbor(i, candidate.distance, true));
    //         break;
    //     }
    // }

    timer.end();
    logger << "NSG tree time: " << timer.elapsed() << "s" << std::endl;
}

void
nsg::NSG::build_internal(DatasetPtr& dataset) {
    {
        nndescent::NNDescent nnd(index_param_, K_);
        nnd.build(dataset);
        graph_ = std::move(nnd.extract_graph());
    }

    {
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
        root =
            search_one_graph(oracle_.get(), visited_list_pool_.get(), graph_, center, 1, L_)[0].id;
        delete[] center;
    }

    logger << "Root: " << root << std::endl;

//     {
//         Graph C(oracle_->size());
// #pragma omp parallel for schedule(dynamic)
//         for (int u = 0; u < graph_.size(); ++u) {
//             auto candidates = search_one_graph_track(
//                 oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[u], L_, root);
//             candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
//             candidates.erase(
//                 std::remove_if(
//                     candidates.begin(), candidates.end(), [u](const Neighbor& n) { return n.id == u; }),
//                 candidates.end());
//             prune(candidates);
//             C[u].candidates_.swap(candidates);
//         }
//         logger << "knnsearch done." << std::endl;
//
// #pragma omp parallel for schedule(dynamic)
//         for (int u = 0; u < graph_.size(); ++u) {
//             graph_[u].candidates_.swap(C[u].candidates_);
//             C[u].candidates_.clear();
//         }
//
// #pragma omp parallel for schedule(dynamic)
//         for (int u = 0; u < graph_.size(); ++u) {
//             for (auto & neighbor : graph_[u].candidates_) {
//                 std::lock_guard lock(graph_[neighbor.id].lock_);
//                 C[neighbor.id].candidates_.emplace_back(u, neighbor.distance, true);
//             }
//         }
//
// #pragma omp parallel for schedule(dynamic)
//         for (int u = 0; u < graph_.size(); ++u) {
//             graph_[u].candidates_.insert(
//                 graph_[u].candidates_.end(), C[u].candidates_.begin(), C[u].candidates_.end());
//             std::sort(graph_[u].candidates_.begin(), graph_[u].candidates_.end());
//             graph_[u].candidates_.erase(
//                 std::unique(graph_[u].candidates_.begin(), graph_[u].candidates_.end()),
//                 graph_[u].candidates_.end());
//             prune(graph_[u].candidates_);
//         }
//     }

#pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < graph_.size(); ++u) {
        if (u % (oracle_->size() / 10) == 0) {
            logger << "Adding " << u << " / " << graph_.size() << std::endl;
        }
        std::vector<Neighbor> candidates = search_one_graph_track(
            oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[u], L_, root);
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        candidates.erase(
            std::remove_if(
                candidates.begin(), candidates.end(), [u](const Neighbor& n) { return n.id == u; }),
            candidates.end());
        prune(candidates);
        {
            std::lock_guard<std::mutex> guard(graph_[u].lock_);
            graph_[u].candidates_.swap(candidates);
        }
        for (auto &neighbor : graph_[u].candidates_) {
            std::lock_guard lock(graph_[neighbor.id].lock_);
            graph_[neighbor.id].addNeighbor(Neighbor(u, neighbor.distance, true));
            prune(graph_[neighbor.id].candidates_);
        }
    }

    tree();
}

Neighbors
nsg::NSG::search(const float* query, unsigned int topk, unsigned int L) const {
    return search_flatten_graph(
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
