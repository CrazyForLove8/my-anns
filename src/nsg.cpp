#include "nsg.h"

nsg::NSG::NSG(DatasetPtr& dataset, unsigned int K, unsigned int L, unsigned int m)
    : Index(dataset), L_(L), m_(m), K_(K) {
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

        auto candidates = track_search(
                oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[i], L_, root);
        for (auto& candidate : candidates) {
            if (graph_[candidate.id].candidates_.size() >= m_) {
                continue;
            }
            graph_[candidate.id].addNeighbor(Neighbor(i, candidate.distance, true));
            break;
        }
    }

    timer.end();
    logger << "NSG tree time: " << timer.elapsed() << "s" << std::endl;
}

void
nsg::NSG::build_internal() {
    nndescent::NNDescent nnd(dataset_, K_);
    nnd.build();
    auto knn_graph = nnd.extract_graph();

    {
        auto* center = new double[oracle_->dim()];
        for (unsigned i = 0; i < oracle_->dim(); ++i) {
            center[i] = 0.0;
        }

        for (unsigned i = 0; i < oracle_->size(); ++i) {
            auto pt = (*oracle_)[i];
            for (unsigned j = 0; j < oracle_->dim(); ++j) {
                center[j] += pt[j];
            }
        }
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center[j] /= oracle_->size();
        }

        auto* center_float = new float[oracle_->dim()];
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center_float[j] = static_cast<float>(center[j]);
        }

        delete[] center;

        root = knn_search(oracle_.get(), visited_list_pool_.get(), knn_graph, center_float, 1, L_)[0].id;

        delete[] center_float;
    }

    logger << "Root: " << root << std::endl;

#pragma omp parallel for schedule(dynamic)
    for (IdType u = 0; u < graph_.size(); ++u) {
        if (u % (oracle_->size() / 10) == 0) {
            logger << "Adding " << u << " / " << graph_.size() << std::endl;
        }
        std::vector<Neighbor> candidates =
            track_search(oracle_.get(), visited_list_pool_.get(), graph_, (*oracle_)[u], L_, root);
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        candidates.erase(
            std::remove_if(
                candidates.begin(), candidates.end(), [u](const Neighbor& n) { return n.id == u; }),
            candidates.end());
        prune(candidates);
        // FIXME 这里加锁应该是先加锁ID小的节点，防止死锁，现在多线程会死锁
        {
            std::lock_guard<std::mutex> guard(graph_[u].lock_);
            graph_[u].candidates_.swap(candidates);
        }
        for (auto &neighbor : graph_[u].candidates_) {
            auto a = std::min(u, neighbor.id);
            auto b = std::max(u, neighbor.id);

            std::scoped_lock lock(graph_[a].lock_, graph_[b].lock_);
            graph_[neighbor.id].addNeighbor(Neighbor(u, neighbor.distance, true));
            prune(graph_[neighbor.id].candidates_);
        }
    }

    tree();
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
