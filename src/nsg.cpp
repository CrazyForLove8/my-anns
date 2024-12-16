#include "nsg.h"

void nsg::NSG::build(Graph &graph,
                     IndexOracle &oracle) {
    auto *center = new float[oracle.dim()];
    for (unsigned i = 0; i < oracle.size(); ++i) {
        auto pt = oracle[i];
        for (unsigned j = 0; j < oracle.dim(); ++j) {
            center[j] += pt[j];
        }
    }
    for (unsigned i = 0; i < oracle.dim(); ++i) {
        center[i] /= oracle.size();
    }
    root = knn_search(oracle, graph, center, 1, L_)[0].id;
    delete center;
    logger << "Root: " << root << std::endl;

    for (int u = 0; u < graph.size(); ++u) {
        if (u % 10000 == 0) {
            logger << "Adding " << u << " / " << graph.size() << std::endl;
        }
        std::vector<Neighbor> candidates = track_search(oracle, graph, oracle[u], root, L_);
        graph[u].candidates_ = prune(oracle, candidates);
    }

    tree(graph, oracle, root);
}

std::vector<Neighbor> nsg::NSG::prune(IndexOracle &oracle,
                                      std::vector<Neighbor> &candidates) {
    std::vector<Neighbor> prunedNeighbors;
    for (auto &&v: candidates) {
        auto flag = false;
        for (auto &&w: prunedNeighbors) {
            if (oracle(w.id, v.id) < v.distance) {
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

void nsg::NSG::tree(Graph &graph,
                    IndexOracle &oracle,
                    int root) {
    /**
     * TODO Here needs to be fixed
     * Too many recursion
     */
    auto dfs = [&](int start,
                   const Graph &g,
                   std::vector<bool> &visited) {
        std::stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int node = s.top();
            s.pop();
            for (const auto &neighbor: g[node].candidates_) {
                if (!visited[neighbor.id]) {
                    visited[neighbor.id] = true;
                    s.push(neighbor.id);
                }
            }
        }
    };
    std::vector<bool> visited(graph.size(), false);
    bool built = false;
    while (!built) {
        dfs(root, graph, visited);
        built = true;
        for (int i = 0; i < graph.size(); ++i) {
            if (!visited[i]) {
                built = false;
                auto candidates = track_search(oracle, graph, oracle[i], root, L_);
                bool added = false;
                int idx = 0;
                for (auto &&candidate: candidates) {
                    if (!visited[candidate.id]) {
                        continue;
                    }
                    graph[candidate.id].addNeighbor(Neighbor(i, candidate.distance, true));
                    added = true;
                    logger << candidate.id << " " << i << std::endl;
                    break;
                }

                if (!added) {
                    std::mt19937 rng(2024);
                    do {
                        idx = rng() % graph.size();
                        if (visited[idx]) {
                            graph[idx].addNeighbor(Neighbor(i, oracle(idx, i), true));
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

