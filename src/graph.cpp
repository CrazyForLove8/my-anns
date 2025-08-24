
#include "graph.h"

using namespace graph;

int seed = 2024;

Node::Node(IdType i, float d) {
    id = i;
    distance = d;
}

Node&
Node::operator=(const Node& other) {
    if (this == &other)
        return *this;
    id = other.id;
    distance = other.distance;
    return *this;
}

Neighbor::Neighbor(IdType i, float d, bool f) {
    id = i;
    distance = d;
    flag = f;
}

Neighbor&
Neighbor::operator=(const Neighbor& other) {
    if (this == &other)
        return *this;
    id = other.id;
    distance = other.distance;
    flag = other.flag;
    return *this;
}

Neighbor::Neighbor(const Neighbor& other) {
    id = other.id;
    distance = other.distance;
    flag = other.flag;
}

Neighborhood&
Neighborhood::operator=(const Neighborhood& other) {
    if (this == &other)
        return *this;
    candidates_.clear();
    new_.clear();
    old_.clear();
    reverse_old_.clear();
    reverse_new_.clear();
    candidates_.reserve(other.candidates_.capacity());
    new_.reserve(other.new_.capacity());
    old_.reserve(other.old_.capacity());
    reverse_old_.reserve(other.reverse_old_.capacity());
    reverse_new_.reserve(other.reverse_new_.capacity());
    std::copy(other.candidates_.begin(), other.candidates_.end(), std::back_inserter(candidates_));
    std::copy(other.new_.begin(), other.new_.end(), std::back_inserter(new_));
    std::copy(other.old_.begin(), other.old_.end(), std::back_inserter(old_));
    std::copy(
        other.reverse_old_.begin(), other.reverse_old_.end(), std::back_inserter(reverse_old_));
    std::copy(
        other.reverse_new_.begin(), other.reverse_new_.end(), std::back_inserter(reverse_new_));
    return *this;
}

Neighborhood::Neighborhood(const Neighborhood& other) {
    new_.clear();
    candidates_.clear();
    old_.clear();
    reverse_old_.clear();
    reverse_new_.clear();
    new_.reserve(other.new_.capacity());
    candidates_.reserve(other.candidates_.capacity());
    old_.reserve(other.old_.capacity());
    reverse_old_.reserve(other.reverse_old_.capacity());
    reverse_new_.reserve(other.reverse_new_.capacity());
    std::copy(other.new_.begin(), other.new_.end(), std::back_inserter(new_));
    std::copy(other.candidates_.begin(), other.candidates_.end(), std::back_inserter(candidates_));
    std::copy(other.old_.begin(), other.old_.end(), std::back_inserter(old_));
    std::copy(
        other.reverse_old_.begin(), other.reverse_old_.end(), std::back_inserter(reverse_old_));
    std::copy(
        other.reverse_new_.begin(), other.reverse_new_.end(), std::back_inserter(reverse_new_));
}

unsigned
Neighborhood::pushHeap(IdType id, float dist) {
    std::lock_guard<std::mutex> guard(lock_);
    if (!candidates_.empty() && dist >= candidates_.front().distance)
        return 0;
    for (auto& candidate : candidates_) {
        if (id == candidate.id)
            return 0;
    }
    if (candidates_.size() < candidates_.capacity()) {
        candidates_.emplace_back(id, dist, true);
        std::push_heap(candidates_.begin(), candidates_.end());
    } else {
        std::pop_heap(candidates_.begin(), candidates_.end());
        candidates_[candidates_.size() - 1] = Neighbor(id, dist, true);
        std::push_heap(candidates_.begin(), candidates_.end());
    }
    return 1;
}

void
Neighborhood::addNeighbor(const Neighbor& nn, int capacity) {
    auto it = std::lower_bound(candidates_.begin(), candidates_.end(), nn);
    if (nn.distance >= greatest_distance)
        return;
    if (it == candidates_.end() || it->id != nn.id) {
        candidates_.insert(it, nn);
    }
    if (capacity > 0 && candidates_.size() > capacity) {
        candidates_.pop_back();
    }
    greatest_distance = candidates_.back().distance;
}

void
Neighborhood::move(Neighborhood& other) {
    candidates_.resize(other.candidates_.size());
    std::move(other.candidates_.begin(), other.candidates_.end(), candidates_.begin());
}

FlattenGraph::FlattenGraph(const Graph& graph) {
    auto total = graph.size();
    offsets.resize(total + 1);
    offsets[0] = 0;
    for (int u = 0; u < total; ++u) {
        offsets[u + 1] = offsets[u] + (int)graph[u].candidates_.size();
    }

    final_graph.resize(offsets.back(), -1);
#pragma omp parallel for
    for (int u = 0; u < total; ++u) {
        auto& pool = graph[u].candidates_;
        int offset = offsets[u];
        for (int i = 0; i < pool.size(); ++i) {
            final_graph[offset + i] = pool[i].id;
        }
    }
}

std::vector<int>
FlattenGraph::operator[](int i) const {
    return {final_graph.begin() + offsets[i], final_graph.begin() + offsets[i + 1]};
}

FlattenHGraph::FlattenHGraph(const HGraph& graph) {
    auto levels = graph.size();
    graphs_.reserve(levels);
    for (int level = 0; level < levels; ++level) {
        graphs_.emplace_back(graph[level]);
        if (graphs_.back().final_graph.empty()) {
            graphs_.pop_back();
            break;
        }
    }
}

FlattenGraph&
FlattenHGraph::operator[](int i) const {
    return const_cast<FlattenGraph&>(graphs_[i]);
}

int
FlattenHGraph::size() const {
    return (int)graphs_.size();
}

// find the first -1 in the vector
int
seekPos(const Neighbors& vec) {
    int left = 0, right = vec.size() - 1;
    if (vec.back().id != std::numeric_limits<IdType>::max()) {
        return right;
    }
    int result = left;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (vec[mid].id == std::numeric_limits<IdType>::max()) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

Neighbors
graph::search(IndexOracle<float>* oracle,
              VisitedListPool* visited_list_pool,
              const FlattenGraph& fg,
              const float* query,
              int topk,
              int search_L,
              IdType entry_id,
              int K0) {
    auto visit_pool_ptr = visited_list_pool->getFreeVisitedList();
    auto visit_list = visit_pool_ptr.get();
    auto* visit_array = visit_list->block_;
    auto visit_tag = visit_list->version_;

    const std::vector<int>& offsets = fg.offsets;
    const std::vector<int>& final_graph = fg.final_graph;
    auto total = oracle->size();
    int L = std::max(search_L, topk);
    Neighbors retset(
        L + 1,
        Neighbor(std::numeric_limits<IdType>::max(), std::numeric_limits<float>::max(), false));
    if (entry_id == std::numeric_limits<IdType>::max()) {
        std::vector<int> init_ids;
        init_ids.reserve(L);
        init_ids.resize(L);
        std::mt19937 rng(seed);
        gen_random(rng, init_ids.data(), L, total);
        for (int i = 0; i < L; i++) {
            int id = init_ids[i];
            float dist = (*oracle)(id, query);
            retset[i] = Neighbor(id, dist, true);
        }
        std::sort(retset.begin(), retset.begin() + L);
    } else {
        auto dist = (*oracle)(entry_id, query);
        retset[0] = Neighbor(entry_id, dist, true);
    }

    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            IdType n = retset[k].id;
            int offset = offsets[n];
            int K = offsets[n + 1] - offset;
            K = K > K0 ? K0 : K;
            for (int m = 0; m < K; ++m) {
                int id = final_graph[offset + m];
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
//                _mm_prefetch((*oracle)[id], _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    continue;

                visit_array[id] = visit_tag;
                float dist = (*oracle)(id, query);
                if (dist >= retset[L - 1].distance)
                    continue;

                Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L, nn);

                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }

    int real_end = seekPos(retset);
    retset.resize(std::min(topk, real_end));

    visited_list_pool->releaseVisitedList(visit_pool_ptr);
    return retset;
}

// search from entry_id
Neighbors
graph::knn_search(IndexOracle<float>* oracle,
                  VisitedListPool* visited_list_pool,
                  Graph& graph,
                  const float* query,
                  int topk,
                  int L,
                  IdType entry_id,
                  int graph_sz) {
    auto visit_pool_ptr = visited_list_pool->getFreeVisitedList();
    auto visit_list = visit_pool_ptr.get();
    auto* visit_array = visit_list->block_;
    auto visit_tag = visit_list->version_;

    Neighbors retset(
        L + 1,
        Neighbor(std::numeric_limits<IdType>::max(), std::numeric_limits<float>::max(), false));
    if (entry_id == std::numeric_limits<IdType>::max()) {
        if (graph_sz == -1) {
            graph_sz = graph.size();
        }
        std::mt19937 rng(seed);
        std::vector<int> init_ids;
        int generate_size = std::min(graph_sz, L);
        init_ids.reserve(generate_size);
        init_ids.resize(generate_size);
        gen_random(rng, init_ids.data(), generate_size, graph_sz);
        for (int i = 0; i < generate_size; i++) {
            int id = init_ids[i];
            float dist = (*oracle)(id, query);
            retset[i] = Neighbor(id, dist, true);
        }
        std::sort(retset.begin(), retset.begin() + generate_size);
    } else {
        auto dist = (*oracle)(entry_id, query);
        retset[0] = Neighbor(entry_id, dist, true);
    }

    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            auto n = retset[k].id;
            std::lock_guard<std::mutex> guard(graph[n].lock_);
            for (const auto& candidate : graph[n].candidates_) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
//                _mm_prefetch(&oracle[id], _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    continue;
                visit_array[id] = visit_tag;
                float dist = (*oracle)(id, query);
                if (dist >= retset[L - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L, nn);
                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    int real_end = seekPos(retset);
    retset.resize(std::min(topk, real_end));

    visited_list_pool->releaseVisitedList(visit_pool_ptr);
    return retset;
}

Neighbors
graph::search_layer(IndexOracle<float>* oracle,
                    VisitedListPool* visited_list_pool,
                    HGraph& hgraph,
                    int layer,
                    const float* query,
                    int topk,
                    int L,
                    IdType entry_id) {
    auto visit_pool_ptr = visited_list_pool->getFreeVisitedList();
    auto visit_list = visit_pool_ptr.get();
    auto* visit_array = visit_list->block_;
    auto visit_tag = visit_list->version_;
    auto& graph = hgraph[layer];
    Neighbors retset(
        L + 1,
        Neighbor(std::numeric_limits<IdType>::max(), std::numeric_limits<float>::max(), false));
    auto dist = (*oracle)(entry_id, query);
    retset[0] = Neighbor(entry_id, dist, true);

    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            auto n = retset[k].id;
            std::lock_guard<std::mutex> lock(hgraph[0][n].lock_);
            for (const auto& candidate : graph[n].candidates_) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
//                _mm_prefetch(&oracle[id], _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    continue;
                visit_array[id] = visit_tag;
                dist = (*oracle)(id, query);
                if (dist >= retset[L - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L, nn);
                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k) {
            k = nk;
        } else {
            ++k;
        }
    }
    int real_end = seekPos(retset);
    retset.resize(std::min(topk, real_end));

    visited_list_pool->releaseVisitedList(visit_pool_ptr);
    return retset;
}

Neighbors
graph::track_search(IndexOracle<float>* oracle,
                    VisitedListPool* visited_list_pool,
                    Graph& graph,
                    const float* query,
                    int L,
                    IdType entry_id) {
    auto visit_pool_ptr = visited_list_pool->getFreeVisitedList();
    auto visit_list = visit_pool_ptr.get();
    auto* visit_array = visit_list->block_;
    auto visit_tag = visit_list->version_;

    Neighbors retset(
        L + 1,
        Neighbor(std::numeric_limits<IdType>::max(), std::numeric_limits<float>::max(), false));
    Neighbors track;
    auto dist = (*oracle)(entry_id, query);
    retset[0] = Neighbor(entry_id, dist, true);
    track.emplace_back(entry_id, dist, true);

    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            auto n = retset[k].id;
            std::lock_guard<std::mutex> lock(graph[n].lock_);
            for (const auto& candidate : graph[n].candidates_) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
//                _mm_prefetch(&oracle[id], _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    continue;
                visit_array[id] = visit_tag;
                dist = (*oracle)(id, query);
                if (dist >= retset[L - 1].distance)
                    continue;
                Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L, nn);
                track.emplace_back(id, dist, true);
                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k) {
            k = nk;
        } else {
            ++k;
        }
    }

    visited_list_pool->releaseVisitedList(visit_pool_ptr);
    std::sort(track.begin(), track.end());
    return track;
}

std::string
graph::saveGraph(Graph& graph, const std::string& filename, const ParamMap& params) {
    if (graph.empty()) {
        return "";
    }
    auto path = get_path(filename);
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the file");
    }
    logger << "Saving graph to " << path << std::endl;

    size_t graph_size = graph.size();
    file.write(reinterpret_cast<const char*>(&graph_size), sizeof(graph_size));

    for (size_t i = 0; i < graph.size(); ++i) {
        std::lock_guard<std::mutex> guard(graph[i].lock_);
        file.write(reinterpret_cast<const char*>(&i), sizeof(i));
        size_t num_neighbors = graph[i].candidates_.size();
        file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));
        for (const auto& neighbor : graph[i].candidates_) {
            file.write(reinterpret_cast<const char*>(&neighbor.id), sizeof(neighbor.id));
        }
    }

    ParamsHelper::write(params, file);
    file.close();

    return path;
}

ParamMap
graph::loadGraph(Graph& graph, const std::string& index_path, const OraclePtr& oracle) {
    std::ifstream file(index_path, std::ios::in | std::ios::binary);

    logger << "Loading graph from " << index_path << std::endl;
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the file for reading: " + index_path);
    }

    size_t graph_size;
    file.read(reinterpret_cast<char*>(&graph_size), sizeof(graph_size));
    graph.resize(graph_size);

    for (size_t i = 0; i < graph_size; ++i) {
        size_t node_id;
        file.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));

        size_t num_neighbors;
        file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));

        graph[node_id].candidates_.resize(num_neighbors);
        for (size_t j = 0; j < num_neighbors; ++j) {
            file.read(reinterpret_cast<char*>(&graph[node_id].candidates_[j].id),
                      sizeof(graph[node_id].candidates_[j].id));
        }
    }

    ParamMap params;
    ParamsHelper::read(params, file);
    file.close();

    logger << "Graph loaded successfully, size: " << graph_size << std::endl;

    if (oracle != nullptr) {
        for (size_t i = 0; i < graph.size(); ++i) {
            for (auto& c : graph[i].candidates_) {
                c.distance = (*oracle)(i, c.id);
                c.flag = true;
            }
        }
    }
    return params;
}

std::string
graph::saveHGraph(HGraph& hgraph, const std::string& filename, const ParamMap& params) {
    if (hgraph.empty()) {
        logger << "HGraph is empty, nothing to save." << std::endl;
        return "";
    }
    std::string full_filepath = get_path(filename);
    std::ofstream file(full_filepath, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the file");
    }

    std::vector<int> levels(hgraph[0].size(), 0);
    for (int i = hgraph.size() - 1; i >= 0; --i) {
        for (int j = 0; j < hgraph[i].size(); ++j) {
            if (hgraph[i][j].candidates_.empty()) {
                continue;
            }
            levels[j] = std::max(levels[j], i);
        }
    }

    uint64_t graph_size = hgraph[0].size();
    // We can save less than the full graph size if specified in params
    auto it = params.find("save_point");
    if (it != params.end()) {
        graph_size = std::get<uint64_t>(it->second);
    }

    uint64_t max_level = hgraph.size() - 1;
    logger << "Saving HGraph to " << full_filepath << std::endl;
    logger << "Graph size: " << graph_size << ", Max level: " << max_level << std::endl;
    file.write(reinterpret_cast<const char*>(&graph_size), sizeof(graph_size));
    file.write(reinterpret_cast<const char*>(&max_level), sizeof(max_level));
    for (uint64_t i = 0; i < graph_size; ++i) {
        std::lock_guard<std::mutex> guard(hgraph[0][i].lock_);
        file.write(reinterpret_cast<const char*>(&i), sizeof(i));
        int level = levels[i];
        file.write(reinterpret_cast<const char*>(&level), sizeof(level));
        for (int j = 0; j <= level; ++j) {
            uint64_t num_candidates = hgraph[j][i].candidates_.size();
            file.write(reinterpret_cast<const char*>(&num_candidates), sizeof(num_candidates));
            for (const auto& neighbor : hgraph[j][i].candidates_) {
                file.write(reinterpret_cast<const char*>(&neighbor.id), sizeof(neighbor.id));
            }
        }
    }

    ParamsHelper::write(params, file);
    file.close();

    return full_filepath;
}

ParamMap
graph::loadHGraph(HGraph& hgraph, const std::string& index_path, const OraclePtr& oracle) {
    std::ifstream file(index_path, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the file for reading: " + index_path);
    }

    uint64_t graph_size, max_level;
    file.read(reinterpret_cast<char*>(&graph_size), sizeof(graph_size));
    file.read(reinterpret_cast<char*>(&max_level), sizeof(max_level));

    hgraph.resize(max_level + 1);
    for (int i = 0; i <= max_level; ++i) {
        hgraph[i].resize(graph_size);
    }

    logger << "Loading HGraph from " << index_path << std::endl;
    logger << "Graph size: " << graph_size << ", Max level: " << max_level << std::endl;
    for (int i = 0; i < graph_size; ++i) {
        uint64_t node_id;
        file.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));
        int level;
        file.read(reinterpret_cast<char*>(&level), sizeof(level));
        for (int j = 0; j <= level; ++j) {
            uint64_t num_neighbors;
            file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
            hgraph[j][node_id].candidates_.resize(num_neighbors);
            for (int k = 0; k < num_neighbors; ++k) {
                file.read(reinterpret_cast<char*>(&hgraph[j][node_id].candidates_[k].id),
                          sizeof(hgraph[j][node_id].candidates_[k].id));
            }
        }
    }

    ParamMap params;
    ParamsHelper::read(params, file);
    file.close();

    if (oracle != nullptr) {
        for (int i = 0; i < static_cast<int>(hgraph.size()); ++i) {
            for (int j = 0; j < static_cast<int>(hgraph[i].size()); ++j) {
                auto& candidates = hgraph[i][j].candidates_;
                candidates.erase(std::remove_if(candidates.begin(),
                                                candidates.end(),
                                                [&](const Neighbor& n) {
                                                    return n.id < 0 || n.id >= graph_size;
                                                }),
                                 candidates.end());

                for (auto& c : candidates) {
                    c.distance = (*oracle)(j, c.id);
                    c.flag = true;
                }
            }
        }
    }

    logger << "HGraph loaded successfully, size: " << graph_size << ", max level: " << max_level
           << std::endl;
    return params;
}

int
graph::checkConnectivity(const Graph& graph) {
    const int n = static_cast<int>(graph.size());
    std::vector<bool> visited(n, false);
    std::stack<int> finishStack;

    auto dfs1 = [&](int start, const Graph& g) {
        std::stack<int> s;
        s.push(start);
        visited[start] = true;

        while (!s.empty()) {
            int node = s.top();
            bool finished = true;
            for (const auto& neighbor : g[node].candidates_) {
                if (!visited[neighbor.id]) {
                    visited[neighbor.id] = true;
                    s.push(neighbor.id);
                    finished = false;
                    break;
                }
            }
            if (finished) {
                s.pop();
                finishStack.push(node);
            }
        }
    };

    auto dfs2 = [&](int start, const Graph& g, std::vector<bool>& visited) {
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

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            dfs1(i, graph);
        }
    }

    Graph reverseGraph(n);
    for (int i = 0; i < n; ++i) {
        for (const auto& neighbor : graph[i].candidates_) {
            reverseGraph[neighbor.id].candidates_.emplace_back(i, neighbor.distance, false);
        }
    }

    std::fill(visited.begin(), visited.end(), false);
    int componentCount = 0;
    while (!finishStack.empty()) {
        int node = finishStack.top();
        finishStack.pop();

        if (!visited[node]) {
            componentCount++;
            dfs2(node, reverseGraph, visited);
        }
    }
    return componentCount;
}
