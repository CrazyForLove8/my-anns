//
// Created by XiaoWu on 2025/9/21.
//

#ifndef MYANNS_VECTORS_H
#define MYANNS_VECTORS_H

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "dtype.h"
#include "graph.h"
#include "io.h"
#include "memory.h"
#include "metric.h"
#include "typing.h"

namespace graph {

template <typename T>
class Vectors;
template <typename T>
using VectorsPtr = std::shared_ptr<Vectors<T> >;

template <typename T>
class Vectors {
    IdType nums_{0};
    DimType dim_{0};

    IOPtr io_{nullptr};
    metric::DistFunc dist_func_{nullptr};

    std::shared_mutex rw_mutex_;

    // std::unordered_set<IdType> deleted_ids_; // deletion should be handled in the index level

public:
    explicit Vectors(const IdType dim,
                     const metric::DISTANCE dist_type,
                     const IOType io_type = IOType::MEMORY_IO)
        : dim_(dim) {
        logger << "Creating Vectors with dimension: " << dim_ << std::endl;
        switch (io_type) {
            case IOType::MEMORY_IO:
                io_ = std::make_shared<MemoryIO>(dim * sizeof(T));
                logger << "Using Memory IO for Vectors." << std::endl;
                break;
            case IOType::FILE_IO:
                logger << "File IO is not yet supported for Vectors." << std::endl;
                // TODO support file io
                break;
            default:
                throw std::invalid_argument("Unsupported IO type");
        }

        switch (dist_type) {
            case metric::DISTANCE::L2:
                dist_func_ = metric::l2_dist;
                logger << "Using L2 distance for Vectors." << std::endl;
                break;
            case metric::DISTANCE::COSINE:
                dist_func_ = metric::angular_dist;
                logger << "Using Cosine distance for Vectors." << std::endl;
                break;
            default:
                throw std::invalid_argument("Unsupported distance type");
        }
    }

    void
    insert(DataPtr vec, IdType id = std::numeric_limits<IdType>::max()) {
        if (vec == nullptr) {
            throw std::invalid_argument("Invalid vector in insert");
        }
        if (id == std::numeric_limits<IdType>::max()) {
            id = nums_;
            ++nums_;
        } else {
            nums_ = std::max(nums_, id + 1);
        }
        io_->write(vec, dim_ * sizeof(T), id);
    }

    void
    insert(MatrixPtr<T> mat, const IdType* ids = nullptr) {
        if (mat == nullptr || mat->size() == 0) {
            throw std::invalid_argument("Invalid matrix in batchInsertVectors");
        }

        const int n = static_cast<int>(mat->size());
        if (ids == nullptr) {
            logger << "Ids not provided, inserting vectors with continuous ids." << std::endl;
            auto cnt = 0;
            logger << "Current number of vectors: " << nums_ << std::endl;
            logger << "Inserting " << n << " vectors." << std::endl;
            {
                std::lock_guard lock(rw_mutex_);
                cnt = nums_;
                nums_ += n;
            }
            for (IdType i = cnt, j = 0; i < cnt + n; ++i, ++j) {
                io_->write(reinterpret_cast<DataPtr>((*mat)[j].get()), dim_ * sizeof(T), i);
            }
        } else {
            for (IdType i = 0; i < n; ++i) {
                this->insert(reinterpret_cast<DataPtr>((*mat)[i].get()), ids[i]);
            }
        }
    }

    void
    insert(DataPtr data, const int n, const IdType* ids = nullptr) {
        if (data == nullptr) {
            throw std::invalid_argument("Invalid data in batchInsertVectors");
        }
        if (ids == nullptr) {
            logger << "Ids not provided, inserting vectors with continuous ids." << std::endl;
            auto cnt = 0;
            logger << "Current number of vectors: " << nums_ << std::endl;
            logger << "Inserting " << n << " vectors." << std::endl;
            {
                std::lock_guard lock(rw_mutex_);
                cnt = nums_;
                nums_ += n;
            }
            io_->write(data, n * dim_ * sizeof(T), cnt);
        } else {
            for (IdType i = 0; i < n; ++i) {
                this->insert(data + i * dim_ * sizeof(T), ids[i]);
            }
        }
    }

    void
    insert(VectorsPtr<T> other,
           const IdType offset = std::numeric_limits<IdType>::max()) {
        if (other == nullptr || other->size() == 0) {
            return;
        }
        if (other->dim() != dim_) {
            throw std::invalid_argument("Dimension mismatch in append");
        }
        IdType cnt = 0;
        {
            std::lock_guard lock(rw_mutex_);
            if (offset == std::numeric_limits<IdType>::max()) {
                cnt = nums_;
                nums_ += other->size();
            } else {
                cnt = offset;
                nums_ = std::max(nums_, offset + other->size());
            }
        }
        for (IdType i = 0; i < other->size(); ++i) {
            io_->write(other->io_->read(dim_ * sizeof(T), i),
                       dim_ * sizeof(T),
                       cnt + i);
        }
    }

    T*
    operator[](const IdType idx) const {
        if (idx >= nums_) {
            throw std::out_of_range("Index out of range in Vectors");
        }
        const auto ptr = io_->read(dim_ * sizeof(T), idx);
        return reinterpret_cast<T*>(ptr);
    }

    T
    operator()(unsigned i, unsigned j) const {
        return dist_func_((*this)[i], (*this)[j], dim_);
    }

    T
    operator()(unsigned i, const T* vec) const {
        return dist_func_(vec, (*this)[i], dim_);
    }

    T
    operator()(const T* vec1, const T* vec2) const {
        return dist_func_(vec1, vec2, dim_);
    }

    [[nodiscard]] DimType
    dim() const {
        return dim_;
    }

    [[nodiscard]] IdType
    size() const {
        return nums_;
    }
};

inline auto
search_flatten_graph(const Vectors<float>* oracle,
                     VisitedListPool* visited_list_pool,
                     const FlattenGraph& fg,
                     const float* query,
                     const int topk,
                     const int search_L,
                     const IdType entry_id = std::numeric_limits<IdType>::max(),
                     const int K0 = 128) -> Neighbors {
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

template <bool lock_free = false>
inline Neighbors
search_one_graph(const Vectors<float>* oracle,
                 VisitedListPool* visited_list_pool,
                 Graph& graph,
                 const float* query,
                 int topk,
                 int L,
                 IdType entry_id = std::numeric_limits<IdType>::max(),
                 int graph_sz = -1) {
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
            auto expand = [&](const auto &candidate) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    return;
                visit_array[id] = visit_tag;

                auto dist_local = (*oracle)(id, query);
                if (dist_local >= retset[L - 1].distance)
                    return;

                Neighbor nn(id, dist_local, true);
                int r = insert_into_pool(retset.data(), L, nn);
                if (r < nk)
                    nk = r;
            };

            if constexpr (!lock_free) {
                std::lock_guard<std::mutex> guard(graph[n].lock_);
                for (const auto &c: graph[n].candidates_) expand(c);
            } else {
                for (const auto &c: graph[n].candidates_) expand(c);
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

    template<bool lock_free = false>
    inline Neighbors
    search_one_graph_with_seed(const Vectors<float> *oracle,
                               VisitedListPool *visited_list_pool,
                               Graph &graph,
                               const float *query,
                               int topk,
                               int L,
                               std::vector<IdType> seed_ids) {
        auto visit_pool_ptr = visited_list_pool->getFreeVisitedList();
        auto visit_list = visit_pool_ptr.get();
        auto *visit_array = visit_list->block_;
        auto visit_tag = visit_list->version_;

        Neighbors retset(
                L + 1,
                Neighbor(std::numeric_limits<IdType>::max(), std::numeric_limits<float>::max(), false));
        for (int i = 0; i < std::min((int) seed_ids.size(), L); i++) {
            if (seed_ids[i] == std::numeric_limits<IdType>::max()) {
                continue;
            }
            IdType id = seed_ids[i];
            float dist = (*oracle)(id, query);
            retset[i] = Neighbor(id, dist, true);
        }
        std::sort(retset.begin(), retset.begin() + std::min((int) seed_ids.size(), L));

        int k = 0;
        while (k < L) {
            int nk = L;
            if (retset[k].flag) {
            retset[k].flag = false;
            auto n = retset[k].id;
            auto expand = [&](const auto& candidate) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    return;
                visit_array[id] = visit_tag;

                auto dist_local = (*oracle)(id, query);
                if (dist_local >= retset[L - 1].distance)
                    return;

                Neighbor nn(id, dist_local, true);
                int r = insert_into_pool(retset.data(), L, nn);
                if (r < nk)
                    nk = r;
            };

            if constexpr (!lock_free) {
                std::lock_guard<std::mutex> guard(graph[n].lock_);
                for (const auto& c : graph[n].candidates_) expand(c);
            } else {
                for (const auto& c : graph[n].candidates_) expand(c);
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

inline Neighbors
search_hgraph_layer(const Vectors<float>* oracle,
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

inline Neighbors
search_hgraph_layer_with_seed(const Vectors<float> *oracle,
                              VisitedListPool *visited_list_pool,
                              HGraph &hgraph,
                              int layer,
                              const float *query,
                              int topk,
                              int L,
                              std::vector<IdType> seed_ids) {
    auto visit_pool_ptr = visited_list_pool->getFreeVisitedList();
    auto visit_list = visit_pool_ptr.get();
    auto *visit_array = visit_list->block_;
    auto visit_tag = visit_list->version_;
    auto &graph = hgraph[layer];
    Neighbors retset(
            L + 1,
            Neighbor(std::numeric_limits<IdType>::max(), std::numeric_limits<float>::max(), false));
    for (int i = 0; i < std::min((int) seed_ids.size(), L); i++) {
        if (seed_ids[i] == std::numeric_limits<IdType>::max()) {
            continue;
        }
        IdType id = seed_ids[i];
        float dist = (*oracle)(id, query);
        retset[i] = Neighbor(id, dist, true);
    }
    std::sort(retset.begin(), retset.begin() + std::min((int) seed_ids.size(), L));

    int k = 0;
    while (k < L) {
        int nk = L;
        if (retset[k].flag) {
            retset[k].flag = false;
            auto n = retset[k].id;
            std::lock_guard<std::mutex> lock(hgraph[0][n].lock_);
            for (const auto &candidate: graph[n].candidates_) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
//                _mm_prefetch(&oracle[id], _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    continue;
                visit_array[id] = visit_tag;
                auto dist = (*oracle)(id, query);
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

template <bool lock_free = false>
inline Neighbors
search_one_graph_track(const Vectors<float>* oracle,
                       VisitedListPool* visited_list_pool,
                       Graph& graph,
                       const float* query,
                       const int L,
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

            auto expand = [&](const auto& candidate) {
                auto id = candidate.id;
#ifdef USE_SSE
                _mm_prefetch(visit_array + id, _MM_HINT_T0);
#endif
                if (visit_array[id] == visit_tag)
                    return;
                visit_array[id] = visit_tag;

                auto dist_local = (*oracle)(id, query);
                if (dist_local >= retset[L - 1].distance)
                    return;

                Neighbor nn(id, dist_local, true);
                int r = insert_into_pool(retset.data(), L, nn);
                track.emplace_back(id, dist_local, true);
                if (r < nk)
                    nk = r;
            };

            if constexpr (!lock_free) {
                std::lock_guard<std::mutex> guard(graph[n].lock_);
                for (const auto& c : graph[n].candidates_) expand(c);
            } else {
                for (const auto& c : graph[n].candidates_) expand(c);
            }
        }
        k = (nk <= k) ? nk : (k + 1);
    }

    visited_list_pool->releaseVisitedList(visit_pool_ptr);
    std::sort(track.begin(), track.end());
    return track;
}

}  // namespace graph

#endif  //MYANNS_VECTORS_H
