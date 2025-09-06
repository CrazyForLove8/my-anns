//
// Created by XiaoWu on 2025/2/14.
//

#ifndef MYANNS_INDEX_H
#define MYANNS_INDEX_H

#include <omp.h>

#include <random>
#include <unordered_set>

#include "dataset.h"
#include "dtype.h"
#include "graph.h"
#include "logger.h"
#include "memory.h"
#include "metric.h"
#include "timer.h"
#include "visittable.h"

using namespace graph;

struct SaveHelper {
    uint8_t save_frequency{0};
    std::string save_path;

    uint64_t save_per_count{0};
    uint64_t total_count{0};
    uint64_t last_save_point{0};

    [[nodiscard]] bool
    is_enabled() const {
        return save_frequency > 0 && !save_path.empty();
    }

    [[nodiscard]] uint64_t
    get_interval() const {
        return save_per_count > 0 ? save_per_count : ((save_frequency + 1) * 100000000);
    }

    [[nodiscard]] bool
    should_save(uint64_t u) const {
        return is_enabled() && u % get_interval() == 0 && u > last_save_point &&
               u + save_per_count <= total_count;
    }
};

class Index {
protected:
    Graph graph_;

    DatasetPtr dataset_;

    OraclePtr oracle_;

    MatrixPtr<float> base_;

    VisitedListPoolPtr visited_list_pool_;

    std::mutex graph_lock_;

    FlattenGraph flatten_graph_;

    IdType cur_size_{0};

    bool built_{false};

    SaveHelper save_helper_;

    virtual void
    build_internal();

    virtual void
    partial_build(IdType start, IdType end);

public:
    Index();

    // TODO In the future, we shall store the original vectors in the index
    explicit Index(DatasetPtr& dataset, bool allocate = true);

    explicit Index(DatasetPtr& dataset, Graph& graph);

    virtual ~Index() = default;

    virtual void
    set_save_helper(const SaveHelper& saveHelper);

    virtual void
    reset(DatasetPtr& dataset);

    virtual void
    build();

    virtual void
    add(DatasetPtr& dataset);

    virtual void
    partial_build(IdType num);

    virtual void
    remove(IdType id);

    virtual Graph&
    extract_graph();

    virtual FlattenGraph&
    extract_flatten_graph();

    virtual DatasetPtr&
    extract_dataset();

    virtual ParamMap
    extract_params();

    virtual void
    load_params(const ParamMap& params);

    /**
     * @brief The basic search function. It initializes with L random nodes and greedily expands the candidates. The results are pruned by the topk.
     * @param query
     * @param topk
     * @param L
     * @return
     */
    virtual Neighbors
    search(const float* query, unsigned int topk, unsigned int L) const;

    virtual void
    print_info() const;
};

using IndexPtr = std::shared_ptr<Index>;

class IndexWrapper : public Index {
public:
    // TODO replace graph with index_path
    explicit IndexWrapper(DatasetPtr& dataset, Graph& graph);

    explicit IndexWrapper(IndexPtr& index);

    IndexWrapper() = default;

    ~IndexWrapper() override = default;

    void
    append(std::vector<IndexPtr>& indexes);
};

// TODO Support IndexFactory

#endif  //MYANNS_INDEX_H
