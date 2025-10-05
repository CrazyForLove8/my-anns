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
#include "vectors.h"
#include "visittable.h"

using namespace graph;

struct IndexParam {
    DimType dim_{0};
    IdType init_size_{1000};
    metric::DISTANCE metric_{metric::DISTANCE::L2};
    IOType io_type_{IOType::FILE_IO};
};

class Index {
protected:
    Graph graph_;

    IndexParam index_param_;

    VectorsPtr<float> oracle_;

    VisitedListPoolPtr visited_list_pool_;

    std::mutex graph_lock_;

    FlattenGraph flatten_graph_;

    IdType cur_size_{0};

    bool built_{false};

    SaveHelper save_helper_;

    virtual void
    build_internal(DatasetPtr& dataset);

    virtual void
    partial_build(IdType start, IdType end);

    virtual void
    resize(IdType new_size);

public:
    explicit Index(const IndexParam& param, bool allocate = true);

    virtual ~Index() = default;

    virtual void
    set_save_helper(const SaveHelper& saveHelper);

    virtual void
    build(DatasetPtr& dataset);

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

    virtual ParamMap
    extract_params();

    virtual VectorsPtr<float>
    extract_vectors();

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

// TODO Support IndexFactory

#endif  //MYANNS_INDEX_H
