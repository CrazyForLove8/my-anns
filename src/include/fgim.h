//
// Created by XiaoWu on 2024/11/24.
//

#ifndef MYANNS_MERGE_H
#define MYANNS_MERGE_H

#include "dataset.h"
#include "dtype.h"
#include "graph.h"
#include "hnsw.h"
#include "index.h"
#include "logger.h"
#include "timer.h"

class FGIM : public Index {
protected:
    uint32_t max_degree_{32};

    uint32_t max_base_degree_{64};

    float sample_rate_{0.3};

    std::vector<uint64_t> offsets_;

    uint64_t start_id_{0};

    uint64_t start_iter_{1};

    uint64_t max_index_size_{0};

    std::string cur_phase_{"1"};

    //TODO we can support index_name_ for each index
    std::string serial_;

    virtual void
    CrossQuery(std::vector<IndexPtr>& indexes);

    virtual void
    Refinement();

    void
    update_neighbors(Graph& graph);

    void
    prune(Graph& graph, bool add = false);

    void
    add_reverse_edge(Graph& graph);

    void
    connect_no_indegree(Graph& graph);

    [[nodiscard]] bool
    are_in_same_index(size_t id1, size_t id2) const;

    void
    load_latest(Graph& graph, const std::filesystem::path& directoryPath = "./graph_output/");

public:
    static constexpr unsigned ITER_MAX = 30;

    static constexpr unsigned SAMPLES = 100;

    static constexpr float THRESHOLD = 0.002;

    FGIM();

    explicit FGIM(unsigned max_degree, float sample_rate = 0.3);

    explicit FGIM(DatasetPtr& dataset,
                  unsigned max_degree = 20,
                  float sample_rate = 0.3,
                  bool allocate = true);

    ~FGIM() override = default;

    virtual void
    combine(std::vector<IndexPtr>& indexes);

    void
    set_serial(const std::string& serial);

    [[nodiscard]] const std::string&
    get_serial() const;

    ParamMap
    extract_params() override;

    void
    print_info() const override;
};

#endif  // MYANNS_MERGE_H
