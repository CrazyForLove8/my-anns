//
// Created by XiaoWu on 2024/11/24.
//

#ifndef MYANNS_MERGE_H
#define MYANNS_MERGE_H

#include "dtype.h"
#include "graph.h"
#include "logger.h"
#include "timer.h"
#include "dataset.h"
#include "index.h"
#include "hnsw.h"

class FGIM : public Index {
protected:
    unsigned max_degree_;

    float sample_rate_;

//        void
//        Sampling(Graph &graph,
//                 const Graph &g1,
//                 const Graph &g2,
//                 OraclePtr &oracle1,
//                 OraclePtr &oracle2,
//                 OraclePtr &oracle);
    virtual void
    CrossQuery(std::vector<IndexPtr> &indexes);

    virtual void
    Refinement();

    void
    update_neighbors(Graph &graph);

    void
    prune(Graph &graph);

    void
    add_reverse_edge(Graph &graph);

    void
    connect_no_indegree(Graph &graph);

public:
    static constexpr unsigned ITER_MAX = 100;

    static constexpr unsigned SAMPLES = 100;

    static constexpr float THRESHOLD = 0.001;

    FGIM();

    explicit FGIM(unsigned max_degree,
                  float sample_rate = 0.3);

    ~FGIM() override = default;

//        /**
//               * @brief FGIM two PGs
//               * @param g1  The first graph
//               * @param oracle1 The distance oracle of the first graph
//               * @param g2  The second nearest neighbor graph
//               * @param oracle2 The distance oracle of the second graph
//               * @param oracle The distance oracle
//               * @return The merged graph
//               */
//        Graph
//        merge(const Graph &g1,
//              OraclePtr &oracle1,
//              const Graph &g2,
//              OraclePtr &oracle2,
//              OraclePtr &oracle);

    virtual void
    Combine(std::vector<IndexPtr> &indexes);
};

#endif  // MYANNS_MERGE_H
