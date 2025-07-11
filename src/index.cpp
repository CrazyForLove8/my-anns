#include "index.h"

Index::Index()
    : dataset_(nullptr),
      base_(nullptr),
      oracle_(nullptr),
      built_(false),
      visited_list_pool_(nullptr) {
}

Index::Index(DatasetPtr& dataset, bool allocate)
    : dataset_(dataset),
      oracle_(dataset->getOracle()),
      base_(dataset->getBasePtr()),
      visited_list_pool_(dataset->getVisitedListPool()),
      built_(false) {
    if (allocate) {
        graph_.reserve(oracle_->size());
        graph_.resize(oracle_->size());
    }
}

Index::Index(DatasetPtr& dataset, Graph& graph)
    : dataset_(dataset),
      oracle_(dataset->getOracle()),
      base_(dataset->getBasePtr()),
      visited_list_pool_(dataset->getVisitedListPool()),
      graph_(std::move(graph)),
      built_(true) {
    flatten_graph_ = FlattenGraph(graph_);
}

void
Index::build_internal() {
    throw std::runtime_error("Index does not support build");
}

void
Index::build() {
    print_info();
    if (built_) {
        logger << "Index is already built, skipping build." << std::endl;
        return;
    }

    Timer timer;
    timer.start();

    build_internal();

    timer.end();
    logger << "Indexing time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

void
Index::set_save_helper(const SaveHelper& saveHelper) {
    save_helper_.save_frequency = saveHelper.save_frequency;
    save_helper_.save_path = saveHelper.save_path;

    save_helper_.total_count = oracle_->size();
    save_helper_.save_per_count = save_helper_.total_count / saveHelper.save_frequency;
}

void
Index::reset(DatasetPtr& dataset) {
    dataset_ = dataset;
    oracle_ = dataset->getOracle();
    base_ = dataset->getBasePtr();
    visited_list_pool_ = dataset->getVisitedListPool();
    graph_.clear();
    graph_.reserve(oracle_->size());
    graph_.resize(oracle_->size());
    built_ = false;
}

Graph&
Index::extract_graph() {
    if (!built_) {
        throw std::runtime_error("Index is not built");
    }
    return graph_;
}

DatasetPtr&
Index::extract_dataset() {
    return dataset_;
}

void
Index::add(DatasetPtr& dataset) {
    throw std::runtime_error("Index does not support add");
}

Neighbors
Index::search(const float* query, unsigned int topk, unsigned int L) const {
    if (!built_) {
        throw std::runtime_error("Index is not built");
    }
    return graph::search(oracle_.get(), visited_list_pool_.get(), flatten_graph_, query, topk, L);
}

void
Index::print_info() const {
    logger << "Indexing settings:" << std::endl;
    logger << "Dataset: " << dataset_->getName() << std::endl;
    logger << "Dataset Size: " << oracle_->size() << std::endl;
#pragma omp parallel
    {
#pragma omp single
        { logger << "Number of threads: " << omp_get_num_threads() << std::endl; }
    }
    print_memory_usage();
}

FlattenGraph&
Index::extract_flatten_graph() {
    if (!built_) {
        throw std::runtime_error("Index is not built");
    }
    return flatten_graph_;
}

ParamMap
Index::extract_params() {
    ParamMap params;
    params["index_type"] = "Index";
    params["built"] = built_ ? 1ULL : 0ULL;
    return params;
}

void
Index::load_params(const ParamMap& params) {
    throw std::runtime_error("Index does not need to load parameters");
}

IndexWrapper::IndexWrapper(DatasetPtr& dataset, Graph& graph) {
    dataset_ = dataset;
    oracle_ = dataset->getOracle();
    base_ = dataset->getBasePtr();
    visited_list_pool_ = dataset->getVisitedListPool();
    graph_ = std::move(graph);
    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

IndexWrapper::IndexWrapper(IndexPtr& index) {
    dataset_ = index->extract_dataset();
    oracle_ = dataset_->getOracle();
    base_ = dataset_->getBasePtr();
    visited_list_pool_ = dataset_->getVisitedListPool();

    graph_.reserve(oracle_->size());
    graph_.resize(oracle_->size());
    auto& graph = index->extract_graph();
    for (size_t i = 0; i < oracle_->size(); ++i) {
        auto& neighbors = graph[i].candidates_;
        graph_[i].candidates_.reserve(neighbors.size());
        for (auto& neighbor : neighbors) {
            graph_[i].candidates_.emplace_back(neighbor.id, neighbor.distance, false);
        }
    }

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}

void
IndexWrapper::append(std::vector<IndexPtr>& indexes) {
    built_ = false;

    std::vector<DatasetPtr> datasets = {dataset_};
    for (auto& index : indexes) {
        datasets.emplace_back(index->extract_dataset());
    }

    dataset_ = Dataset::aggregate(datasets);
    oracle_ = dataset_->getOracle();
    visited_list_pool_ = dataset_->getVisitedListPool();
    base_ = dataset_->getBasePtr();
    graph_.reserve(oracle_->size());

    for (auto& index : indexes) {
        auto& graph = index->extract_graph();
        for (auto& neighborhood : graph) {
            graph_.emplace_back(neighborhood);
        }
    }

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
}
