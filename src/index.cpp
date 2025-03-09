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

void
Index::build_internal() {
    throw std::runtime_error("Index does not support build");
}

void
Index::build() {
    Timer timer;
    timer.start();

    build_internal();

    timer.end();
    logger << "Indexing time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
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
Index::extractGraph() {
    return graph_;
}

DatasetPtr&
Index::extractDataset() {
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
