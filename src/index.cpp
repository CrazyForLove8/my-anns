#include "index.h"

Index::Index(const IndexParam& param, const bool allocate) : index_param_(param) {
    oracle_ = std::make_shared<Vectors<float> >(param.dim_, param.metric_, param.io_type_);
    visited_list_pool_ = VisitedListPool::getInstance(index_param_.init_size_);
    if (allocate) {
        graph_.resize(index_param_.init_size_);
    }
}

void
Index::build_internal(DatasetPtr& dataset) {
    throw std::runtime_error("Index does not support building");
}

void
Index::build(DatasetPtr& dataset) {
    print_info();
    if (built_) {
        logger << "Index is already built, skipping build." << std::endl;
        return;
    }

    Timer timer;
    timer.start();

    oracle_->insert(dataset->getBasePtr());
    this->resize(oracle_->size());

    build_internal(dataset);

    timer.end();
    logger << "Indexing time: " << timer.elapsed() << "s" << std::endl;

    flatten_graph_ = FlattenGraph(graph_);
    built_ = true;
    cur_size_ += oracle_->size();
}

void
Index::set_save_helper(const SaveHelper& saveHelper) {
    save_helper_.save_frequency = saveHelper.save_frequency;
    save_helper_.save_path = saveHelper.save_path;

    save_helper_.total_count = oracle_->size();
    save_helper_.save_per_count = save_helper_.total_count / saveHelper.save_frequency;
}

Graph&
Index::extract_graph() {
    if (!built_) {
        throw std::runtime_error("Index is not built");
    }
    return graph_;
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
    return search_flatten_graph(
        oracle_.get(), visited_list_pool_.get(), flatten_graph_, query, topk, L);
}

void
Index::print_info() const {
    logger << "Indexing settings:" << std::endl;
    logger << "Index Size: " << oracle_->size() << std::endl;
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
    params["built"] = (uint64_t)(built_ ? 1ULL : 0ULL);
    return params;
}

VectorsPtr<float>
Index::extract_vectors() {
    return oracle_;
}

void
Index::load_params(const ParamMap& params) {
    throw std::runtime_error("Index does not need to load parameters");
}

void
Index::remove(IdType id) {
    throw std::runtime_error("Index does not support remove operation");
}

void
Index::partial_build(IdType start, IdType end) {
    throw std::runtime_error("Index does not support partial build");
}

void
Index::resize(const IdType new_size) {
    visited_list_pool_ = VisitedListPool::getInstance(new_size);
    graph_.resize(new_size);
}

void
Index::partial_build(IdType num) {
    print_info();
    if (built_) {
        logger << "Index is already built, skipping build." << std::endl;
        return;
    }
    auto start = cur_size_;
    auto end = cur_size_ == 1 ? num : cur_size_ + num;
    if (end > oracle_->size()) {
        num = oracle_->size() - start;
        end = oracle_->size();
    }

    Timer timer;
    timer.start();
    this->partial_build(start, end);
    timer.end();

    cur_size_ += num;
    if (cur_size_ == oracle_->size()) {
        logger << "Partial build completed, total size: " << cur_size_ << std::endl;
        flatten_graph_ = FlattenGraph(graph_);
        built_ = true;
        logger << "Index built successfully." << std::endl;
    } else {
        logger << "Partial build completed, total size: " << cur_size_
               << ", but not all points are added yet." << std::endl;
        built_ = false;
    }
    logger << "Partial build consumed " << timer.elapsed() << " s." << std::endl;
}
