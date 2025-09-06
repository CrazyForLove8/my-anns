#include "vamana.h"

diskann::Vamana::Vamana(DatasetPtr& dataset, float alpha, int L, int R)
    : Index(dataset), alpha_(alpha), L_(L), R_(R) {
    std::mt19937 rng(std::random_device{}());
    for (auto u = 0; u < oracle_->size(); ++u) {
        std::uniform_int_distribution<IdType> distrib(0, oracle_->size() - 1);
        for (int i = 0; i < R_; ++i) {
            auto v = distrib(rng);
            if (u == v) {
                continue;
            }
            float dist = (*oracle_)(u, v);
            graph_[u].candidates_.emplace_back(v, dist, false);
        }
        std::sort(graph_[u].candidates_.begin(), graph_[u].candidates_.end());
    }

    std::vector<float> center(oracle_->dim(), 0);
    for (auto i = 0; i < oracle_->size(); ++i) {
        auto pt = (*oracle_)[i];
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center[j] += pt.get()[j];
        }
    }
    for (unsigned i = 0; i < oracle_->dim(); ++i) {
        center[i] /= static_cast<float>(oracle_->size());
    }
    float minimum = std::numeric_limits<float>::max();
    for (auto x = 0; x < oracle_->size(); ++x) {
        auto dist = (*oracle_)(x, center.data());
        if (dist < minimum) {
            minimum = dist;
            root = x;
        }
    }
}

diskann::Vamana::Vamana(
    DatasetPtr& dataset, std::vector<IdType>& permutation, float alpha, int L, int R)
    : Index(dataset), alpha_(alpha), L_(L), R_(R) {
    auto n = permutation.size();
    std::mt19937 rng(std::random_device{}());
    for (int u = 0; u < n; ++u) {
        for (int i = 0; i < R_; ++i) {
            auto v = permutation[rng() % n];
            if (permutation[u] == v) {
                continue;
            }
            float dist = (*oracle_)(permutation[u], v);
            graph_[permutation[u]].candidates_.emplace_back(v, dist, false);
        }
        std::sort(graph_[permutation[u]].candidates_.begin(),
                  graph_[permutation[u]].candidates_.end());
    }

    std::vector<float> center(oracle_->dim(), 0);
    for (unsigned i = 0; i < n; ++i) {
        auto pt = (*oracle_)[permutation[i]];
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center[j] += pt.get()[j];
        }
    }
    for (unsigned i = 0; i < oracle_->dim(); ++i) {
        center[i] /= static_cast<float>(n);
    }
    auto minimum = std::numeric_limits<float>::max();
    for (int x = 0; x < n; ++x) {
        auto dist = (*oracle_)(permutation[x], center.data());
        if (dist < minimum) {
            minimum = dist;
            root = permutation[x];
        }
    }
}

void
diskann::Vamana::set_alpha(float alpha) {
    this->alpha_ = alpha;
}

void
diskann::Vamana::set_L(int L) {
    this->L_ = L;
}

void
diskann::Vamana::set_R(int R) {
    this->R_ = R;
}

void
diskann::Vamana::RobustPrune(float alpha, IdType point, Neighbors& candidates) {
    candidates.insert(
        candidates.end(), graph_[point].candidates_.begin(), graph_[point].candidates_.end());
    auto it = std::find(candidates.begin(), candidates.end(), Neighbor(point, 0, false));
    if (it != candidates.end()) {
        candidates.erase(it);
    }
    graph_[point].candidates_.clear();
    while (!candidates.empty()) {
        auto min_it = std::min_element(candidates.begin(), candidates.end());
        auto p_star_ = *min_it;
        graph_[point].candidates_.push_back(p_star_);
        if (graph_[point].candidates_.size() >= R_) {
            break;
        }
        candidates.erase(std::remove_if(candidates.begin(),
                                        candidates.end(),
                                        [&](const Neighbor& p_prime_) {
                                            return alpha * (*oracle_)(p_star_.id, p_prime_.id) <=
                                                   p_prime_.distance;
                                        }),
                         candidates.end());
    }
}

void
diskann::Vamana::partial_build(graph::IdType start, graph::IdType end) {
    std::vector<int> permutation(end - start);
    std::iota(permutation.begin(), permutation.end(), start);
    std::shuffle(permutation.begin(), permutation.end(), std::mt19937(std::random_device()()));

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < permutation.size(); ++i) {
        if (i % (permutation.size() / 10) == 0) {
            logger << "Processing " << i << " / " << graph_.size() << std::endl;
        }
        auto res = track_search(oracle_.get(),
                                visited_list_pool_.get(),
                                graph_,
                                (*oracle_)[permutation[i]].get(),
                                L_,
                                root);
        res.erase(
            std::remove_if(
                res.begin(), res.end(), [&](const Neighbor& n) { return n.id == permutation[i]; }),
            res.end());
        {
            std::lock_guard<std::mutex> guard(graph_[permutation[i]].lock_);
            RobustPrune(1.0f, permutation[i], res);
        }
        for (auto& j : graph_[permutation[i]].candidates_) {
            std::lock_guard<std::mutex> neighbor_guard(graph_[j.id].lock_);
            if (graph_[j.id].candidates_.size() + 1 > R_) {
                Neighbors rev = {Neighbor(permutation[i], j.distance, false)};
                RobustPrune(alpha_, j.id, rev);
            } else {
                graph_[j.id].addNeighbor(Neighbor(permutation[i], j.distance, false));
            }
        }
    }
}

void
diskann::Vamana::partial_build(graph::IdType num) {
    Index::partial_build(num);
}

void
diskann::Vamana::build_internal() {
    this->partial_build(0, oracle_->size());
}

void
diskann::Vamana::partial_build(std::vector<IdType>& permutation) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < permutation.size(); ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << permutation.size() << std::endl;
        }
        auto res = track_search(oracle_.get(),
                                visited_list_pool_.get(),
                                graph_,
                                (*oracle_)[permutation[i]].get(),
                                L_,
                                root);
        {
            std::lock_guard<std::mutex> guard(graph_[permutation[i]].lock_);
            RobustPrune(1.0f, permutation[i], res);
        }
        for (auto& j : graph_[permutation[i]].candidates_) {
            std::lock_guard<std::mutex> neighbor_guard(graph_[j.id].lock_);
            if (graph_[j.id].candidates_.size() + 1 > R_) {
                Neighbors rev = {Neighbor(permutation[i], j.distance, false)};
                RobustPrune(alpha_, j.id, rev);
            } else {
                graph_[j.id].addNeighbor(Neighbor(permutation[i], j.distance, false));
            }
        }
    }

    built_ = true;
}

void
diskann::Vamana::print_info() const {
    Index::print_info();
    logger << "Vamana Index Info:" << std::endl;
    logger << "Alpha: " << alpha_ << std::endl;
    logger << "L: " << L_ << std::endl;
    logger << "R: " << R_ << std::endl;
}

ParamMap
diskann::Vamana::extract_params() {
    ParamMap params = Index::extract_params();
    params["alpha"] = alpha_;
    params["L"] = (uint64_t)L_;
    params["R"] = (uint64_t)R_;
    return params;
}

diskann::DiskANN::DiskANN(DatasetPtr& dataset, float alpha, int L, int R, int k, int ell)
    : Index(dataset), alpha_(alpha), L_(L), R_(R), k_(k), ell_(ell) {
}

void
diskann::DiskANN::build_internal() {
    auto kmeans = std::make_shared<Kmeans>(dataset_, k_);
    kmeans->Run();

    std::mt19937 rng(std::random_device{}());
    std::vector<IndexPtr> indexes;
    for (int k = 0; k < k_; ++k) {
        logger << "Indexing subset for cluster " << k << " / " << k_ << std::endl;
        std::vector<uint32_t> permutation;
        permutation.reserve(oracle_->size() * ell_ / k_);
        for (int i = 0; i < oracle_->size(); ++i) {
            auto centers = kmeans->NearestCenter(i, ell_);
            if (std::find(centers.begin(), centers.end(), k) != centers.end()) {
                permutation.push_back(i);
            }
        }
        std::shuffle(permutation.begin(), permutation.end(), rng);
        auto vamana = std::make_shared<Vamana>(dataset_, permutation, alpha_, L_, R_);
        logger << "Constructing sub-index for cluster " << k << " with " << permutation.size()
               << " points." << std::endl;
        vamana->partial_build(permutation);
        indexes.emplace_back(vamana);
    }

    for (auto& index : indexes) {
        auto& graph = index->extract_graph();
        for (int i = 0; i < oracle_->size(); ++i) {
            graph_[i].candidates_.insert(graph_[i].candidates_.end(),
                                         graph[i].candidates_.begin(),
                                         graph[i].candidates_.end());
        }
    }

    for (int i = 0; i < oracle_->size(); ++i) {
        std::sort(graph_[i].candidates_.begin(), graph_[i].candidates_.end());
    }
}
