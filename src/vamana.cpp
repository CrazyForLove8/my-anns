#include "vamana.h"

diskann::Vamana::Vamana(DatasetPtr& dataset, float alpha, int L, int R)
    : Index(dataset), alpha_(alpha), L_(L), R_(R) {
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

//void
//diskann::Vamana::build() {
//    Timer timer;
//    timer.start();
//
//    int n = oracle_->size();
//    std::mt19937 rng(2024);
//    for (int u = 0; u < n; ++u) {
//        std::vector<int> init_(R_);
//        gen_random(rng, init_.data(), R_, n);
//        graph_[u].M_ = R_;
//        for (auto& v : init_) {
//            if (u == v) {
//                continue;
//            }
//            float dist = (*oracle_)(u, v);
//            graph_[u].candidates_.emplace_back(v, dist, false);
//        }
//        std::sort(graph_[u].candidates_.begin(), graph_[u].candidates_.end());
//    }
//
//    auto* center = new float[oracle_->dim()];
//    for (unsigned i = 0; i < oracle_->size(); ++i) {
//        auto pt = (*oracle_)[i];
//        for (unsigned j = 0; j < oracle_->dim(); ++j) {
//            center[j] += pt[j];
//        }
//    }
//    for (unsigned i = 0; i < oracle_->dim(); ++i) {
//        center[i] /= oracle_->size();
//    }
//
//    unsigned root = 0;
//    float minimum = std::numeric_limits<float>::max();
//    for (int x = 0; x < oracle_->size(); ++x) {
//        auto dist = (*oracle_)(x, center);
//        if (dist < minimum) {
//            minimum = dist;
//            root = x;
//        }
//    }
//    delete[] center;
//
//    std::vector<int> permutation(n);
//    std::iota(permutation.begin(), permutation.end(), 0);
//    std::shuffle(permutation.begin(), permutation.end(), rng);
//    for (int i = 0; i < n; ++i) {
//        if (i % 10000 == 0) {
//            logger << "Processing " << i << " / " << graph_.size() << std::endl;
//        }
//        auto res = track_search(oracle_.get(), graph_, (*oracle_)[permutation[i]], root, L_);
//        RobustPrune(1.0f, permutation[i], res);
//        for (auto& j : graph_[permutation[i]].candidates_) {
//            if (graph_[j.id].candidates_.size() + 1 > R_) {
//                graph_[j.id].candidates_.emplace_back(permutation[i], j.distance, false);
//                RobustPrune(alpha_, j.id, graph_[j.id].candidates_);
//            } else {
//                graph_[j.id].addNeighbor(Neighbor(permutation[i], j.distance, false));
//            }
//        }
//    }
//
//    timer.end();
//    logger << "Construction time: " << timer.elapsed() << "s" << std::endl;
//}

void
diskann::Vamana::RobustPrune(float alpha, int point, Neighbors& candidates) {
    candidates.insert(
        candidates.begin(), graph_[point].candidates_.begin(), graph_[point].candidates_.end());
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
diskann::Vamana::build_internal() {
    int n = oracle_->size();
    std::mt19937 rng(2024);
    for (int u = 0; u < n; ++u) {
        std::vector<int> init_(R_);
        gen_random(rng, init_.data(), R_, n);
        graph_[u].M_ = R_;
        for (auto& v : init_) {
            if (u == v) {
                continue;
            }
            float dist = (*oracle_)(u, v);
            graph_[u].candidates_.emplace_back(v, dist, false);
        }
        std::sort(graph_[u].candidates_.begin(), graph_[u].candidates_.end());
    }

    auto* center = new float[oracle_->dim()];
    for (unsigned i = 0; i < oracle_->size(); ++i) {
        auto pt = (*oracle_)[i];
        for (unsigned j = 0; j < oracle_->dim(); ++j) {
            center[j] += pt[j];
        }
    }
    for (unsigned i = 0; i < oracle_->dim(); ++i) {
        center[i] /= oracle_->size();
    }

    unsigned root = 0;
    float minimum = std::numeric_limits<float>::max();
    for (int x = 0; x < oracle_->size(); ++x) {
        auto dist = (*oracle_)(x, center);
        if (dist < minimum) {
            minimum = dist;
            root = x;
        }
    }
    delete[] center;

    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    for (int i = 0; i < n; ++i) {
        if (i % 10000 == 0) {
            logger << "Processing " << i << " / " << graph_.size() << std::endl;
        }
        auto res = track_search(oracle_.get(), graph_, (*oracle_)[permutation[i]], root, L_);
        RobustPrune(1.0f, permutation[i], res);
        for (auto& j : graph_[permutation[i]].candidates_) {
            if (graph_[j.id].candidates_.size() + 1 > R_) {
                graph_[j.id].candidates_.emplace_back(permutation[i], j.distance, false);
                RobustPrune(alpha_, j.id, graph_[j.id].candidates_);
            } else {
                graph_[j.id].addNeighbor(Neighbor(permutation[i], j.distance, false));
            }
        }
    }
}
