#include "kmeans.h"

int Kmeans::maxIteration_ = 100;

Kmeans::Kmeans(DatasetPtr& dataset, uint8_t k, float threshold)
    : oracle_(dataset->getOracle()),
      k_(k),
      pointNumber_((int)dataset->getOracle()->size()),
      threshold_(threshold) {
    if (k <= 0 || k > std::numeric_limits<uint8_t>::max()) {
        throw std::runtime_error("k must be in (0, 255]");
    }
    centers_.resize(k_);
    points_.resize(pointNumber_);
}

//void
//Kmeans::Init() {
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_int_distribution<IdType> distribution(0, pointNumber_ - 1);
//    std::unordered_set<IdType> init_ids;
//    while (init_ids.size() < k_) {
//        auto id = distribution(gen);
//        init_ids.insert(id);
//    }
//    auto it = init_ids.begin();
//    for (int i = 0; i < k_; i++) {
//        auto id = *it;
//        centers_[i].data_ = std::shared_ptr<float[]>(new float[oracle_->dim()]);
//        memcpy(centers_[i].data_.get(), (*oracle_)[id].get(), oracle_->dim() * sizeof(float));
//        ++it;
//    }
//
//#pragma omp parallel for schedule(dynamic, 256)
//    for (int i = 0; i < pointNumber_; i++) {
//        points_[i] = NearestCenter(i, 1)[0];
//    }
//}

void
Kmeans::Init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<IdType> uni_dist(0, pointNumber_ - 1);

    IdType first_id = uni_dist(gen);
    centers_[0].data_ = std::shared_ptr<float[]>(new float[oracle_->dim()]);
    memcpy(centers_[0].data_.get(), (*oracle_)[first_id].get(), oracle_->dim() * sizeof(float));

    std::vector<double> minDist(pointNumber_, std::numeric_limits<double>::max());

    for (int c = 1; c < k_; ++c) {
        double total = 0.0;

#pragma omp parallel for reduction(+ : total) schedule(dynamic, 256)
        for (int i = 0; i < pointNumber_; ++i) {
            double d = (*oracle_)(i, centers_[c - 1].data_.get());
            if (d < minDist[i]) {
                minDist[i] = d;
            }
            total += minDist[i];
        }

        std::uniform_real_distribution<double> dist(0, total);
        double r = dist(gen);

        double prefix = 0.0;
        IdType next_id = 0;
        for (int i = 0; i < pointNumber_; ++i) {
            prefix += minDist[i];
            if (prefix >= r) {
                next_id = i;
                break;
            }
        }

        centers_[c].data_ = std::shared_ptr<float[]>(new float[oracle_->dim()]);
        memcpy(centers_[c].data_.get(), (*oracle_)[next_id].get(), oracle_->dim() * sizeof(float));
    }

#pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < pointNumber_; i++) {
        points_[i] = NearestCenter(i, 1)[0];
    }
}

std::vector<int>
Kmeans::NearestCenter(IdType p, uint8_t ell) {
    std::vector<std::pair<uint8_t, float> > distance;
    distance.reserve(k_);
    for (uint8_t k = 0; k < k_; k++) {
        distance.emplace_back(k, (*oracle_)(p, centers_[k].data_.get()));
    }
    std::nth_element(
        distance.begin(), distance.begin() + ell, distance.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    std::vector<int> result;
    result.reserve(ell);
    for (int i = 0; i < ell; i++) {
        result.push_back(distance[i].first);
    }
    return result;
}

void
Kmeans::Cluster() {
#pragma omp parallel for schedule(dynamic, 256)
    for (int p = 0; p < pointNumber_; p++) {
        int best_center = 0;
        float min_dist = (*oracle_)(p, centers_[0].data_.get());
        for (int k = 1; k < k_; k++) {
            float dist = (*oracle_)(p, centers_[k].data_.get());
            if (dist < min_dist) {
                min_dist = dist;
                best_center = k;
            }
        }
        points_[p] = best_center;
    }
}

void
Kmeans::Center() {
    std::vector<int> count(k_, 0);

    for (auto& c : centers_) {
        memset(c.data_.get(), 0, oracle_->dim() * sizeof(float));
    }
    for (int i = 0; i < pointNumber_; ++i) {
        for (int d = 0; d < oracle_->dim(); ++d) {
            centers_[points_[i]].data_[d] += (*oracle_)[i].get()[d];
        }
        count[points_[i]]++;
    }

    for (int c = 0; c < k_; ++c) {
        for (int d = 0; d < oracle_->dim(); ++d) {
            centers_[c].data_[d] /= (float)std::max(1, count[c]);
        }
    }
}

void
Kmeans::Run() {
    Timer t;
    t.start();
    Init();
    std::vector<std::vector<float> > oldCenter(k_);
    for (int i = 0; i < k_; ++i) {
        oldCenter[i].resize(oracle_->dim());
    }

    float last_sum = std::numeric_limits<float>::max();
    int cnt = 0;
    for (int iteration = 0; iteration < maxIteration_; iteration++) {
        for (int x = 0; x < k_; ++x) {
            memcpy(oldCenter[x].data(), centers_[x].data_.get(), oracle_->dim() * sizeof(float));
        }

        Cluster();
        Center();

        float sum = 0;
        for (int k = 0; k < k_; k++) {
            auto dt = (*oracle_)(centers_[k].data_.get(), oldCenter[k].data());
            sum += dt;
        }
        logger << "iteration " << iteration << " sum " << sum << std::endl;
        if (sum < threshold_) {
            logger << "Converged after " << iteration << " iterations" << std::endl;
            break;
        }
        if (sum > last_sum) {
            cnt++;
            if (cnt >= 5) {
                logger << "Converged after " << iteration << " iterations" << std::endl;
                break;
            }
        }
        last_sum = sum;
    }

    t.end();
    logger << "Kmeans clustering completed in " << t.elapsed() << " seconds." << std::endl;
}
