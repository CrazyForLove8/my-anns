//
// Created by XiaoWu on 2025/3/29.
//

#ifndef MYANNS_KMEANS_H
#define MYANNS_KMEANS_H

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <random>

#include "dataset.h"
#include "timer.h"

struct Centroid {
    std::shared_ptr<float[]> data_{nullptr};
};

class Kmeans {
private:
    std::vector<Centroid> centers_;
    std::vector<uint8_t> points_;

    uint8_t k_;
    IdType pointNumber_;
    static int maxIteration_;
    float threshold_{1e-3};

    OraclePtr& oracle_;

    void
    Init();

    void
    Cluster();

    void
    Center();

public:
    Kmeans(DatasetPtr& dataset, uint8_t k, float threshold = 1e-4);

    /**
     * Get the ell nearest centers of the p
     * @param p
     * @param ell
     * @return
     */
    std::vector<int>
    NearestCenter(IdType p, uint8_t ell);

    void
    Run();
};

#endif  //MYANNS_KMEANS_H
