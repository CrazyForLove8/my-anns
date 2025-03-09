//
// Created by XiaoWu on 2025/2/13.
//

#include "evaluator.h"
#include "fgim.h"
#include "hnsw.h"
#include "logger.h"
#include "nndescent.h"
#include "nsw.h"
#include "taumng.h"
#include "timer.h"
#include "vamana.h"

void
test_nndescent(DatasetPtr& dataset) {
    nndescent::NNDescent nndescent(dataset, 20);

    nndescent.build();

    eval(nndescent, dataset, 200);
}

void
test_vamana(DatasetPtr& dataset) {
    diskann::Vamana vamana(dataset, 1.2, 100, 80);

    vamana.build();

    eval(vamana, dataset, 200);
}

void
test_taumng(DatasetPtr& dataset) {
    nndescent::NNDescent nndescent(dataset, 20);

    nndescent.build();

    auto graph = nndescent.extractGraph();

    taumng::TauMNG taumng(dataset, graph, 10, 80, 100);

    taumng.build();

    eval(taumng, dataset, 200);
}

void
test_nsw(DatasetPtr& dataset) {
    nsw::NSW nsw(dataset, 32, 100);

    nsw.build();

    eval(nsw, dataset, 200);
}

void
test_hnsw(DatasetPtr& dataset) {
    hnsw::HNSW hnsw(dataset, 20, 200);

    hnsw.build();

    eval(hnsw, dataset);
}

int
main() {
    Log::setVerbose(true);

    auto dataset = Dataset::getInstance("sift", "1m");

    test_hnsw(dataset);
}
