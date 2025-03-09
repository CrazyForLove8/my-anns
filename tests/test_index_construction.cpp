#include "evaluator.h"
#include "nndescent.h"
#include "nsw.h"
#include "taumng.h"
#include "vamana.h"

void
test_nndescent(OraclePtr& oracle,
               const Matrix<float>& query,
               const std::vector<std::vector<unsigned int>>& groundTruth,
               unsigned int K) {
    nndescent::NNDescent nndescent(oracle, 20);

    nndescent.build();

    auto graph = nndescent.extractGraph();

    evaluate(graph, K, query, groundTruth, oracle.get());
}

void
test_vamana(OraclePtr& oracle,
            const Matrix<float>& query,
            const std::vector<std::vector<unsigned int>>& groundTruth,
            unsigned int K) {
    diskann::Vamana vamana(oracle, 1.2, 100, 80);

    vamana.build();

    auto graph = vamana.extractGraph();

    evaluate(graph, K, query, groundTruth, oracle.get());
}

void
test_taumng(OraclePtr& oracle,
            const Matrix<float>& query,
            const std::vector<std::vector<unsigned int>>& groundTruth,
            unsigned int K) {
    nndescent::NNDescent nndescent(oracle, 20);

    nndescent.build();

    auto graph = nndescent.extractGraph();

    taumng::TauMNG taumng(oracle, graph, 10, 80, 100);

    taumng.build();

    graph = taumng.extractGraph();

    evaluate(graph, K, query, groundTruth, oracle.get());
}

void
test_nsw(OraclePtr& oracle,
         const Matrix<float>& query,
         const std::vector<std::vector<unsigned int>>& groundTruth,
         unsigned int K) {
    nsw::NSW nsw(oracle, 32, 100);

    nsw.build();

    auto graph = nsw.extractGraph();

    evaluate(graph, K, query, groundTruth, oracle.get());
}

int
main(int argc, char** argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <algorithm>" << std::endl;
        std::cerr << "Algorithm: nndescent, vamana, taumng, nsw" << std::endl;
        return 1;
    }
    // Recall@K
    int K = 10;

    // Change the path to your dataset
    std::string base_path = "../../datasets/sift/sift_base.fvecs";
    std::string query_path = "../../datasets/sift/sift_query.fvecs";
    std::string groundtruth_path = "../../datasets/sift/sift_groundtruth.ivecs";

    // Load the dataset and groundtruth
    Matrix<float> base;
    base.load(base_path);
    Matrix<float> query;
    query.load(query_path);
    auto oracle = MatrixOracle<float, metric::l2>::getInstance(base);
    auto groundTruth = loadGroundTruth(groundtruth_path, query.size());

    if (std::string(argv[1]) == "nndescent") {
        test_nndescent(oracle, query, groundTruth, K);
    } else if (std::string(argv[1]) == "vamana") {
        test_vamana(oracle, query, groundTruth, K);
    } else if (std::string(argv[1]) == "taumng") {
        test_taumng(oracle, query, groundTruth, K);
    } else if (std::string(argv[1]) == "nsw") {
        test_nsw(oracle, query, groundTruth, K);
    } else {
        std::cerr << "Unimplemented algorithm: " << argv[1] << std::endl;
        return 1;
    }

    return 0;
}
