#include "evaluator.h"
#include "fgim.h"

using namespace graph;

int
main(int argc, char** argv) {
    // set verbose to true if you want to see more information
    Log::setVerbose(false);

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <graph 1> <dataset 1> <graph 2> <dataset 2> <metric> <output>" << std::endl;
        std::cerr << "Metric: l2, cosine" << std::endl;
        return 1;
    }

    std::string graph1 = argv[1];
    std::string dataset1 = argv[2];
    std::string graph2 = argv[3];
    std::string dataset2 = argv[4];
    std::string output = argv[5];

    Graph g1, g2;
    loadGraph(g1, graph1);
    loadGraph(g2, graph2);

    Matrix<float> d1, d2;
    d1.load(dataset1);
    d2.load(dataset2);
    auto oracle1 = MatrixOracle<float, metric::l2>::getInstance(d1);
    auto oracle2 = MatrixOracle<float, metric::l2>::getInstance(d2);

    if (d1.dim() != d2.dim()) {
        std::cerr << "Dataset dimension mismatch" << std::endl;
        return 1;
    }

    if (g1.size() != d1.size() || g2.size() != d2.size()) {
        std::cerr << "Graph and dataset size mismatch" << std::endl;
        return 1;
    }

    Matrix<float> merged;
    mergeMatrix(d1, d2, merged);

    auto oracle_merged = MatrixOracle<float, metric::l2>::getInstance(merged);

    FGIM merge;
    auto graph = merge.merge(g1, oracle1, g2, oracle2, oracle_merged);

    saveGraph(graph, output);

    return 0;
}