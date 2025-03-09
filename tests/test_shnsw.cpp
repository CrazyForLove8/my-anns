#include "evaluator.h"
#include "hnsw.h"
#include "shnsw.h"
#include "timer.h"

using namespace graph;

int
main() {
    std::string noise = "1_3_001";

    Log::setVerbose(true);
    int K = 10;

    /**
   * 0: sift 10k
   * 1: sift 1m
   * 2: perturbed sift 10k 10k+30k
   */
    int dataset = 3;

    std::string base_path, query_path, groundtruth_path;

    switch (dataset) {
        case 0:
            base_path = "/root/datasets/sift/10k/sift_base.fvecs";
            query_path = "/root/datasets/sift/10k/sift_query.fvecs";
            groundtruth_path = "/root/datasets/sift/10k/sift_groundtruth.ivecs";
            break;
        case 1:
            base_path = "/root/datasets/sift/1m/sift_base.fvecs";
            query_path = "/root/datasets/sift/1m/sift_query.fvecs";
            groundtruth_path = "/root/datasets/sift/1m/sift_groundtruth.ivecs";
            break;
        case 2:
            base_path = "../../datasets/siftsmall/siftsmall_base_preprocessed.fvecs";
            query_path = "../../datasets/siftsmall/siftsmall_query.fvecs";
            groundtruth_path = "../../datasets/siftsmall/siftsmall_groundtruth.ivecs";
            break;
        case 3:
            base_path = "/root/datasets/sift_perturbed/10k/sift_base_" + noise + ".fvecs";
            query_path = "/root/datasets/sift/10k/sift_query.fvecs";
            groundtruth_path = "/root/datasets/sift/10k/sift_groundtruth.ivecs";
            break;
        default:
            std::cerr << "Unknown dataset" << std::endl;
            return 0;
    }

    Matrix<float> base;
    base.load(base_path);
    Matrix<float> query;
    query.load(query_path);
    MatrixOracle<float, metric::l2> oracle(base);
    auto groundTruth = loadGroundTruth(groundtruth_path, query.size());

    //    dhnsw::DHNSW dhnsw(8, 200);
    shnsw::SHNSW shnsw(8, 200, 1);
    auto graph = shnsw.build(oracle);

    for (int level = 0; level < graph.size(); level++) {
        int cnt = 0;
        for (auto& i : graph[level]) {
            cnt += !i.candidates_.empty();
        }
        std::cout << "Level " << level << " has " << cnt << " nodes" << std::endl;
    }

    std::unordered_map<int, int> duplicate_map;
    int current_cluster_id = 0;

    for (int i = 0; i < oracle.size(); ++i) {
        if (duplicate_map.count(i))
            continue;

        duplicate_map[i] = current_cluster_id;

        for (int j = i + 1; j < oracle.size(); ++j) {
            if (oracle(i, j) < 1) {
                duplicate_map[j] = current_cluster_id;
            }
        }
        current_cluster_id++;
    }
    std::unordered_map<int, std::vector<int>> cluster_map;

    for (const auto& entry : duplicate_map) {
        int point_id = entry.first;
        int cluster_id = entry.second;
        cluster_map[cluster_id].push_back(point_id);
    }

    std::unordered_set<int> seen_clusters;

    std::vector<unsigned> search_Ls = {
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    size_t qsize = query.size();
    for (auto L : search_Ls) {
        float recall = 0;
        double qps = 0;
        //        float ilad = 0, ilmd = std::numeric_limits<float>::max();
        auto runs = 5;
        for (int x = 0; x < runs; ++x) {
            Timer timer;
            timer.start();
            float local_recall = 0;
            //            float local_ilad = 0;
            //            float local_ilmd = std::numeric_limits<float>::max();
            //#pragma omp parallel for reduction(+:local_recall)
            for (size_t i = 0; i < qsize; ++i) {
                auto result = shnsw.HNSW_search(graph, oracle, query[i], K, L);
                if (result.size() > K) {
                    std::cerr << "Result size: " << result.size() << " is bigger than K"
                              << std::endl;
                    return 0;
                }
                std::unordered_set<unsigned> gt(groundTruth[i].begin(), groundTruth[i].begin() + K);
                size_t correct = 0;
                for (const auto& res : result) {
                    int cluster_id = duplicate_map[res.id];
                    if (seen_clusters.count(cluster_id)) {
                        continue;
                    }
                    auto cluster = cluster_map[cluster_id];
                    for (auto& point_id : cluster) {
                        if (gt.find(point_id) != gt.end()) {
                            correct++;
                            break;
                        }
                    }
                    seen_clusters.insert(cluster_id);
                }

                //                for (int x = 0; x < result.size(); ++x) {
                //                    for (int y = x + 1; y < result.size(); ++y) {
                //                        auto dist = oracle(result[x].id,
                //                        result[y].id); local_ilad += dist; local_ilmd
                //                        = std::min(local_ilmd, dist);
                //                    }
                //                }
                local_recall += static_cast<float>(correct);
                seen_clusters.clear();
            }
            timer.end();
            //            ilad = std::max(ilad, local_ilad / (qsize * K * (K - 1) /
            //            2)); ilmd = std::min(ilmd, local_ilmd);
            qps = std::max(qps, (double)qsize / timer.elapsed());
            recall = std::max(local_recall / (static_cast<float>(qsize * K)), recall);
        }
        std::cout << "L: " << L << " recall: " << recall << " qps: " << qps << std::endl;
        //        std::cout << "L: " << L << " ilad: " << ilad << " ilmd: " << ilmd
        //        << std::endl;
    }
}