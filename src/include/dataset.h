//
// Created by XiaoWu on 2025/3/1.
//

#ifndef MYANNS_DATASET_H
#define MYANNS_DATASET_H

#include <unordered_set>

#include "dtype.h"
#include "visittable.h"

using namespace graph;

namespace graph {

std::vector<std::vector<unsigned int>>
loadGroundTruth(const std::string& filename, unsigned int qsize, unsigned int K = 100);

/* Default is L2 */
enum class DISTANCE { L2, COSINE, JACCARD, HAMMING };

class Dataset {
private:
    std::string name_;
    std::string size_;

    DISTANCE distance_;

    MatrixPtr<float> base_;
    MatrixPtr<float> query_;
    MatrixPtr<int> groundTruth_;

    OraclePtr oracle_;

    VisitedListPoolPtr visited_list_pool_;

    bool full_dataset_;

    void
    load();

public:
    Dataset();

    /**
                 * @brief Get the instance of the dataset
                 * @param name sift, gist, deep, msong, glove, crawl
                 * @param size 10k, 100k, 1m, 10m or empty for the default size
                 * @return
                 */
    static std::shared_ptr<Dataset>
    getInstance(const std::string& name, const std::string& size);

    std::string&
    getName();

    std::string&
    getSize();

    Matrix<float>&
    getBase();

    MatrixPtr<float>&
    getBasePtr();

    Matrix<float>&
    getQuery();

    Matrix<int>&
    getGroundTruth();

    OraclePtr&
    getOracle();

    DISTANCE&
    getDistance();

    VisitedListPoolPtr&
    getVisitedListPool();

    void
    split(std::vector<std::shared_ptr<Dataset>>& datasets, unsigned int num);

    void
    merge(std::vector<std::shared_ptr<Dataset>>& datasets);

    static std::shared_ptr<Dataset>
    aggregate(std::vector<std::shared_ptr<Dataset>>& datasets);
};

using DatasetPtr = std::shared_ptr<Dataset>;

}  // namespace graph

#endif  //MYANNS_DATASET_H
