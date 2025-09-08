#include "annslib.h"

std::string experiment = "1.1";

void
mergeExp1_1(DatasetPtr& dataset) {
    Log::redirect(experiment + "_" + dataset->getName() + "_ours");
    std::cout << "Exp1.1: Merge 2 HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);

    datasets.insert(datasets.begin(), dataset);
    auto merged_dataset = Dataset::aggregate(datasets);

    std::vector<int> params = {16};
    for (auto& param : params) {
        omp_set_num_threads(20);
        std::vector<IndexPtr> vec(datasets.size());
        vec[0] = std::make_shared<hnsw::HNSW>(dataset, param, 200);
        vec[0]->build();
        for (size_t i = 1; i < datasets.size(); i++) {
            vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], param, 200);
            vec[i]->build();
        }
        omp_set_num_threads(1);
        {
            std::cout << "Parameter: Max degree: " << param << std::endl;
            MGraph mgraph(merged_dataset, param, 200);
            mgraph.combine(vec);
            recall(mgraph, merged_dataset, 200);
        }
    }

    dataset = merged_dataset;
}

void
mergeExp1_2(DatasetPtr& dataset) {
    Log::redirect(experiment + "_" + dataset->getName() + "_baseline");
    std::cout << "Exp1.2: Merge 2 HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    std::vector<std::pair<int, int> > params = {
        {6, 250}, {8, 250}, {10, 250}, {12, 250}, {14, 250}, {16, 250}};
    for (auto param : params) {
        omp_set_num_threads(20);
        auto datasets = std::vector<DatasetPtr>();
        int num_splits = 2;
        std::cout << "Number of splits: " << num_splits << std::endl;
        dataset->split(datasets, num_splits);

        std::vector<IndexPtr> vec(datasets.size() + 1);
        std::cout << "Parameter: Max degree: " << param.first << std::endl;
        std::cout << "Parameter: ef_construction: " << param.second << std::endl;
        vec[0] = std::make_shared<hnsw::HNSW>(dataset, param.first, param.second);
        vec[0]->build();
        {
            omp_set_num_threads(1);
            Timer timer;
            timer.start();
            for (auto& data : datasets) {
                vec[0]->add(data);
            }
            timer.end();
            std::cout << "Total adding time: " << timer.elapsed() << "s" << std::endl;
            recall(vec[0], vec[0]->extract_dataset(), 200);
        }
    }
}

void
mergeExp2_1(DatasetPtr& dataset) {
    Log::redirect("2.1_HNSW_K100_" + dataset->getName() + "_ours");
    std::cout << "Exp2.1: Merge 2 HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);

    datasets.insert(datasets.begin(), dataset);
    auto merged_dataset = Dataset::aggregate(datasets);

    std::vector<int> params = {64};
    for (auto& param : params) {
        std::vector<IndexPtr> vec(datasets.size());
        vec[0] = std::make_shared<hnsw::HNSW>(dataset, 32, 200);
        vec[0]->build();
        for (size_t i = 1; i < datasets.size(); i++) {
            vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], 32, 200);
            vec[i]->build();
        }
        {
            std::cout << "Parameter: Max degree: " << param << std::endl;
            std::shared_ptr<MGraph> mgraph = std::make_shared<MGraph>(merged_dataset, param, 200);
            mgraph->combine(vec);
            recall(mgraph, merged_dataset, -1, 100);
        }
    }

    dataset = merged_dataset;
}

void
mergeExp2_2(DatasetPtr& dataset) {
    Log::redirect("2.2_NSW_K100_" + dataset->getName() + "_baseline");
    std::cout << "Exp2.2: Merge 2 NSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    std::vector<std::pair<int, int> > params = {{64, 200}};
    for (auto param : params) {
        auto datasets = std::vector<DatasetPtr>();
        int num_splits = 2;
        std::cout << "Number of splits: " << num_splits << std::endl;
        dataset->split(datasets, num_splits);

        std::vector<IndexPtr> vec(datasets.size() + 1);
        std::cout << "Parameter: Max degree: " << param.first << std::endl;
        std::cout << "Parameter: ef_construction: " << param.second << std::endl;
        vec[0] = std::make_shared<nsw::NSW>(dataset, param.first, param.second);
        vec[0]->build();
        {
            Timer timer;
            timer.start();
            for (auto& data : datasets) {
                vec[0]->add(data);
            }
            timer.end();
            std::cout << "Total adding time: " << timer.elapsed() << "s" << std::endl;
            recall(vec[0], vec[0]->extract_dataset(), -1, 100);
        }
    }
}

void
mergeExp3_1(DatasetPtr& dataset) {
    Log::redirect("15_Vamana_" + dataset->getName() + "_baseline");
    std::cout << "Exp2.1: Merge 2 Vamana indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;
    auto datasets = std::vector<DatasetPtr>();
    dataset->split(datasets, 2);
    datasets.insert(datasets.begin(), dataset);

    int param = 64;

    std::vector<IndexPtr> indexes;
    for (auto& data : datasets) {
        auto vamana = std::make_shared<diskann::Vamana>(data, 1.2, 200, param);
        vamana->build();
        indexes.push_back(vamana);
    }

    auto merged_dataset = Dataset::aggregate(datasets);
    {
        MGraph mgraph(merged_dataset, param, 200);
        mgraph.combine(indexes);
        recall(mgraph, merged_dataset);
    }
}

void
mergeExp5_1(DatasetPtr& dataset) {
    Log::redirect("5.1_" + dataset->getName() + "_ours");
    std::cout << "Exp5.1: Merge several HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    std::vector<int> num_splits_list = {7};

    for (auto split : num_splits_list) {
        auto datasets = std::vector<DatasetPtr>();
        std::cout << "Number of splits: " << split << std::endl;
        dataset->split(datasets, split);

        datasets.insert(datasets.begin(), dataset);
        auto merged_dataset = Dataset::aggregate(datasets);

        omp_set_num_threads(20);
        int max_degree = 20;
        std::vector<IndexPtr> vec(datasets.size());
        vec[0] = std::make_shared<hnsw::HNSW>(dataset, 20, 200);
        vec[0]->build();
        for (size_t i = 1; i < datasets.size(); i++) {
            vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], 20, 200);
            vec[i]->build();
        }

        omp_set_num_threads(1);
        {
            MGraph mgraph(merged_dataset, max_degree, 200);
            mgraph.combine(vec);
            recall(mgraph, merged_dataset, 200);
        }

        dataset = merged_dataset;
    }
}

void
mergeExp5_2(DatasetPtr& dataset) {
    Log::redirect("5.1_" + dataset->getName() + "_baseline");
    std::cout << "Exp5.2: Merge several HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    std::vector<int> num_splits_list = {3, 4, 5, 6, 7};

    for (auto split : num_splits_list) {
        auto datasets = std::vector<DatasetPtr>();
        std::cout << "Number of splits: " << split << std::endl;
        dataset->split(datasets, split);

        omp_set_num_threads(20);
        int max_degree = 32;
        auto hnsw = std::make_shared<hnsw::HNSW>(dataset, max_degree, 200);
        hnsw->build();
        omp_set_num_threads(1);
        {
            Timer timer;
            timer.start();
            for (auto& data : datasets) {
                hnsw->add(data);
            }
            timer.end();
            std::cout << "Total adding time: " << timer.elapsed() << "s" << std::endl;
            recall(hnsw, hnsw->extract_dataset(), 200);
        }

        dataset = hnsw->extract_dataset();
    }
}

void
mergeExp6_1(DatasetPtr& dataset) {
    Log::redirect("6.1_" + dataset->getName() + "_random");
    std::cout << "Exp6.1: With Refinement\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);

    datasets.insert(datasets.begin(), dataset);
    auto merged_dataset = Dataset::aggregate(datasets);

    omp_set_num_threads(20);
    std::vector<IndexPtr> vec(datasets.size());
    vec[0] = std::make_shared<hnsw::HNSW>(dataset, 32, 200);
    vec[0]->build();
    for (size_t i = 1; i < datasets.size(); i++) {
        vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], 32, 200);
        vec[i]->build();
    }

    omp_set_num_threads(1);
    {
        MGraph mgraph(merged_dataset, 32, 200);
        mgraph.combine(vec);
        //        recall(mgraph, merged_dataset, 200);
    }
}

void
mergeExp6_2(DatasetPtr& dataset) {
    Log::redirect("6.2_" + dataset->getName() + "_connectivity");
    std::cout << "Exp6.2: ALL\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);

    datasets.insert(datasets.begin(), dataset);
    auto merged_dataset = Dataset::aggregate(datasets);

    omp_set_num_threads(20);
    std::vector<IndexPtr> vec(datasets.size());
    int param = 32;
    vec[0] = std::make_shared<hnsw::HNSW>(dataset, param, 200);
    vec[0]->build();
    for (size_t i = 1; i < datasets.size(); i++) {
        vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], param, 200);
        vec[i]->build();
    }

    {
        std::cout << "Parameter: Max degree: " << param << std::endl;
        std::cout << "Parameter: ef_construction: " << param << std::endl;
        FGIM mgraph(merged_dataset, param);
        mgraph.combine(vec);
        std::cout << checkConnectivity(mgraph.extract_graph()) << std::endl;
    }
}

void
mergeExp6_3(DatasetPtr& dataset) {
    Log::redirect("6.3" + dataset->getName() + "_repair");
    std::cout << "Exp6.3: Repair no in-degree\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    auto datasets = dataset->subsets(num_splits);

    for (int it = 0; it < 2; ++it) {
        std::cout << "Trying with " << it + 1 << " iterations" << std::endl;
        std::vector<IndexPtr> vec(datasets.size());
        int max_degree;
        if (dataset->getName() == "sift" || dataset->getName() == "msong" ||
            dataset->getName() == "deep") {
            max_degree = 20;
        } else {
            max_degree = 32;
        }
        for (size_t i = 0; i < datasets.size(); i++) {
            vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], max_degree, 200);
            vec[i]->build();
        }
        MGraph mgraph(dataset, max_degree, 200);
        mgraph.combine(vec);
        recall(mgraph, dataset);

        auto& graph = mgraph.extract_hgraph();

        std::cout << checkConnectivity(graph[0]) << std::endl;
    }
}

void
exp_multiple(DatasetPtr& dataset) {
    Log::redirect("mul_" + dataset->getName() + "_ours");
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";

    auto split = {3, 4, 5, 6, 7};
    int max_degree;
    if (dataset->getName() == "crawl" || dataset->getName() == "gist" ||
        dataset->getName() == "glove") {
        max_degree = 32;
    } else {
        max_degree = 16;
    }
    std::cout << "Our method" << std::endl;
    for (auto num_split : split) {
        std::cout << "Number of splits: " << num_split << std::endl;
        auto datasets = dataset->subsets(num_split);

        std::vector<IndexPtr> vec(datasets.size());
        for (size_t i = 0; i < datasets.size(); i++) {
            vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], max_degree, 200);
            vec[i]->build();
        }

        MGraph mgraph(dataset, max_degree, 200);
        mgraph.combine(vec);
        recall(mgraph, dataset);
        max_degree += 2;
    }

    //    std::cout << "Baseline method" << std::endl;
    //    for (auto num_split : split) {
    //        std::cout << "Number of splits: " << num_split << std::endl;
    //        auto datasets = dataset->subsets(num_split);
    //
    //        auto hnsw = std::make_shared<hnsw::HNSW>(datasets[0], max_degree, 200);
    //        hnsw->build();
    //
    //        auto another =
    //            std::make_shared<hnsw::HNSW>(dataset, hnsw->extract_hgraph(), true, max_degree, 200);
    //        another->partial_build();
    //
    //        recall(another, dataset);
    //    }
}

void
mergeExp9(DatasetPtr& dataset) {
    std::cout << "Exp1.1: Merge 2 HNSW indexes\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);

    datasets.insert(datasets.begin(), dataset);
    auto merged_dataset = Dataset::aggregate(datasets);

    std::vector<IndexPtr> vec(datasets.size());
    vec[0] = std::make_shared<hnsw::HNSW>(dataset, 20, 200);
    vec[0]->build();
    for (size_t i = 1; i < datasets.size(); i++) {
        vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], 20, 200);
        vec[i]->build();
    }

    std::vector<int> params = {16};
    for (auto& param : params) {
        omp_set_num_threads(1);
        {
            std::cout << "Parameter: Max degree: " << param << std::endl;
            MGraph mgraph(merged_dataset, param, 200);
            mgraph.combine(vec);
            recall(mgraph, merged_dataset, 200);
        }
    }
}

void
mergeExp10(DatasetPtr& dataset) {
    Log::redirect("10_" + dataset->getName() + "_ours");
    std::cout << "Exp10: Varying L\n";
    std::cout << "Current Time: " << Log::getTimestamp() << "\n";
    std::cout << "Dataset name: " << dataset->getName() << " size: " << dataset->getSize()
              << std::endl;

    auto datasets = std::vector<DatasetPtr>();

    int num_splits = 2;
    std::cout << "Number of splits: " << num_splits << std::endl;
    dataset->split(datasets, num_splits);

    datasets.insert(datasets.begin(), dataset);
    auto merged_dataset = Dataset::aggregate(datasets);

    std::vector<IndexPtr> vec(datasets.size());
    vec[0] = std::make_shared<hnsw::HNSW>(dataset, 20, 200);
    vec[0]->build();
    for (size_t i = 1; i < datasets.size(); i++) {
        vec[i] = std::make_shared<hnsw::HNSW>(datasets[i], 20, 200);
        vec[i]->build();
    }

    std::vector<int> params = {16};
    for (auto& param : params) {
        omp_set_num_threads(1);
        {
            std::cout << "Parameter: Max degree: " << param << std::endl;
            MGraph mgraph(merged_dataset, param, 200);
            mgraph.combine(vec);
            recall(mgraph, merged_dataset, 200);
        }
    }
}

void
mergeExp_nsw_and_hnsw_build(DatasetPtr& dataset) {
    Log::redirect("nsw_hnsw_build_" + dataset->getName());
    omp_set_num_threads(1);

    auto subsets = dataset->subsets(2);

    auto nsw = std::make_shared<nsw::NSW>(subsets[0], 32, 200);
    Timer sub_timer, total_timer;
    sub_timer.start();
    total_timer.start();
    nsw->build();
    sub_timer.end();
    std::cout << "NSW first part build time: " << sub_timer.elapsed() << "s" << std::endl;
    nsw = std::make_shared<nsw::NSW>(subsets[1], 32, 200);
    sub_timer.start();
    nsw->build();
    sub_timer.end();
    std::cout << "NSW second part build time: " << sub_timer.elapsed() << "s" << std::endl;
    total_timer.end();
    std::cout << "Total NSW build time: " << total_timer.elapsed() << "s" << std::endl;

    auto hnsw = std::make_shared<hnsw::HNSW>(subsets[0], 16, 200);
    sub_timer.start();
    total_timer.start();
    hnsw->build();
    sub_timer.end();
    std::cout << "HNSW first part build time: " << sub_timer.elapsed() << "s" << std::endl;
    hnsw = std::make_shared<hnsw::HNSW>(subsets[1], 16, 200);
    sub_timer.start();
    hnsw->build();
    sub_timer.end();
    std::cout << "HNSW second part build time: " << sub_timer.elapsed() << "s" << std::endl;
    total_timer.end();
    std::cout << "Total HNSW build time: " << total_timer.elapsed() << "s" << std::endl;
}

int
main() {
    Log::setVerbose(true);
    auto dataset = Dataset::getInstance("msong", "1m");
    mergeExp_nsw_and_hnsw_build(dataset);

    int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }

    return 0;
}