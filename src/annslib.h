//
// Created by XiaoWu on 2025/3/2.
//

#ifndef MYANNS_ANNSLIB_H
#define MYANNS_ANNSLIB_H

#include "evaluator.h"
#include "fgim.h"
#include "hnsw.h"
#include "memory.h"
#include "mgraph.h"
#include "nndescent.h"
#include "nnmerge.h"
#include "nsg.h"
#include "nsw.h"
#include "output.h"
#include "taumng.h"
#include "vamana.h"

inline IndexParam
getParam(const DatasetPtr& dataset) {
    const auto dim = dataset->getBase().dim();
    const auto metric = dataset->getDistance();

    IndexParam param;
    param.dim_ = dim;
    param.metric_ = metric;
    param.io_type_ = IOType::MEMORY_IO;
    return param;
}

#endif  //MYANNS_ANNSLIB_H
