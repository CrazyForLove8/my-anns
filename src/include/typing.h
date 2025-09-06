//
// Created by XiaoWu on 2025/9/5.
//

#ifndef MYANNS_TYPING_H
#define MYANNS_TYPING_H

#include <string>
#include <unordered_map>

#include "variant"

namespace graph {

using IdType = uint32_t;

using Value = std::variant<uint64_t, double, std::string>;
using ParamMap = std::unordered_map<std::string, Value>;

}  // namespace graph

#endif  //MYANNS_TYPING_H
