//
// Created by XiaoWu on 2025/9/5.
//

#ifndef MYANNS_TYPING_H
#define MYANNS_TYPING_H

#include <string>
#include <unordered_map>
#include <variant>

#ifdef __GNUC__
#ifdef __AVX__
#define ALIGNMENT 32
#else
#ifdef __SSE2__
#define ALIGNMENT 16
#else
#define ALIGNMENT 4
#endif
#endif
#endif

#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

namespace graph {

using IdType = uint32_t;
using DimType = uint16_t;

using DataPtr = uint8_t*;

using Value = std::variant<uint64_t, double, std::string>;
using ParamMap = std::unordered_map<std::string, Value>;

}  // namespace graph

#endif  //MYANNS_TYPING_H
