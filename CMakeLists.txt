cmake_minimum_required(VERSION 3.22.1)
project(myanns)

include_directories(${PROJECT_SOURCE_DIR}/src/include)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
    message(FATAL_ERROR "Please install OpenMP")
endif ()

find_package(fmt)
if (NOT fmt_FOUND)
    message(FATAL_ERROR "Please install fmt")
endif ()

SET(CMAKE_CXX_FLAGS "-fpic -DHAVE_CXX0X -fopenmp -ftree-vectorize -Ofast -march=native -msse4.2 -mavx2 -mfma")

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)

target_link_libraries(myanns fmt::fmt)
