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

SET(CMAKE_CXX_FLAGS "-std=c++17 -DNDEBUG -fpic -DHAVE_CXX0X -fopenmp -ftree-vectorize -march=native -msse4.2 -mavx2 -mfma")

set(SANITIZER_TYPE "none")

if (SANITIZER_TYPE STREQUAL "address")
    message(STATUS "Using AddressSanitizer (-fsanitize=address)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

elseif (SANITIZER_TYPE STREQUAL "thread")
    message(STATUS "Using ThreadSanitizer (-fsanitize=thread)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")

else ()
    message(STATUS "No sanitizer: enabling -Ofast optimization")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
endif ()

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)
