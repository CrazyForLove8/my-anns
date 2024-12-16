add_executable(test_merging_implemented_algorithms test_merging_implemented_algorithms.cpp)
target_link_libraries(test_merging_implemented_algorithms ${PROJECT_NAME})

add_executable(test_merging_graphs test_merging_graphs.cpp)
target_link_libraries(test_merging_graphs ${PROJECT_NAME})

add_executable(test_index_construction test_index_construction.cpp)
target_link_libraries(test_index_construction ${PROJECT_NAME})

add_executable(my_test my_test.cpp)
target_link_libraries(my_test ${PROJECT_NAME})

add_executable(test_algorithm test_algorithm.cpp)
target_link_libraries(test_algorithm ${PROJECT_NAME})

add_executable(test_duplicate test_shnsw.cpp)
target_link_libraries(test_duplicate ${PROJECT_NAME})

add_executable(test_dhnsw test_dhnsw.cpp)
target_link_libraries(test_dhnsw ${PROJECT_NAME})
