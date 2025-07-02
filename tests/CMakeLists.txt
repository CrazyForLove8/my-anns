add_executable(my_test my_test.cpp)
target_link_libraries(my_test ${PROJECT_NAME})

add_executable(fgim_exp fgim_exp.cpp)
target_link_libraries(fgim_exp ${PROJECT_NAME})