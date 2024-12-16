//
// Created by XiaoWu on 2024/12/12.
//

#include <iostream>
#include <random>

std::uniform_real_distribution<double> distribution(0.0, 1.0);
std::default_random_engine random_engine_(2024);

double get_random(){
    return -log(distribution(random_engine_)) * 1 / log(1.0 * 10);
}

int main(){
    double r = get_random();
    std::cout << r << std::endl;

    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;
    r = get_random();
    std::cout << r << std::endl;

    return 0;
}