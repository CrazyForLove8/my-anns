//
// Created by root on 25-6-17.
//

#ifndef OUTPUT_H
#define OUTPUT_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dtype.h"

std::string
filename_separator();

std::string
get_path(std::string filename);

std::string
append(const std::string& filename, const std::string& suffix);

std::string
get_suffix(const std::string& filename, int n = -1);

bool
check_if_exist(const std::string& filename);

void
check_and_remove(const std::string& filename);

void
check_prefix_and_remove(const std::string& file_path_with_prefix);

class ParamsHelper {
public:
    static void
    write(const graph::ParamMap& map, std::ofstream& out);

    static void
    read(graph::ParamMap& map, std::ifstream& in);
};

class CsvLogger {
public:
    explicit CsvLogger(const std::string& filePath, int precision = 4);

    ~CsvLogger();

    bool
    writeHeader(const std::vector<std::string>& headers);

    template <typename T>
    bool
    writeRow(const std::vector<T>& values) {
        if (!ofs_.is_open()) {
            return false;
        }

        for (size_t i = 0; i < values.size(); ++i) {
            ofs_ << toCsvString(values[i]);
            if (i < values.size() - 1) {
                ofs_ << ",";
            }
        }
        ofs_ << "\n";
        ofs_.flush();
        return true;
    }

    bool
    isOpen() const;

private:
    std::string filePath_;
    std::ofstream ofs_;
    bool headerWritten_;
    int precision_;

    std::string
    toCsvString(const std::string& value);

    template <typename T>
    std::string
    toCsvString(T value) {
        if
            constexpr(std::is_floating_point_v<T>) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(precision_) << value;
                return oss.str();
            }
        else if
            constexpr(std::is_same_v<T, std::string>) {
                return toCsvString(value);
            }
        else {
            std::ostringstream oss;
            oss << value;
            return oss.str();
        }
    }

    CsvLogger(const CsvLogger&) = delete;
    CsvLogger&
    operator=(const CsvLogger&) = delete;
};

#endif  //OUTPUT_H
