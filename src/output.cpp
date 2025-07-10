#include "output.h"

using namespace graph;

std::string graph_output_dir = "./graph_output/";

std::string
filename_separator() {
    return "_";
}

std::string
get_path(std::string filename){
    if (filename.find(".bin") == std::string::npos) {
        filename += ".bin";
    }
    std::filesystem::path p(filename);
    if (p.is_absolute()) {
        std::filesystem::path parent_dir = p.parent_path();
        std::string stem = p.stem().string();
        std::string extension = p.extension().string();

        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir)) {
            std::filesystem::create_directories(parent_dir);
        }

        if (std::filesystem::exists(p)) {
            std::mt19937 rng(std::random_device{}());
            std::string new_relative_filename = stem + "_" + std::to_string(rng()) + extension;
            filename = (parent_dir / new_relative_filename).string();
        } else {
            filename = p.string();
        }
        return filename;
    }
    if (!std::filesystem::exists(graph_output_dir)) {
        std::filesystem::create_directories(graph_output_dir);
    }
    std::filesystem::path pp(graph_output_dir + filename);
    if (std::filesystem::exists(pp)) {
        std::mt19937 rng(std::random_device{}());
        filename = pp.stem().string() + "_" + std::to_string(rng()) + pp.extension().string();
    }
    return graph_output_dir + filename;
}

std::string
append(const std::string& filename, const std::string& suffix) {
    std::filesystem::path p(filename);
    return p.parent_path().string() + "/" + p.stem().string() + suffix + p.extension().string();
}

std::string
get_suffix(const std::string& filename, const int n) {
    std::filesystem::path p(filename);
    std::string stem = p.stem().string();
    if (n < 0) {
        auto last_underscore_pos = stem.rfind(filename_separator());
        if (last_underscore_pos != std::string::npos) {
            return stem.substr(last_underscore_pos + 1);
        }
        return "";
    }
    if (n >= stem.size()) {
        return stem;
    }

    return stem.substr(stem.size() - n);
}

void check_and_remove(const std::string& filename){
    std::filesystem::path p(filename);
    if (std::filesystem::exists(p)) {
        logger << "Removing existing file: " << p << std::endl;
        std::filesystem::remove(p);
    }
}

void check_prefix_and_remove(const std::string& file_path_with_prefix) {
    std::filesystem::path full_path(file_path_with_prefix);
    std::filesystem::path directory = full_path.parent_path();
    if (directory.empty()) {
        directory = std::filesystem::current_path();
    }
    std::string prefix_to_match = full_path.stem().string();
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().stem().string();

            if (filename.rfind(prefix_to_match, 0) == 0) {
                try {
                    std::filesystem::remove(entry.path());
                } catch (const std::filesystem::filesystem_error& e) {
                    logger << "Error removing file: " << entry.path() << " - " << e.what() << std::endl;
                }
            }
        }
    }
}

CsvLogger::CsvLogger(const std::string& filePath, int precision)
    : filePath_(filePath), headerWritten_(false), precision_(precision) {
    ofs_.open(filePath_, std::ios::out | std::ios::app);
    if (!ofs_.is_open()) {
        std::cerr << "Error: CsvLogger - Could not open file " << filePath_ << std::endl;
    }
}

CsvLogger::~CsvLogger() {
    if (ofs_.is_open()) {
        ofs_.close();
    }
}

bool
CsvLogger::writeHeader(const std::vector<std::string>& headers) {
    if (!ofs_.is_open()) {
        std::cerr << "Error: CsvLogger - Cannot write header, file not open." << std::endl;
        return false;
    }
    if (headerWritten_) {
        return true;
    }

    for (size_t i = 0; i < headers.size(); ++i) {
        ofs_ << toCsvString(headers[i]);
        if (i < headers.size() - 1) {
            ofs_ << ",";
        }
    }
    ofs_ << "\n";
    ofs_.flush();
    headerWritten_ = true;
    return true;
}

bool
CsvLogger::isOpen() const {
    return ofs_.is_open();
}

std::string
CsvLogger::toCsvString(const std::string& value) {
    std::string escaped_value = value;
    if (escaped_value.find(',') != std::string::npos ||
        escaped_value.find('\n') != std::string::npos ||
        escaped_value.find('"') != std::string::npos) {
        size_t pos = escaped_value.find('"');
        while (pos != std::string::npos) {
            escaped_value.replace(pos, 1, "\"\"");
            pos = escaped_value.find('"', pos + 2);
        }
        escaped_value = "\"" + escaped_value + "\"";
    }
    return escaped_value;
}
