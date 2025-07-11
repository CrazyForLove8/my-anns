#include "output.h"

using namespace graph;

std::string graph_output_dir = "./graph_output/";

std::string
filename_separator() {
    return "_";
}

std::string
get_path(std::string filename) {
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

        check_and_remove(p.string());
        filename = p.string();
        return filename;
    }
    if (!std::filesystem::exists(graph_output_dir)) {
        std::filesystem::create_directories(graph_output_dir);
    }
    std::filesystem::path pp(graph_output_dir + filename);
    check_and_remove(pp.string());
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

bool
check_if_exist(const std::string& filename) {
    std::filesystem::path p(filename);
    return std::filesystem::exists(p) && std::filesystem::is_regular_file(p);
}

void
check_and_remove(const std::string& filename) {
    std::filesystem::path p(filename);
    if (std::filesystem::exists(p)) {
        logger << "Removing existing file: " << p << std::endl;
        std::filesystem::remove(p);
    }
}

void
check_prefix_and_remove(const std::string& file_path_with_prefix) {
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
                    logger << "Error removing file: " << entry.path() << " - " << e.what()
                           << std::endl;
                }
            }
        }
    }
}

void
ParamsHelper::write(const ParamMap& map, std::ofstream& out) {
    size_t count = map.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& [key, val] : map) {
        size_t keyLen = key.size();
        out.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
        out.write(key.data(), keyLen);

        if (std::holds_alternative<uint64_t>(val)) {
            uint8_t type = 0;
            out.write(reinterpret_cast<const char*>(&type), sizeof(type));
            auto i = std::get<uint64_t>(val);
            out.write(reinterpret_cast<const char*>(&i), sizeof(i));
        } else if (std::holds_alternative<double_t>(val)) {
            uint8_t type = 1;
            out.write(reinterpret_cast<const char*>(&type), sizeof(type));
            auto f = std::get<double_t>(val);
            out.write(reinterpret_cast<const char*>(&f), sizeof(f));
        } else if (std::holds_alternative<std::string>(val)) {
            uint8_t type = 2;
            out.write(reinterpret_cast<const char*>(&type), sizeof(type));
            auto b = std::get<std::string>(val);
            size_t b_size = b.size();
            out.write(reinterpret_cast<const char*>(&b_size), sizeof(b_size));
            out.write(b.data(), b_size);
        }
    }

    out.close();
}

void
ParamsHelper::read(ParamMap& map, std::ifstream& in) {
    size_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    for (size_t i = 0; i < count; ++i) {
        size_t keyLen;
        in.read(reinterpret_cast<char*>(&keyLen), sizeof(keyLen));
        std::string key(keyLen, '\0');
        in.read(&key[0], keyLen);

        uint8_t type;
        in.read(reinterpret_cast<char*>(&type), sizeof(type));

        if (type == 0) {
            uint64_t value;
            in.read(reinterpret_cast<char*>(&value), sizeof(value));
            map[key] = value;
        } else if (type == 1) {
            double_t value;
            in.read(reinterpret_cast<char*>(&value), sizeof(value));
            map[key] = value;
        } else if (type == 2) {
            size_t str_size;
            in.read(reinterpret_cast<char*>(&str_size), sizeof(str_size));
            std::string value(str_size, '\0');
            in.read(&value[0], str_size);
            map[key] = value;
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