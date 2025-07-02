#include "annslib.h"

void
test_file() {
    Log::setVerbose(true);
    Log::redirect("test_log.txt");

    logger << "Testing logger functionality." << std::endl;
}

void
test_nofile() {
    Log::setVerbose(true);
    Log::redirect();

    logger << "Testing logger without file redirection." << std::endl;
}

void
test_absolute_path() {
    Log::setVerbose(true);
    Log::redirect("/root/mount/my-anns/output/logs/test_log.txt");

    logger << "Testing logger with absolute path." << std::endl;
}

void
test_relative_path() {
    Log::setVerbose(true);
    Log::redirect("output/logs/test_log.txt");

    logger << "Testing logger with relative path." << std::endl;
}

void
test_double_redirect() {
    Log::setVerbose(true);
    Log::redirect("test_log.txt");
    logger << "Testing single redirection of logger." << std::endl;
    Log::redirect("test_log.txt");
    logger << "Testing double redirection of logger." << std::endl;
}

void
test_no_extension() {
    Log::setVerbose(true);
    Log::redirect("test_log");

    logger << "Testing logger with no file extension." << std::endl;
}

int
main() {
    test_no_extension();
    return 0;
}
