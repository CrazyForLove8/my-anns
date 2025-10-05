//
// Created by XiaoWu on 2025/3/4.
//

#include "annslib.h"

#define MULTI_THREAD 1

int
main() {
    Log::setVerbose(true);

int ret = std::system("mpv /mnt/c/Windows/Media/Alarm01.wav");
    if (ret != 0) {
        std::cerr << "Warning: System command failed with exit code " << ret << std::endl;
    }
}