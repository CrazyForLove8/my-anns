#include "timer.h"

using namespace graph;

Timer::Timer() {
    started = false;
    _elapsed = 0.0;
}

void
Timer::start() {
    if (started) {
        throw std::runtime_error("Timer already started");
    }
    started = true;
    _start = std::chrono::steady_clock::now();
}

void
Timer::end() {
    if (!started) {
        std::cerr << "Warning: Timer was not started. Cannot end." << std::endl;
    }
    started = false;
    _end = std::chrono::steady_clock::now();
    _elapsed = std::chrono::duration_cast<std::chrono::duration<double> >(_end - _start).count();
}

double
Timer::elapsed() const {
    return started ? std::chrono::duration_cast<std::chrono::duration<double> >(
                         std::chrono::steady_clock::now() - _start)
                         .count()
                   : _elapsed;
}
