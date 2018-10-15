/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"
#include "Utils.h"

#include <algorithm>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/select.h>
#endif

#include "GTP.h"

Utils::ThreadPool thread_pool;

bool Utils::input_pending(void) {
#ifdef HAVE_SELECT
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(0,&read_fds);
    struct timeval timeout{0,0};
    select(1,&read_fds,nullptr,nullptr,&timeout);
    return FD_ISSET(0, &read_fds);
#else
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;

    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
        if (!pipe) {
            SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
            FlushConsoleInputBuffer(inh);
        }
    }

    if (pipe) {
        if (!PeekNamedPipe(inh, nullptr, 0, nullptr, &dw, nullptr)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw;
    } else {
        if (!GetNumberOfConsoleInputEvents(inh, &dw)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw > 1;
    }
    return false;
#endif
}

static std::mutex IOmutex;

void Utils::myprintf(const char *fmt, ...) {
    if (cfg_quiet) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        va_start(ap, fmt);
        vfprintf(cfg_logfile_handle, fmt, ap);
        va_end(ap);
    }
}

static void gtp_fprintf(FILE* file, const std::string& prefix,
                        const char *fmt, va_list ap) {
    fprintf(file, "%s ", prefix.c_str());
    vfprintf(file, fmt, ap);
    fprintf(file, "\n\n");
}

static void gtp_base_printf(int id, std::string prefix,
                            const char *fmt, va_list ap) {
    if (id != -1) {
        prefix += std::to_string(id);
    }

    gtp_fprintf(stdout, prefix, fmt, ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        gtp_fprintf(cfg_logfile_handle, prefix, fmt, ap);
    }
}

void Utils::gtp_printf(int id, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "=", fmt, ap);
    va_end(ap);
}

void Utils::gtp_fail_printf(int id, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "?", fmt, ap);
    va_end(ap);
}

void Utils::log_input(const std::string& input) {
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        fprintf(cfg_logfile_handle, ">>%s\n", input.c_str());
    }
}

size_t Utils::ceilMultiple(size_t a, size_t b) {
    if (a % b == 0) {
        return a;
    }

    auto ret = a + (b - a % b);
    return ret;
}

float Utils::sigmoid_interval_avg(float alpkt, float beta, float s, float t) {
    if (s>t) {
	std::swap(s,t);
    }
    const auto h = beta * (t - s);
    
    if (h < 0.001f) {
	return sigmoid(alpkt,beta,(s+t)/2).first;
    }

    #ifndef NDEBUG
    if (std::abs(s) + std::abs(t) > 2000.0f) {
	myprintf("Warning: integration interval out of bound: [%f,%f].\n", s, t);
    }
    #endif

    const auto a = std::abs(alpkt+s);
    const auto b = std::abs(alpkt+t);

    const auto main_term = (alpkt+s)*(alpkt+t) > 0 ?
	( alpkt+s > 0 ? 1.0f : 0.0f ) :
	0.5f + 0.5f*(b-a)/(t-s);

    const auto aa = log_sigmoid(a*beta) / h;
    const auto bb = log_sigmoid(b*beta) / h;

    //    myprintf("integral: alpkt=%f, beta=%f, s=%f, t=%f\n"
    //	     "h=%f, a=%f, b=%f, main_term=%f, aa=%f, bb=%f\n",
    //	     alpkt, beta, s, t, h, a, b, main_term, aa, bb);
    return main_term - bb + aa;
}

float Utils::log_sigmoid(float x) {
    // Returns log(1/(1+exp(-x))
    // When sigmoid is about 1, log(sigmoid) is about 0 but not very precise
    // This function provides a robust estimate in that case.
    // Its argument should be beta*(alpha+bonus)
    if (x>10.0f) {
        const auto ul = std::exp(-x);
        const auto ll = 1.0f/(1.0f+std::exp(x));
        return -std::sqrt(ul*ll);
    }
    return std::log(sigmoid(x,1.0f,0.0f).first);
}

float Utils::median(std::vector<float> & sample){
    sort(sample.begin(), sample.end());
    const auto len = sample.size();

    if(len % 2) {
        return sample[len/2];
    }

    return 0.5 * sample[len/2] + 0.5 * sample[len/2-1];
}
