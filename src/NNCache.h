/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Michael O and contributors

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef NNCACHE_H_INCLUDED
#define NNCACHE_H_INCLUDED

#include "config.h"

#ifdef USE_STAND_SHARED_MUTEX
#include <shared_mutex>
#else
#include "SharedMutex.h"
#endif

#include <array>
#include <deque>
#include <memory>
#include <unordered_map>


class NNCache {
public:

    // Maximum size of the cache in number of items.
    static constexpr int MAX_CACHE_COUNT = 150'000;

    // Minimum size of the cache in number of items.
    static constexpr int MIN_CACHE_COUNT = 6'000;

    struct Netresult {
        // 19x19 board positions
        std::array<float, NUM_INTERSECTIONS> policy;

        // pass
        float policy_pass;

        // winrate
        float value;

        // sigmoid alpha
        float alpha;

        // sigmoid beta
        float beta;

        bool is_sai;

        Netresult() : policy_pass(0.0f), value(0.0f), alpha(0.0f), beta(0.0f), is_sai(false) {
            policy.fill(0.0f);
        }
    };

    static constexpr size_t ENTRY_SIZE =
          sizeof(Netresult)
        + sizeof(std::uint64_t)
        + sizeof(std::unique_ptr<Netresult>);

    NNCache(int size = MAX_CACHE_COUNT);  // ~ 208MiB

    // Set a reasonable size gives max number of playouts
    void set_size_from_playouts(int max_playouts);

    // Resize NNCache
    void resize(int size);
    void clear();

    // Try and find an existing entry.
    bool lookup(std::uint64_t hash, Netresult & result);

    // Insert a new entry.
    void insert(std::uint64_t hash,
                const Netresult& result);

    // Return the hit rate ratio.
    std::pair<int, int> hit_rate() const {
        return {m_hits, m_lookups};
    }

    void dump_stats();

    // Return the estimated memory consumption of the cache.
    size_t get_estimated_size();
private:
#ifdef USE_STAND_SHARED_MUTEX
    std::shared_mutex m_mutex;
#else
    SharedMutex m_mutex;
#endif
    size_t m_size;

    // Statistics
    int m_hits{0};
    int m_lookups{0};
    int m_inserts{0};

    struct Entry {
        Entry(const Netresult& r)
            : result(r) {}
        Netresult result;  // ~ 1.4KiB
    };

    // Map from hash to {features, result}
    std::unordered_map<std::uint64_t, std::unique_ptr<const Entry>> m_cache;
    // Order entries were added to the map.
    std::deque<size_t> m_order;
};

#endif
