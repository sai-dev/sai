#ifndef SHARED_MUTEX_H_INCLUDE
#define SHARED_MUTEX_H_INCLUDE

#include <atomic>
#include <thread>
#include <chrono>
#include <cassert>

class SharedMutex {
public:
    SharedMutex() {};

    void lock();
    void unlock();

    void lock_shared();
    void unlock_shared();

private:
    bool acquire_exclusive_lock();
    int get_share_counter();

    void take_break();

    std::atomic<int> m_share_counter{0};
    std::atomic<bool> m_exclusive{false};
    std::chrono::microseconds m_wait_microseconds{0};
};

inline void SharedMutex::take_break() {
    std::this_thread::sleep_for(m_wait_microseconds);
}

inline bool SharedMutex::acquire_exclusive_lock() {
    bool expected = false;
    return m_exclusive.compare_exchange_weak(expected, true);
}

inline int SharedMutex::get_share_counter() {
    return m_share_counter.load();
}

inline void SharedMutex::lock() {
    while (!acquire_exclusive_lock()) {
        take_break();
    }
    while (get_share_counter() > 0) {
        take_break();
    }
}

inline void SharedMutex::unlock() {
    m_exclusive.store(false);
}

inline void SharedMutex::lock_shared() {
    while (true) {
        m_share_counter.fetch_add(1);
        if (m_exclusive.load()) {
            m_share_counter.fetch_sub(1);
	    } else {
		    break;
        }
        while (m_exclusive.load()) {
            take_break();
        }
    }
}

inline void SharedMutex::unlock_shared() {
    m_share_counter.fetch_sub(1);
}

enum class lock_t {
    X_LOCK,
    S_LOCK
};

template<lock_t T>
class LockGuard {
public:
    LockGuard(SharedMutex &sm);
    ~LockGuard();

private:
    SharedMutex &m_sm;
};

template<>
inline LockGuard<lock_t::X_LOCK>::LockGuard(SharedMutex& sm) : m_sm(sm) {
    m_sm.lock();
}

template<>
inline LockGuard<lock_t::X_LOCK>::~LockGuard() {
    m_sm.unlock();
}

template<>
inline LockGuard<lock_t::S_LOCK>::LockGuard(SharedMutex& sm) : m_sm(sm) {
    m_sm.lock_shared();
}

template<>
inline LockGuard<lock_t::S_LOCK>::~LockGuard() {
    m_sm.unlock_shared();
}
#endif
