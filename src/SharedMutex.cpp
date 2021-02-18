#include "SharedMutex.h"

template<> 
LockGuard<lock_t::X_LOCK>::LockGuard(SharedMutex &sm) : m_sm(sm) {
    m_sm.lock();
}

template<> 
LockGuard<lock_t::X_LOCK>::~LockGuard() {
    m_sm.unlock();
}

template<> 
LockGuard<lock_t::S_LOCK>::LockGuard(SharedMutex &sm) : m_sm(sm) {
    m_sm.lock_shared();
}

template<> 
LockGuard<lock_t::S_LOCK>::~LockGuard() {
    m_sm.unlock_shared();
}
