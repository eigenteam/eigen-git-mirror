// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_THREADS) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_THREAD_POOL_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_THREAD_POOL_H

namespace Eigen {

// Use the SimpleThreadPool by default. We'll switch to the new non blocking
// thread pool later.
#ifdef EIGEN_USE_NONBLOCKING_THREAD_POOL
template <typename Env> using ThreadPoolTempl = NonBlockingThreadPoolTempl<Env>;
typedef NonBlockingThreadPool ThreadPool;
#else
template <typename Env> using ThreadPoolTempl = SimpleThreadPoolTempl<Env>;
typedef SimpleThreadPool ThreadPool;
#endif


// Barrier is an object that allows one or more threads to wait until
// Notify has been called a specified number of times.
class Barrier {
 public:
  Barrier(unsigned int count) : state_(count << 1), notified_(false) {
    eigen_assert(((count << 1) >> 1) == count);
  }
  ~Barrier() {
    eigen_assert((state_>>1) == 0);
  }

  void Notify() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
      eigen_assert(((v + 2) & ~1) != 0);
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    std::unique_lock<std::mutex> l(mu_);
    eigen_assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return;
    std::unique_lock<std::mutex> l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<unsigned int> state_;  // low bit is waiter flag
  bool notified_;
};


// Notification is an object that allows a user to to wait for another
// thread to signal a notification that an event has occurred.
//
// Multiple threads can wait on the same Notification object,
// but only one caller must call Notify() on the object.
struct Notification : Barrier {
  Notification() : Barrier(1) {};
};


// Runs an arbitrary function and then calls Notify() on the passed in
// Notification.
template <typename Function, typename... Args> struct FunctionWrapperWithNotification
{
  static void run(Notification* n, Function f, Args... args) {
    f(args...);
    if (n) {
      n->Notify();
    }
  }
};

template <typename Function, typename... Args> struct FunctionWrapperWithBarrier
{
  static void run(Barrier* b, Function f, Args... args) {
    f(args...);
    if (b) {
      b->Notify();
    }
  }
};

template <typename SyncType>
static EIGEN_STRONG_INLINE void wait_until_ready(SyncType* n) {
  if (n) {
    n->Wait();
  }
}


// Build a thread pool device on top the an existing pool of threads.
struct ThreadPoolDevice {
  // The ownership of the thread pool remains with the caller.
  ThreadPoolDevice(ThreadPoolInterface* pool, size_t num_cores) : pool_(pool), num_threads_(num_cores) { }

  EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }

  EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }

  EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
    ::memcpy(dst, src, n);
  }
  EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }

  EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
    ::memset(buffer, c, n);
  }

  EIGEN_STRONG_INLINE size_t numThreads() const {
    return num_threads_;
  }

  EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
    return l1CacheSize();
  }

  EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // The l3 cache size is shared between all the cores.
    return l3CacheSize() / num_threads_;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
    // Should return an enum that encodes the ISA supported by the CPU
    return 1;
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE Notification* enqueue(Function&& f, Args&&... args) const {
    Notification* n = new Notification();
    std::function<void()> func =
      std::bind(&FunctionWrapperWithNotification<Function, Args...>::run, n, f, args...);
    pool_->Schedule(func);
    return n;
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE void enqueue_with_barrier(Barrier* b,
                                                Function&& f,
                                                Args&&... args) const {
    std::function<void()> func = std::bind(
        &FunctionWrapperWithBarrier<Function, Args...>::run, b, f, args...);
    pool_->Schedule(func);
  }

  template <class Function, class... Args>
  EIGEN_STRONG_INLINE void enqueueNoNotification(Function&& f, Args&&... args) const {
    std::function<void()> func = std::bind(f, args...);
    pool_->Schedule(func);
  }

  // parallelFor executes f with [0, size) arguments in parallel and waits for
  // completion. Block size is choosen between min_block_size and
  // 2 * min_block_size to achieve the best parallel efficiency.
  // If min_block_size == -1, parallelFor uses block size of 1.
  // If hard_align > 0, block size is aligned to hard_align.
  // If soft_align > hard_align, block size is aligned to soft_align provided
  // that it does not increase block size too much.
  void parallelFor(Index size, Index min_block_size, Index hard_align,
                   Index soft_align,
                   std::function<void(Index, Index)> f) const {
    if (size <= 1 || (min_block_size != -1 && size < min_block_size) ||
        numThreads() == 1) {
      f(0, size);
      return;
    }

    Index block_size = 1;
    Index block_count = size;
    if (min_block_size != -1) {
      // Calculate block size based on (1) estimated cost and (2) parallel
      // efficiency. We want blocks to be not too small to mitigate
      // parallelization overheads; not too large to mitigate tail effect and
      // potential load imbalance and we also want number of blocks to be evenly
      // dividable across threads.
      min_block_size = numext::maxi<Index>(min_block_size, 1);
      block_size = numext::mini(min_block_size, size);
      // Upper bound on block size:
      const Index max_block_size = numext::mini(min_block_size * 2, size);
      block_size = numext::mini(
          alignBlockSize(block_size, hard_align, soft_align), size);
      block_count = divup(size, block_size);
      // Calculate parallel efficiency as fraction of total CPU time used for
      // computations:
      double max_efficiency =
          static_cast<double>(block_count) /
          (divup<int>(block_count, numThreads()) * numThreads());
      // Now try to increase block size up to max_block_size as long as it
      // doesn't decrease parallel efficiency.
      for (Index prev_block_count = block_count; prev_block_count > 1;) {
        // This is the next block size that divides size into a smaller number
        // of blocks than the current block_size.
        Index coarser_block_size = divup(size, prev_block_count - 1);
        coarser_block_size =
            alignBlockSize(coarser_block_size, hard_align, soft_align);
        if (coarser_block_size > max_block_size) {
          break;  // Reached max block size. Stop.
        }
        // Recalculate parallel efficiency.
        const Index coarser_block_count = divup(size, coarser_block_size);
        eigen_assert(coarser_block_count < prev_block_count);
        prev_block_count = coarser_block_count;
        const double coarser_efficiency =
            static_cast<double>(coarser_block_count) /
            (divup<int>(coarser_block_count, numThreads()) * numThreads());
        if (coarser_efficiency + 0.01 >= max_efficiency) {
          // Taking it.
          block_size = coarser_block_size;
          block_count = coarser_block_count;
          if (max_efficiency < coarser_efficiency) {
            max_efficiency = coarser_efficiency;
          }
        }
      }
    }

    // Recursively divide size into halves until we reach block_size.
    // Division code rounds mid to block_size, so we are guaranteed to get
    // block_count leaves that do actual computations.
    Barrier barrier(block_count);
    std::function<void(Index, Index)> handleRange;
    handleRange = [=, &handleRange, &barrier, &f](Index first, Index last) {
      if (last - first <= block_size) {
        // Single block or less, execute directly.
        f(first, last);
        barrier.Notify();
        return;
      }
      // Split into halves and submit to the pool.
      Index mid = first + divup((last - first) / 2, block_size) * block_size;
      pool_->Schedule([=, &handleRange]() { handleRange(mid, last); });
      pool_->Schedule([=, &handleRange]() { handleRange(first, mid); });
    };
    handleRange(0, size);
    barrier.Wait();
  }

 private:
  static Index alignBlockSize(Index size, Index hard_align, Index soft_align) {
    if (soft_align > hard_align && size >= 4 * soft_align) {
      // Align to soft_align, if it won't increase size by more than 25%.
      return (size + soft_align - 1) & ~(soft_align - 1);
    }
    if (hard_align > 0) {
      return (size + hard_align - 1) & ~(hard_align - 1);
    }
    return size;
  }


  ThreadPoolInterface* pool_;
  size_t num_threads_;
};


}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_THREAD_POOL_H
