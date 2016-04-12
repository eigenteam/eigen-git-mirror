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

// This defines an interface that ThreadPoolDevice can take to use
// custom thread pools underneath.
class ThreadPoolInterface {
 public:
  virtual void Schedule(std::function<void()> fn) = 0;

  virtual ~ThreadPoolInterface() {}
};

// The implementation of the ThreadPool type ensures that the Schedule method
// runs the functions it is provided in FIFO order when the scheduling is done
// by a single thread.
// Environment provides a way to create threads and also allows to intercept
// task submission and execution.
template <typename Environment>
class ThreadPoolTempl : public ThreadPoolInterface {
 public:
  // Construct a pool that contains "num_threads" threads.
  explicit ThreadPoolTempl(int num_threads, Environment env = Environment())
      : env_(env), threads_(num_threads), waiters_(num_threads) {
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(env.CreateThread([this]() { WorkerLoop(); }));
    }
  }

  // Wait until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPoolTempl() {
    {
      // Wait for all work to get done.
      std::unique_lock<std::mutex> l(mu_);
      while (!pending_.empty()) {
        empty_.wait(l);
      }
      exiting_ = true;

      // Wakeup all waiters.
      for (auto w : waiters_) {
        w->ready = true;
        w->task.f = nullptr;
        w->cv.notify_one();
      }
    }

    // Wait for threads to finish.
    for (auto t : threads_) {
      delete t;
    }
  }

  // Schedule fn() for execution in the pool of threads. The functions are
  // executed in the order in which they are scheduled.
  void Schedule(std::function<void()> fn) {
    Task t = env_.CreateTask(std::move(fn));
    std::unique_lock<std::mutex> l(mu_);
    if (waiters_.empty()) {
      pending_.push_back(std::move(t));
    } else {
      Waiter* w = waiters_.back();
      waiters_.pop_back();
      w->ready = true;
      w->task = std::move(t);
      w->cv.notify_one();
    }
  }

 protected:
  void WorkerLoop() {
    std::unique_lock<std::mutex> l(mu_);
    Waiter w;
    Task t;
    while (!exiting_) {
      if (pending_.empty()) {
        // Wait for work to be assigned to me
        w.ready = false;
        waiters_.push_back(&w);
        while (!w.ready) {
          w.cv.wait(l);
        }
        t = w.task;
        w.task.f = nullptr;
      } else {
        // Pick up pending work
        t = std::move(pending_.front());
        pending_.pop_front();
        if (pending_.empty()) {
          empty_.notify_all();
        }
      }
      if (t.f) {
        mu_.unlock();
        env_.ExecuteTask(t);
        t.f = nullptr;
        mu_.lock();
      }
    }
  }

 private:
  typedef typename Environment::Task Task;
  typedef typename Environment::EnvThread Thread;

  struct Waiter {
    std::condition_variable cv;
    Task task;
    bool ready;
  };

  Environment env_;
  std::mutex mu_;
  MaxSizeVector<Thread*> threads_;  // All threads
  MaxSizeVector<Waiter*> waiters_;  // Stack of waiting threads.
  std::deque<Task> pending_;          // Queue of pending work
  std::condition_variable empty_;          // Signaled on pending_.empty()
  bool exiting_ = false;
};

struct StlThreadEnvironment {
  struct Task {
    std::function<void()> f;
  };

  // EnvThread constructor must start the thread,
  // destructor must join the thread.
  class EnvThread {
   public:
    EnvThread(std::function<void()> f) : thr_(f) {}
    ~EnvThread() { thr_.join(); }

   private:
    std::thread thr_;
  };

  EnvThread* CreateThread(std::function<void()> f) { return new EnvThread(f); }
  Task CreateTask(std::function<void()> f) { return Task{std::move(f)}; }
  void ExecuteTask(const Task& t) { t.f(); }
};

typedef ThreadPoolTempl<StlThreadEnvironment> ThreadPool;


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

 private:
  ThreadPoolInterface* pool_;
  size_t num_threads_;
};


}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_THREAD_POOL_H
