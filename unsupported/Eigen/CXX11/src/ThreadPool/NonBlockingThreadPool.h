// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
#define EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H


namespace Eigen {

template <typename Environment>
class NonBlockingThreadPoolTempl : public Eigen::ThreadPoolInterface {
 public:
  typedef typename Environment::Task Task;
  typedef RunQueue<Task, 1024> Queue;

  NonBlockingThreadPoolTempl(int num_threads, Environment env = Environment())
      : env_(env),
        threads_(num_threads),
        queues_(num_threads),
        waiters_(num_threads),
        blocked_(),
        spinning_(),
        done_(),
        ec_(waiters_) {
    for (int i = 0; i < num_threads; i++) queues_.push_back(new Queue());
    for (int i = 0; i < num_threads; i++)
      threads_.push_back(env_.CreateThread([this, i]() { WorkerLoop(i); }));
  }

  ~NonBlockingThreadPoolTempl() {
    done_.store(true, std::memory_order_relaxed);
    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    ec_.Notify(true);

    // Join threads explicitly to avoid destruction order issues.
    for (size_t i = 0; i < threads_.size(); i++) delete threads_[i];
    for (size_t i = 0; i < threads_.size(); i++) delete queues_[i];
  }

  void Schedule(std::function<void()> fn) {
    Task t = env_.CreateTask(std::move(fn));
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue* q = queues_[pt->index];
      t = q->PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      Queue* q = queues_[Rand(&pt->rand) % queues_.size()];
      t = q->PushBack(std::move(t));
    }
    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t.f)
      ec_.Notify(false);
    else
      env_.ExecuteTask(t);  // Push failed, execute directly.
  }

 private:
  typedef typename Environment::EnvThread Thread;

  struct PerThread {
    bool inited;
    NonBlockingThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    unsigned index;         // Worker thread index in pool.
    unsigned rand;          // Random generator state.
  };

  Environment env_;
  MaxSizeVector<Thread*> threads_;
  MaxSizeVector<Queue*> queues_;
  std::vector<EventCount::Waiter> waiters_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> spinning_;
  std::atomic<bool> done_;
  EventCount ec_;

  // Main worker thread loop.
  void WorkerLoop(unsigned index) {
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->index = index;
    Queue* q = queues_[index];
    EventCount::Waiter* waiter = &waiters_[index];
    std::vector<Task> stolen;
    for (;;) {
      Task t;
      if (!stolen.empty()) {
        t = std::move(stolen.back());
        stolen.pop_back();
      }
      if (!t.f) t = q->PopFront();
      if (!t.f) {
        if (Steal(&stolen)) {
          t = std::move(stolen.back());
          stolen.pop_back();
          while (stolen.size()) {
            Task t1 = q->PushFront(std::move(stolen.back()));
            stolen.pop_back();
            if (t1.f) {
              // There is not much we can do in this case. Just execute the
              // remaining directly.
              stolen.push_back(std::move(t1));
              break;
            }
          }
        }
      }
      if (t.f) {
        env_.ExecuteTask(t);
        continue;
      }
      // Leave one thread spinning. This reduces latency.
      if (!spinning_ && !spinning_.exchange(true)) {
        bool nowork = true;
        for (int i = 0; i < 1000; i++) {
          if (!OutOfWork()) {
            nowork = false;
            break;
          }
        }
        spinning_ = false;
        if (!nowork) continue;
      }
      if (!WaitForWork(waiter)) return;
    }
  }

  // Steal tries to steal work from other worker threads in best-effort manner.
  bool Steal(std::vector<Task>* stolen) {
    if (queues_.size() == 1) return false;
    PerThread* pt = GetPerThread();
    unsigned lastq = pt->index;
    for (unsigned i = queues_.size(); i > 0; i--) {
      unsigned victim = Rand(&pt->rand) % queues_.size();
      if (victim == lastq && queues_.size() > 2) {
        i++;
        continue;
      }
      // Steal half of elements from a victim queue.
      // It is typical to steal just one element, but that assumes that work is
      // recursively subdivided in halves so that the stolen element is exactly
      // half of work. If work elements are equally-sized, then is makes sense
      // to steal half of elements at once and then work locally for a while.
      if (queues_[victim]->PopBackHalf(stolen)) return true;
      lastq = victim;
    }
    // Just to make sure that we did not miss anything.
    for (unsigned i = queues_.size(); i > 0; i--)
      if (queues_[i - 1]->PopBackHalf(stolen)) return true;
    return false;
  }

  // WaitForWork blocks until new work is available, or if it is time to exit.
  bool WaitForWork(EventCount::Waiter* waiter) {
    // We already did best-effort emptiness check in Steal, so prepare blocking.
    ec_.Prewait(waiter);
    // Now do reliable emptiness check.
    if (!OutOfWork()) {
      ec_.CancelWait(waiter);
      return true;
    }
    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    blocked_++;
    if (done_ && blocked_ == threads_.size()) {
      ec_.CancelWait(waiter);
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (!OutOfWork()) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        blocked_--;
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(true);
      return false;
    }
    ec_.CommitWait(waiter);
    blocked_--;
    return true;
  }

  bool OutOfWork() {
    for (unsigned i = 0; i < queues_.size(); i++)
      if (!queues_[i]->Empty()) return false;
    return true;
  }

  PerThread* GetPerThread() {
    static thread_local PerThread per_thread_;
    PerThread* pt = &per_thread_;
    if (pt->inited) return pt;
    pt->inited = true;
    pt->rand = std::hash<std::thread::id>()(std::this_thread::get_id());
    return pt;
  }

  static unsigned Rand(unsigned* state) {
    return *state = *state * 1103515245 + 12345;
  }
};

typedef NonBlockingThreadPoolTempl<StlThreadEnvironment> NonBlockingThreadPool;

}  // namespace Eigen

#endif  // EIGEN_CXX11_THREADPOOL_NONBLOCKING_THREAD_POOL_H
