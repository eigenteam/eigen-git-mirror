// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_THREADPOOL_THREAD_POOL_INTERFACE_H
#define EIGEN_CXX11_THREADPOOL_THREAD_POOL_INTERFACE_H

namespace Eigen {

// This defines an interface that ThreadPoolDevice can take to use
// custom thread pools underneath.
class ThreadPoolInterface {
 public:
  virtual void Schedule(std::function<void()> fn) = 0;

  virtual ~ThreadPoolInterface() {}
};

}  // namespace Eigen

#endif  // EIGEN_CXX11_THREADPOOL_THREAD_POOL_INTERFACE_H
