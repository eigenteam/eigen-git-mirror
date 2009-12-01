// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_BENCH_TIMER_H
#define EIGEN_BENCH_TIMER_H

#if defined(_WIN32) || defined(__CYGWIN__)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <numeric>

namespace Eigen
{

/** Elapsed time timer keeping the best try.
  *
  * On POSIX platforms we use clock_gettime with CLOCK_PROCESS_CPUTIME_ID.
  * On Windows we use QueryPerformanceCounter
  *
  * Important: on linux, you must link with -lrt
  */
class BenchTimer
{
public:

  BenchTimer() 
  { 
#if defined(_WIN32) || defined(__CYGWIN__)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    m_frequency = (double)freq.QuadPart;
#endif
    reset(); 
  }

  ~BenchTimer() {}

  inline void reset(void) {m_best = 1e6;}
  inline void start(void) {m_start = getTime();}
  inline void stop(void)
  {
    m_best = std::min(m_best, getTime() - m_start);
  }

  /** Return the best elapsed time in seconds.
    */
  inline double value(void)
  {
    return m_best;
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  inline double getTime(void)
#else
  static inline double getTime(void)
#endif
  {
#ifdef WIN32
    LARGE_INTEGER query_ticks;
    QueryPerformanceCounter(&query_ticks);
    return query_ticks.QuadPart/m_frequency;
#else
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
#endif
  }

protected:
#if defined(_WIN32) || defined(__CYGWIN__)
  double m_frequency;
#endif
  double m_best, m_start;

};

}

#endif // EIGEN_BENCH_TIMER_H
