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

#ifndef EIGEN_BENCH_TIMERR_H
#define EIGEN_BENCH_TIMERR_H

#if defined(_WIN32) || defined(__CYGWIN__)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#endif

#include <cmath>
#include <cstdlib>
#include <numeric>

namespace Eigen
{

enum {
  CPU_TIMER = 0,
  REAL_TIMER = 1
};

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

  inline void reset()
  {
    m_bests.fill(1e9);
    m_totals.setZero();
  }
  inline void start()
  {
    m_starts[CPU_TIMER]  = getCpuTime();
    m_starts[REAL_TIMER] = getRealTime();
  }
  inline void stop()
  {
    m_times[CPU_TIMER] = getCpuTime() - m_starts[CPU_TIMER];
    m_times[REAL_TIMER] = getRealTime() - m_starts[REAL_TIMER];
    m_bests = m_bests.cwiseMin(m_times);
    m_totals += m_times;
  }

  /** Return the elapsed time in seconds between the last start/stop pair
    */
  inline double value(int TIMER = CPU_TIMER)
  {
    return m_times[TIMER];
  }

  /** Return the best elapsed time in seconds
    */
  inline double best(int TIMER = CPU_TIMER)
  {
    return m_bests[TIMER];
  }

  /** Return the total elapsed time in seconds.
    */
  inline double total(int TIMER = CPU_TIMER)
  {
    return m_totals[TIMER];
  }

  inline double getCpuTime()
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

  inline double getRealTime()
  {
#ifdef WIN32
	SYSTEMTIME st;
	GetSystemTime(&st);
	return (double)st.wSecond + 1.e-6 * (double)st.wMilliseconds;
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (double)tv.tv_sec + 1.e-6 * (double)tv.tv_usec;
#endif
  }

protected:
#if defined(_WIN32) || defined(__CYGWIN__)
  double m_frequency;
#endif
  Vector2d m_starts;
  Vector2d m_times;
  Vector2d m_bests;
  Vector2d m_totals;

};

#define BENCH(TIMER,TRIES,REP,CODE) { \
    TIMER.reset(); \
    for(int uglyvarname1=0; uglyvarname1<TRIES; ++uglyvarname1){ \
      TIMER.start(); \
      for(int uglyvarname2=0; uglyvarname2<REP; ++uglyvarname2){ \
        CODE; \
      } \
      TIMER.stop(); \
    } \
  }

}

#endif // EIGEN_BENCH_TIMERR_H
