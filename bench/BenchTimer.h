// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#else
#define NOMINMAX
#include <windows.h>
#endif

#include <cstdlib>
#include <numeric>

namespace Eigen
{

/** Elapsed time timer keeping the best try.
  */
class BenchTimer
{
public:

  BenchTimer() 
  { 
#ifdef WIN32
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

#ifdef WIN32
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
      struct timeval tv;
      struct timezone tz;
      gettimeofday(&tv, &tz);
      return (double)tv.tv_sec + 1.e-6 * (double)tv.tv_usec;
#endif
  }

protected:
#ifdef WIN32
  double m_frequency;
#endif
  double m_best, m_start;

};

}

#endif // EIGEN_BENCH_TIMER_H
