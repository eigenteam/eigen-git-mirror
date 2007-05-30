/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: Timer.h,v 1.5 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_UTIL_TIMER_H
#define TVMET_UTIL_TIMER_H

#if defined(TVMET_HAVE_SYS_TIME_H) && defined(TVMET_HAVE_UNISTD_H)
#  include <sys/time.h>
#  include <sys/resource.h>
#  include <unistd.h>
#else
#  include <ctime>
#endif

namespace tvmet {

namespace util {

/**
   \class Timer Timer.h "tvmet/util/Timer.h"
   \brief A quick& dirty portable timer, measures elapsed time.

   It is recommended that implementations measure wall clock rather than CPU
   time since the intended use is performance measurement on systems where
   total elapsed time is more important than just process or CPU time.

   The accuracy of timings depends on the accuracy of timing information
   provided by the underlying platform, and this varies from platform to
   platform.
*/

class Timer
{
  Timer(const Timer&);
  Timer& operator=(const Timer&);

public: // types
  typedef double					time_t;

public:
  /** starts the timer immediatly. */
  Timer() { m_start_time = getTime(); }

  /** restarts the timer */
  void restart() { m_start_time = getTime(); }

  /** return elapsed time in seconds */
  time_t elapsed() const { return (getTime() - m_start_time); }

private:
  time_t getTime() const {
#if defined(TVMET_HAVE_SYS_TIME_H) && defined(TVMET_HAVE_UNISTD_H)
    getrusage(RUSAGE_SELF, &m_rusage);
    time_t sec = m_rusage.ru_utime.tv_sec; // user, no system time
    time_t usec  = m_rusage.ru_utime.tv_usec; // user, no system time
    return sec + usec/1e6;
#else
    return static_cast<time_t>(std::clock()) / static_cast<time_t>(CLOCKS_PER_SEC);
#endif
  }

private:
#if defined(TVMET_HAVE_SYS_TIME_H) && defined(TVMET_HAVE_UNISTD_H)
  mutable struct rusage 				m_rusage;
#endif
  time_t 						m_start_time;
};

} // namespace util

} // namespace tvmet

#endif // TVMET_UTIL_TIMER_H

// Local Variables:
// mode:C++
// End:
