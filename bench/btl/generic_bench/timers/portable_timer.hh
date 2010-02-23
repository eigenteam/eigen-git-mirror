//=====================================================
// File   :  portable_timer.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)> from boost lib
// Copyright (C) EDF R&D,  lun sep 30 14:23:17 CEST 2002
//=====================================================
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
//  simple_time extracted from the boost library
//
#ifndef _PORTABLE_TIMER_HH
#define _PORTABLE_TIMER_HH

#include <ctime>
#include <cstdlib>

#include <time.h>


#define USEC_IN_SEC 1000000


//  timer  -------------------------------------------------------------------//

//  A timer object measures CPU time.
#ifdef _MSC_VER

#define NOMINMAX
#include <windows.h>

/*#ifndef hr_timer
#include "hr_time.h"
#define hr_timer
#endif*/

 class Portable_Timer
 {
  public:

   typedef struct {
    LARGE_INTEGER start;
    LARGE_INTEGER stop;
   } stopWatch;


   Portable_Timer()
   {
	 startVal.QuadPart = 0;
	 stopVal.QuadPart = 0;
	 QueryPerformanceFrequency(&frequency);
   }

   void start() { QueryPerformanceCounter(&startVal); }

   void stop() { QueryPerformanceCounter(&stopVal); }

   double elapsed() {
	 LARGE_INTEGER time;
     time.QuadPart = stopVal.QuadPart - startVal.QuadPart;
     return LIToSecs(time);
   }

   double user_time() { return elapsed(); }


 private:

   double LIToSecs(LARGE_INTEGER& L) {
     return ((double)L.QuadPart /(double)frequency.QuadPart) ;
   }

   LARGE_INTEGER startVal;
   LARGE_INTEGER stopVal;
   LARGE_INTEGER frequency;


 }; // Portable_Timer

#else

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/times.h>

class Portable_Timer
{
 public:

  Portable_Timer( void )
//   :_utime_sec_start(-1),
// 		_utime_usec_start(-1),
// 		_utime_sec_stop(-1),
// 		_utime_usec_stop(-1)/*,
//         m_prev_cs(-1)*/
  {
  }


  void   start()
  {
//     int status=getrusage(RUSAGE_SELF, &resourcesUsage) ;
//     _utime_sec_start  =  resourcesUsage.ru_utime.tv_sec ;
//     _utime_usec_start =  resourcesUsage.ru_utime.tv_usec ;

    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    m_start_time = double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);

  }

  void stop()
  {
//     int status=getrusage(RUSAGE_SELF, &resourcesUsage) ;
//     _utime_sec_stop  =  resourcesUsage.ru_utime.tv_sec ;
//     _utime_usec_stop =  resourcesUsage.ru_utime.tv_usec ;

    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    m_stop_time = double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);

  }

  double elapsed()
  {
    return  user_time();//double(_stop_time - _start_time) / CLOCKS_PER_SEC;
  }

  double user_time()
  {
//     std::cout << m_prev_cs << "\n";
//     long tot_utime_sec=_utime_sec_stop-_utime_sec_start;
//     long tot_utime_usec=_utime_usec_stop-_utime_usec_start;
//     return double(tot_utime_sec)+ double(tot_utime_usec)/double(USEC_IN_SEC) ;
    return m_stop_time - m_start_time;
  }


private:

//   struct rusage resourcesUsage ;

//   long _utime_sec_start ;
//   long _utime_usec_start ;

//   long _utime_sec_stop ;
//   long _utime_usec_stop ;

//   long m_prev_cs;

//   std::clock_t _start_time;
//   std::clock_t _stop_time;

  double m_stop_time, m_start_time;

}; // Portable_Timer

#endif

#endif  // PORTABLE_TIMER_HPP
