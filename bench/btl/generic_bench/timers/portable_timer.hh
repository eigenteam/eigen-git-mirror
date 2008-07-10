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
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/times.h>


#define USEC_IN_SEC 1000000


//  timer  -------------------------------------------------------------------//

//  A timer object measures CPU time.

class Portable_Timer
{
 public:

  Portable_Timer( void ):_utime_sec_start(-1),
		_utime_usec_start(-1),
		_utime_sec_stop(-1),
		_utime_usec_stop(-1)
  {
  }


  void   start()
  {

    int status=getrusage(RUSAGE_SELF, &resourcesUsage) ;

    _start_time = std::clock();

    _utime_sec_start  =  resourcesUsage.ru_utime.tv_sec ;
    _utime_usec_start =  resourcesUsage.ru_utime.tv_usec ;

  }

  void stop()
  {

    int status=getrusage(RUSAGE_SELF, &resourcesUsage) ;

    _stop_time = std::clock();

    _utime_sec_stop  =  resourcesUsage.ru_utime.tv_sec ;
    _utime_usec_stop =  resourcesUsage.ru_utime.tv_usec ;

  }

  double elapsed()
  {
    return  double(_stop_time - _start_time) / CLOCKS_PER_SEC;
  }

  double user_time()
  {
    long tot_utime_sec=_utime_sec_stop-_utime_sec_start;
    long tot_utime_usec=_utime_usec_stop-_utime_usec_start;
    return double(tot_utime_sec)+ double(tot_utime_usec)/double(USEC_IN_SEC) ;
  }


private:

  struct rusage resourcesUsage ;

  long _utime_sec_start ;
  long _utime_usec_start ;

  long _utime_sec_stop ;
  long _utime_usec_stop ;

  std::clock_t _start_time;
  std::clock_t _stop_time;

}; // Portable_Timer


#endif  // PORTABLE_TIMER_HPP
