//=====================================================
// File   :  portable_perf_analyzer.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  mar d�c 3 18:59:35 CET 2002
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
#ifndef _PORTABLE_PERF_ANALYZER_HH
#define _PORTABLE_PERF_ANALYZER_HH

#include "utilities.h"
#include "timers/portable_timer.hh"

template <class Action>
class Portable_Perf_Analyzer{
public:
  Portable_Perf_Analyzer( void ):_nb_calc(1),_chronos(){
    MESSAGE("Portable_Perf_Analyzer Ctor");
  };
  Portable_Perf_Analyzer( const Portable_Perf_Analyzer & ){
    INFOS("Copy Ctor not implemented");
    exit(0);
  };
  ~Portable_Perf_Analyzer( void ){
    MESSAGE("Portable_Perf_Analyzer Dtor");
  };

  BTL_DONT_INLINE  double eval_mflops(int size)
  {
    Action action(size);

    double time_action = 0;
    action.initialize();
    time_action = time_calculate(action);
    while (time_action < MIN_TIME)
    {
      //Action action(size);
      _nb_calc *= 2;
      action.initialize();
      time_action = time_calculate(action);
    }

    // optimize
    for (int i=1; i<NB_TRIES; ++i)
    {
      Action _action(size);
      std::cout << " " << _action.nb_op_base()*_nb_calc/(time_action*1e6) << " ";
      _action.initialize();
      time_action = std::min(time_action, time_calculate(_action));
    }

    time_action = time_action / (double(_nb_calc));

    // check
    if (BtlConfig::Instance.checkResults)
    {
      action.initialize();
      action.calculate();
      action.check_result();
    }
    return action.nb_op_base()/(time_action*1000000.0);
  }

  BTL_DONT_INLINE double time_calculate(Action & action)
  {
    // time measurement
    action.calculate();
    _chronos.start();
    for (int ii=0;ii<_nb_calc;ii++)
    {
      action.calculate();
    }
    _chronos.stop();
    return _chronos.user_time();
  }

  unsigned long long get_nb_calc( void )
  {
    return _nb_calc;
  }


private:
  unsigned long long _nb_calc;
  Portable_Timer _chronos;

};

#endif //_PORTABLE_PERF_ANALYZER_HH

