/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2007 Benoit Jacob <jacob@math.jussieu.fr>
 *
 * Based on Tvmet source code, http://tvmet.sourceforge.net,
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
 * $Id: Traits.h,v 1.11 2004/11/04 18:10:35 opetzold Exp $
 */

#ifndef TVMET_NUMERIC_TRAITS_H
#define TVMET_NUMERIC_TRAITS_H

#if defined(EIGEN_USE_COMPLEX)
#  include <complex>
#endif

#include <cmath>

#include <tvmet/TraitsBase.h>

namespace tvmet {

/**
 * \class Traits Traits.h "tvmet/Traits.h"
 * \brief Traits for standard types.
 *
 */
template<typename T>
struct Traits : public TraitsBase<T>
{
  typedef TraitsBase<T>                 Base;
  typedef typename Base::value_type     value_type;
  typedef typename Base::real_type      real_type;
  typedef typename Base::float_type     float_type;
  typedef typename Base::argument_type  argument_type;
  
  using Base::isFloat;
  using Base::isComplex;
  using Base::epsilon;
  using Base::abs;
  using Base::real;
  using Base::imag;
  using Base::conj;
  using Base::sqrt;
  
  static value_type random()
  {
    value_type x;
    pickRandom(x);
    return x;
  }
  /**
    * Short version: returns true if the absolute value of \a a is much smaller
    * than that of \a b.
    *
    * Full story: returns(abs(a) <= abs(b) * epsilon());
    */
  static bool isNegligible(argument_type a, argument_type b)
  {
    if(isFloat())
      return(abs(a) <= abs(b) * epsilon());
    else
      return(a==0);
  }
  
  /**
    * Short version: returns true if \a a is approximately zero.
    *
    * Full story: returns isNegligible( a, static_cast<T>(1) );
    */
  static bool isZero(argument_type a)
  {
    return isNegligible(a, static_cast<value_type>(1));
  }
  
  /**
    * Short version: returns true if a is very close to b, false otherwise.
    *
    * Full story: returns abs( a - b ) <= min( abs(a), abs(b) ) * epsilon<T>.
    */
  static bool isApprox(argument_type a, argument_type b)
  {
    if(isFloat())
      return(abs( a - b ) <= std::min(abs(a), abs(b)) * epsilon());
    else
      return(a==b);
  }
  
  /**
    * Short version: returns true if a is smaller or approximately equalt to b, false otherwise.
    *
    * Full story: returns a <= b || isApprox(a, b);
    */
  static bool isLessThan( argument_type a, argument_type b )
  {
    assert(!isComplex());
    if(isFloat())
      return(a <= b || isApprox(a, b));
    else
      return(a<=b);
  }

};


} // namespace tvmet

#endif //  TVMET_NUMERIC_TRAITS_H
