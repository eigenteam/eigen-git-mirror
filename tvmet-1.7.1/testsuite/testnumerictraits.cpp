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
 * $Id: SelfTest.cc,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#include "main.h"

template<typename T> struct TestNumericTraits
{
  void real()
  {
    T x = someRandom<T>();
    typedef typename NumericTraits<T>::real_type real_type;
    real_type r = NumericTraits<T>::real(x);
    TEST_APPROX(r, x);
  }

  void imag()
  {
    T x = someRandom<T>();
    typedef typename NumericTraits<T>::real_type real_type;
    real_type r = NumericTraits<T>::imag(x);
    TEST_ZERO(r);
  }
  
  void conj()
  {
    T x = someRandom<T>();
    typedef typename NumericTraits<T>::real_type conj_type;
    conj_type r = NumericTraits<T>::conj(x);
    TEST_APPROX(r, x);
  }

  void abs()
  {
    T x = someRandom<T>();
    typedef typename NumericTraits<T>::real_type value_type;
    value_type r1 = NumericTraits<T>::abs(x);
    value_type r2 = NumericTraits<T>::abs(-x);
    TEST_APPROX(r1, r2);
  }
  
  void sqrt()
  {
    T x = someRandom<T>();
    T a = NumericTraits<T>::abs(x);
    T b = NumericTraits<T>::sqrt(a);
    // T could be an integer type, so b*b=a is not necessarily true
    TEST_LESSTHAN(b*b, a);
    TEST_LESSTHAN(a, (b+1)*(b+1));
  }
  
  TestNumericTraits()
  {
    real();
    imag();
    conj();
    abs();
    sqrt();
  }
};

void TvmetTestSuite::testNumericTraits()
{
  TestNumericTraits<int>();
  TestNumericTraits<float>();
  TestNumericTraits<double>();
}
