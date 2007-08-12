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

template<typename T> struct TestTraits
{
  void real_imag_conj_abs()
  {
    T x = Traits<T>::random();
    typedef typename Traits<T>::real_type real;
    real r = Traits<T>::real(x);
    real i = Traits<T>::imag(x);
    T c = Traits<T>::conj(x);
    real a = Traits<T>::abs(x);
    
    // a must be real
    TEST_APPROX(a, Traits<real>::real(a));
    TEST_APPROX(a, Traits<T>::real(a));
    TEST_ZERO(Traits<real>::imag(a));
    TEST_ZERO(Traits<T>::imag(a));
    
    // check Pythagora's formula
    if(Traits<T>::isFloat() || !Traits<T>::isComplex()) TEST_APPROX(r*r + i*i, a*a);
    
    // check complex conjugation
    TEST_APPROX(-i, Traits<T>::imag(c));
  }

  void imag()
  {
    T x = Traits<T>::random();
    typedef typename Traits<T>::real_type real_type;
    real_type r = Traits<T>::imag(x);
    TEST_ZERO(r);
  }
  
  void conj()
  {
    T x = Traits<T>::random();
    typedef typename Traits<T>::real_type conj_type;
    conj_type r = Traits<T>::conj(x);
    TEST_APPROX(r, x);
  }

  void abs()
  {
    T x = Traits<T>::random();
    typedef typename Traits<T>::real_type value_type;
    value_type r1 = Traits<T>::abs(x);
    value_type r2 = Traits<T>::abs(-x);
    TEST_APPROX(r1, r2);
  }
  
  void sqrt()
  {
    // only test compilation here
    T x = Traits<T>::random();
    Traits<T>::sqrt(x);
  }
  
  void isApprox()
  {
    T x = Traits<T>::random();
    T e = T(Traits<T>::epsilon()) / T(10);
    TEST(Traits<T>::isApprox(x,x));
    TEST(Traits<T>::isApprox(x,x+e));
    TEST(!Traits<T>::isApprox(x,x+T(1)));
  }
  
  void isNegligible()
  {
    T zero(0), one(1), x = Traits<T>::random();
    TEST(Traits<T>::isNegligible(zero, zero));
    TEST(Traits<T>::isNegligible(zero, one));
    TEST(!Traits<T>::isNegligible(one, one));
    if(x != zero) {
      TEST(Traits<T>::isNegligible(zero, x));
    }
    TEST(!Traits<T>::isNegligible(one, x));
  }
  
  
  void isZero()
  {
    T zero(0), one(1), x = Traits<T>::random(), y = Traits<T>::random();
    TEST(Traits<T>::isZero(zero));
    TEST(!Traits<T>::isZero(one));
    TEST(Traits<T>::isZero((x+y)*(x+y)-x*x-y*y-x*y-y*x));
  }
  
  void isLessThan()
  {
    if(Traits<T>::isComplex()) {
      T x = Traits<T>::random(), y = Traits<T>::random();
      TEST(!Traits<T>::isLessThan(x,y));
    }
    else {
      T one(1), x = Traits<T>::random();
      T e = T(Traits<T>::epsilon()) / T(10);
      TEST(Traits<T>::isLessThan(x, x+one));
      TEST(!Traits<T>::isLessThan(x+one, x));
      TEST(Traits<T>::isLessThan(x, x+e));
    }
  }
  
  TestTraits()
  {
    real_imag_conj_abs();
    sqrt();
    isApprox();
    isNegligible();
    isZero();
    isLessThan();
  }
};

void TvmetTestSuite::testTraits()
{
  TestTraits<int>();
  TestTraits<float>();
  TestTraits<double>();
  TestTraits<std::complex<int> >();
  TestTraits<std::complex<float> >();
  TestTraits<std::complex<double> >();
}
