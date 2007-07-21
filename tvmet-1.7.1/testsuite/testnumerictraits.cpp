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

template<typename T, int n> struct TestNumericTraits
{
  const T m_real;
  const T m_imag;
  const T m_conj;
  const T m_abs_Q1;
  
  void real()
  {
    typedef typename tvmet::NumericTraits<T>::base_type real_type;
    real_type r = tvmet::NumericTraits<T>::real(m_real);
    QVERIFY( r == m_real );
  }

  void imag()
  {
    typedef typename tvmet::NumericTraits<T>::base_type real_type;
    real_type r = tvmet::NumericTraits<T>::real(m_real);
    QVERIFY( r == m_real );
  }
  
  void conj()
  {
    typedef typename tvmet::NumericTraits<T>::base_type conj_type;
    conj_type r = tvmet::NumericTraits<T>::conj(m_conj);
    QVERIFY( r == m_conj );
  }

  void abs()
  {
    typedef typename tvmet::NumericTraits<T>::base_type value_type;
    value_type r1 = tvmet::NumericTraits<T>::abs(m_abs_Q1);
    value_type r2 = tvmet::NumericTraits<T>::abs(-m_abs_Q1);
    QVERIFY( r1 == m_abs_Q1 );
    QVERIFY( r2 == m_abs_Q1 );
  }
  
  void sqrt()
  {
    typedef typename tvmet::NumericTraits<T>::base_type value_type;
    value_type r1 = tvmet::NumericTraits<T>::sqrt(m_real);
    value_type r2 = tvmet::NumericTraits<T>::sqrt(m_imag);
    QVERIFY( r1 == 2 );
    QVERIFY( r2 == 3 );
  }
  
  void norm1()
  {
    typedef typename tvmet::NumericTraits<T>::base_type value_type;
    value_type r = tvmet::NumericTraits<T>::norm_1(m_real);
    QVERIFY( r == tvmet::NumericTraits<T>::abs(m_real) );
  }
  
  void norm2()
  {
    typedef typename tvmet::NumericTraits<T>::base_type value_type;
    value_type r = tvmet::NumericTraits<T>::norm_2(m_real);
    QVERIFY( r == tvmet::NumericTraits<T>::abs(m_real) );
  }
  
  void normInf()
  {
    typedef typename tvmet::NumericTraits<T>::base_type value_type;
    value_type r = tvmet::NumericTraits<T>::norm_inf(m_real);
    QVERIFY( r == tvmet::NumericTraits<T>::abs(m_real) );
  }
  
  void equals()
  {
    typedef typename tvmet::NumericTraits<T>::base_type value_type;
    value_type lhs, rhs;
    lhs = rhs = 47;
    QVERIFY( true == tvmet::NumericTraits<T>::equals(lhs,rhs) );
    // a not very intelligent test
    rhs += 1;
    QVERIFY( false == tvmet::NumericTraits<T>::equals(lhs,rhs) );
  }

  
  TestNumericTraits() : m_real(4), m_imag(9), m_conj(16), m_abs_Q1(7)
  {
    real();
    imag();
    conj();
    abs();
    sqrt();
    norm1();
    norm2();
    normInf();
    equals();
  }
};

void TvmetTestSuite::testNumericTraits()
{
  TestNumericTraits<double,1>();
  TestNumericTraits<int,   2>();
  TestNumericTraits<float, 3>();
  TestNumericTraits<double,4>();
}
