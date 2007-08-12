/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
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
 
/* This file is mostly a duplication of util/Fuzzy.h, but with 1000 times
 * bigger epsilon. This prevents false positives in tests.
 */

#ifndef EIGEN_TESTSUITE_COMPARE_H
#define EIGEN_TESTSUITE_COMPARE_H

#include <QtTest/QtTest>

#include <cstdlib>
#include <cmath>
#include <complex>

#include <tvmet/Traits.h>

template<typename T> inline typename tvmet::Traits<T>::real_type test_epsilon()
{ return static_cast<typename tvmet::Traits<T>::real_type>(0); }
template<> inline float test_epsilon<float>() { return 1e-2f; }
template<> inline double test_epsilon<double>() { return 1e-8; }
template<> inline float test_epsilon<std::complex<float> >() { return test_epsilon<float>(); }
template<> inline double test_epsilon<std::complex<double> >() { return test_epsilon<double>(); }

/**
  * Short version: returns true if the absolute value of \a a is much smaller
  * than that of \a b.
  *
  * Full story: returns ( abs(a) <= abs(b) * test_epsilon<T> ).
  */
template<typename T> bool test_isNegligible( const T& a, const T& b )
{
  return( tvmet::Traits<T>::abs(a)
            <= tvmet::Traits<T>::abs(b)
             * test_epsilon<T>() );
}

/**
  * Short version: returns true if \a a is approximately zero.
  *
  * Full story: returns test_isNegligible( a, static_cast<T>(1) );
  */
template<typename T> bool test_isZero( const T& a )
{
  return test_isNegligible( a, static_cast<T>(1) );
}

/**
  * Short version: returns true if a is very close to b, false otherwise.
  *
  * Full story: returns abs( a - b ) <= min( abs(a), abs(b) ) * test_epsilon<T>.
  */
template<typename T> bool test_isApprox( const T& a, const T& b )
{
  return( tvmet::Traits<T>::abs( a - b )
          <= std::min( tvmet::Traits<T>::abs(a),
                       tvmet::Traits<T>::abs(b) ) * test_epsilon<T>() );
}

/**
  * Short version: returns true if a is smaller or approximately equalt to b, false otherwise.
  *
  * Full story: returns a <= b || test_isApprox(a, b);
  */
template<typename T> bool test_isLessThan( const T& a, const T& b )
{
  return( tvmet::Traits<T>::isLessThan_nonfuzzy(a, b) || test_isApprox(a, b) );
}

#define TEST(a)              QVERIFY(a)
#define TEST_NEGLIGIBLE(a,b) QVERIFY(test_isNegligible(a,b))
#define TEST_ZERO(a)         QVERIFY(test_isZero(a))
#define TEST_APPROX(a,b)     QVERIFY(test_isApprox(a,b))
#define TEST_LESSTHAN(a,b)   QVERIFY(test_isLessThan(a,b))

#endif // EIGEN_TESTSUITE_COMPARE_H
