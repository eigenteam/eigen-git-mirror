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
 * $Id: TestUnFunc.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_UNFUNC_H
#define TVMET_TEST_UNFUNC_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

/**
 * generic
 */
template <class T>
class TestUnFunc : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestUnFunc );
  CPPUNIT_TEST( fn_abs );
  CPPUNIT_TEST( Round );
  CPPUNIT_TEST( Arc );
  CPPUNIT_TEST( Log );
  CPPUNIT_TEST( Nan );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  TestUnFunc()
    : vZero(0), vOne(1), vMinusOne(-1), vTwo(2), vE(M_E),
      mZero(0), mOne(1), mMinusOne(-1), mTwo(2), mE(M_E),
      scalar(10)
  { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void fn_abs();
  void Round();
  void Arc();
  void Log();
  void Nan();

private:
  const vector_type vZero;
  const vector_type vOne;
  const vector_type vMinusOne;
  const vector_type vTwo;
  const vector_type vE;
  vector_type v1, v1b;
  vector_type vBig;	/**< vector 10x bigger than v1 */

private:
  const matrix_type mZero;
  const matrix_type mOne;
  const matrix_type mMinusOne;
  const matrix_type mTwo;
  const matrix_type mE;
  matrix_type m1, m1b;
  matrix_type mBig;	/**< matrix 10x bigger than m1 */

private:
  const T scalar;
};


/**
 * specialized for int's (it doesn't support all binary functions, like sqrt(int))
 */
template <>
class TestUnFunc<int> : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestUnFunc<int> );
  CPPUNIT_TEST( fn_abs );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<int, 3>			vector_type;
  typedef tvmet::Matrix<int, 3, 3>		matrix_type;

public:
  TestUnFunc();

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void fn_abs();

private:
  const vector_type vZero, vOne, vMinusOne, vTwo;

private:
  const matrix_type mZero, mOne, mMinusOne, mTwo;
};


/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestUnFunc<T>::setUp() {
  v1 = 1,2,3;
  v1b = v1;		// same as v1, cctor test done in checkInternal
  vBig = 10,20,30;

  m1 = 1,4,7,
       2,5,8,
       3,6,9;
  m1b = m1;		// same as m1, cctor test done in checkInternal
  mBig = 10,40,70,
         20,50,80,
         30,60,90;
}

template <class T>
void TestUnFunc<T>::tearDown() {

}


/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


template <class T>
void
TestUnFunc<T>::fn_abs() {
  vector_type v, tv;
  matrix_type m, tm;
}


template <class T>
void
TestUnFunc<T>::Round() {
  vector_type v, tv;
  matrix_type m, tm;

  // abs
  v = abs(vMinusOne);
  m = abs(mMinusOne);
  CPPUNIT_ASSERT( all_elements(v == vOne) );
  CPPUNIT_ASSERT( all_elements(m == mOne) );

#if 0  // XXX cbrt not in std ?!
  v = cbrt(vOne);
  m = cbrt(mOne);
  CPPUNIT_ASSERT( all_elements(v == std::cbrt(1)) );
  CPPUNIT_ASSERT( all_elements(m == std::cbrt(1)) );
#endif

  // ceil
  tv = vOne + 0.5;
  tm = mOne + 0.5;
  v = ceil(tv);
  m = ceil(tm);
  CPPUNIT_ASSERT( all_elements(v == vTwo) );
  CPPUNIT_ASSERT( all_elements(m == mTwo) );

  // floor
  tv = vTwo - 0.5;
  tm = mTwo - 0.5;
  v = floor(tv);
  m = floor(tm);
  CPPUNIT_ASSERT( all_elements(v == vOne) );
  CPPUNIT_ASSERT( all_elements(m == mOne) );

#if 0  // XXX rint not in std ?!
  tv = vTwo - 0.5;
  tm = mTwo - 0.5;
  v = rint(tv);
  m = rint(tm);
  CPPUNIT_ASSERT( all_elements(v == vOne) );
  CPPUNIT_ASSERT( all_elements(m == mOne) );
#endif
}

template <class T>
void
TestUnFunc<T>::Arc() {
  vector_type v, tv;
  matrix_type m, tm;

  // sin
  tv = M_PI/2.0;
  tm = M_PI/2.0;
  v = sin(tv);
  m = sin(tm);
  CPPUNIT_ASSERT( all_elements(v == vOne) );
  CPPUNIT_ASSERT( all_elements(m == mOne) );

  // cos
  tv = 2.0*M_PI;
  tm = 2.0*M_PI;
  v = cos(tv);
  m = cos(tm);
  CPPUNIT_ASSERT( all_elements(v == vOne) );
  CPPUNIT_ASSERT( all_elements(m == mOne) );

  // tan
  tv = M_PI/4.0;
  tm = M_PI/4.0;
  v = tan(tv);
  m = tan(tm);
  // precision problems, using element wise compare
  CPPUNIT_ASSERT( all_elements(v == tan(M_PI/4.0) ) ); // this failed by OP
  CPPUNIT_ASSERT( all_elements(m == tan(M_PI/4.0) ) ); // this not ...

  // asin
  v = asin(vOne);
  m = asin(mOne);
  // precision problems, using element wise compare
  CPPUNIT_ASSERT( all_elements(v == M_PI/2.0 ) );
  CPPUNIT_ASSERT( all_elements(m == M_PI/2.0 ) );

  // acos
  v = acos(vOne);
  m = acos(mOne);
  CPPUNIT_ASSERT( all_elements(v == 0.0) );
  CPPUNIT_ASSERT( all_elements(m == 0.0) );

  // atan
  v = atan(vOne);
  m = atan(mOne);
  CPPUNIT_ASSERT( all_elements(v == M_PI/4.0) );
  CPPUNIT_ASSERT( all_elements(m == M_PI/4.0) );
}

template <class T>
void
TestUnFunc<T>::Log() {
  vector_type v, tv;
  matrix_type m, tm;

  // exp
  tv = vOne;
  tm = mOne;
  v = exp(tv);
  m = exp(tm);
  CPPUNIT_ASSERT( all_elements(v == vE) );
  CPPUNIT_ASSERT( all_elements(m == mE) );

  // log naturalis
  tv = vE;
  tm = mE;
  v = log(tv);
  m = log(tm);
  CPPUNIT_ASSERT( all_elements(v == vOne) );
  CPPUNIT_ASSERT( all_elements(m == mOne) );

  // log10
  tv = vOne;
  tm = mOne;
  v = log10(tv);
  m = log10(tm);
  // precision problems, using element wise compare
  CPPUNIT_ASSERT( all_elements(v == log10(1.0)) );
  CPPUNIT_ASSERT( all_elements(m == log10(1.0)) );

  // sqrt
  tv = 9;
  tm = 9;
  v = sqrt(tv);
  m = sqrt(tm);
  CPPUNIT_ASSERT( all_elements(v == 3) );
  CPPUNIT_ASSERT( all_elements(m == 3) );
}

template <class T>
void
TestUnFunc<T>::Nan() {
#ifdef HAVE_IEEE_MATH
  vector_type v;
  matrix_type m;

  // isnan
  v = NAN;
  m = NAN;
  CPPUNIT_ASSERT( all_elements(isnan(v)) );
  CPPUNIT_ASSERT( all_elements(isnan(m)) );

  CPPUNIT_ASSERT( all_elements(!isnan(v1)) );
  CPPUNIT_ASSERT( all_elements(!isnan(vBig)) );
  CPPUNIT_ASSERT( all_elements(!isnan(vOne)) );
  CPPUNIT_ASSERT( all_elements(!isnan(vZero)) );
  CPPUNIT_ASSERT( all_elements(!isnan(m1)) );
  CPPUNIT_ASSERT( all_elements(!isnan(mBig)) );
  CPPUNIT_ASSERT( all_elements(!isnan(mOne)) );
  CPPUNIT_ASSERT( all_elements(!isnan(mZero)) );

  // isinf(1)
  v = HUGE_VAL;
  m = HUGE_VAL;
  CPPUNIT_ASSERT( all_elements(isinf(v) > 0) ); 	// == 1
  CPPUNIT_ASSERT( all_elements(isinf(m) > 0) );		// == 1

  v = -HUGE_VAL;
  m = -HUGE_VAL;

  CPPUNIT_ASSERT( all_elements(isinf(v) < 0) ); 	// == -1
  CPPUNIT_ASSERT( all_elements(isinf(m) < 0) );		// == -1

  // isinf(2)
  CPPUNIT_ASSERT( all_elements(!isinf(v1)) );
  CPPUNIT_ASSERT( all_elements(!isinf(vBig)) );
  CPPUNIT_ASSERT( all_elements(!isinf(vOne)) );
  CPPUNIT_ASSERT( all_elements(!isinf(vZero)) );
  CPPUNIT_ASSERT( all_elements(!isinf(m1)) );
  CPPUNIT_ASSERT( all_elements(!isinf(mBig)) );
  CPPUNIT_ASSERT( all_elements(!isinf(mOne)) );
  CPPUNIT_ASSERT( all_elements(!isinf(mZero)) );

#if 0  // XXX finite not in std ?!
  v = NAN;
  m = NAN;
  CPPUNIT_ASSERT( all_elements(finite(v) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(m) != 0) );

  CPPUNIT_ASSERT( all_elements(finite(v1) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(vBig) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(vOne) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(vZero) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(m1) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(mBig) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(mOne) != 0) );
  CPPUNIT_ASSERT( all_elements(finite(mZero) != 0) );

  v = HUGE_VAL;
  m = HUGE_VAL;
  CPPUNIT_ASSERT( all_elements(finite(v) == 0) );
  CPPUNIT_ASSERT( all_elements(finite(m) == 0) );

  v = -HUGE_VAL;
  m = -HUGE_VAL;

  CPPUNIT_ASSERT( all_elements(finite(v) == 0) );
  CPPUNIT_ASSERT( all_elements(finite(m) == 0) );
#endif

#endif // HAVE_IEEE_MATH
}



#endif // TVMET_TEST_UNFUNC_H

// Local Variables:
// mode:C++
// End:
