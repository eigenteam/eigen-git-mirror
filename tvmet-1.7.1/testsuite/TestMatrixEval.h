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
 * $Id: TestMatrixEval.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_MATRIX_EVAL_H
#define TVMET_TEST_MATRIX_EVAL_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Matrix.h>

#include <cassert>

template <class T>
class TestMatrixEval : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestMatrixEval );
  CPPUNIT_TEST( Greater );
  CPPUNIT_TEST( Less );
  CPPUNIT_TEST( GreaterOrEqual );
  CPPUNIT_TEST( LessOrEqual );
  CPPUNIT_TEST( Equal );
  CPPUNIT_TEST( NotEqual );
  CPPUNIT_TEST( LogicalAnd );
  CPPUNIT_TEST( LogicalOr );

  // others
  CPPUNIT_TEST( AllElements );
  CPPUNIT_TEST( AnyElements );
  CPPUNIT_TEST( Eval3 );
  CPPUNIT_TEST( EvalPod3 );

  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  TestMatrixEval()
    : mOne(1), mZero(0) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void Greater();
  void Less();
  void GreaterOrEqual();
  void LessOrEqual();
  void Equal();
  void NotEqual();
  void LogicalAnd();
  void LogicalOr();

  void AllElements();
  void AnyElements();
  void Eval3();
  void EvalPod3();

private:
  matrix_type m1;
  matrix_type mBig;	/**< matrix bigger than m1 */
  const matrix_type mOne;
  const matrix_type mZero;
};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestMatrixEval<T>::setUp () {
  m1 = 1,4,7,
       2,5,8,
       3,6,9;
  mBig = 10,40,70,
         20,50,80,
         30,60,90;
}

template <class T>
void TestMatrixEval<T>::tearDown() { }

/*****************************************************************************
 * Implementation Part II
 * these are elemental - therefore we use std::assert
 ****************************************************************************/

/*
 * on SelfTest, we have the guarantee, that the container holds the
 * expected values. Now check comparing operation using tvmet's
 * eval function. This is the basic for all further test since it's
 * the way we check the correctness. The other way would be element wise
 * compare as in SelfTest, urgh...
 */
template <class T>
void
TestMatrixEval<T>::Greater() {
  // all test are element wise !
  assert( all_elements(mBig >  m1) );
}

template <class T>
void
TestMatrixEval<T>::Less() {
  // all test are element wise !
  assert( all_elements(m1   <  mBig) );
}

template <class T>
void
TestMatrixEval<T>::GreaterOrEqual() {
  // all test are element wise !
  assert( all_elements(mBig >= m1) );
  assert( all_elements(m1   >= m1) );
  assert( all_elements(mBig >= mBig) );
  assert( all_elements(mOne >= T(1)) );
  assert( all_elements(mZero>= T(0)) );
}

template <class T>
void
TestMatrixEval<T>::LessOrEqual() {
  // all test are element wise !
  assert( all_elements(m1   <= mBig) );
  assert( all_elements(m1   <= m1) );
  assert( all_elements(mBig <= mBig) );
  assert( all_elements(mOne <= T(1)) );
  assert( all_elements(mZero<= T(0)) );
}

template <class T>
void
TestMatrixEval<T>::Equal() {
  // all test are element wise !
  assert( all_elements(m1   == m1) );
  assert( all_elements(mBig == mBig) );
  assert( all_elements(mOne == T(1)) );
  assert( all_elements(mZero == T(0)) );
}

template <class T>
void
TestMatrixEval<T>::NotEqual() {
  // all test are element wise !
  assert( all_elements(m1   != mBig) );
}

template <class T>
void
TestMatrixEval<T>::LogicalAnd() {
  // TODO: implement
}

template <class T>
void
TestMatrixEval<T>::LogicalOr() {
  // TODO: implement
}

/*****************************************************************************
 * Implementation Part III
 * test on generell and eval functions
 ****************************************************************************/

template <class T>
void
TestMatrixEval<T>::AllElements() {
  // true cases
  CPPUNIT_ASSERT( all_elements(mBig > T(0)) );
  CPPUNIT_ASSERT( all_elements(mBig >= T(1)) );

  CPPUNIT_ASSERT( all_elements(mBig < T(1000)) );
  CPPUNIT_ASSERT( all_elements(mBig <= T(1000)) );

  CPPUNIT_ASSERT( all_elements(T(0) < mBig) );		// possible, I newer would write it
  CPPUNIT_ASSERT( all_elements(T(1000) > mBig) );	// possible, I newer would write it

  CPPUNIT_ASSERT( all_elements(mOne == T(1)) );
  CPPUNIT_ASSERT( all_elements(mZero == T(0)) );

  CPPUNIT_ASSERT( all_elements(mBig != T(1000)) );

  // false cases
  CPPUNIT_ASSERT( !all_elements(mBig < T(0)) );
}


template <class T>
void
TestMatrixEval<T>::AnyElements() {
  // true cases
  CPPUNIT_ASSERT( any_elements(mBig > T(0)) );
  CPPUNIT_ASSERT( any_elements(mBig >= T(1)) );

  CPPUNIT_ASSERT( any_elements(mBig < T(1000)) );
  CPPUNIT_ASSERT( any_elements(mBig <= T(1000)) );

  CPPUNIT_ASSERT( any_elements(T(2) < m1) );	// possible, I newer would write it
  CPPUNIT_ASSERT( any_elements(T(2) > m1) );	// possible, I newer would write it

  CPPUNIT_ASSERT( any_elements(mOne == T(1)) );
  CPPUNIT_ASSERT( any_elements(mZero == T(0)) );

  CPPUNIT_ASSERT( any_elements(mBig != T(1000)) );

  // false cases
  CPPUNIT_ASSERT( !any_elements(mBig < T(2)) );
  CPPUNIT_ASSERT( !any_elements(mOne == T(0)) );
  CPPUNIT_ASSERT( !any_elements(mZero == T(1)) );
}


template <class T>
void
TestMatrixEval<T>::Eval3() {
  matrix_type v;
  T a(1);	// scalar

  // XprMatrix<E1, Rows, Cols> ? Matrix<T2, Rows, Cols> : Matrix<T3, Rows, Cols>
  v = eval( m1 < mBig, m1, mBig);
  CPPUNIT_ASSERT( all_elements(v == m1) );

  v = eval( m1 > mBig, m1, mBig);
  CPPUNIT_ASSERT( all_elements(v == mBig) );

  // XprMatrix<E1, Rows, Cols> ? Matrix<T2, Rows, Cols> : XprMatrix<E3, Rows, Cols>
  v = eval( m1 < mBig, m1, a*mBig);
  CPPUNIT_ASSERT( all_elements(v == m1) );

  v = eval( m1 > mBig, m1, a*mBig);
  CPPUNIT_ASSERT( all_elements(v == mBig) );

  // XprMatrix<E1, Rows, Cols> ? XprMatrix<E2, Rows, Cols> : Matrix<T3, Rows, Cols>
  v = eval( m1 < mBig, a*m1, mBig);
  CPPUNIT_ASSERT( all_elements(v == m1) );

  v = eval( m1 > mBig, a*m1, mBig);
  CPPUNIT_ASSERT( all_elements(v == mBig) );

  // XprMatrix<E1, Rows, Cols> ? XprMatrix<E2, Rows, Cols> : XprMatrix<E3, Rows, Cols>
  v = eval( m1 < mBig, a*m1, a*mBig);
  CPPUNIT_ASSERT( all_elements(v == m1) );

  v = eval( m1 > mBig, a*m1, a*mBig);
  CPPUNIT_ASSERT( all_elements(v == mBig) );
}


template <class T>
void
TestMatrixEval<T>::EvalPod3() {
  matrix_type v;
  T a(1);	// scalar

  // XprMatrix<E, Rows, Cols> ? POD1 : POD2
  v = eval( m1 < mBig, T(0), T(1));
  CPPUNIT_ASSERT( all_elements(v == T(0)) );

  v = eval( m1 > mBig, T(0), T(1));
  CPPUNIT_ASSERT( all_elements(v == T(1)) );

  // XprMatrix<E1, Rows, Cols> ? POD : XprMatrix<E3, Rows, Cols>
  v = eval( m1 < mBig, 1, a*mBig);
  CPPUNIT_ASSERT( all_elements(v == mOne) );

  v = eval( m1 > mBig, 1, a*mBig);
  CPPUNIT_ASSERT( all_elements(v == mBig) );

  // XprMatrix<E1, Rows, Cols> ? XprMatrix<E2, Rows, Cols> : POD
  v = eval( m1 < mBig, a*m1, T(1));
  CPPUNIT_ASSERT( all_elements(v == m1) );

  v = eval( m1 > mBig, a*m1, T(1));
  CPPUNIT_ASSERT( all_elements(v == mOne) );

}


#endif // TVMET_TEST_MATRIX_EVAL_H

// Local Variables:
// mode:C++
// End:
