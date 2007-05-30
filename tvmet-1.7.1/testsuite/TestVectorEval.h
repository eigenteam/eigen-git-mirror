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
 * $Id: TestVectorEval.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_VECTOR_EVAL_H
#define TVMET_TEST_VECTOR_EVAL_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>

#include <cassert>

template <class T>
class TestVectorEval : public CppUnit::TestFixture
{
  // basic tests
  CPPUNIT_TEST_SUITE( TestVectorEval );
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
  typedef tvmet::Vector<T, 3>			vector_type;

public:
  TestVectorEval()
    : vOne(1), vZero(0) { }

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
  vector_type v1;
  vector_type vBig;	/**< vector bigger than v1 */
  const vector_type vOne;
  const vector_type vZero;

};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestVectorEval<T>::setUp () {
  v1 = 1,2,3;
  vBig = 10,20,30;
}

template <class T>
void TestVectorEval<T>::tearDown() { }

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
TestVectorEval<T>::Greater() {
  // all test are element wise !
  assert( all_elements(vBig >  v1) );
}

template <class T>
void
TestVectorEval<T>::Less() {
  // all test are element wise !
  assert( all_elements(v1   <  vBig) );
}

template <class T>
void
TestVectorEval<T>::GreaterOrEqual() {
  // all test are element wise !
  assert( all_elements(vBig >= v1) );
  assert( all_elements(v1   >= v1) );
  assert( all_elements(vBig >= vBig) );
  assert( all_elements(vOne >= T(1)) );
  assert( all_elements(vZero>= T(0)) );
}

template <class T>
void
TestVectorEval<T>::LessOrEqual() {
  // all test are element wise !
  assert( all_elements(v1   <= vBig) );
  assert( all_elements(v1   <= v1) );
  assert( all_elements(vBig <= vBig) );
  assert( all_elements(vOne <= T(1)) );
  assert( all_elements(vZero<= T(0)) );
}

template <class T>
void
TestVectorEval<T>::Equal() {
  // all test are element wise !
  assert( all_elements(v1   == v1) );
  assert( all_elements(vBig == vBig) );
  assert( all_elements(vOne == 1) );
  assert( all_elements(vZero == T(0)) );
}

template <class T>
void
TestVectorEval<T>::NotEqual() {
  // all test are element wise !
  assert( all_elements(v1   != vBig) );
}

template <class T>
void
TestVectorEval<T>::LogicalAnd() {
  // TODO: implement
}

template <class T>
void
TestVectorEval<T>::LogicalOr() {
  // TODO: implement
}

/*****************************************************************************
 * Implementation Part III
 * test on generell and eval functions
 ****************************************************************************/

template <class T>
void
TestVectorEval<T>::AllElements() {
  // true cases
  CPPUNIT_ASSERT( all_elements(vBig > T(0)) );
  CPPUNIT_ASSERT( all_elements(vBig >= T(1)) );

  CPPUNIT_ASSERT( all_elements(vBig < T(1000)) );
  CPPUNIT_ASSERT( all_elements(vBig <= T(1000)) );

  CPPUNIT_ASSERT( all_elements(T(0) < vBig) );		// possible, I newer would write it
  CPPUNIT_ASSERT( all_elements(T(1000) > vBig) );	// possible, I newer would write it

  CPPUNIT_ASSERT( all_elements(vOne == T(1)) );
  CPPUNIT_ASSERT( all_elements(vZero == T(0)) );

  CPPUNIT_ASSERT( all_elements(vBig != T(1000)) );

  // false cases
  CPPUNIT_ASSERT( !all_elements(vBig < T(0)) );
}


template <class T>
void
TestVectorEval<T>::AnyElements() {
  // true cases
  CPPUNIT_ASSERT( any_elements(vBig > T(0)) );
  CPPUNIT_ASSERT( any_elements(vBig >= T(1)) );

  CPPUNIT_ASSERT( any_elements(vBig < T(1000)) );
  CPPUNIT_ASSERT( any_elements(vBig <= T(1000)) );

  CPPUNIT_ASSERT( any_elements(T(2) < v1) );	// possible, I newer would write it
  CPPUNIT_ASSERT( any_elements(T(2) > v1) );	// possible, I newer would write it

  CPPUNIT_ASSERT( any_elements(vOne == T(1)) );
  CPPUNIT_ASSERT( any_elements(vZero == T(0)) );

  CPPUNIT_ASSERT( any_elements(vBig != T(1000)) );

  // false cases
  CPPUNIT_ASSERT( !any_elements(vBig < T(2)) );
  CPPUNIT_ASSERT( !any_elements(vOne == T(0)) );
  CPPUNIT_ASSERT( !any_elements(vZero == T(1)) );
}


template <class T>
void
TestVectorEval<T>::Eval3() {
  vector_type v;
  T a(1);	// scalar

  // XprVector<E1, Sz> ? Vector<T2, Sz> : Vector<T3, Sz>
  v = eval( v1 < vBig, v1, vBig);
  CPPUNIT_ASSERT( all_elements(v == v1) );

  v = eval( v1 > vBig, v1, vBig);
  CPPUNIT_ASSERT( all_elements(v == vBig) );

  // XprVector<E1, Sz> ? Vector<T2, Sz> : XprVector<E3, Sz>
  v = eval( v1 < vBig, v1, a*vBig);
  CPPUNIT_ASSERT( all_elements(v == v1) );

  v = eval( v1 > vBig, v1, a*vBig);
  CPPUNIT_ASSERT( all_elements(v == vBig) );

  // XprVector<E1, Sz> ? XprVector<E2, Sz> : Vector<T3, Sz>
  v = eval( v1 < vBig, a*v1, vBig);
  CPPUNIT_ASSERT( all_elements(v == v1) );

  v = eval( v1 > vBig, a*v1, vBig);
  CPPUNIT_ASSERT( all_elements(v == vBig) );

  // XprVector<E1, Sz> ? XprVector<E2, Sz> : XprVector<E3, Sz>
  v = eval( v1 < vBig, a*v1, a*vBig);
  CPPUNIT_ASSERT( all_elements(v == v1) );

  v = eval( v1 > vBig, a*v1, a*vBig);
  CPPUNIT_ASSERT( all_elements(v == vBig) );
}


template <class T>
void
TestVectorEval<T>::EvalPod3() {
  vector_type v;
  T a(1);	// scalar

  // XprVector<E, Sz> ? POD1 : POD2
  v = eval( v1 < vBig, T(0), T(1));
  CPPUNIT_ASSERT( all_elements(v == T(0)) );

  v = eval( v1 > vBig, T(0), T(1));
  CPPUNIT_ASSERT( all_elements(v == T(1)) );

  // XprVector<E1, Sz> ? POD : XprVector<E3, Sz>
  v = eval( v1 < vBig, 1, a*vBig);
  CPPUNIT_ASSERT( all_elements(v == vOne) );

  v = eval( v1 > vBig, 1, a*vBig);
  CPPUNIT_ASSERT( all_elements(v == vBig) );

  // XprVector<E1, Sz> ? XprVector<E2, Sz> : POD
  v = eval( v1 < vBig, a*v1, T(1));
  CPPUNIT_ASSERT( all_elements(v == v1) );

  v = eval( v1 > vBig, a*v1, T(1));
  CPPUNIT_ASSERT( all_elements(v == vOne) );
}


#endif // TVMET_TEST_VECTOR_EVAL_H

// Local Variables:
// mode:C++
// End:
