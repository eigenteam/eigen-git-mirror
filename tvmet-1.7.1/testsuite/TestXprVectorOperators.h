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
 * $Id: TestXprVectorOperators.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_XPR_VECTOROPS_H
#define TVMET_TEST_XPR_VECTOROPS_H

#include <limits>

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/util/General.h>

template <class T>
class TestXprVectorOperators : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestXprVectorOperators );
  CPPUNIT_TEST( scalarOps1 );
  CPPUNIT_TEST( scalarOps2 );
  CPPUNIT_TEST( globalXprVectorOps );
  CPPUNIT_TEST( negate );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;

public:
  TestXprVectorOperators()
    : vZero(0), vOne(1), scalar(10), scalar2(2) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void scalarOps1();
  void scalarOps2();
  void globalXprVectorOps();
  void negate();

private:
  const vector_type vZero;
  const vector_type vOne;
  vector_type v1;
  vector_type vBig;	/**< vector 10x bigger than v1 */

private:
  const T scalar;
  const T scalar2;
};


/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/


template <class T>
void TestXprVectorOperators<T>::setUp() {
  v1 = 1,2,3;
  vBig = 10,20,30;
}


template <class T>
void TestXprVectorOperators<T>::tearDown() { }


/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


/*
 * global math operators with scalars
 * Note: checked against member operators since they are allready checked
 */
template <class T>
void
TestXprVectorOperators<T>::scalarOps1() {
  vector_type r1(v1), r2(v1), r3(v1), r4(vBig);
  vector_type t1(0), t2(0), t3(0), t4(0);

  r1 += scalar;
  r2 -= scalar;
  r3 *= scalar;
  r4 /= scalar;

  // all element wise
  t1 = T(1)*v1 + scalar;
  t2 = T(1)*v1 - scalar;
  t3 = T(1)*v1 * scalar;
  t4 = T(1)*vBig / scalar;

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * global math operators with scalars, part II
 * Note: checked against member operators since they are allready checked
 */
template <class T>
void
TestXprVectorOperators<T>::scalarOps2() {
  vector_type r1(v1), r2(v1);
  vector_type t1(0), t2(0);

  r1 += scalar;
  r2 *= scalar;

  // all element wise
  t1 = scalar + T(1)*v1;
  t2 = scalar * T(1)*v1;

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
}


/*
 * global math operators with vector expressions
 */
template <class T>
void
TestXprVectorOperators<T>::globalXprVectorOps() {
  vector_type r1(v1), r2(v1), r3(v1), r4(v1);
  vector_type t1(0), t2(0), t3(0), t4(0);
  vector_type v2(v1);

  CPPUNIT_ASSERT( all_elements( v1 == v2) );

  r1 += v1;
  r2 -= v1;
  r3 *= v1;

  {
    using namespace tvmet::element_wise;
    r4 /= v1;
  }

  CPPUNIT_ASSERT( all_elements(r2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(r4 == T(1)) );

  t1 = T(1)*v1 + T(1)*v2;
  t2 = T(1)*v1 - T(1)*v2;
  t3 = T(1)*v1 * T(1)*v2;

  {
    using namespace tvmet::element_wise;
    t4 = (T(1)*v1) / (T(1)*v1);
  }

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * negate operators
 */
template <class T>
void
TestXprVectorOperators<T>::negate() {
  vector_type v2;

  v2 = -(T(1)*vOne);

  CPPUNIT_ASSERT( all_elements(v2 == T(-1)) );
}

#endif // TVMET_TEST_XPR_VECTOROPS_H

// Local Variables:
// mode:C++
// End:
