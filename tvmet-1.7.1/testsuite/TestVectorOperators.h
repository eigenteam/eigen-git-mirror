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
 * $Id: TestVectorOperators.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_VECTORPOS_H
#define TVMET_TEST_VECTORPOS_H

#include <limits>

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/util/General.h>

template <class T>
class TestVectorOperators : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestVectorOperators );
  CPPUNIT_TEST( scalarUpdAssign1 );
  CPPUNIT_TEST( scalarUpdAssign2 );
  CPPUNIT_TEST( scalarUpdAssign3 );
  CPPUNIT_TEST( scalarOps1 );
  CPPUNIT_TEST( scalarOps2 );
  CPPUNIT_TEST( globalVectorOps1 );
  CPPUNIT_TEST( globalVectorOps2 );
  CPPUNIT_TEST( globalVectorOps3 );
  CPPUNIT_TEST( negate );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;

public:
  TestVectorOperators()
    : vZero(0), vOne(1), scalar(10), scalar2(2) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void scalarUpdAssign1();
  void scalarUpdAssign2();
  void scalarUpdAssign3();
  void scalarOps1();
  void scalarOps2();
  void globalVectorOps1();
  void globalVectorOps2();
  void globalVectorOps3();
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
void TestVectorOperators<T>::setUp() {
  v1 = 1,2,3;
  vBig = 10,20,30;
}


template <class T>
void TestVectorOperators<T>::tearDown() { }


/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


/*
 * member math operators with scalars;
 */
template <class T>
void
TestVectorOperators<T>::scalarUpdAssign1() {
  // all these functions are element wise
  vector_type t1(v1), t2(v1), t3(v1), t4(vBig);

  t1 += scalar;
  t2 -= scalar;
  t3 *= scalar;
  t4 /= scalar;

  CPPUNIT_ASSERT(t1(0) == (v1(0)+scalar) && t1(1) == (v1(1)+scalar) && t1(2) == (v1(2)+scalar));
  CPPUNIT_ASSERT(t2(0) == (v1(0)-scalar) && t2(1) == (v1(1)-scalar) && t2(2) == (v1(2)-scalar));
  CPPUNIT_ASSERT( all_elements(t3 == vBig) );
  CPPUNIT_ASSERT( all_elements(t4 == v1) );
}


/*
 * member math operators with Vectors
 */
template <class T>
void
TestVectorOperators<T>::scalarUpdAssign2() {
  // all these functions are element wise
  vector_type t1(v1), t2(v1), t3(v1), t4(v1);

  t1 += v1;
  t2 -= v1;
  t3 *= v1;

  {
    using namespace tvmet::element_wise;
    t4 /= v1;
  }

  CPPUNIT_ASSERT( all_elements(t1 == 2*v1) );
  CPPUNIT_ASSERT( all_elements(t2 == vZero) );
  CPPUNIT_ASSERT(t3(0) == (v1(0)*v1(0)) && t3(1) == (v1(1)*v1(1)) && t3(2) == (v1(2)*v1(2)));
  CPPUNIT_ASSERT( all_elements(t4 == 1) );
}


/*
 * member math operators with XprVector
 */
template <class T>
void
TestVectorOperators<T>::scalarUpdAssign3() {
  // all these functions are element wise
  vector_type t1(v1), t2(v1), t3(v1), t4(v1);

  t1 += T(1)*v1;
  t2 -= T(1)*v1;
  t3 *= T(1)*v1;

  {
    using namespace tvmet::element_wise;
    t4 /= T(1)*v1;
  }

  CPPUNIT_ASSERT( all_elements(t1 == 2*v1) );
  CPPUNIT_ASSERT( all_elements(t2 == vZero) );
  CPPUNIT_ASSERT(t3(0) == (v1(0)*v1(0)) && t3(1) == (v1(1)*v1(1)) && t3(2) == (v1(2)*v1(2)));
  CPPUNIT_ASSERT( all_elements(t4 == vOne) );
}


/*
 * global math operators with scalars
 * Note: checked against member operators since they are allready checked
 */
template <class T>
void
TestVectorOperators<T>::scalarOps1() {
  vector_type r1(v1), r2(v1), r3(v1), r4(vBig);
  vector_type t1(0), t2(0), t3(0), t4(0);

  r1 += scalar;
  r2 -= scalar;
  r3 *= scalar;
  r4 /= scalar;

  // all element wise
  t1 = v1 + scalar;
  t2 = v1 - scalar;
  t3 = v1 * scalar;
  t4 = vBig / scalar;

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
TestVectorOperators<T>::scalarOps2() {
  vector_type r1(v1), r2(v1);
  vector_type t1(0), t2(0);

  r1 += scalar;
  r2 *= scalar;

  // all element wise
  t1 = scalar + v1;
  t2 = scalar * v1;

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
}


/*
 * global math operators with vectors
 */
template <class T>
void
TestVectorOperators<T>::globalVectorOps1() {
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

  t1 = v1 + v2;
  t2 = v1 - v2;
  t3 = v1 * v2;

  {
    using namespace tvmet::element_wise;
    t4 = v1 / v1;
  }

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * global math operators with vectors and xpr
 */
template <class T>
void
TestVectorOperators<T>::globalVectorOps2() {
  vector_type r1(v1), r2(v1), r3(v1), r4(v1);
  vector_type t1(0), t2(0), t3(0), t4(0);
  vector_type v2(v1);

  CPPUNIT_ASSERT( all_elements( v1 == v2) );

  r1 += T(1)*v1;
  r2 -= T(1)*v1;
  r3 *= T(1)*v1;

  {
    using namespace tvmet::element_wise;
    r4 /= T(1)*v1;
  }

  CPPUNIT_ASSERT( all_elements(r2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(r4 == T(1)) );

  t1 = v1 + T(1)*v2;
  t2 = v1 - T(1)*v2;
  t3 = v1 * T(1)*v2;

  {
    using namespace tvmet::element_wise;
    t4 = v1 / ( T(1)*v1 );
  }

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * global math operators with vectors with xpr
 */
template <class T>
void
TestVectorOperators<T>::globalVectorOps3() {
  vector_type r1(v1), r2(v1), r3(v1), r4(v1);
  vector_type t1(0), t2(0), t3(0), t4(0);
  vector_type v2(v1);

  CPPUNIT_ASSERT( all_elements( v1 == v2) );

  r1 += T(1)*v1;
  r2 -= T(1)*v1;
  r3 *= T(1)*v1;

  {
    using namespace tvmet::element_wise;
    r4 /= T(1)*v1;
  }

  CPPUNIT_ASSERT( all_elements(r2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(r4 == T(1)) );

  t1 = (T(1)*v1) + v2;
  t2 = (T(1)*v1) - v2;
  t3 = (T(1)*v1) * v2;

  {
    using namespace tvmet::element_wise;
    t4 = (T(1)*v1) / v1;
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
TestVectorOperators<T>::negate() {
  vector_type v2;

  v2 = -vOne;

  CPPUNIT_ASSERT( all_elements(v2 == T(-1)) );
}


/*****************************************************************************
 * Implementation Part II (specialized for ints)
 ****************************************************************************/


/*
 * member math operators with scalars
 */
template <>
void
TestVectorOperators<int>::scalarUpdAssign1() {
  // all these functions are element wise
  vector_type t1(v1), t2(v1), t3(v1), t4(vBig);
  vector_type t5(v1), t6(vBig), t7(vBig), t8(vBig), t9(vBig);
  vector_type t10(v1), t11(v1);

  t1 += scalar;
  t2 -= scalar;
  t3 *= scalar;
  t4 /= scalar;

  t5 %= scalar;
  t6 %= scalar;
  t7 ^= scalar;
  t8 &= scalar;
  t9 |= scalar;
  t10 <<= scalar;
  t11 >>= scalar2;

  CPPUNIT_ASSERT(t1(0) == (v1(0)+scalar) && t1(1) == (v1(1)+scalar) && t1(2) == (v1(2)+scalar));
  CPPUNIT_ASSERT(t2(0) == (v1(0)-scalar) && t2(1) == (v1(1)-scalar) && t2(2) == (v1(2)-scalar));
  CPPUNIT_ASSERT( all_elements(t3 == vBig) );
  CPPUNIT_ASSERT( all_elements(t4 == v1) );
  CPPUNIT_ASSERT( all_elements(t5 == v1) );
  CPPUNIT_ASSERT( all_elements(t6 == vZero) );
  CPPUNIT_ASSERT(t7(0) == (vBig(0)^scalar)  && t7(1) == (vBig(1)^scalar) && t7(2) == (vBig(2)^scalar) );
  CPPUNIT_ASSERT(t8(0) == (vBig(0)&scalar)  && t8(1) == (vBig(1)&scalar) && t8(2) == (vBig(2)&scalar) );
  CPPUNIT_ASSERT(t9(0) == (vBig(0)|scalar)  && t9(1) == (vBig(1)|scalar) && t9(2) == (vBig(2)|scalar) );
  CPPUNIT_ASSERT(t10(0) == (v1(0)<<scalar)  && t10(1) == (v1(1)<<scalar) && t10(2) == (v1(2)<<scalar) );
  CPPUNIT_ASSERT(t11(0) == (v1(0)>>scalar2) && t11(1) == (v1(1)>>scalar2) && t11(2) == (v1(2)>>scalar2) );
}

/*
 * TODO: implement other UpdAssign functions, esp. for bit ops
 * (since functions above are working, all others should work)
 */


#endif // TVMET_TEST_VECTORPOS_H

// Local Variables:
// mode:C++
// End:
