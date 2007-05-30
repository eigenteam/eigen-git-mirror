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
 * $Id: TestXprVectorFunctions.h,v 1.2 2005/03/25 07:12:07 opetzold Exp $
 */

#ifndef TVMET_TEST_XPR_VECTORFUNC_H
#define TVMET_TEST_XPR_VECTORFUNC_H

#include <limits>

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/util/General.h>

template <class T>
class TestXprVectorFunctions : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestXprVectorFunctions );
  CPPUNIT_TEST( scalarFuncs1 );
  CPPUNIT_TEST( scalarFuncs2 );
  CPPUNIT_TEST( globalXprVectorFuncs );
  CPPUNIT_TEST( fn_sum );
  CPPUNIT_TEST( fn_product );
  CPPUNIT_TEST( fn_dot );
  CPPUNIT_TEST( fn_cross );
  CPPUNIT_TEST( fn_norm );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;

public:
  TestXprVectorFunctions()
    : vZero(0), vOne(1), scalar(10), scalar2(2) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void scalarFuncs1();
  void scalarFuncs2();
  void globalXprVectorFuncs();
  void fn_sum();
  void fn_product();
  void fn_dot();
  void fn_cross();
  void fn_norm();

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
void TestXprVectorFunctions<T>::setUp() {
  v1 = 1,2,3;
  vBig = 10,20,30;
}


template <class T>
void TestXprVectorFunctions<T>::tearDown() { }


/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


/*
 * global math operators with scalars
 * function(XprVector, scalar)
 */
template <class T>
void
TestXprVectorFunctions<T>::scalarFuncs1() {
  vector_type r1(v1), r2(v1), r3(v1), r4(vBig);
  vector_type t1(0), t2(0), t3(0), t4(0);

  r1 += scalar;
  r2 -= scalar;
  r3 *= scalar;
  r4 /= scalar;

  // all element wise
  t1 = add(T(1)*v1, scalar);
  t2 = sub(T(1)*v1, scalar);
  t3 = mul(T(1)*v1, scalar);
  t4 = div(T(1)*vBig, scalar);

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * global math operators with scalars, part II
 * function(scalar, XprVector)
 */
template <class T>
void
TestXprVectorFunctions<T>::scalarFuncs2() {
  vector_type r1(v1), r2(v1);
  vector_type t1(0), t2(0);

  r1 += scalar;
  r2 *= scalar;

  // all element wise
  t1 = add(scalar, T(1)*v1);
  t2 = mul(scalar, T(1)*v1);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
}


/*
 * global math operators with xpr vectors (using functions)
 */
template <class T>
void
TestXprVectorFunctions<T>::globalXprVectorFuncs() {
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

  t1 = add(T(1)*v1, T(1)*v2);
  t2 = sub(T(1)*v1, T(1)*v2);
  t3 = mul(T(1)*v1, T(1)*v2);
  t4 = tvmet::element_wise::div(T(1)*v1, T(1)*v2);

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * sum of vector
 */
template <class T>
void
TestXprVectorFunctions<T>::fn_sum() {
  T t = sum(scalar*v1);	// alias vBig

  CPPUNIT_ASSERT( t == (vBig(0) + vBig(1) + vBig(2)));
}


/*
 * product of vector
 */
template <class T>
void
TestXprVectorFunctions<T>::fn_product() {
  T t = product(scalar*v1); // alias vBig

  CPPUNIT_ASSERT( t == (vBig(0) * vBig(1) * vBig(2)));
}

/*
 * dot product
 */
template <class T>
void
TestXprVectorFunctions<T>::fn_dot() {
  vector_type v2(v1);

  T t1 = dot(T(1)*v1, T(1)*v2);
  CPPUNIT_ASSERT( t1 == 14 );

  T t2 = dot(T(1)*v1, T(1)*vBig);
  CPPUNIT_ASSERT( t2 == 140 );

  T t3 = dot(T(1)*v1, T(1)*vOne);
  CPPUNIT_ASSERT( t3 == 6 );

  T t4 = dot(T(1)*v1, T(1)*vZero);
  CPPUNIT_ASSERT( t4 == 0 );

  T t5 = dot(T(1)*v1, vOne);
  CPPUNIT_ASSERT( t5 == 6 );

  T t6 = dot(vOne, T(1)*v1);
  CPPUNIT_ASSERT( t6 == 6 );
}


/*
 * cross product
 */
template <class T>
void
TestXprVectorFunctions<T>::fn_cross() {
  vector_type v2(v1);

  vector_type t1 = cross(T(1)*v1, T(1)*v2);
  CPPUNIT_ASSERT( all_elements(t1 == vZero) );	// orthogonal vectors

  vector_type t2 = cross(T(1)*v1, T(1)*vBig);
  CPPUNIT_ASSERT( all_elements(t2 == vZero) );	// orthogonal vectors

  const vector_type r(-1,2,-1);
  vector_type t3 = cross(T(1)*v1, T(1)*vOne);
  CPPUNIT_ASSERT( all_elements(t3 == r) );

  vector_type t4 = cross(T(1)*v1, T(1)*vZero);
  CPPUNIT_ASSERT( all_elements(t4 == vZero) );

  vector_type t5 = cross(T(1)*v2, v1);		// orthogonal
  CPPUNIT_ASSERT( all_elements(t5 == vZero) );

  vector_type t6 = cross(v1, T(1)*v2);		// orthogonal
  CPPUNIT_ASSERT( all_elements(t6 == vZero) );
}


/*
 * norm
 * Note: norm2 for ints specialized
 */
template <class T>
void
TestXprVectorFunctions<T>::fn_norm() {
  vector_type v2;
  vector_type r;
  vector_type t5;

  T t1 = norm1(T(1)*v1);
  T t2 = norm1(-v1);
  T t3 = norm2(T(1)*v1);
  T t4 = norm2(-v1);

  CPPUNIT_ASSERT( t1 == sum(v1) );
  CPPUNIT_ASSERT( t2 == sum(v1) );
  CPPUNIT_ASSERT( std::abs(t3 - std::sqrt(static_cast<typename tvmet::NumericTraits<T>::float_type>(14)))
		  < std::numeric_limits<T>::epsilon() );
  CPPUNIT_ASSERT( std::abs(t4 - std::sqrt(static_cast<typename tvmet::NumericTraits<T>::float_type>(14)))
		  < std::numeric_limits<T>::epsilon() );

  r = v1/norm2(v1); 	// norm2 is checked before
  t5 = normalize(T(1)*v1);

  CPPUNIT_ASSERT( all_elements(t5 == r) );
}


/*****************************************************************************
 * Implementation Part II (specialized for ints)
 ****************************************************************************/


/*
 * norm on int specialized due to rounding problems
 */
template <>
void
TestXprVectorFunctions<int>::fn_norm() {
  vector_type v2;

  int t1 = norm1(int(1)*v1);
  int t2 = norm1(-v1);
  CPPUNIT_ASSERT( t1 == sum(v1) );
  CPPUNIT_ASSERT( t2 == sum(v1) );
}


#endif // TVMET_TEST_XPR_VECTORFUNC_H

// Local Variables:
// mode:C++
// End:
