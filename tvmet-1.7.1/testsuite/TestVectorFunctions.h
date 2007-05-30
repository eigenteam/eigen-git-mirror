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
 * $Id: TestVectorFunctions.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_VECTORFUNC_H
#define TVMET_TEST_VECTORFUNC_H

#include <limits>

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/util/General.h>

template <class T>
class TestVectorFunctions : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestVectorFunctions );
  CPPUNIT_TEST( scalarUpdAssign1 );
  CPPUNIT_TEST( scalarUpdAssign2 );
  CPPUNIT_TEST( scalarUpdAssign3 );
  CPPUNIT_TEST( scalarFuncs1 );
  CPPUNIT_TEST( scalarFuncs2 );
  CPPUNIT_TEST( globalVectorFuncs1 );
  CPPUNIT_TEST( globalVectorFuncs2 );
  CPPUNIT_TEST( globalVectorFuncs3 );
  CPPUNIT_TEST( fn_sum );
  CPPUNIT_TEST( fn_product );
  CPPUNIT_TEST( fn_dot );
  CPPUNIT_TEST( fn_cross );
  CPPUNIT_TEST( fn_norm );
  CPPUNIT_TEST( extremum );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;

public:
  TestVectorFunctions()
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
  void scalarFuncs1();
  void scalarFuncs2();
  void globalVectorFuncs1();
  void globalVectorFuncs2();
  void globalVectorFuncs3();
  void fn_sum();
  void fn_product();
  void fn_dot();
  void fn_cross();
  void fn_norm();
  void extremum();

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
void TestVectorFunctions<T>::setUp() {
  v1 = 1,2,3;
  vBig = 10,20,30;
}


template <class T>
void TestVectorFunctions<T>::tearDown() { }


/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


/*
 * member math operators with scalars
 * Since we use it to compare results, these tests are elemental.
 */
template <class T>
void
TestVectorFunctions<T>::scalarUpdAssign1() {
  // all these functions are element wise
  vector_type t1(v1), t2(v1), t3(v1), t4(vBig);

  t1 += scalar;
  t2 -= scalar;
  t3 *= scalar;
  t4 /= scalar;

  assert(t1(0) == (v1(0)+scalar) && t1(1) == (v1(1)+scalar) && t1(2) == (v1(2)+scalar));
  assert(t2(0) == (v1(0)-scalar) && t2(1) == (v1(1)-scalar) && t2(2) == (v1(2)-scalar));
  assert( all_elements(t3 == vBig) );
  assert( all_elements(t4 == v1) );
}


/*
 * member math operators with Vectors
 * Since we use it to compare results, these tests are elemental.
 */
template <class T>
void
TestVectorFunctions<T>::scalarUpdAssign2() {
  // all these functions are element wise
  vector_type t1(v1), t2(v1), t3(v1), t4(v1);

  t1 += v1;
  t2 -= v1;
  t3 *= v1;

  {
    using namespace tvmet::element_wise;
    t4 /= v1;
  }

  assert( all_elements(t1 == 2*v1) );
  assert( all_elements(t2 == vZero) );
  assert(t3(0) == (v1(0)*v1(0)) && t3(1) == (v1(1)*v1(1)) && t3(2) == (v1(2)*v1(2)));
  assert( all_elements(t4 == 1) );
}


/*
 * member math operators with XprVector
 * Since we use it to compare results, these tests are elemental.
 */
template <class T>
void
TestVectorFunctions<T>::scalarUpdAssign3() {
  // all these functions are element wise
  vector_type t1(v1), t2(v1), t3(v1), t4(v1);

  t1 += T(1)*v1;
  t2 -= T(1)*v1;
  t3 *= T(1)*v1;

  {
    using namespace tvmet::element_wise;
    t4 /= T(1)*v1;
  }

  assert( all_elements(t1 == 2*v1) );
  assert( all_elements(t2 == vZero) );
  assert(t3(0) == (v1(0)*v1(0)) && t3(1) == (v1(1)*v1(1)) && t3(2) == (v1(2)*v1(2)));
  assert( all_elements(t4 == vOne) );
}


/*
 * global math operators with scalars
 * Note: checked against member operators since they are allready checked
 */
template <class T>
void
TestVectorFunctions<T>::scalarFuncs1() {
  vector_type r1(v1), r2(v1), r3(v1), r4(vBig);
  vector_type t1(0), t2(0), t3(0), t4(0);

  r1 += scalar;
  r2 -= scalar;
  r3 *= scalar;
  r4 /= scalar;

  // all element wise
  t1 = add(v1, scalar);
  t2 = sub(v1, scalar);
  t3 = mul(v1, scalar);
  t4 = div(vBig, scalar);

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
TestVectorFunctions<T>::scalarFuncs2() {
  vector_type r1(v1), r2(v1);
  vector_type t1(0), t2(0);

  r1 += scalar;
  r2 *= scalar;

  // all element wise
  t1 = add(scalar, v1);
  t2 = mul(scalar, v1);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
}


/*
 * global math operators with vectors (using functions)
 */
template <class T>
void
TestVectorFunctions<T>::globalVectorFuncs1() {
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

  t1 = add(v1, v2);
  t2 = sub(v1, v2);
  t3 = mul(v1, v2);
  t4 = tvmet::element_wise::div(v1, v1);

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * global math operators with vectors and xpr (using functions)
 */
template <class T>
void
TestVectorFunctions<T>::globalVectorFuncs2() {
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

  t1 = add(v1, T(1)*v2);
  t2 = sub(v1, T(1)*v2);
  t3 = mul(v1, T(1)*v2);
  t4 = tvmet::element_wise::div(v1, T(1)*v1);

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * global math operators with vectors with xpr (using functions)
 */
template <class T>
void
TestVectorFunctions<T>::globalVectorFuncs3() {
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

  t1 = add(T(1)*v1, v2);
  t2 = sub(T(1)*v1, v2);
  t3 = mul(T(1)*v1, v2);
  t4 = tvmet::element_wise::div(T(1)*v1, v1);

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
TestVectorFunctions<T>::fn_sum() {
  T t = sum(v1);

  CPPUNIT_ASSERT( t == (v1(0) + v1(1) + v1(2)));
}


/*
 * prod of vector
 */
template <class T>
void
TestVectorFunctions<T>::fn_product() {
  T t = product(v1);

  CPPUNIT_ASSERT( t == (v1(0) * v1(1) * v1(2)));
}


/*
 * dot product
 */
template <class T>
void
TestVectorFunctions<T>::fn_dot() {
  vector_type v2(v1);

  T t1 = dot(v1, v2);
  CPPUNIT_ASSERT( t1 == 14 );

  T t2 = dot(v1, vBig);
  CPPUNIT_ASSERT( t2 == 140 );

  T t3 = dot(v1, vOne);
  CPPUNIT_ASSERT( t3 == 6 );

  T t4 = dot(v1, vZero);
  CPPUNIT_ASSERT( t4 == 0 );
}


/*
 * cross product
 */
template <class T>
void
TestVectorFunctions<T>::fn_cross() {
  vector_type v2(v1);

  vector_type t1 = cross(v1, v2);
  CPPUNIT_ASSERT( all_elements(t1 == vZero) );	// orthogonal vectors

  vector_type t2 = cross(v1, vBig);
  CPPUNIT_ASSERT( all_elements(t2 == vZero) );	// orthogonal vectors

  const vector_type r(-1,2,-1);
  vector_type t3 = cross(v1, vOne);
  CPPUNIT_ASSERT( all_elements(t3 == r) );

  vector_type t4 = cross(v1, vZero);
  CPPUNIT_ASSERT( all_elements(t4 == vZero) );
}


/*
 * norm
 * Note: norm2 for ints specialized
 */
template <class T>
void
TestVectorFunctions<T>::fn_norm() {
  vector_type v2;
  vector_type r;
  vector_type t5;

  v2 = -v1;		// norm can't handle XprVector<> yet

  T t1 = norm1(v1);
  T t2 = norm1(v2);
  T t3 = norm2(v1);
  T t4 = norm2(v2);

  CPPUNIT_ASSERT( t1 == sum(v1) );
  CPPUNIT_ASSERT( t2 == sum(v1) );
  CPPUNIT_ASSERT( std::abs(t3 - std::sqrt(static_cast<typename tvmet::NumericTraits<T>::float_type>(14)))
		  < std::numeric_limits<T>::epsilon() );
  CPPUNIT_ASSERT( std::abs(t4 - std::sqrt(static_cast<typename tvmet::NumericTraits<T>::float_type>(14)))
		  < std::numeric_limits<T>::epsilon() );

  r = v1/norm2(v1); 	// norm2 is checked before
  t5 = normalize(v1);

  CPPUNIT_ASSERT( all_elements(t5 == r) );
}


/*
 * min/max functions
 */
template <class T>
void
TestVectorFunctions<T>::extremum() {
  CPPUNIT_ASSERT(max(v1) == 3);
  CPPUNIT_ASSERT(min(v1) == 1);

  CPPUNIT_ASSERT(max(vBig) == 30);
  CPPUNIT_ASSERT(min(vBig) == 10);

  CPPUNIT_ASSERT(maximum(v1).value() == 3);
  CPPUNIT_ASSERT(maximum(v1).index() == 2);

  CPPUNIT_ASSERT(minimum(v1).value() == 1);
  CPPUNIT_ASSERT(minimum(v1).index() == 0);

  CPPUNIT_ASSERT(maximum(vBig).value() == 30);
  CPPUNIT_ASSERT(maximum(vBig).index() == 2);

  CPPUNIT_ASSERT(minimum(vBig).value() == 10);
  CPPUNIT_ASSERT(minimum(vBig).index() == 0);
}


/*****************************************************************************
 * Implementation Part II (specialized for ints)
 ****************************************************************************/


/*
 * norm on int specialized due to rounding problems
 */
template <>
void
TestVectorFunctions<int>::fn_norm() {
  vector_type v2;
  v2 = -v1;

  int t1 = norm1(v1);
  int t2 = norm1(v2);
  CPPUNIT_ASSERT( t1 == sum(v1) );
  CPPUNIT_ASSERT( t2 == sum(v1) );
}


#endif // TVMET_TEST_VECTORFUNC_H

// Local Variables:
// mode:C++
// End:
