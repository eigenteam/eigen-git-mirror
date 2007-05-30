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
 * $Id: TestMathOps.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_MATHOPS_H
#define TVMET_TEST_MATHOPS_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>
#include <tvmet/util/General.h>

template <class T>
class TestMathOps : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestMathOps );
  CPPUNIT_TEST( ScalarAssign );
  CPPUNIT_TEST( Assign );
  CPPUNIT_TEST( ScalarOps );
  CPPUNIT_TEST( Ops1 );
  CPPUNIT_TEST( Ops2 );
  CPPUNIT_TEST( VectorOps );
  CPPUNIT_TEST( VectorOps2 );
  CPPUNIT_TEST( VectorNorm2 );
  CPPUNIT_TEST( MatrixOps );
  CPPUNIT_TEST( MatrixVector1 );
  CPPUNIT_TEST( MatrixVector2 );
  CPPUNIT_TEST( MatrixTransMatrix );
  CPPUNIT_TEST( MatrixTransVector );
  CPPUNIT_TEST( MatrixRowVector );
  CPPUNIT_TEST( MatrixColVector );
  CPPUNIT_TEST( MatrixDiagVector );
  CPPUNIT_TEST( MatrixMatrixVector );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  TestMathOps()
    : vZero(0), vOne(1), mZero(0), mOne(1), scalar(10) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void ScalarAssign();
  void Assign();
  void ScalarOps();
  void Ops1();
  void Ops2();
  void VectorOps();
  void VectorOps2();
  void VectorNorm2();
  void MatrixOps();
  void MatrixVector1();
  void MatrixVector2();
  void MatrixTransMatrix();
  void MatrixTransVector();
  void MatrixRowVector();
  void MatrixColVector();
  void MatrixDiagVector();
  void MatrixMatrixVector();

private:
  const vector_type vZero;
  const vector_type vOne;
  vector_type v1, v1b;
  vector_type vBig;	/**< vector 10x bigger than v1 */

private:
  const matrix_type mZero;
  const matrix_type mOne;
  matrix_type m1, m1b;
  matrix_type mBig;	/**< matrix 10x bigger than m1 */

private:
  vector_type m1_r0, m1_r1, m1_r2;	// row vectors
  vector_type m1_c0, m1_c1, m1_c2;	// col vectors

private:
  const T scalar;
};

/*****************************************************************************
 * Implementation
 ****************************************************************************/

/*
 * cppunit part
 */
template <class T>
void TestMathOps<T>::setUp() {
  v1 = 1,2,3;
  v1b = v1;		// same as v1, cctor test done in checkInternal
  vBig = 10,20,30;

  m1 = 1,4,7,
       2,5,8,
       3,6,9;

  m1_r0 = 1,4,7;
  m1_r1 = 2,5,8;
  m1_r2 = 3,6,9;

  m1_c0 = 1,2,3;
  m1_c1 = 4,5,6;
  m1_c2 = 7,8,9;

  m1b = m1;		// same as m1, cctor test done in checkInternal

  mBig = 10,40,70,
         20,50,80,
         30,60,90;

}

template <class T>
void TestMathOps<T>::tearDown() {

}

/*
 * regressions
 */
template <class T>
void
TestMathOps<T>::ScalarAssign() {
  {
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
  {
    matrix_type t1(m1), t2(m1), t3(m1), t4(mBig);

    t1 += scalar;
    t2 -= scalar;
    t3 *= scalar;
    t4 /= scalar;

    CPPUNIT_ASSERT(t1(0,0) == (m1(0,0)+scalar) && t1(0,1) == (m1(0,1)+scalar) && t1(0,2) == (m1(0,2)+scalar) &&
	   t1(1,0) == (m1(1,0)+scalar) && t1(1,1) == (m1(1,1)+scalar) && t1(1,2) == (m1(1,2)+scalar) &&
	   t1(2,0) == (m1(2,0)+scalar) && t1(2,1) == (m1(2,1)+scalar) && t1(2,2) == (m1(2,2)+scalar));
    CPPUNIT_ASSERT(t2(0,0) == (m1(0,0)-scalar) && t2(0,1) == (m1(0,1)-scalar) && t2(0,2) == (m1(0,2)-scalar) &&
	   t2(1,0) == (m1(1,0)-scalar) && t2(1,1) == (m1(1,1)-scalar) && t2(1,2) == (m1(1,2)-scalar) &&
	   t2(2,0) == (m1(2,0)-scalar) && t2(2,1) == (m1(2,1)-scalar) && t2(2,2) == (m1(2,2)-scalar));
    CPPUNIT_ASSERT( all_elements(t3 == mBig) );
    CPPUNIT_ASSERT( all_elements(t4 == m1) );
  }
}

template <class T>
void
TestMathOps<T>::Assign() {
  {
    vector_type t1(vZero), t2(v1), t3(v1);

    t1 += v1;
    t2 -= v1;
    t3 *= v1;

    CPPUNIT_ASSERT( all_elements(t1 == v1) );
    CPPUNIT_ASSERT( all_elements(t2 == vZero) );
    CPPUNIT_ASSERT(t3(0) == (v1(0)*v1(0)) && t3(1) == (v1(1)*v1(1)) && t3(2) == (v1(2)*v1(2)));
  }
  {
    matrix_type t1(mZero), t2(m1), t3(m1);

    t1 += m1;
    t2 -= m1;

    CPPUNIT_ASSERT( all_elements(t1 == m1) );
    CPPUNIT_ASSERT( all_elements(t2 == mZero) );
  }
}

template <class T>
void
TestMathOps<T>::ScalarOps() {
  {
    vector_type t1(v1), t2(v1), t3(v1), t4(vBig);
    vector_type r1(v1), r2(v1);
    r1 += scalar;
    r2 -= scalar;

    t1 = t1 + scalar;
    t2 = t2 - scalar;
    t3 = t3 * scalar;
    t4 = t4 / scalar;

    CPPUNIT_ASSERT( all_elements(t1 == r1) );
    CPPUNIT_ASSERT( all_elements(t2 == r2) );
    CPPUNIT_ASSERT( all_elements(t3 == vBig) );
    CPPUNIT_ASSERT( all_elements(t4 == v1) );
  }
  {
    matrix_type t1(m1), t2(m1), t3(m1), t4(mBig);
    matrix_type r1(m1), r2(m1);
    r1 += scalar;
    r2 -= scalar;

    t1 = t1 + scalar;
    t2 = t2 - scalar;
    t3 = t3 * scalar;
    t4 = t4 / scalar;

    CPPUNIT_ASSERT( all_elements(t1 == r1) );
    CPPUNIT_ASSERT( all_elements(t2 == r2) );
    CPPUNIT_ASSERT( all_elements(t3 == mBig) );
    CPPUNIT_ASSERT( all_elements(t4 == m1) );
  }
}

template <class T>
void
TestMathOps<T>::Ops1() {
  {
    vector_type t1(0), t2(0), t3(0);
    vector_type r(v1);
    r *= v1;

    t1 = v1 + v1;
    t2 = v1 - v1;
    t3 = v1 * v1;

    CPPUNIT_ASSERT( all_elements(t1 == T(2)*v1) );
    CPPUNIT_ASSERT( all_elements(t2 == vZero) );
    CPPUNIT_ASSERT( all_elements(t3 == r) );
  }
  {
    matrix_type t1(0), t2(0);
    t1 = m1 + m1;
    t2 = m1 - m1;

    CPPUNIT_ASSERT( all_elements(t1 == T(2)*m1) );
    CPPUNIT_ASSERT( all_elements(t2 == mZero) );
  }
}

template <class T>
void
TestMathOps<T>::Ops2() {
  const vector_type vMinusOne(-1);
  const matrix_type mMinusOne(-1);

  // negate operator
  {
    vector_type t1, t2;

    t1 = abs(v1);
    CPPUNIT_ASSERT( all_elements(t1 == v1) );

    t1 = -vOne;
    CPPUNIT_ASSERT( all_elements(t1 == vMinusOne) );
  }
  {
    matrix_type t1, t2;

    t1 = abs(m1);
    CPPUNIT_ASSERT( all_elements(t1 == m1) );

    t1 = -mOne;
    CPPUNIT_ASSERT( all_elements(t1 == mMinusOne) );

  }
}

template <class T>
void
TestMathOps<T>::VectorOps() {

}

template <class T>
void
TestMathOps<T>::VectorOps2() {
}

template <class T>
void
TestMathOps<T>::VectorNorm2() {
  // casts for int vectors, as well as for complex<> since
  // norm2 returns sum_type
  CPPUNIT_ASSERT( norm2(v1) == static_cast<T>(std::sqrt(14.0)));
}

template <class T>
void
TestMathOps<T>::MatrixOps() {
  matrix_type t1, t2, t3;
  matrix_type r1, r2, r3;

  tvmet::util::Gemm(m1, m1, r1);
  tvmet::util::Gemm(m1, mBig, r2);
  tvmet::util::Gemm(mBig, m1, r3);
  CPPUNIT_ASSERT( all_elements(r2 == r3) );

  t1 = m1 * m1;
  CPPUNIT_ASSERT( all_elements(t1 == r1) );

  t2 = m1 * mBig;
  CPPUNIT_ASSERT( all_elements(t2 == r2) );

  t3 = mBig * m1;
  CPPUNIT_ASSERT( all_elements(t3 == r3) );

  t3 = trans(t1);
  CPPUNIT_ASSERT( any_elements(t3 != t1) ); // XXX very simple test
  t2 = trans(t3);
  CPPUNIT_ASSERT( all_elements(t1 == t2) );

  // trace return sum_type, therefore the cast for complex<>
  CPPUNIT_ASSERT( static_cast<T>(trace(m1)) == static_cast<T>(15) );
}

template <class T>
void
TestMathOps<T>::MatrixVector1() {

  vector_type t1, t2;
  vector_type vr1(0), vr2(0);	// clear it before use due to util::Gemv algo

  // Matrix-Vector
  tvmet::util::Gemv(m1, v1, vr1);
  tvmet::util::Gemv(mBig, vBig, vr2);

  t1 = m1 * v1;
  t2 = mBig * vBig;

  CPPUNIT_ASSERT( all_elements(t1 == vr1) );
  CPPUNIT_ASSERT( all_elements(t2 == vr2) );
}

template <class T>
void
TestMathOps<T>::MatrixVector2() {

  vector_type t1, t2;
  vector_type vr(0), v2(0);	// clear it before use due to util::Gemv algo

  // Matrix-XprVector
  v2 = v1 * vBig;
  tvmet::util::Gemv(m1, v2, vr);

  t1 = m1 * (v1*vBig);

  CPPUNIT_ASSERT( all_elements(t1 == vr) );
}

template <class T>
void
TestMathOps<T>::MatrixTransMatrix() {
  // greatings to
  {
    matrix_type m1t, Mr, M2;

    // trans() and prod() is checked before!
    m1t = trans(m1);
    Mr  = prod(m1t, mBig);

    M2  = MtM_prod(m1, mBig);

    CPPUNIT_ASSERT( all_elements(Mr == M2) );
  }
}

template <class T>
void
TestMathOps<T>::MatrixTransVector() {
  // greatings to
  {
    matrix_type Mt;
    vector_type vr, y;

    // trans() and prod() is checked before!
    Mt = trans(m1);
    vr = Mt*v1;
    y  = Mtx_prod(m1, v1);

    CPPUNIT_ASSERT( all_elements(vr == y) );
  }
}

template <class T>
void
TestMathOps<T>::MatrixRowVector() {
  vector_type r0, r1, r2;

  r0 = row(m1, 0);
  r1 = row(m1, 1);
  r2 = row(m1, 2);

  CPPUNIT_ASSERT( all_elements(r0 == m1_r0) );
  CPPUNIT_ASSERT( all_elements(r1 == m1_r1) );
  CPPUNIT_ASSERT( all_elements(r2 == m1_r2) );
}

template <class T>
void
TestMathOps<T>::MatrixColVector() {
  vector_type c0, c1, c2;

  c0 = col(m1, 0);
  c1 = col(m1, 1);
  c2 = col(m1, 2);

  CPPUNIT_ASSERT( all_elements(c0 == m1_c0) );
  CPPUNIT_ASSERT( all_elements(c1 == m1_c1) );
  CPPUNIT_ASSERT( all_elements(c2 == m1_c2) );
}

template <class T>
void
TestMathOps<T>::MatrixDiagVector() {
  vector_type vd, t;

  vd = T(1), T(5), T(9);

  t = diag(m1);

  CPPUNIT_ASSERT( all_elements(vd == t) );
}

template <class T>
void
TestMathOps<T>::MatrixMatrixVector() {
  {
    vector_type t1;
    vector_type vr1(0), vr2(0);	// clear it before use due to util::Gemv algo

    // Matrix-Vector-Vector, referenz is using two ops
    tvmet::util::Gemv(m1, v1, vr1);
    tvmet::util::Gevvmul(vr1, vBig, vr2);

    t1 = m1 * v1 * vBig;
    CPPUNIT_ASSERT( all_elements(t1 == vr2) );
  }
#if 0
  {
    // XXX not working due to missing operators for (XprMatrix, Vector)
    vector_type t;
    matrix_type vr1;
    vector_type vr2;

    // Matrix-Matrix-Vector
    tvmet::util::Gemm(m1, mBig, vr1);
    tvmet::util::Gemv(vr1, v1, vr2);

  }
#endif
}

#endif // TVMET_TEST_MATHOPS_H

// Local Variables:
// mode:C++
// End:
