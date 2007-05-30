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
 * $Id: TestMatrixFunctions.h,v 1.2 2004/07/06 06:24:23 opetzold Exp $
 */

#ifndef TVMET_TEST_MATRIXFUNC_H
#define TVMET_TEST_MATRIXFUNC_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>
#include <tvmet/util/General.h>

template <class T>
class TestMatrixFunctions : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestMatrixFunctions );
  CPPUNIT_TEST( scalarUpdAssign1 );
  CPPUNIT_TEST( scalarUpdAssign2 );
  CPPUNIT_TEST( scalarUpdAssign3 );
  CPPUNIT_TEST( scalarOps1 );
  CPPUNIT_TEST( scalarOps2 );
  CPPUNIT_TEST( globalMatrixFuncs1 );
  CPPUNIT_TEST( globalMatrixFuncs2 );
  CPPUNIT_TEST( globalMatrixFuncs3 );
  CPPUNIT_TEST( fn_prod1 );
  CPPUNIT_TEST( fn_prod2 );
  CPPUNIT_TEST( fn_prod3 );
  CPPUNIT_TEST( fn_trans );
  CPPUNIT_TEST( fn_MtM_prod );
  CPPUNIT_TEST( fn_MMt_prod );
  CPPUNIT_TEST( fn_prodTrans );
  CPPUNIT_TEST( fn_trace );
  CPPUNIT_TEST( rowVector );
  CPPUNIT_TEST( colVector );
  CPPUNIT_TEST( fn_diag );
  CPPUNIT_TEST( extremum );
  CPPUNIT_TEST( identity_matrix );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  TestMatrixFunctions()
    : mZero(0), mOne(1), scalar(10), scalar2(2) { }

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
  void globalMatrixFuncs1();
  void globalMatrixFuncs2();
  void globalMatrixFuncs3();
  void fn_prod1();
  void fn_prod2();
  void fn_prod3();
  void fn_trans();
  void fn_MtM_prod();
  void fn_MMt_prod();
  void fn_prodTrans();
  void fn_trace();
  void rowVector();
  void colVector();
  void fn_diag();
  void extremum();
  void identity_matrix();

private:
  const matrix_type mZero;
  const matrix_type mOne;
  matrix_type m1;
  matrix_type mBig;	/**< matrix 10x bigger than m1 */

private:
  vector_type m1_r0, m1_r1, m1_r2;	// row vectors
  vector_type m1_c0, m1_c1, m1_c2;	// col vectors

private:
  const T scalar;
  const T scalar2;
};


/*****************************************************************************
 * Implementation Part I (cppunit part)
 *** *************************************************************************/


template <class T>
void TestMatrixFunctions<T>::setUp() {
  m1 = 1,4,7,
       2,5,8,
       3,6,9;

  m1_r0 = 1,4,7;
  m1_r1 = 2,5,8;
  m1_r2 = 3,6,9;

  m1_c0 = 1,2,3;
  m1_c1 = 4,5,6;
  m1_c2 = 7,8,9;

  mBig = 10,40,70,
         20,50,80,
         30,60,90;
}

template <class T>
void TestMatrixFunctions<T>::tearDown() { }

/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


/*
 * member math operators with scalars
 * Since we use it to compare results, these tests are elemental.
 */
template <class T>
void
TestMatrixFunctions<T>::scalarUpdAssign1() {
  // all these functions are element wise
  matrix_type t1(m1), t2(m1), t3(m1), t4(mBig);

  t1 += scalar;
  t2 -= scalar;
  t3 *= scalar;
  t4 /= scalar;

  assert(t1(0,0) == (m1(0,0)+scalar) && t1(0,1) == (m1(0,1)+scalar) && t1(0,2) == (m1(0,2)+scalar) &&
	 t1(1,0) == (m1(1,0)+scalar) && t1(1,1) == (m1(1,1)+scalar) && t1(1,2) == (m1(1,2)+scalar) &&
	 t1(2,0) == (m1(2,0)+scalar) && t1(2,1) == (m1(2,1)+scalar) && t1(2,2) == (m1(2,2)+scalar));
  assert(t2(0,0) == (m1(0,0)-scalar) && t2(0,1) == (m1(0,1)-scalar) && t2(0,2) == (m1(0,2)-scalar) &&
	 t2(1,0) == (m1(1,0)-scalar) && t2(1,1) == (m1(1,1)-scalar) && t2(1,2) == (m1(1,2)-scalar) &&
	 t2(2,0) == (m1(2,0)-scalar) && t2(2,1) == (m1(2,1)-scalar) && t2(2,2) == (m1(2,2)-scalar));
  assert( all_elements(t3 == mBig) );
  assert( all_elements(t4 == m1) );
}


/*
 * member math operators with Matrizes
 * Since we use it to compare results, these tests are elemental.
 */
template <class T>
void
TestMatrixFunctions<T>::scalarUpdAssign2() {
  // all these functions are element wise
  matrix_type t1(m1), t2(m1), t3(m1), t4(m1);

  t1 += m1;
  t2 -= m1;

  {
    using namespace tvmet::element_wise;

    t3 *= m1;
    t4 /= m1;
  }

  assert(t1(0,0) == (m1(0,0)*2) && t1(0,1) == (m1(0,1)*2) && t1(0,2) == (m1(0,2)*2) &&
	 t1(1,0) == (m1(1,0)*2) && t1(1,1) == (m1(1,1)*2) && t1(1,2) == (m1(1,2)*2) &&
	 t1(2,0) == (m1(2,0)*2) && t1(2,1) == (m1(2,1)*2) && t1(2,2) == (m1(2,2)*2));
  assert( all_elements(t2 == mZero) );
  assert(t3(0,0) == (m1(0,0)*m1(0,0)) && t3(0,1) == (m1(0,1)*m1(0,1)) && t3(0,2) == (m1(0,2)*m1(0,2)) &&
	 t3(1,0) == (m1(1,0)*m1(1,0)) && t3(1,1) == (m1(1,1)*m1(1,1)) && t3(1,2) == (m1(1,2)*m1(1,2)) &&
	 t3(2,0) == (m1(2,0)*m1(2,0)) && t3(2,1) == (m1(2,1)*m1(2,1)) && t3(2,2) == (m1(2,2)*m1(2,2)));
  assert( all_elements(t4 == mOne) );
}


/*
 * member math operators with XprMatrizes
 * Since we use it to compare results, these tests are elemental.
 */
template <class T>
void
TestMatrixFunctions<T>::scalarUpdAssign3() {
  // all these functions are element wise
  matrix_type t1(m1), t2(m1), t3(m1), t4(m1);

  t1 += T(1)*m1;
  t2 -= T(1)*m1;

  {
    using namespace tvmet::element_wise;

    t3 *= T(1)*m1;
    t4 /= T(1)*m1;
  }

  assert( all_elements(t1 == 2*m1) );
  assert( all_elements(t2 == mZero) );
  assert(t3(0,0) == (m1(0,0)*m1(0,0)) && t3(0,1) == (m1(0,1)*m1(0,1)) && t3(0,2) == (m1(0,2)*m1(0,2)) &&
		 t3(1,0) == (m1(1,0)*m1(1,0)) && t3(1,1) == (m1(1,1)*m1(1,1)) && t3(1,2) == (m1(1,2)*m1(1,2)) &&
		 t3(2,0) == (m1(2,0)*m1(2,0)) && t3(2,1) == (m1(2,1)*m1(2,1)) && t3(2,2) == (m1(2,2)*m1(2,2)));
  assert( all_elements(t4 == mOne) );
}


/*
 * global math operators with scalars
 * Note: checked against member operators since they are allready checked
 */
template <class T>
void
TestMatrixFunctions<T>::scalarOps1() {
  matrix_type r1(m1), r2(m1);
  matrix_type t1(0), t2(0), t3(0), t4(0);

  r1 += scalar;
  r2 -= scalar;

  t1 = add(m1, scalar);
  t2 = sub(m1, scalar);
  t3 = mul(m1, scalar);
  t4 = div(mBig, scalar);

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == mBig) );
  CPPUNIT_ASSERT( all_elements(t4 == m1) );
}


/*
 * global math operators with scalars, part II
 * Note: checked against member operators since they are allready checked
 */
template <class T>
void
TestMatrixFunctions<T>::scalarOps2() {
  matrix_type r1(m1), r2(m1);
  matrix_type t1(0), t2(0);

  r1 += scalar;
  r2 *= scalar;

  t1 = add(scalar, m1);
  t2 = mul(scalar, m1);

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
}


/*
 * global math operators with matrizes
 */
template <class T>
void
TestMatrixFunctions<T>::globalMatrixFuncs1() {
  matrix_type t1(0), t2(0), t3(0), t4(0);

  t1 = add(m1, m1);
  t2 = sub(m1, m1);

  {
    using namespace tvmet::element_wise;

    t3 = mul(m1, mOne);
    t4 = div(m1, mOne);
  }

  CPPUNIT_ASSERT( all_elements(t1 == 2*m1) );
  CPPUNIT_ASSERT( all_elements(t2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(t3 == m1) );
  CPPUNIT_ASSERT( all_elements(t4 == m1) );
}


/*
 * global math operators with matrizes and xpr
 */
template <class T>
void
TestMatrixFunctions<T>::globalMatrixFuncs2() {
  matrix_type r1(m1), r2(m1), r3(m1), r4(m1);
  matrix_type t1(0), t2(0), t3(0), t4(0);

  r1 += T(1)*m1;
  r2 -= T(1)*m1;

  {
    using namespace tvmet::element_wise;

    r3 *= T(1)*m1;
    r4 /= T(1)*m1;
  }

  CPPUNIT_ASSERT( all_elements(r2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(r4 == T(1)) );

  t1 = add(m1, m1*T(1));
  t2 = sub(m1, m1*T(1));

  {
    using namespace tvmet::element_wise;

    t3 = mul(m1, m1*T(1));
    t4 = div(m1, m1*T(1));
  }

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * global math operators with matrizes and xpr
 */
template <class T>
void
TestMatrixFunctions<T>::globalMatrixFuncs3() {
  matrix_type r1(m1), r2(m1), r3(m1), r4(m1);
  matrix_type t1(0), t2(0), t3(0), t4(0);

  r1 += T(1)*m1;
  r2 -= T(1)*m1;

  {
    using namespace tvmet::element_wise;

    r3 *= T(1)*m1;
    r4 /= T(1)*m1;
  }

  CPPUNIT_ASSERT( all_elements(r2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(r4 == T(1)) );

  t1 = add(T(1)*m1, m1);
  t2 = sub(T(1)*m1, m1);

  {
    using namespace tvmet::element_wise;

    t3 = mul(T(1)*m1, m1);
    t4 = div(T(1)*m1, m1);
  }

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
  CPPUNIT_ASSERT( all_elements(t4 == r4) );
}


/*
 * product functions with matrizes
 */
template <class T>
void
TestMatrixFunctions<T>::fn_prod1() {
  matrix_type t1, t2, t3;
  matrix_type r1, r2, r3;

  tvmet::util::Gemm(m1, m1, r1);
  tvmet::util::Gemm(m1, mBig, r2);
  tvmet::util::Gemm(mBig, m1, r3);
  CPPUNIT_ASSERT( all_elements(r2 == r3) );

  t1 = prod(m1, m1);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );

  t2 = prod(m1, mBig);
  CPPUNIT_ASSERT( all_elements(t2 == r2) );

  t3 = prod(mBig, m1);
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
}


/*
 * product functions with matrizes and xpr
 * Note: Take care on aliasing!
 */
template <class T>
void
TestMatrixFunctions<T>::fn_prod2() {
  matrix_type r1(0), rm(0);
  matrix_type m2(m1);
  matrix_type t1;

  rm = scalar*m1;

  tvmet::util::Gemm(m1, rm, r1);

  t1 = prod(m1, scalar*m2 /* alias mBig */);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );
}


/*
 * product functions with matrizes
 * Note: Take care on aliasing!
 */
template <class T>
void
TestMatrixFunctions<T>::fn_prod3() {
  matrix_type r1(0), rm(0);
  matrix_type m2(m1);
  matrix_type t1;

  rm = scalar*m1;

  tvmet::util::Gemm(rm, m1, r1);

  t1 = prod(scalar*m1 /* alias mBig */, m2);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );
}


/*
 * transpose functions with matrizes
 */
template <class T>
void
TestMatrixFunctions<T>::fn_trans() {
  matrix_type t1, t2;

  t1 = trans(m1);
  CPPUNIT_ASSERT( any_elements(t1 != m1) ); // XXX not very clever test

  t2 = trans(t1);	// transpose back
  CPPUNIT_ASSERT( all_elements(t2 == m1) );
}


/*
 * matrix function M^T * M
 */
template <class T>
void
TestMatrixFunctions<T>::fn_MtM_prod() {
  matrix_type m1t, r1;
  matrix_type m2;

  // trans() and prod() is checked before!
  m1t = trans(m1);
  r1  = prod(m1t, mBig);

  m2  = MtM_prod(m1, mBig);

  CPPUNIT_ASSERT( all_elements(r1 == m2) );
}


/*
 * matrix function M * M^T
 */
template <class T>
void
TestMatrixFunctions<T>::fn_MMt_prod() {
  matrix_type m1t, r1;
  matrix_type m2;

  // trans() and prod() is checked before!
  m1t = trans(m1);
  r1  = prod(mBig, m1t);

  m2  = MMt_prod(mBig, m1);

  CPPUNIT_ASSERT( all_elements(r1 == m2) );
}


/*
 * matrix function (M * M)^T
 */
template <class T>
void
TestMatrixFunctions<T>::fn_prodTrans() {
  matrix_type r1, r1t;
  matrix_type m2;

  // trans() and prod() is checked before!
  r1  = prod(m1, mBig);
  r1t = trans(r1);

  m2  = trans_prod(m1, mBig);

  CPPUNIT_ASSERT( all_elements(r1t == m2) );
}


/*
 * trace
 */
template <class T>
void
TestMatrixFunctions<T>::fn_trace() {
  T t1 = trace(m1);
  T t2 = trace(mBig);

  CPPUNIT_ASSERT( t1 == (m1(0,0)+m1(1,1)+m1(2,2)) );
  CPPUNIT_ASSERT( t2 == (mBig(0,0)+mBig(1,1)+mBig(2,2)) );
}


/*
 * matrix row vector
 */
template <class T>
void
TestMatrixFunctions<T>::rowVector() {
  vector_type r0, r1, r2;

  r0 = row(m1, 0);
  r1 = row(m1, 1);
  r2 = row(m1, 2);

  CPPUNIT_ASSERT( all_elements(r0 == m1_r0) );
  CPPUNIT_ASSERT( all_elements(r1 == m1_r1) );
  CPPUNIT_ASSERT( all_elements(r2 == m1_r2) );
}


/*
 * matrix col vector
 */
template <class T>
void
TestMatrixFunctions<T>::colVector() {
  vector_type c0, c1, c2;

  c0 = col(m1, 0);
  c1 = col(m1, 1);
  c2 = col(m1, 2);

  CPPUNIT_ASSERT( all_elements(c0 == m1_c0) );
  CPPUNIT_ASSERT( all_elements(c1 == m1_c1) );
  CPPUNIT_ASSERT( all_elements(c2 == m1_c2) );
}


/*
 * matrix diag vector
 */
template <class T>
void
TestMatrixFunctions<T>::fn_diag() {
  vector_type r, v;

  r = 1, 5, 9;

  v = diag(m1);

  CPPUNIT_ASSERT( all_elements(r == v) );
}


/*
 * extremums
 */
template <class T>
void
TestMatrixFunctions<T>::extremum() {
  CPPUNIT_ASSERT(max(m1) == 9);
  CPPUNIT_ASSERT(min(m1) == 1);

  CPPUNIT_ASSERT(max(mBig) == 90);
  CPPUNIT_ASSERT(min(mBig) == 10);

  CPPUNIT_ASSERT(maximum(m1).value() == 9);
  CPPUNIT_ASSERT(maximum(m1).row() == 2);
  CPPUNIT_ASSERT(maximum(m1).col() == 2);

  CPPUNIT_ASSERT(minimum(m1).value() == 1);
  CPPUNIT_ASSERT(minimum(m1).row() == 0);
  CPPUNIT_ASSERT(minimum(m1).col() == 0);

  CPPUNIT_ASSERT(maximum(mBig).value() == 90);
  CPPUNIT_ASSERT(maximum(mBig).row() == 2);
  CPPUNIT_ASSERT(maximum(mBig).col() == 2);

  CPPUNIT_ASSERT(minimum(mBig).value() == 10);
  CPPUNIT_ASSERT(minimum(mBig).row() == 0);
  CPPUNIT_ASSERT(minimum(mBig).col() == 0);
}


/*
 * identity
 */
template <class T>
void
TestMatrixFunctions<T>::identity_matrix() {
  // XXX strange, why does we have to specify the namespace here?
  // got error: identifier "identity" is undefined
  matrix_type E( tvmet::identity<matrix_type>() );

  CPPUNIT_ASSERT( E(0,0) == 1 &&
		  E(1,1) == 1 &&
		  E(2,2) == 1);

  CPPUNIT_ASSERT( E(0,1) == 0 &&
		  E(0,2) == 0 &&
		  E(1,0) == 0 &&
		  E(1,2) == 0 &&
		  E(2,0) == 0 &&
		  E(2,1) == 0);
}


/*****************************************************************************
 * Implementation Part II (specialized for ints)
 ****************************************************************************/


/*
 * member math operators with scalars
 */
template <>
void
TestMatrixFunctions<int>::scalarUpdAssign1() {
  // all these functions are element wise
  matrix_type t1(m1), t2(m1), t3(m1), t4(mBig);
  matrix_type t5(m1), t6(mBig), t7(mBig), t8(mBig), t9(mBig);
  matrix_type t10(m1), t11(m1);

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

  CPPUNIT_ASSERT(t1(0,0) == (m1(0,0)+scalar) && t1(0,1) == (m1(0,1)+scalar) && t1(0,2) == (m1(0,2)+scalar) &&
		 t1(1,0) == (m1(1,0)+scalar) && t1(1,1) == (m1(1,1)+scalar) && t1(1,2) == (m1(1,2)+scalar) &&
		 t1(2,0) == (m1(2,0)+scalar) && t1(2,1) == (m1(2,1)+scalar) && t1(2,2) == (m1(2,2)+scalar));

  CPPUNIT_ASSERT(t2(0,0) == (m1(0,0)-scalar) && t2(0,1) == (m1(0,1)-scalar) && t2(0,2) == (m1(0,2)-scalar) &&
		 t2(1,0) == (m1(1,0)-scalar) && t2(1,1) == (m1(1,1)-scalar) && t2(1,2) == (m1(1,2)-scalar) &&
		 t2(2,0) == (m1(2,0)-scalar) && t2(2,1) == (m1(2,1)-scalar) && t2(2,2) == (m1(2,2)-scalar));

  CPPUNIT_ASSERT( all_elements(t3 == mBig) );

  CPPUNIT_ASSERT( all_elements(t4 == m1) );

  CPPUNIT_ASSERT( all_elements(t5 == m1) );
  CPPUNIT_ASSERT( all_elements(t6 == mZero) );

  CPPUNIT_ASSERT(t7(0,0) == (mBig(0,0)^scalar) && t7(0,1) == (mBig(0,1)^scalar) && t7(0,2) == (mBig(0,2)^scalar) &&
		 t7(1,0) == (mBig(1,0)^scalar) && t7(1,1) == (mBig(1,1)^scalar) && t7(1,2) == (mBig(1,2)^scalar) &&
		 t7(2,0) == (mBig(2,0)^scalar) && t7(2,1) == (mBig(2,1)^scalar) && t7(2,2) == (mBig(2,2)^scalar));

  CPPUNIT_ASSERT(t8(0,0) == (mBig(0,0)&scalar) && t8(0,1) == (mBig(0,1)&scalar) && t8(0,2) == (mBig(0,2)&scalar) &&
		 t8(1,0) == (mBig(1,0)&scalar) && t8(1,1) == (mBig(1,1)&scalar) && t8(1,2) == (mBig(1,2)&scalar) &&
		 t8(2,0) == (mBig(2,0)&scalar) && t8(2,1) == (mBig(2,1)&scalar) && t8(2,2) == (mBig(2,2)&scalar));

  CPPUNIT_ASSERT(t9(0,0) == (mBig(0,0)|scalar) && t9(0,1) == (mBig(0,1)|scalar) && t9(0,2) == (mBig(0,2)|scalar) &&
		 t9(1,0) == (mBig(1,0)|scalar) && t9(1,1) == (mBig(1,1)|scalar) && t9(1,2) == (mBig(1,2)|scalar) &&
		 t9(2,0) == (mBig(2,0)|scalar) && t9(2,1) == (mBig(2,1)|scalar) && t9(2,2) == (mBig(2,2)|scalar));

  CPPUNIT_ASSERT(t10(0,0) == (m1(0,0)<<scalar) && t10(0,1) == (m1(0,1)<<scalar) && t10(0,2) == (m1(0,2)<<scalar) &&
		 t10(1,0) == (m1(1,0)<<scalar) && t10(1,1) == (m1(1,1)<<scalar) && t10(1,2) == (m1(1,2)<<scalar) &&
		 t10(2,0) == (m1(2,0)<<scalar) && t10(2,1) == (m1(2,1)<<scalar) && t10(2,2) == (m1(2,2)<<scalar));

  CPPUNIT_ASSERT(t11(0,0) == (m1(0,0)>>scalar2) && t11(0,1) == (m1(0,1)>>scalar2) && t11(0,2) == (m1(0,2)>>scalar2) &&
		 t11(1,0) == (m1(1,0)>>scalar2) && t11(1,1) == (m1(1,1)>>scalar2) && t11(1,2) == (m1(1,2)>>scalar2) &&
		 t11(2,0) == (m1(2,0)>>scalar2) && t11(2,1) == (m1(2,1)>>scalar2) && t11(2,2) == (m1(2,2)>>scalar2));
}

/*
 * TODO: implement other UpdAssign functions, esp. for bit ops
 * (since functions above are working, all others should work)
 */


#endif // TVMET_TEST_MATRIXFUNC_H

// Local Variables:
// mode:C++
// End:
