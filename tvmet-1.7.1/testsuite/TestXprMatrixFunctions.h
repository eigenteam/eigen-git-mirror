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
 * $Id: TestXprMatrixFunctions.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_XPR_MATRIXFUNC_H
#define TVMET_TEST_XPR_MATRIXFUNC_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>
#include <tvmet/util/General.h>

template <class T>
class TestXprMatrixFunctions : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestXprMatrixFunctions );
  CPPUNIT_TEST( scalarFuncs1 );
  CPPUNIT_TEST( scalarFuncs2 );
  CPPUNIT_TEST( globalXprMatrixFuncs1 );
  CPPUNIT_TEST( globalXprMatrixFuncs2 );
  CPPUNIT_TEST( globalXprMatrixFuncs3 );
  CPPUNIT_TEST( fn_prod1 );
  CPPUNIT_TEST( fn_prod2 );
  CPPUNIT_TEST( fn_prod3 );
  CPPUNIT_TEST( fn_trans );
  CPPUNIT_TEST( fn_MtM_prod );
  CPPUNIT_TEST( fn_MMt_prod );
  CPPUNIT_TEST( fn_prodTrans );
  CPPUNIT_TEST( fn_trace );
  CPPUNIT_TEST( rowVector1 );
  CPPUNIT_TEST( rowVector2 );
  CPPUNIT_TEST( colVector1 );
  CPPUNIT_TEST( colVector2 );
  CPPUNIT_TEST( fn_diag1 );
  CPPUNIT_TEST( fn_diag2 );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  TestXprMatrixFunctions()
    : mZero(0), mOne(1), scalar(10), scalar2(2) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void scalarFuncs1();
  void scalarFuncs2();
  void globalXprMatrixFuncs1();
  void globalXprMatrixFuncs2();
  void globalXprMatrixFuncs3();
  void fn_prod1();
  void fn_prod2();
  void fn_prod3();
  void fn_trans();
  void fn_MtM_prod();
  void fn_MMt_prod();
  void fn_prodTrans();
  void fn_trace();
  void rowVector1();
  void rowVector2();
  void colVector1();
  void colVector2();
  void fn_diag1();
  void fn_diag2();

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
void TestXprMatrixFunctions<T>::setUp() {
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
void TestXprMatrixFunctions<T>::tearDown() { }


/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


/*
 * global math operators with scalars
 * Note: checked against member operators since they are allready checked
 */
template <class T>
void
TestXprMatrixFunctions<T>::scalarFuncs1() {
  matrix_type r1(m1), r2(m1);
  matrix_type t1(0), t2(0), t3(0), t4(0);

  r1 += scalar;
  r2 -= scalar;

  t1 = add(T(1)*m1, scalar);
  t2 = sub(T(1)*m1, scalar);
  t3 = mul(T(1)*m1, scalar);
  t4 = div(T(1)*mBig, scalar);

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
TestXprMatrixFunctions<T>::scalarFuncs2() {
  matrix_type r1(m1), r2(m1);
  matrix_type t1(0), t2(0);

  r1 += scalar;
  r2 *= scalar;

  t1 = add(scalar, T(1)*m1);
  t2 = mul(scalar, T(1)*m1);

  CPPUNIT_ASSERT( all_elements(t1 == r1) );
  CPPUNIT_ASSERT( all_elements(t2 == r2) );
}


/*
 * global math operators with matrizes
 */
template <class T>
void
TestXprMatrixFunctions<T>::globalXprMatrixFuncs1() {
  matrix_type t1(0), t2(0), t3(0), t4(0);
  matrix_type m2(m1);

  t1 = add(T(1)*m1, T(1)*m2);
  t2 = sub(T(1)*m1, T(1)*m2);

  {
    using namespace tvmet::element_wise;

    t3 = mul(T(1)*m1, T(1)*mOne);
    t4 = div(T(1)*m1, T(1)*mOne);
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
TestXprMatrixFunctions<T>::globalXprMatrixFuncs2() {
  matrix_type r1(m1), r2(m1), r3(m1), r4(m1);
  matrix_type t1(0), t2(0), t3(0), t4(0);
  matrix_type m2(m1);

  r1 += T(1)*m1;
  r2 -= T(1)*m1;

  {
    using namespace tvmet::element_wise;

    r3 *= T(1)*m1;
    r4 /= T(1)*m1;
  }

  CPPUNIT_ASSERT( all_elements(r2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(r4 == T(1)) );

  t1 = add(T(1)*m1, m2*T(1));
  t2 = sub(T(1)*m1, m2*T(1));

  {
    using namespace tvmet::element_wise;

    t3 = mul(T(1)*m1, m2*T(1));
    t4 = div(T(1)*m1, m2*T(1));
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
TestXprMatrixFunctions<T>::globalXprMatrixFuncs3() {
  matrix_type r1(m1), r2(m1), r3(m1), r4(m1);
  matrix_type t1(0), t2(0), t3(0), t4(0);
  matrix_type m2(m1);

  r1 += T(1)*m1;
  r2 -= T(1)*m1;

  {
    using namespace tvmet::element_wise;

    r3 *= T(1)*m1;
    r4 /= T(1)*m1;
  }

  CPPUNIT_ASSERT( all_elements(r2 == T(0)) );
  CPPUNIT_ASSERT( all_elements(r4 == T(1)) );

  t1 = add(T(1)*m1, m2*T(1));
  t2 = sub(T(1)*m1, m2*T(1));

  {
    using namespace tvmet::element_wise;

    t3 = mul(T(1)*m1, m2*T(1));
    t4 = div(T(1)*m1, m2*T(1));
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
TestXprMatrixFunctions<T>::fn_prod1() {
  matrix_type t1, t2, t3;
  matrix_type r1, r2, r3;
  matrix_type m2(m1);

  tvmet::util::Gemm(m1, m1, r1);
  tvmet::util::Gemm(m1, mBig, r2);
  tvmet::util::Gemm(mBig, m1, r3);
  CPPUNIT_ASSERT( all_elements(r2 == r3) );

  t1 = prod(T(1)*m1, T(1)*m2);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );

  t2 = prod(T(1)*m1, T(1)*mBig);
  CPPUNIT_ASSERT( all_elements(t2 == r2) );

  t3 = prod(T(1)*mBig, T(1)*m1);
  CPPUNIT_ASSERT( all_elements(t3 == r3) );
}


/*
 * product functions with matrizes and xpr
 * Note: Take care on aliasing!
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_prod2() {
  matrix_type r1(0), rm(0);
  matrix_type m2(m1);
  matrix_type t1;

  rm = scalar*m1;

  tvmet::util::Gemm(m1, rm, r1);

  t1 = prod(T(1)*m1, scalar*m2 /* alias mBig */);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );
}


/*
 * product functions with matrizes
 * Note: Take care on aliasing!
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_prod3() {
  matrix_type r1(0), rm(0);
  matrix_type m2(m1);
  matrix_type t1;

  rm = scalar*m1;

  tvmet::util::Gemm(rm, m1, r1);

  t1 = prod(scalar*m1 /* alias mBig */, T(1)*m2);
  CPPUNIT_ASSERT( all_elements(t1 == r1) );
}


/*
 * transpose functions with matrizes
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_trans() {
  matrix_type t1, t2;

  t1 = trans(T(1)*m1);
  CPPUNIT_ASSERT( any_elements(t1 == m1) ); // XXX not very clever

  t2 = trans(T(1)*t1);	// transpose back
  CPPUNIT_ASSERT( all_elements(t2 == m1) );
}


/*
 * matrix function M^T * M
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_MtM_prod() {
  matrix_type m1t, r1;
  matrix_type m2;

  // trans() and prod() is checked before!
  m1t = trans(m1);
  r1  = prod(m1t, mBig);

  m2  = MtM_prod(T(1)*m1, T(1)*mBig);

  CPPUNIT_ASSERT( all_elements(r1 == m2) );
}


/*
 * matrix function M * M^T
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_MMt_prod() {
  matrix_type m1t, r1;
  matrix_type m2;

  // trans() and prod() is checked before!
  m1t = trans(m1);
  r1  = prod(mBig, m1t);

  m2  = MMt_prod(T(1)*mBig, T(1)*m1);

  CPPUNIT_ASSERT( all_elements(r1 == m2) );
}


/*
 * matrix function (M * M)^T
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_prodTrans() {
  matrix_type r1, r1t;
  matrix_type m2;

  // trans() and prod() is checked before!
  r1  = prod(m1, mBig);
  r1t = trans(r1);

  m2  = trans_prod(T(1)*m1, T(1)*mBig);

  CPPUNIT_ASSERT( all_elements(r1t == m2) );
}


/*
 * trace
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_trace() {
//   declaration on trace not yet.
//   T t1 = trace(T(1)*m1);
//   T t2 = trace(T(1)*mBig);

//   CPPUNIT_ASSERT( t1 == (m1(0,0)+m1(1,1)+m1(2,2)) );
//   CPPUNIT_ASSERT( t2 == (mBig(0,0)+mBig(1,1)+mBig(2,2)) );
}


/*
 * matrix row vector I
 */
template <class T>
void
TestXprMatrixFunctions<T>::rowVector1() {
  vector_type r0, r1, r2;

   r0 =  row(m1+m1, 0);
   r1 =  row(m1+m1, 1);
   r2 =  row(m1+m1, 2);

  CPPUNIT_ASSERT( all_elements(r0 == 2*m1_r0) );
  CPPUNIT_ASSERT( all_elements(r1 == 2*m1_r1) );
  CPPUNIT_ASSERT( all_elements(r2 == 2*m1_r2) );
}


/*
 * matrix row vector II
 * g++ produce wrong results only for row0
 */
template <class T>
void
TestXprMatrixFunctions<T>::rowVector2() {
  vector_type r0, r1, r2;

   r0 =  row(T(1)*m1, 0);
   r1 =  row(T(1)*m1, 1);
   r2 =  row(T(1)*m1, 2);

  CPPUNIT_ASSERT( all_elements(r0 == m1_r0) );
  CPPUNIT_ASSERT( all_elements(r1 == m1_r1) );
  CPPUNIT_ASSERT( all_elements(r2 == m1_r2) );
}


/*
 * matrix col vector I
 */
template <class T>
void
TestXprMatrixFunctions<T>::colVector1() {
  vector_type c0, c1, c2;

  c0 = col(m1+m1, 0);
  c1 = col(m1+m1, 1);
  c2 = col(m1+m1, 2);

  CPPUNIT_ASSERT( all_elements(c0 == 2*m1_c0) );
  CPPUNIT_ASSERT( all_elements(c1 == 2*m1_c1) );
  CPPUNIT_ASSERT( all_elements(c2 == 2*m1_c2) );
}


/*
 * matrix col vector II
 * g++ produce wrong results only for col0
 */
template <class T>
void
TestXprMatrixFunctions<T>::colVector2() {
  vector_type c0, c1, c2;

  c0 = col(T(1)*m1, 0);
  c1 = col(T(1)*m1, 1);
  c2 = col(T(1)*m1, 2);

  CPPUNIT_ASSERT( all_elements(c0 == m1_c0) );
  CPPUNIT_ASSERT( all_elements(c1 == m1_c1) );
  CPPUNIT_ASSERT( all_elements(c2 == m1_c2) );
}


/*
 * matrix diag vector I
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_diag1() {
  vector_type r, v;

  r = 2*diag(m1);

  v = diag(m1+m1);

  CPPUNIT_ASSERT( all_elements(r == v) );
}


/*
 * matrix diag vector II
 * g++ produce wrong results opposite to diag1
 */
template <class T>
void
TestXprMatrixFunctions<T>::fn_diag2() {
  vector_type r, v;

  r = diag(m1);

  v = diag(T(1)*m1);

  CPPUNIT_ASSERT( all_elements(r == v) );
}


#endif // TVMET_TEST_XPR_MATRIXFUNC_H


// Local Variables:
// mode:C++
// End:
