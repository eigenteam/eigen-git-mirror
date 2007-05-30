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
 * $Id: TestXpr.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_XPR_H
#define TVMET_TEST_XPR_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>
#include <tvmet/util/General.h>

template <class T>
class TestXpr : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestXpr );
  CPPUNIT_TEST( fn_MMProd );
  CPPUNIT_TEST( fn_MMMProd );
  CPPUNIT_TEST( fn_MVProd );
  CPPUNIT_TEST( op_MMProd );
  CPPUNIT_TEST( op_MMMProd );
  CPPUNIT_TEST( op_MVProd );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  TestXpr()
    : vZero(0), vOne(1), mZero(0), mOne(1), scalar(10) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void fn_MMProd();
  void fn_MMMProd();
  void fn_MVProd();
  void op_MMProd();
  void op_MMMProd();
  void op_MVProd();

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
  const T scalar;
};

/*****************************************************************************
 * Implementation part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestXpr<T>::setUp() {
  v1 = 1,2,3;
  v1b = v1;
  vBig = 10,20,30;

  m1 = 1,4,7,
       2,5,8,
       3,6,9;
  m1b = m1;
  mBig = 10,40,70,
         20,50,80,
         30,60,90;
}

template <class T>
void TestXpr<T>::tearDown() { }

/*****************************************************************************
 * Implementation part II
 ****************************************************************************/

/*
 * XprMatrix - XprMatrix
 */
template <class T>
void
TestXpr<T>::fn_MMProd() {
  matrix_type mr1(0), mr2(0), mr(0);
  matrix_type m;

  tvmet::util::Gemm(m1, mOne, mr1);
  tvmet::util::Gemm(m1, mOne, mr2);
  tvmet::util::Gemm(mr1, mr2, mr);

  // XprMatrix * XprMatrix
  m = prod(prod(m1,mOne), prod(m1,mOne));

  CPPUNIT_ASSERT( all_elements(mr == m) );
}

/*
 * Matrix - XprMatrix - XprMatrix
 */
template <class T>
void
TestXpr<T>::fn_MMMProd() {
  matrix_type m;
  matrix_type rhs(0), r(0);

  tvmet::util::Gemm(m1, m1, rhs);
  tvmet::util::Gemm(m1, rhs, r);

  // Matrix * XprMatrix * XprMatrix
  m = prod(m1, prod(m1, m1));

  CPPUNIT_ASSERT( all_elements(r == m) );
}

/*
 * XprMatrix - XprVector
 */
template <class T>
void
TestXpr<T>::fn_MVProd() {
  matrix_type mr1(0);
  vector_type vr(0), v;

  tvmet::util::Gemm(m1, mOne, mr1);
  tvmet::util::Gemv(mr1, v1, vr);

  // XprMatrix * XprVector
  v = prod(prod(m1,mOne), mul(v1,vOne));

  CPPUNIT_ASSERT( all_elements(vr == v) );
}

/*
 * XprMatrix - XprMatrix
 */
template <class T>
void
TestXpr<T>::op_MMProd() {
  matrix_type mr1(0), mr2(0), mr(0), m;

  tvmet::util::Gemm(m1, mOne, mr1);
  tvmet::util::Gemm(m1, mOne, mr2);
  tvmet::util::Gemm(mr1, mr2, mr);

  // XprMatrix * XprMatrix
  m = (m1*mOne)*(m1*mOne);

  CPPUNIT_ASSERT( all_elements(mr == m) );
}

/*
 * Matrix - XprMatrix - XprMatrix
 */
template <class T>
void
TestXpr<T>::op_MMMProd() {
  matrix_type m;
  matrix_type rhs(0), r(0);

  tvmet::util::Gemm(m1, m1, rhs);
  tvmet::util::Gemm(m1, rhs, r);

  // Matrix * XprMatrix * XprMatrix
  m = m1 * m1 * m1;

  CPPUNIT_ASSERT( all_elements(r == m) );
}

/*
 * XprMatrix - XprVector
 */
template <class T>
void
TestXpr<T>::op_MVProd() {
  matrix_type mr1(0);
  vector_type vr(0), v;

  tvmet::util::Gemm(m1, mOne, mr1);
  tvmet::util::Gemv(mr1, v1, vr);

  // XprMatrix * XprVector
  v = (m1*mOne)*(v1*vOne);

  CPPUNIT_ASSERT( all_elements(vr == v) );
}

#endif // TVMET_TEST_XPR_H

// Local Variables:
// mode:C++
// End:
