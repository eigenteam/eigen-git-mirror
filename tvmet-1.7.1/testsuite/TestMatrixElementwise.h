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
 * $Id: TestMatrixElementwise.h,v 1.2 2005/03/09 11:11:53 opetzold Exp $
 */

#ifndef TVMET_TEST_MATRIX_ELEMENTWISE_H
#define TVMET_TEST_MATRIX_ELEMENTWISE_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

template <class T>
class TestMatrixElementwise : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestMatrixElementwise );
  CPPUNIT_TEST( sqr_add );
  CPPUNIT_TEST( sqr_xpr_add );
  CPPUNIT_TEST( sqr_mul );
  CPPUNIT_TEST( sqr_xpr_mul );
  CPPUNIT_TEST( nsqr_add );
  CPPUNIT_TEST( nsqr_xpr_add );
  CPPUNIT_TEST( nsqr_mul );
  CPPUNIT_TEST( nsqr_xpr_mul );
  CPPUNIT_TEST_SUITE_END();

public:
  TestMatrixElementwise() { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void sqr_add();
  void sqr_xpr_add();
  void sqr_mul();
  void sqr_xpr_mul();

  void nsqr_add();
  void nsqr_xpr_add();
  void nsqr_mul();
  void nsqr_xpr_mul();
};

/*****************************************************************************
 * Implementation part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestMatrixElementwise<T>::setUp() { }

template <class T>
void TestMatrixElementwise<T>::tearDown() { }

/*****************************************************************************
 * Implementation part II, square matrices
 ****************************************************************************/

template <class T>
void TestMatrixElementwise<T>::sqr_add() {
  using namespace tvmet;

  Matrix<T, 3, 3>		M1, M2, Mr1, Mr2;

  M1 = 2;
  M2 = 2;

  Mr1 = M1 + M2;
  Mr2 = add(M1, M1);

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


template <class T>
void TestMatrixElementwise<T>::sqr_xpr_add() {
  using namespace tvmet;

  Matrix<T, 3, 3>		M1, M2, Mr1, Mr2;

  M1 = 1;
  M2 = 1;

  T c1 = 1;
  T c2 = 1;

  Mr1 = (c1+M1) + (c2+M2);
  Mr2 = add(add(c1,M1), add(c2,M1));

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


template <class T>
void TestMatrixElementwise<T>::sqr_mul() {
  using namespace tvmet;

  Matrix<T, 3, 3>		M1, M2, Mr1, Mr2;

  M1 = 2;
  M2 = 2;

  Mr1 = element_wise::operator*(M1, M2);
  Mr2 = element_wise::mul(M1, M2);

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


template <class T>
void TestMatrixElementwise<T>::sqr_xpr_mul() {
  using namespace tvmet;

  Matrix<T, 3, 3>		M1, M2, Mr1, Mr2;

  M1 = 2;
  M2 = 2;

  T c1 = 1;
  T c2 = 1;

  Mr1 = element_wise::operator*(c1*M1, c2*M2);
  Mr2 = element_wise::mul(mul(c1, M1), mul(c2, M2));

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


/*****************************************************************************
 * Implementation part II, non square matrices
 ****************************************************************************/

template <class T>
void TestMatrixElementwise<T>::nsqr_add() {
  using namespace tvmet;

  Matrix<T, 4, 3>		M1, M2, Mr1, Mr2;

  M1 = 2;
  M2 = 2;

  Mr1 = M1 + M2;
  Mr2 = add(M1, M1);

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


template <class T>
void TestMatrixElementwise<T>::nsqr_xpr_add() {
  using namespace tvmet;

  Matrix<T, 4, 3>		M1, M2, Mr1, Mr2;

  M1 = 1;
  M2 = 1;

  T c1 = 1;
  T c2 = 1;

  Mr1 = (c1+M1) + (c2+M2);
  Mr2 = add(add(c1,M1), add(c2,M1));

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


template <class T>
void TestMatrixElementwise<T>::nsqr_mul() {
  using namespace tvmet;

  Matrix<T, 4, 3>		M1, M2, Mr1, Mr2;

  M1 = 2;
  M2 = 2;

  Mr1 = element_wise::operator*(M1, M2);
  Mr2 = element_wise::mul(M1, M2);

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


template <class T>
void TestMatrixElementwise<T>::nsqr_xpr_mul() {
  using namespace tvmet;

  Matrix<T, 4, 3>		M1, M2, Mr1, Mr2;

  M1 = 2;
  M2 = 2;

  T c1 = 1;
  T c2 = 1;

  Mr1 = element_wise::operator*(c1*M1, c2*M2);
  Mr2 = element_wise::mul(mul(c1, M1), mul(c2, M2));

  CPPUNIT_ASSERT( all_elements(Mr1 == 4) );
  CPPUNIT_ASSERT( all_elements(Mr2 == 4) );
}


#endif // TVMET_TEST_MATRIX_ELEMENTWISE_H

// Local Variables:
// mode:C++
// End:
