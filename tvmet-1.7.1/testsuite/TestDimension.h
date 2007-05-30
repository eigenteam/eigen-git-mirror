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
 * $Id: TestDimension.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_DIMENSION_H
#define TVMET_TEST_DIMENSION_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

template <class T>
class TestDimension : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestDimension );
  CPPUNIT_TEST( small_add );
  CPPUNIT_TEST( Mtx );
  CPPUNIT_TEST( MtM );
  CPPUNIT_TEST( MMt );
  CPPUNIT_TEST( MMt );
  CPPUNIT_TEST( trans_MM );
  CPPUNIT_TEST( Row );
  CPPUNIT_TEST( Col );
  CPPUNIT_TEST_SUITE_END();

public:
  TestDimension() { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void small_add();
  void Mtx();
  void MtM();
  void MMt();
  void trans_MM();
  void Row();
  void Col();
};

/*****************************************************************************
 * Implementation part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestDimension<T>::setUp() { }

template <class T>
void TestDimension<T>::tearDown() { }

/*****************************************************************************
 * Implementation part II
 ****************************************************************************/

template <class T>
void TestDimension<T>::small_add() {
  using namespace tvmet;

  Matrix<double, 5, 3>		M1, M2, M3;

  M1 =
    1,1,1,
    1,1,1,
    1,1,1,
    1,1,1,
    1,1,1;
  M2 = M1;

  M3 = M1 + M2;

  CPPUNIT_ASSERT( all_elements(M3 == 2) );
}


template <class T>
void TestDimension<T>::Mtx() {
  using namespace tvmet;

  Matrix<double, 6, 3>		M1;

  Vector<double, 6> 		v1;
  Vector<double, 3> 		r(0), v2(0);

  M1 =
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10,11,12,
    13,14,15,
    16,17,18;
  v1 = 1,2,3,4,5,6;

  r = trans(M1)*v1;

  v2 = Mtx_prod(M1, v1);

  CPPUNIT_ASSERT( all_elements(r == v2) );
}


template <class T>
void TestDimension<T>::MtM() {
  using namespace tvmet;

  Matrix<double, 6, 3>		M1;
  Matrix<double, 6, 2>  	M2;
  Matrix<double, 3, 2>		r(0), M3(0);

  M1 =
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10,11,12,
    13,14,15,
    16,17,18;
  M2 =
    1, 2,
    3, 4,
    5, 6,
    7, 8,
    9,10,
    11,12;

  r = prod(trans(M1),M2);

  M3 = MtM_prod(M1, M2);

  CPPUNIT_ASSERT( all_elements(r == M3) );
}


template <class T>
void TestDimension<T>::MMt() {
  using namespace tvmet;

  Matrix<double, 3, 4>		M1;
  Matrix<double, 2, 4>  	M2;
  Matrix<double, 3, 2>		M3(0), r(0);

  M1 =
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10,11,12;
  M2 =
    1,2,3,4,
    5,6,7,8;

  r = M1*trans(M2);

  M3 = MMt_prod(M1,M2);

  CPPUNIT_ASSERT( all_elements(r == M3) );
}


template <class T>
void TestDimension<T>::trans_MM() {
  using namespace tvmet;

  Matrix<double, 6, 3>		M1;
  Matrix<double, 3, 6>  	M2;
  Matrix<double, 6, 6>		r(0), M3(0);

  M1 =
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10,11,12,
    13,14,15,
    16,17,18;
  M2 =
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10,11,12,
    13,14,15,16,17,18;

  r = trans(prod(M1, M2));

  M3 = trans_prod(M1, M2);

  CPPUNIT_ASSERT( all_elements(r == M3) );
}


template <class T>
void TestDimension<T>::Row() {
  using namespace tvmet;

  Matrix<double, 6, 3>		M;
  Vector<double, 3>		v;
  Vector<double, 3>		r0(1,2,3);
  Vector<double, 3>		r5(16,17,18);

  M =
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10,11,12,
    13,14,15,
    16,17,18;

  v = row(M, 0);
  CPPUNIT_ASSERT( all_elements(v == r0) );

  v = row(M, 5);
  CPPUNIT_ASSERT( all_elements(v == r5) );
}


template <class T>
void TestDimension<T>::Col() {
  using namespace tvmet;

  Matrix<double, 3, 6>		M;
  Vector<double, 3>		v;
  Vector<double, 3>		c0(1,7,13);
  Vector<double, 3>		c5(6,12,18);

  M =
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10,11,12,
    13,14,15,16,17,18;

  v = col(M, 0);
  CPPUNIT_ASSERT( all_elements(v == c0) );

  v = col(M, 5);
  CPPUNIT_ASSERT( all_elements(v == c5) );
}

#endif // TVMET_TEST_DIMENSION_H

// Local Variables:
// mode:C++
// End:
