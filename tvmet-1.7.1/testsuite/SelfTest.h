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
 * $Id: SelfTest.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_SELFTEST_H
#define TVMET_SELFTEST_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

#include <cassert>

template <class T>
class SelfTest : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( SelfTest );
  CPPUNIT_TEST( basics1 );
  CPPUNIT_TEST( basics2 );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  SelfTest()
    : vZero(0), vOne(1), mZero(0), mOne(1) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void basics1();
  void basics2();

private: // vectors
  const vector_type vZero;
  const vector_type vOne;
  vector_type v1;

private: // matrizes
  const matrix_type mZero;
  const matrix_type mOne;
  matrix_type m1;
};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void SelfTest<T>::setUp () {
  v1 = 1,2,3;

  m1 = 1,4,7,
       2,5,8,
       3,6,9;
}

template <class T>
void SelfTest<T>::tearDown() { }

/*****************************************************************************
 * Implementation Part II
 * these are elemental - therefore we use std::assert
 ****************************************************************************/

/*
 * We have to guarantee that the vectors and matrizes these are what
 * they should be. E.g. vOne: all elements should be equal to 1.
 * Implicit we check on right construction here due to the way of
 * construction.
 */
template <class T>
void
SelfTest<T>::basics1()
{
  // all elements of vZero have to be equal to zero:
  assert(vZero(0) == T(0) && vZero(1) == T(0) && vZero(2) == T(0));

  // all elements of vOne have to be equal to 1:
  assert(vOne(0) == T(1) && vOne(1) == T(1) && vOne(2) == T(1));

  // all elements of mZero have to be equal to zero:
  assert(mZero(0,0) == T(0) && mZero(0,1) == T(0) && mZero(0,2) == T(0) &&
	 mZero(1,0) == T(0) && mZero(1,1) == T(0) && mZero(1,2) == T(0) &&
	 mZero(2,0) == T(0) && mZero(2,1) == T(0) && mZero(2,2) == T(0));

  // all elements of mOne have to be equal to 1:
  assert(mOne(0,0) == T(1) && mOne(0,1) == T(1) && mOne(0,2) == T(1) &&
	 mOne(1,0) == T(1) && mOne(1,1) == T(1) && mOne(1,2) == T(1) &&
	 mOne(2,0) == T(1) && mOne(2,1) == T(1) && mOne(2,2) == T(1));
}


/*
 * We have to guarantee that the vectors and matrizes these are what
 * they should be. E.g. v1: all elements should increase by 1.
 * Implicit we check the CommaInitializer due to the way of
 * construction.
 */
template <class T>
void
SelfTest<T>::basics2()
{
  // all elements of v1 should increase
  assert(v1(0) == T(1) && v1(1) == T(2) && v1(2) == T(3));

  // all elements of m1 should increase column-wise
  assert(m1(0,0) == T(1) && m1(0,1) == T(4) && m1(0,2) == T(7) &&
	 m1(1,0) == T(2) && m1(1,1) == T(5) && m1(1,2) == T(8) &&
	 m1(2,0) == T(3) && m1(2,1) == T(6) && m1(2,2) == T(9));
}

#endif // TVMET_SELFTEST_H

// Local Variables:
// mode:C++
// End:
