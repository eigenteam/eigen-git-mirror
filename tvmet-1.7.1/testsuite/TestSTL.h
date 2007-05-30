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
 * $Id: TestSTL.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_STL_H
#define TVMET_TEST_STL_H

#include <vector>
#include <algorithm>

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

/**
 * gernell test case
 */
template <class T>
class TestSTL : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestSTL );
  CPPUNIT_TEST( vector_ctor );
  CPPUNIT_TEST( vector_copy );
  CPPUNIT_TEST( matrix_ctor );
  CPPUNIT_TEST( matrix_copy );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

  typedef std::vector<T>			stlvec;

public:
  TestSTL() { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void vector_ctor();
  void vector_copy();

  void matrix_ctor();
  void matrix_copy();

private:
  stlvec					stl_v1;
  stlvec					stl_v2;
  vector_type					v1;
  matrix_type					m1;
};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 *** *************************************************************************/

template <class T>
void TestSTL<T>::setUp() {
  stl_v1.push_back(static_cast<T>(1));
  stl_v1.push_back(static_cast<T>(2));
  stl_v1.push_back(static_cast<T>(3));

  stl_v2.push_back(static_cast<T>(1));
  stl_v2.push_back(static_cast<T>(2));
  stl_v2.push_back(static_cast<T>(3));
  stl_v2.push_back(static_cast<T>(4));
  stl_v2.push_back(static_cast<T>(5));
  stl_v2.push_back(static_cast<T>(6));
  stl_v2.push_back(static_cast<T>(7));
  stl_v2.push_back(static_cast<T>(8));
  stl_v2.push_back(static_cast<T>(9));
}


template <class T>
void TestSTL<T>::tearDown() {

}


/*****************************************************************************
 * Implementation Part II (vector)
 ****************************************************************************/


template <class T>
void
TestSTL<T>::vector_ctor() {
  vector_type v(stl_v1.begin(), stl_v1.end());

  CPPUNIT_ASSERT( std::equal(stl_v1.begin(), stl_v1.end(), v.begin()) == true );
}


template <class T>
void
TestSTL<T>::vector_copy() {
  vector_type v;

  std::copy(stl_v1.begin(), stl_v1.end(), v.begin());

  CPPUNIT_ASSERT( std::equal(stl_v1.begin(), stl_v1.end(), v.begin()) == true );
}


/*****************************************************************************
 * Implementation Part II (matrix)
 ****************************************************************************/


template <class T>
void
TestSTL<T>::matrix_ctor() {
  matrix_type m(stl_v2.begin(), stl_v2.end());

  CPPUNIT_ASSERT( std::equal(stl_v2.begin(), stl_v2.end(), m.begin()) == true );
}


template <class T>
void
TestSTL<T>::matrix_copy() {
  matrix_type m;

  std::copy(stl_v2.begin(), stl_v2.end(), m.begin());

  CPPUNIT_ASSERT( std::equal(stl_v2.begin(), stl_v2.end(), m.begin()) == true );
}


#endif // TVMET_TEST_STL_H

// Local Variables:
// mode:C++
// End:
