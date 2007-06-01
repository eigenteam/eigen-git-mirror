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
 * $Id: TestConstruction.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_CONSTRUCTION_H
#define TVMET_TEST_CONSTRUCTION_H

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

template <class T>
class TestConstruction : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestConstruction );
  CPPUNIT_TEST( vector_ctor1 );
  CPPUNIT_TEST( vector_ctor2 );
  CPPUNIT_TEST( vector_ctor3 );
  CPPUNIT_TEST( vector_ctor4 );
  CPPUNIT_TEST( vector_ctor5 );
  CPPUNIT_TEST( vector_ctor6 );
  CPPUNIT_TEST( vector_cctor );
  CPPUNIT_TEST( matrix_ctor1 );
  CPPUNIT_TEST( matrix_ctor2 );
  CPPUNIT_TEST( matrix_ctor3 );
  CPPUNIT_TEST( matrix_ctor4 );
  CPPUNIT_TEST( matrix_ctor5 );
  CPPUNIT_TEST( matrix_cctor );
  CPPUNIT_TEST_SUITE_END();

private:
  enum { dim = 3 };
  typedef tvmet::Vector<T, dim>			vector_type;
  typedef tvmet::Matrix<T, dim, dim>		matrix_type;

public:
  TestConstruction()
    : vZero(0), vOne(1), mZero(0), mOne(1) { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void vector_ctor1();
  void vector_ctor2();
  void vector_ctor3();
  void vector_ctor4();
  void vector_ctor5();
  void vector_ctor6();
  void vector_cctor();

  void matrix_ctor1();
  void matrix_ctor2();
  void matrix_ctor3();
  void matrix_ctor4();
  void matrix_ctor5();
  void matrix_cctor();

private:
  const vector_type vZero;
  const vector_type vOne;
  vector_type v1;

private:
  const matrix_type mZero;
  const matrix_type mOne;
  matrix_type m1;
};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestConstruction<T>::setUp () {
  v1 = 1,2,3;

  m1 = 1,4,7,
       2,5,8,
       3,6,9;
}

template <class T>
void TestConstruction<T>::tearDown() { }

/*****************************************************************************
 * Implementation Part II (Vectors)
 ****************************************************************************/

/*
 * Vector (InputIterator first, InputIterator last)
 */
template <class T>
void
TestConstruction<T>::vector_ctor1() {
  T data[] = {1,2,3};

  int sz = sizeof(data)/sizeof(T);
  T* first = data;
  T* last = data + sz;

  vector_type v(first, last);

  CPPUNIT_ASSERT( all_elements(v == v1) );
}

/*
 * Vector (InputIterator first, int sz)
 */
template <class T>
void
TestConstruction<T>::vector_ctor2() {
  T data[] = {1,2,3};

  int sz = sizeof(data)/sizeof(T);
  T* first = data;

  vector_type v(first, sz);

  CPPUNIT_ASSERT( all_elements(v == v1) );
}

/*
 * Vector (value_type rhs)
 */
template <class T>
void
TestConstruction<T>::vector_ctor3() {

  vector_type one(static_cast<T>(1.0));
  vector_type zero(static_cast<T>(0.0));

  CPPUNIT_ASSERT( all_elements(one  == vOne) );
  CPPUNIT_ASSERT( all_elements(zero == vZero) );
}

/*
 * Vector (value_type x0, value_type x1, value_type x2)
 * TODO: check for other length too.
 */
template <class T>
void
TestConstruction<T>::vector_ctor4() {
  vector_type v(1,2,3);

  CPPUNIT_ASSERT( all_elements(v == v1) );
}

/*
 * Vector (XprVector< E, Sz > expr)
 * Note: a little bit dangerous, since we haven't check expr yet.
 */
template <class T>
void
TestConstruction<T>::vector_ctor5() {
  vector_type v(v1 - v1);

  CPPUNIT_ASSERT( all_elements(v == vZero) );
}

/*
 * operator=(const Vector< T2, Sz > &)
 */
template <class T>
void
TestConstruction<T>::vector_ctor6() {
  vector_type v;
  v = v1;

  CPPUNIT_ASSERT( all_elements(v == v1) );
}

/*
 * Vector (const this_type &rhs)
 */
template <class T>
void
TestConstruction<T>::vector_cctor() {
  vector_type v(v1);

  CPPUNIT_ASSERT( all_elements(v == v1) );
}


/*****************************************************************************
 * Implementation Part III (Matrizes)
 ****************************************************************************/

/*
 * Matrix (InputIterator first, InputIterator last)
 */
template <class T>
void
TestConstruction<T>::matrix_ctor1() {
  T data[] = { 1,4,7,
	       2,5,8,
	       3,6,9 };

  int sz = sizeof(data)/sizeof(T);
  T* first = data;
  T* last = data + sz;

  matrix_type m(first, last);

  CPPUNIT_ASSERT( all_elements(m == m1) );
}

/*
 * Matrix (InputIterator first, int sz)
 */
template <class T>
void
TestConstruction<T>::matrix_ctor2() {
  T data[] = { 1,4,7,
	       2,5,8,
	       3,6,9 };

  int sz = sizeof(data)/sizeof(T);
  T* first = data;

  matrix_type m(first, sz);

  CPPUNIT_ASSERT( all_elements(m == m1) );
}

/*
 * Matrix (value_type rhs)
 */
template <class T>
void
TestConstruction<T>::matrix_ctor3() {
  matrix_type one(static_cast<T>(1.0));
  matrix_type zero(static_cast<T>(0.0));

  CPPUNIT_ASSERT( all_elements(one  == mOne) );
  CPPUNIT_ASSERT( all_elements(zero == mZero) );
}

/*
 * Matrix (XprMatrix< E, Rows, Cols > expr)
 * Note: a little bit dangerous, since we haven't check expr yet.
 */
template <class T>
void
TestConstruction<T>::matrix_ctor4() {
  matrix_type m(m1 - m1);

  CPPUNIT_ASSERT( all_elements(m == mZero) );
}

/*
 * operator= (value_type rhs)
 */
template <class T>
void
TestConstruction<T>::matrix_ctor5() {
  matrix_type m;
  m = m1;

  CPPUNIT_ASSERT( all_elements(m == m1) );
}

/*
 * Matrix (const this_type &rhs)
 */
template <class T>
void
TestConstruction<T>::matrix_cctor() {
  matrix_type m(m1);

  CPPUNIT_ASSERT( all_elements(m == m1) );
}

#endif // TVMET_TEST_CONSTRUCTION_H

// Local Variables:
// mode:C++
// End:
