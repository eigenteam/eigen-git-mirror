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
 * $Id: TestComplexMatrix.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TEST_COMPLEX_MATRIX_H
#define TEST_COMPLEX_MATRIX_H

#include <limits>
#include <algorithm>
#include <complex>

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Matrix.h>
#include <tvmet/util/General.h>
#include <tvmet/util/Incrementor.h>


template <class T>
class TestComplexMatrix : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestComplexMatrix );
  CPPUNIT_TEST( RealImag );
  CPPUNIT_TEST( Abs );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef T					value_type;
  typedef std::complex<T>			complex_type;
  typedef tvmet::Matrix<complex_type, 3, 3>	complex_matrix;
  typedef tvmet::Matrix<value_type, 3, 3>	real_matrix;

public:
  TestComplexMatrix()
    { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void RealImag();
  void Abs();

private:
  complex_matrix m1;

private:
  real_matrix m1_real;
};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestComplexMatrix<T>::setUp() {
  // real part is equal to complex part, ranging from 0 to N=Rows*Cols
  std::generate(m1.begin(), m1.end(),
		tvmet::util::Incrementor<typename complex_matrix::value_type>());

  std::generate(m1_real.begin(), m1_real.end(),
		tvmet::util::Incrementor<typename real_matrix::value_type>());
}

template <class T>
void TestComplexMatrix<T>::tearDown() { }

/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/


/*
 * real and imaginary parts, real and imag parts are equal
 */
template <class T>
void
TestComplexMatrix<T>::RealImag() {
  real_matrix r;

  r = real(m1);
  CPPUNIT_ASSERT( all_elements( r == m1_real ) );

  r = imag(m1);
  CPPUNIT_ASSERT( all_elements( r == m1_real ) );
}


/*
 * abs
 */
template <class T>
void
TestComplexMatrix<T>::Abs() {
  real_matrix m, r;

  m = abs(m1);

  r = sqrt(pow(real(m1), 2) + pow(imag(m1), 2));

  // we do have a precision problem
  // CPPUNIT_ASSERT( all_elements( m == r ) );

  real_matrix eps(m - r);
  CPPUNIT_ASSERT( all_elements( abs(eps) < std::numeric_limits<T>::epsilon() ) );
  //std::cout << eps << std::endl;
}

#endif // TEST_COMPLEX_MATRIX_H

// Local Variables:
// mode:C++
// End:
