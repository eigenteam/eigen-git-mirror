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
 * $Id: TestNumericTraits.h,v 1.2 2004/11/04 18:12:40 opetzold Exp $
 */

#ifndef TVMET_TEST_NUMERIC_TRAITS_H
#define TVMET_TEST_NUMERIC_TRAITS_H

#include <cppunit/extensions/HelperMacros.h>

#include <complex>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

#include <cassert>

template <class T>
class TestNumericTraits : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestNumericTraits );
  CPPUNIT_TEST( Real );
  CPPUNIT_TEST( Imag );
  CPPUNIT_TEST( Conj );
  CPPUNIT_TEST( Abs );
  CPPUNIT_TEST( Sqrt );
  CPPUNIT_TEST( Norm_1 );
  CPPUNIT_TEST( Norm_2 );
  CPPUNIT_TEST( Norm_Inf );
  CPPUNIT_TEST( Equals );
  CPPUNIT_TEST_SUITE_END();

private:
  typedef tvmet::Vector<T, 3>			vector_type;
  typedef tvmet::Matrix<T, 3, 3>		matrix_type;

public:
  TestNumericTraits()
    : m_real(4), m_imag(9),
      m_conj(16),
      m_abs_Q1(7), m_abs_Q2(-m_abs_Q1)
  { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  void Real();
  void Imag();
  void Conj();
  void Abs();
  void Sqrt();
  void Norm_1();
  void Norm_2();
  void Norm_Inf();
  void Equals();

protected:
  // Helper
  void AbsHelper(tvmet::dispatch<true>, const T&);
  void AbsHelper(tvmet::dispatch<false>, const T&);

private:
  const T					m_real;
  const T					m_imag;
  const T					m_conj;
  const T					m_abs_Q1;
  const T					m_abs_Q2;
};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestNumericTraits<T>::setUp () { }

template <class T>
void TestNumericTraits<T>::tearDown() { }

/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/

template <class T>
void
TestNumericTraits<T>::Real()
{
  typedef typename tvmet::NumericTraits<T>::base_type real_type;

  real_type r = tvmet::NumericTraits<T>::real(m_real);

  CPPUNIT_ASSERT( r == m_real );
}


template <class T>
void
TestNumericTraits<T>::Imag()
{
  typedef typename tvmet::NumericTraits<T>::base_type imag_type;

  imag_type r = tvmet::NumericTraits<T>::imag(m_imag);

  CPPUNIT_ASSERT( r == 0 );
}


// conj only for signed types !!
template <> void TestNumericTraits<unsigned char>::Conj() { }
template <> void TestNumericTraits<unsigned short int>::Conj() { }
template <> void TestNumericTraits<unsigned int>::Conj() { }
template <> void TestNumericTraits<unsigned long>::Conj() { }

template <class T>
void
TestNumericTraits<T>::Conj()
{
  typedef typename tvmet::NumericTraits<T>::base_type conj_type;

  conj_type r = tvmet::NumericTraits<T>::conj(m_conj);

  CPPUNIT_ASSERT( r == m_conj );
}


template <class T>
void
TestNumericTraits<T>::Abs()
{
  typedef typename tvmet::NumericTraits<T>::base_type value_type;

  enum {
    is_signed = std::numeric_limits<value_type>::is_signed
  };

  value_type r1 = tvmet::NumericTraits<T>::abs(m_abs_Q1);
  value_type r2 = tvmet::NumericTraits<T>::abs(m_abs_Q2);

  CPPUNIT_ASSERT( r1 == m_abs_Q1 );

  // result depends on signed type
  AbsHelper(tvmet::dispatch<is_signed>(), r2);
}


template <class T>
void
TestNumericTraits<T>::AbsHelper(tvmet::dispatch<true>, const T& r)
{
  // signed type
  CPPUNIT_ASSERT( r == (m_abs_Q1) );
}


template <class T>
void
TestNumericTraits<T>::AbsHelper(tvmet::dispatch<false>, const T& r)
{
  // unsigned type
  CPPUNIT_ASSERT( r == T(-m_abs_Q1) );
}


template <class T>
void
TestNumericTraits<T>::Sqrt()
{
  typedef typename tvmet::NumericTraits<T>::base_type value_type;

  value_type r1 = tvmet::NumericTraits<T>::sqrt(m_real);
  value_type r2 = tvmet::NumericTraits<T>::sqrt(m_imag);

  CPPUNIT_ASSERT( r1 == 2 );
  CPPUNIT_ASSERT( r2 == 3 );
}


template <class T>
void
TestNumericTraits<T>::Norm_1()
{
  typedef typename tvmet::NumericTraits<T>::base_type value_type;

  value_type r = tvmet::NumericTraits<T>::norm_1(m_real);

  CPPUNIT_ASSERT( r == tvmet::NumericTraits<T>::abs(m_real) );
}


template <class T>
void
TestNumericTraits<T>::Norm_2()
{
  typedef typename tvmet::NumericTraits<T>::base_type value_type;

  value_type r = tvmet::NumericTraits<T>::norm_2(m_real);

  CPPUNIT_ASSERT( r == tvmet::NumericTraits<T>::abs(m_real) );
}


template <class T>
void
TestNumericTraits<T>::Norm_Inf()
{
  typedef typename tvmet::NumericTraits<T>::base_type value_type;

  value_type r = tvmet::NumericTraits<T>::norm_inf(m_real);

  CPPUNIT_ASSERT( r == tvmet::NumericTraits<T>::abs(m_real) );
}


template <class T>
void
TestNumericTraits<T>::Equals()
{
  typedef typename tvmet::NumericTraits<T>::base_type value_type;

  value_type lhs, rhs;

  lhs = rhs = 47;

  CPPUNIT_ASSERT( true == tvmet::NumericTraits<T>::equals(lhs,rhs) );

  // a not very intelligent test
  rhs += 1;

  CPPUNIT_ASSERT( false == tvmet::NumericTraits<T>::equals(lhs,rhs) );
}


#endif // TVMET_TEST_NUMERIC_TRAITS_H


// Local Variables:
// mode:C++
// End:
