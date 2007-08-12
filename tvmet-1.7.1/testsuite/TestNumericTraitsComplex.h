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
 * $Id: TestTraitsComplex.h,v 1.2 2004/11/04 18:12:40 opetzold Exp $
 */

#ifndef TVMET_TEST_NUMERIC_TRAITS_H
#define TVMET_TEST_NUMERIC_TRAITS_H

#include <cppunit/extensions/HelperMacros.h>

#include <complex>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

#include <cassert>

template <class T>
class TestTraitsComplex : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestTraitsComplex );
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
  typedef tvmet::Vector<T, 3>				vector_type;
  typedef tvmet::Matrix<T, 3, 3>			matrix_type;

public:
  TestTraitsComplex()
    : m_p_real( 3), m_p_imag( 4),
      m_n_real(-3), m_n_imag(-4),
      m_z1(m_p_real, m_p_imag),
      m_z2(m_n_real, m_p_imag),
      m_z3(m_n_real, m_n_imag),
      m_z4(m_p_real, m_n_imag)
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

private:
  // Helper
  void AbsHelper(tvmet::dispatch<true>,
		 typename tvmet::Traits<T>::base_type);
  void AbsHelper(tvmet::dispatch<false>,
		 typename tvmet::Traits<T>::base_type);
  void SqrtHelper(tvmet::dispatch<true>);
  void SqrtHelper(tvmet::dispatch<false>);
  void NormHelper(tvmet::dispatch<true>,
		  typename tvmet::Traits<T>::base_type);
  void NormHelper(tvmet::dispatch<false>,
		  typename tvmet::Traits<T>::base_type);


private:
  typedef typename tvmet::Traits<T>::base_type 	base_type;
  typedef T						value_type;

  const base_type					m_p_real;
  const base_type					m_p_imag;
  const base_type					m_n_real;
  const base_type					m_n_imag;

  // complex quadrant I ... IV
  const value_type					m_z1;
  const value_type					m_z2;
  const value_type					m_z3;
  const value_type					m_z4;
};

/*****************************************************************************
 * Implementation Part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestTraitsComplex<T>::setUp () { }

template <class T>
void TestTraitsComplex<T>::tearDown() { }

/*****************************************************************************
 * Implementation Part II
 ****************************************************************************/

template <class T>
void
TestTraitsComplex<T>::Real()
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  base_type r1 = tvmet::Traits<T>::real(m_z1);
  base_type r2 = tvmet::Traits<T>::real(m_z2);
  base_type r3 = tvmet::Traits<T>::real(m_z3);
  base_type r4 = tvmet::Traits<T>::real(m_z4);

  CPPUNIT_ASSERT( r1 == m_p_real );
  CPPUNIT_ASSERT( r2 == m_n_real );
  CPPUNIT_ASSERT( r3 == m_n_real );
  CPPUNIT_ASSERT( r4 == m_p_real );
}


template <class T>
void
TestTraitsComplex<T>::Imag()
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  base_type i1 = tvmet::Traits<T>::imag(m_z1);
  base_type i2 = tvmet::Traits<T>::imag(m_z2);
  base_type i3 = tvmet::Traits<T>::imag(m_z3);
  base_type i4 = tvmet::Traits<T>::imag(m_z4);

  CPPUNIT_ASSERT( i1 == m_p_imag );
  CPPUNIT_ASSERT( i2 == m_p_imag );
  CPPUNIT_ASSERT( i3 == m_n_imag );
  CPPUNIT_ASSERT( i4 == m_n_imag );
}


// conj only for signed types !!
template <> void TestTraitsComplex<std::complex<unsigned char> >::Conj() { }
template <> void TestTraitsComplex<std::complex<unsigned short int> >::Conj() { }
template <> void TestTraitsComplex<std::complex<unsigned int> >::Conj() { }
template <> void TestTraitsComplex<std::complex<unsigned long> >::Conj() { }


template <class T>
void
TestTraitsComplex<T>::Conj()
{
  typedef typename tvmet::Traits<T>::value_type value_type;
  typedef typename tvmet::Traits<T>::base_type base_type;

  enum {
    is_signed = std::numeric_limits<base_type>::is_signed
  };

  // conjugate
  value_type conj_z1 = tvmet::Traits<T>::conj(m_z1);
  value_type conj_z2 = tvmet::Traits<T>::conj(m_z2);
  value_type conj_z3 = tvmet::Traits<T>::conj(m_z3);
  value_type conj_z4 = tvmet::Traits<T>::conj(m_z4);

  // real part
  base_type r1 = tvmet::Traits<T>::real(conj_z1);
  base_type r2 = tvmet::Traits<T>::real(conj_z2);
  base_type r3 = tvmet::Traits<T>::real(conj_z3);
  base_type r4 = tvmet::Traits<T>::real(conj_z4);

  // imag part
  base_type i1 = tvmet::Traits<T>::imag(conj_z1);
  base_type i2 = tvmet::Traits<T>::imag(conj_z2);
  base_type i3 = tvmet::Traits<T>::imag(conj_z3);
  base_type i4 = tvmet::Traits<T>::imag(conj_z4);

  // check on real part; real is tested before
  CPPUNIT_ASSERT( r1 == tvmet::Traits<T>::real(m_z1) );
  CPPUNIT_ASSERT( r2 == tvmet::Traits<T>::real(m_z2) );
  CPPUNIT_ASSERT( r3 == tvmet::Traits<T>::real(m_z3) );
  CPPUNIT_ASSERT( r4 == tvmet::Traits<T>::real(m_z4) );

  // check on imag part
  CPPUNIT_ASSERT( i1 == -tvmet::Traits<T>::imag(m_z1) );
  CPPUNIT_ASSERT( i2 == -tvmet::Traits<T>::imag(m_z2) );
  CPPUNIT_ASSERT( i3 == -tvmet::Traits<T>::imag(m_z3) );
  CPPUNIT_ASSERT( i4 == -tvmet::Traits<T>::imag(m_z4) );
}


template <class T>
void
TestTraitsComplex<T>::Abs()
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  enum {
    is_signed = std::numeric_limits<base_type>::is_signed
  };

  base_type a1 = tvmet::Traits<T>::abs(m_z1);
  base_type a2 = tvmet::Traits<T>::abs(m_z2);
  base_type a3 = tvmet::Traits<T>::abs(m_z3);
  base_type a4 = tvmet::Traits<T>::abs(m_z4);

  // result depends on signed type
  AbsHelper(tvmet::dispatch<is_signed>(), a1);
  AbsHelper(tvmet::dispatch<is_signed>(), a2);
  AbsHelper(tvmet::dispatch<is_signed>(), a3);
  AbsHelper(tvmet::dispatch<is_signed>(), a4);
}


template <class T>
void
TestTraitsComplex<T>::AbsHelper(tvmet::dispatch<true>,
				typename tvmet::Traits<T>::base_type r)
{
  // signed type
  CPPUNIT_ASSERT( r == 5 );
}


template <class T>
void
TestTraitsComplex<T>::AbsHelper(tvmet::dispatch<false>,
				typename tvmet::Traits<T>::base_type r)
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  base_type x = m_z1.real();	// sign doesn't matter on abs()
  base_type y = m_z1.imag();	// sign doesn't matter on abs()

  // unsigned type
  CPPUNIT_ASSERT( r == static_cast<base_type>(
			 tvmet::Traits<base_type>::sqrt(x * x + y * y))
		);
}


template <class T>
void
TestTraitsComplex<T>::Sqrt()
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  enum {
    is_signed = std::numeric_limits<base_type>::is_signed
  };

  // delegate tests
  SqrtHelper(tvmet::dispatch<is_signed>());
}


template <class T>
void
TestTraitsComplex<T>::SqrtHelper(tvmet::dispatch<true>)
{
  // signed type
  typedef typename tvmet::Traits<T>::value_type value_type;

  // sqrt
  value_type z1 = tvmet::Traits<T>::sqrt(m_z1);
  value_type z2 = tvmet::Traits<T>::sqrt(m_z2);
  value_type z3 = tvmet::Traits<T>::sqrt(m_z3);
  value_type z4 = tvmet::Traits<T>::sqrt(m_z4);

  CPPUNIT_ASSERT( z1 == value_type(2,1) );
  CPPUNIT_ASSERT( z2 == value_type(1,2) );
  CPPUNIT_ASSERT( z3 == value_type(1,-2) );
  CPPUNIT_ASSERT( z4 == value_type(2,-1) );

}


template <class T>
void
TestTraitsComplex<T>::SqrtHelper(tvmet::dispatch<false>)
{
  // unsigned type

  /* XXX
   * very dirty - we assume we calculate right
   * on "negative" complex types */

  typedef typename tvmet::Traits<T>::value_type value_type;

  // sqrt
  value_type z1 = tvmet::Traits<T>::sqrt(m_z1);
  value_type z2 = tvmet::Traits<T>::sqrt(m_z2);

  CPPUNIT_ASSERT( z1 == value_type(2,1) );
  CPPUNIT_ASSERT( z2 == value_type(1,2) );
}


template <class T>
void
TestTraitsComplex<T>::Norm_1()
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  enum {
    is_signed = std::numeric_limits<base_type>::is_signed
  };

  // norm_1
  base_type n1 = tvmet::Traits<T>::norm_1(m_z1);
  base_type n2 = tvmet::Traits<T>::norm_1(m_z2);
  base_type n3 = tvmet::Traits<T>::norm_1(m_z3);
  base_type n4 = tvmet::Traits<T>::norm_1(m_z4);

  // result depends on signed type
  NormHelper(tvmet::dispatch<is_signed>(), n1);
  NormHelper(tvmet::dispatch<is_signed>(), n2);
  NormHelper(tvmet::dispatch<is_signed>(), n3);
  NormHelper(tvmet::dispatch<is_signed>(), n4);
}


template <class T>
void
TestTraitsComplex<T>::Norm_2()
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  enum {
    is_signed = std::numeric_limits<base_type>::is_signed
  };

  // norm_2
  base_type n1 = tvmet::Traits<T>::norm_2(m_z1);
  base_type n2 = tvmet::Traits<T>::norm_2(m_z2);
  base_type n3 = tvmet::Traits<T>::norm_2(m_z3);
  base_type n4 = tvmet::Traits<T>::norm_2(m_z4);

  // result depends on signed type
  NormHelper(tvmet::dispatch<is_signed>(), n1);
  NormHelper(tvmet::dispatch<is_signed>(), n2);
  NormHelper(tvmet::dispatch<is_signed>(), n3);
  NormHelper(tvmet::dispatch<is_signed>(), n4);
}


template <class T>
void
TestTraitsComplex<T>::Norm_Inf()
{
  typedef typename tvmet::Traits<T>::base_type base_type;

  enum {
    is_signed = std::numeric_limits<base_type>::is_signed
  };

  // norm_inf
  base_type n1 = tvmet::Traits<T>::norm_inf(m_z1);
  base_type n2 = tvmet::Traits<T>::norm_inf(m_z2);
  base_type n3 = tvmet::Traits<T>::norm_inf(m_z3);
  base_type n4 = tvmet::Traits<T>::norm_inf(m_z4);

  // result depends on signed type
  NormHelper(tvmet::dispatch<is_signed>(), n1);
  NormHelper(tvmet::dispatch<is_signed>(), n2);
  NormHelper(tvmet::dispatch<is_signed>(), n3);
  NormHelper(tvmet::dispatch<is_signed>(), n4);
}

template <class T>
void
TestTraitsComplex<T>::NormHelper(tvmet::dispatch<true>,
				typename tvmet::Traits<T>::base_type)
{
  // XXX To be implement
}


template <class T>
void
TestTraitsComplex<T>::NormHelper(tvmet::dispatch<false>,
				typename tvmet::Traits<T>::base_type)
{
  // XXX To be implement
}


template <class T>
void
TestTraitsComplex<T>::Equals()
{
  // XXX this test is to simple

  typedef typename tvmet::Traits<T>::value_type value_type;

  value_type lhs, rhs;

  {
    lhs = rhs = m_z1;

    CPPUNIT_ASSERT( true == tvmet::Traits<T>::equals(lhs,rhs) );

    rhs += m_z1;

    CPPUNIT_ASSERT( false == tvmet::Traits<T>::equals(lhs,rhs) );
  }
  {
    lhs = rhs = m_z2;

    CPPUNIT_ASSERT( true == tvmet::Traits<T>::equals(lhs,rhs) );

    rhs += m_z2;

    CPPUNIT_ASSERT( false == tvmet::Traits<T>::equals(lhs,rhs) );
  }
  {
    lhs = rhs = m_z3;

    CPPUNIT_ASSERT( true == tvmet::Traits<T>::equals(lhs,rhs) );

    rhs += m_z3;

    CPPUNIT_ASSERT( false == tvmet::Traits<T>::equals(lhs,rhs) );
  }
  {
    lhs = rhs = m_z4;

    CPPUNIT_ASSERT( true == tvmet::Traits<T>::equals(lhs,rhs) );

    rhs += m_z4;

    CPPUNIT_ASSERT( false == tvmet::Traits<T>::equals(lhs,rhs) );
  }
}


#endif // TVMET_TEST_NUMERIC_TRAITS_H


// Local Variables:
// mode:C++
// End:
