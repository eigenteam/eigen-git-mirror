// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_TEST_MAIN_H
#define EIGEN_TEST_MAIN_H

#include <QtTest/QtTest>

#define EIGEN_INTERNAL_DEBUGGING
#include <Eigen/Core.h>

#include <cstdlib>
#include <ctime>

#define DEFAULT_REPEAT 50

#define VERIFY(a) QVERIFY(a)
#define VERIFY_IS_APPROX(a, b) QVERIFY(test_isApprox(a, b))
#define VERIFY_IS_NOT_APPROX(a, b) QVERIFY(!test_isApprox(a, b))
#define VERIFY_IS_MUCH_SMALLER_THAN(a, b) QVERIFY(test_isMuchSmallerThan(a, b))
#define VERIFY_IS_NOT_MUCH_SMALLER_THAN(a, b) QVERIFY(!test_isMuchSmallerThan(a, b))
#define VERIFY_IS_APPROX_OR_LESS_THAN(a, b) QVERIFY(test_isApproxOrLessThan(a, b))
#define VERIFY_IS_NOT_APPROX_OR_LESS_THAN(a, b) QVERIFY(!test_isApproxOrLessThan(a, b))

namespace Eigen {

template<typename T> inline typename NumTraits<T>::Real test_precision();
template<> inline int test_precision<int>() { return 0; }
template<> inline float test_precision<float>() { return 1e-2f; }
template<> inline double test_precision<double>() { return 1e-5; }
template<> inline float test_precision<std::complex<float> >() { return test_precision<float>(); }
template<> inline double test_precision<std::complex<double> >() { return test_precision<double>(); }

inline bool test_isApprox(const int& a, const int& b)
{ return isApprox(a, b, test_precision<int>()); }
inline bool test_isMuchSmallerThan(const int& a, const int& b)
{ return isMuchSmallerThan(a, b, test_precision<int>()); }
inline bool test_isApproxOrLessThan(const int& a, const int& b)
{ return isApproxOrLessThan(a, b, test_precision<int>()); }

inline bool test_isApprox(const float& a, const float& b)
{ return isApprox(a, b, test_precision<float>()); }
inline bool test_isMuchSmallerThan(const float& a, const float& b)
{ return isMuchSmallerThan(a, b, test_precision<float>()); }
inline bool test_isApproxOrLessThan(const float& a, const float& b)
{ return isApproxOrLessThan(a, b, test_precision<float>()); }

inline bool test_isApprox(const double& a, const double& b)
{ return isApprox(a, b, test_precision<double>()); }
inline bool test_isMuchSmallerThan(const double& a, const double& b)
{ return isMuchSmallerThan(a, b, test_precision<double>()); }
inline bool test_isApproxOrLessThan(const double& a, const double& b)
{ return isApproxOrLessThan(a, b, test_precision<double>()); }

inline bool test_isApprox(const std::complex<float>& a, const std::complex<float>& b)
{ return isApprox(a, b, test_precision<std::complex<float> >()); }
inline bool test_isMuchSmallerThan(const std::complex<float>& a, const std::complex<float>& b)
{ return isMuchSmallerThan(a, b, test_precision<std::complex<float> >()); }

inline bool test_isApprox(const std::complex<double>& a, const std::complex<double>& b)
{ return isApprox(a, b, test_precision<std::complex<double> >()); }
inline bool test_isMuchSmallerThan(const std::complex<double>& a, const std::complex<double>& b)
{ return isMuchSmallerThan(a, b, test_precision<std::complex<double> >()); }

template<typename Scalar, typename Derived1, typename Derived2>
inline bool test_isApprox(const MatrixBase<Scalar, Derived1>& m1,
                   const MatrixBase<Scalar, Derived2>& m2)
{
  return m1.isApprox(m2, test_precision<Scalar>());
}

template<typename Scalar, typename Derived1, typename Derived2>
inline bool test_isMuchSmallerThan(const MatrixBase<Scalar, Derived1>& m1,
                                   const MatrixBase<Scalar, Derived2>& m2)
{
  return m1.isMuchSmallerThan(m2, test_precision<Scalar>());
}

template<typename Scalar, typename Derived>
inline bool test_isMuchSmallerThan(const MatrixBase<Scalar, Derived>& m,
                                   const typename NumTraits<Scalar>::Real& s)
{
  return m.isMuchSmallerThan(s, test_precision<Scalar>());
}

class EigenTest : public QObject
{
    Q_OBJECT

  public:
    EigenTest(int repeat) : m_repeat(repeat) {}

  private slots:
    void testBasicStuff();
    void testAdjoint();
    void testSubmatrices();
    void testMiscMatrices();
    void testSmallVectors();
  protected:
    int m_repeat;
};

} // end namespace Eigen

#endif // EIGEN_TEST_MAIN_H
