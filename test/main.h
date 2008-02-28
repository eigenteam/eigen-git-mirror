// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either 
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of 
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public 
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_TEST_MAIN_H
#define EIGEN_TEST_MAIN_H

#include <QtTest/QtTest>

//#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER RowMajor
#define EIGEN_INTERNAL_DEBUGGING
#include <Eigen/Core>

#include <cstdlib>
#include <ctime>

#define DEFAULT_REPEAT 50

#define VERIFY(a) QVERIFY(a)
#define VERIFY_IS_APPROX(a, b) QVERIFY(test_ei_isApprox(a, b))
#define VERIFY_IS_NOT_APPROX(a, b) QVERIFY(!test_ei_isApprox(a, b))
#define VERIFY_IS_MUCH_SMALLER_THAN(a, b) QVERIFY(test_ei_isMuchSmallerThan(a, b))
#define VERIFY_IS_NOT_MUCH_SMALLER_THAN(a, b) QVERIFY(!test_ei_isMuchSmallerThan(a, b))
#define VERIFY_IS_APPROX_OR_LESS_THAN(a, b) QVERIFY(test_ei_isApproxOrLessThan(a, b))
#define VERIFY_IS_NOT_APPROX_OR_LESS_THAN(a, b) QVERIFY(!test_ei_isApproxOrLessThan(a, b))

namespace Eigen {

template<typename T> inline typename NumTraits<T>::Real test_precision();
template<> inline int test_precision<int>() { return 0; }
template<> inline float test_precision<float>() { return 1e-2f; }
template<> inline double test_precision<double>() { return 1e-5; }
template<> inline float test_precision<std::complex<float> >() { return test_precision<float>(); }
template<> inline double test_precision<std::complex<double> >() { return test_precision<double>(); }

inline bool test_ei_isApprox(const int& a, const int& b)
{ return ei_isApprox(a, b, test_precision<int>()); }
inline bool test_ei_isMuchSmallerThan(const int& a, const int& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<int>()); }
inline bool test_ei_isApproxOrLessThan(const int& a, const int& b)
{ return ei_isApproxOrLessThan(a, b, test_precision<int>()); }

inline bool test_ei_isApprox(const float& a, const float& b)
{ return ei_isApprox(a, b, test_precision<float>()); }
inline bool test_ei_isMuchSmallerThan(const float& a, const float& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<float>()); }
inline bool test_ei_isApproxOrLessThan(const float& a, const float& b)
{ return ei_isApproxOrLessThan(a, b, test_precision<float>()); }

inline bool test_ei_isApprox(const double& a, const double& b)
{ return ei_isApprox(a, b, test_precision<double>()); }
inline bool test_ei_isMuchSmallerThan(const double& a, const double& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<double>()); }
inline bool test_ei_isApproxOrLessThan(const double& a, const double& b)
{ return ei_isApproxOrLessThan(a, b, test_precision<double>()); }

inline bool test_ei_isApprox(const std::complex<float>& a, const std::complex<float>& b)
{ return ei_isApprox(a, b, test_precision<std::complex<float> >()); }
inline bool test_ei_isMuchSmallerThan(const std::complex<float>& a, const std::complex<float>& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<std::complex<float> >()); }

inline bool test_ei_isApprox(const std::complex<double>& a, const std::complex<double>& b)
{ return ei_isApprox(a, b, test_precision<std::complex<double> >()); }
inline bool test_ei_isMuchSmallerThan(const std::complex<double>& a, const std::complex<double>& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<std::complex<double> >()); }

template<typename Scalar, typename Derived1, typename Derived2>
inline bool test_ei_isApprox(const MatrixBase<Scalar, Derived1>& m1,
                   const MatrixBase<Scalar, Derived2>& m2)
{
  return m1.isApprox(m2, test_precision<Scalar>());
}

template<typename Scalar, typename Derived1, typename Derived2>
inline bool test_ei_isMuchSmallerThan(const MatrixBase<Scalar, Derived1>& m1,
                                   const MatrixBase<Scalar, Derived2>& m2)
{
  return m1.isMuchSmallerThan(m2, test_precision<Scalar>());
}

template<typename Scalar, typename Derived>
inline bool test_ei_isMuchSmallerThan(const MatrixBase<Scalar, Derived>& m,
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
    void testLinearStructure();
    void testProduct();
    void testAdjoint();
    void testSubmatrices();
    void testMiscMatrices();
    void testSmallVectors();
    void testMap();
  protected:
    int m_repeat;
};

} // end namespace Eigen

#endif // EIGEN_TEST_MAIN_H
