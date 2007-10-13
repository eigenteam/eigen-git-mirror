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

#ifndef EI_TEST_MAIN_H
#define EI_TEST_MAIN_H

#include <QtTest/QtTest>
#include "../src/Core.h"

using namespace Eigen;

#include <cstdlib>
#include <ctime>

using namespace std;

class EigenTest : public QObject
{
    Q_OBJECT

  public:
    EigenTest();

  private slots:
    void testVectorOps();
    void testMatrixOps();
    void testMatrixManip();
};

template<typename T> inline typename Eigen::NumTraits<T>::Real TestEpsilon();
template<> inline int TestEpsilon<int>() { return 0; }
template<> inline float TestEpsilon<float>() { return 1e-2f; }
template<> inline double TestEpsilon<double>() { return 1e-4; }
template<> inline int TestEpsilon<std::complex<int> >() { return TestEpsilon<int>(); }
template<> inline float TestEpsilon<std::complex<float> >() { return TestEpsilon<float>(); }
template<> inline double TestEpsilon<std::complex<double> >() { return TestEpsilon<double>(); }

template<typename T> bool TestMuchSmallerThan(const T& a, const T& b)
{
  return NumTraits<T>::isMuchSmallerThan(a, b, TestEpsilon<T>());
}

template<typename Scalar, typename Derived, typename OtherDerived>
bool TestMuchSmallerThan(
  const Object<Scalar, Derived>& a,
  const Object<Scalar, OtherDerived>& b)
{
  return a.isMuchSmallerThan(b, TestEpsilon<Scalar>());
}

template<typename T> bool TestApprox(const T& a, const T& b)
{
  return NumTraits<T>::isApprox(a, b, TestEpsilon<T>());
}

template<typename Scalar, typename Derived, typename OtherDerived>
bool TestApprox(
  const Object<Scalar, Derived>& a,
  const Object<Scalar, OtherDerived>& b)
{
  return a.isApprox(b, TestEpsilon<Scalar>());
}

template<typename T> bool TestApproxOrLessThan(const T& a, const T& b)
{
  return NumTraits<T>::isApproxOrLessThan(a, b, TestEpsilon<T>());
}

template<typename Scalar, typename Derived, typename OtherDerived>
bool TestApproxOrLessThan(
  const Object<Scalar, Derived>& a,
  const Object<Scalar, OtherDerived>& b)
{
  return a.isApproxOrLessThan(b, TestEpsilon<Scalar>());
}

#define QVERIFY_MUCH_SMALLER_THAN(a, b) QVERIFY(TestMuchSmallerThan(a, b))
#define QVERIFY_APPROX(a, b) QVERIFY(TestApprox(a, b))
#define QVERIFY_APPROX_OR_LESS_THAN(a, b) QVERIFY(TestApproxOrLessThan(a, b))

#endif // EI_TEST_MAIN_H
