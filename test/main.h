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

USING_EIGEN_DATA_TYPES

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

template<typename T> bool TestNegligible(const T& a, const T& b)
{
  return(Abs(a) <= Abs(b) * TestEpsilon<T>());
}

template<typename T> bool TestApprox(const T& a, const T& b)
{
  if(Eigen::NumTraits<T>::IsFloat)
    return(Abs(a - b) <= std::min(Abs(a), Abs(b)) * TestEpsilon<T>());
  else
    return(a == b);
}

template<typename T> bool TestLessThanOrApprox(const T& a, const T& b)
{
  if(Eigen::NumTraits<T>::IsFloat)
    return(a < b || Approx(a, b));
  else
    return(a <= b);
}

#define QVERIFY_NEGLIGIBLE(a, b) QVERIFY(TestNegligible(a, b))
#define QVERIFY_APPROX(a, b) QVERIFY(TestApprox(a, b))
#define QVERIFY_LESS_THAN_OR_APPROX(a, b) QVERIFY(TestLessThanOrApprox(a, b))

#endif // EI_TEST_MAIN_H
