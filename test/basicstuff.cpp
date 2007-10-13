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

#include "main.h"

template<typename MatrixType> void basicStuff(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  int rows = m.rows();
  int cols = m.cols();
  
  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             m3;
  VectorType v1 = VectorType::random(rows, 1),
             v2 = VectorType::random(rows, 1);

  Scalar s1 = NumTraits<Scalar>::random(),
         s2 = NumTraits<Scalar>::random();
  
  QVERIFY(v1.isApprox(v1));
  QVERIFY((v1-v1).isMuchSmallerThan(v1));
  QVERIFY((v1-v1).isMuchSmallerThan(v1.norm()));
  QVERIFY(m1.isApprox(m1));
  QVERIFY((m1-m1).isMuchSmallerThan(m1));
  
  QVERIFY((m1+m1).isApprox(2 * m1));
  QVERIFY((m1 * s1).isApprox(s1 * m1));
  QVERIFY(((m1 + m2) * s1).isApprox(s1 * m1 + s1 * m2));
  QVERIFY(((s1 + s2) * m1).isApprox(m1 * s1 + m1 * s2));
  
  
  
  m3 = m2;
  QVERIFY((m3 += m1).isApprox(m1 + m2));
  m3 = m2;
  QVERIFY((m3 -= m1).isApprox(-m1 + m2));
  m3 = m2;
  QVERIFY((m3 *= s1).isApprox(s1 * m2));
  m3 = m2;
  if(NumTraits<Scalar>::HasFloatingPoint
     && s1 != static_cast<Scalar>(0))
    QVERIFY((m3 /= s1).isApprox(m2 / s1));
  
  QVERIFY(((m1 * m1.transpose()) * m2).isApprox(m1 * (m1.transpose() * m2)));
  m3 = m1;
  m3 *= (m1.transpose() * m2);
  QVERIFY(m3.isApprox(m1 * (m1.transpose() * m2)));
  QVERIFY(m3.isApprox(m1.lazyProduct(m1.transpose() * m2)));
}

void EigenTest::testBasicStuff()
{
  basicStuff(Matrix<float, 1, 1>());
  basicStuff(Matrix<complex<int>, 2, 5>());
  basicStuff(Matrix<complex<double>, 4, 4>());
  basicStuff(MatrixXcf(3, 3));
  basicStuff(MatrixXi(8, 12));
  basicStuff(MatrixXd(20, 20));
}
