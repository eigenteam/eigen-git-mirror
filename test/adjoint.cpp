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

namespace Eigen {

template<typename MatrixType> void adjoint(const MatrixType& m)
{
  /* this test covers the following files:
     Transpose.h Conjugate.h Dot.h
  */

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  int rows = m.rows();
  int cols = m.cols();
  
  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::zero(rows, cols),
             identity = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::identity(rows),
             square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::random(rows, rows);
  VectorType v1 = VectorType::random(rows),
             v2 = VectorType::random(rows),
             v3 = VectorType::random(rows),
             vzero = VectorType::zero(rows);

  Scalar s1 = random<Scalar>(),
         s2 = random<Scalar>();
  
  // check involutivity of adjoint, transpose, conjugate
  QVERIFY(m1.transpose().transpose().isApprox(m1));
  QVERIFY(m1.conjugate().conjugate().isApprox(m1));
  QVERIFY(m1.adjoint().adjoint().isApprox(m1));
  
  // check basic compatibility of adjoint, transpose, conjugate
  QVERIFY(m1.transpose().conjugate().adjoint().isApprox(m1));
  QVERIFY(m1.adjoint().conjugate().transpose().isApprox(m1));
  if(!NumTraits<Scalar>::IsComplex) QVERIFY(m1.adjoint().transpose().isApprox(m1));
  
  // check multiplicative behavior
  QVERIFY((m1.transpose() * m2).transpose().isApprox(m2.transpose() * m1));
  QVERIFY((m1.adjoint() * m2).adjoint().isApprox(m2.adjoint() * m1));
  QVERIFY((m1.transpose() * m2).conjugate().isApprox(m1.adjoint() * m2.conjugate()));
  QVERIFY((s1 * m1).transpose().isApprox(s1 * m1.transpose()));
  QVERIFY((s1 * m1).conjugate().isApprox(conj(s1) * m1.conjugate()));
  QVERIFY((s1 * m1).adjoint().isApprox(conj(s1) * m1.adjoint()));
  
  // check basic properties of dot, norm, norm2
  typedef typename NumTraits<Scalar>::Real RealScalar;
  QVERIFY(isApprox((s1 * v1 + s2 * v2).dot(v3), s1 * v1.dot(v3) + s2 * v2.dot(v3)));
  QVERIFY(isApprox(v3.dot(s1 * v1 + s2 * v2), conj(s1) * v3.dot(v1) + conj(s2) * v3.dot(v2)));
  QVERIFY(isApprox(conj(v1.dot(v2)), v2.dot(v1)));
  QVERIFY(isApprox(abs(v1.dot(v1)), v1.norm2()));
  if(NumTraits<Scalar>::HasFloatingPoint)
    QVERIFY(isApprox(v1.norm2(), v1.norm() * v1.norm()));
  QVERIFY(isMuchSmallerThan(abs(vzero.dot(v1)), static_cast<RealScalar>(1)));
  if(NumTraits<Scalar>::HasFloatingPoint)
    QVERIFY(isMuchSmallerThan(vzero.norm(), static_cast<RealScalar>(1)));
  
  // check compatibility of dot and adjoint
  QVERIFY(isApprox(v1.dot(square * v2), (square.adjoint() * v1).dot(v2)));
}

void EigenTest::testAdjoint()
{
  adjoint(Matrix<float, 1, 1>());
  adjoint(Matrix4cd());
  adjoint(MatrixXcf(3, 3));
  adjoint(MatrixXi(8, 12));
  adjoint(MatrixXd(20, 20));
}

} // namespace Eigen
