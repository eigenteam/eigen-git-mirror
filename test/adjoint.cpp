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
  typedef Matrix<Scalar, MatrixType::Traits::RowsAtCompileTime, 1> VectorType;
  int rows = m.rows();
  int cols = m.cols();
  
  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::zero(rows, cols),
             identity = Matrix<Scalar, MatrixType::Traits::RowsAtCompileTime, MatrixType::Traits::RowsAtCompileTime>
                              ::identity(rows),
             square = Matrix<Scalar, MatrixType::Traits::RowsAtCompileTime, MatrixType::Traits::RowsAtCompileTime>
                              ::random(rows, rows);
  VectorType v1 = VectorType::random(rows),
             v2 = VectorType::random(rows),
             v3 = VectorType::random(rows),
             vzero = VectorType::zero(rows);

  Scalar s1 = random<Scalar>(),
         s2 = random<Scalar>();
  
  // check involutivity of adjoint, transpose, conjugate
  VERIFY_IS_APPROX(m1.transpose().transpose(),              m1);
  VERIFY_IS_APPROX(m1.conjugate().conjugate(),              m1);
  VERIFY_IS_APPROX(m1.adjoint().adjoint(),                  m1);
  
  // check basic compatibility of adjoint, transpose, conjugate
  VERIFY_IS_APPROX(m1.transpose().conjugate().adjoint(),    m1);
  VERIFY_IS_APPROX(m1.adjoint().conjugate().transpose(),    m1);
  if(!NumTraits<Scalar>::IsComplex)
    VERIFY_IS_APPROX(m1.adjoint().transpose(),              m1);
  
  // check multiplicative behavior
  VERIFY_IS_APPROX((m1.transpose() * m2).transpose(),       m2.transpose() * m1);
  VERIFY_IS_APPROX((m1.adjoint() * m2).adjoint(),           m2.adjoint() * m1);
  VERIFY_IS_APPROX((m1.transpose() * m2).conjugate(),       m1.adjoint() * m2.conjugate());
  VERIFY_IS_APPROX((s1 * m1).transpose(),                   s1 * m1.transpose());
  VERIFY_IS_APPROX((s1 * m1).conjugate(),                   conj(s1) * m1.conjugate());
  VERIFY_IS_APPROX((s1 * m1).adjoint(),                     conj(s1) * m1.adjoint());
  
  // check basic properties of dot, norm, norm2
  typedef typename NumTraits<Scalar>::Real RealScalar;
  VERIFY_IS_APPROX((s1 * v1 + s2 * v2).dot(v3),      s1 * v1.dot(v3) + s2 * v2.dot(v3));
  VERIFY_IS_APPROX(v3.dot(s1 * v1 + s2 * v2),        conj(s1)*v3.dot(v1)+conj(s2)*v3.dot(v2));
  VERIFY_IS_APPROX(conj(v1.dot(v2)),                 v2.dot(v1));
  VERIFY_IS_APPROX(abs(v1.dot(v1)),                  v1.norm2());
  if(NumTraits<Scalar>::HasFloatingPoint)
    VERIFY_IS_APPROX(v1.norm2(),                     v1.norm() * v1.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(abs(vzero.dot(v1)),    static_cast<RealScalar>(1));
  if(NumTraits<Scalar>::HasFloatingPoint)
    VERIFY_IS_MUCH_SMALLER_THAN(vzero.norm(),        static_cast<RealScalar>(1));
  
  // check compatibility of dot and adjoint
  VERIFY_IS_APPROX(v1.dot(square * v2),              (square.adjoint() * v1).dot(v2));
  
  // like in testBasicStuff, test operator() to check const-qualification
  int r = random<int>(0, rows-1),
      c = random<int>(0, cols-1);
  VERIFY_IS_APPROX(m1.conjugate()(r,c), conj(m1(r,c)));
  VERIFY_IS_APPROX(m1.adjoint()(c,r), conj(m1(r,c)));
  
}

void EigenTest::testAdjoint()
{
  for(int i = 0; i < m_repeat; i++) {
    adjoint(Matrix<float, 1, 1>());
    adjoint(Matrix4d());
    adjoint(MatrixXcf(3, 3));
    adjoint(MatrixXi(8, 12));
    adjoint(MatrixXcd(20, 20));
  }
}

} // namespace Eigen
