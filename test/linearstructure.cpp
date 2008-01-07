// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

template<typename MatrixType> void linearStructure(const MatrixType& m)
{
  /* this test covers the following files:
     Sum.h Difference.h Opposite.h ScalarMultiple.h
  */

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::Traits::RowsAtCompileTime, 1> VectorType;
  
  int rows = m.rows();
  int cols = m.cols();
  
  // this test relies a lot on Random.h, and there's not much more that we can do
  // to test it, hence I consider that we will have tested Random.h
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
             vzero = VectorType::zero(rows);

  Scalar s1 = random<Scalar>(),
         s2 = random<Scalar>();
  
  int r = random<int>(0, rows-1),
      c = random<int>(0, cols-1);
  
  VERIFY_IS_APPROX(-(-m1),                  m1);
  VERIFY_IS_APPROX(m1+m1,                   2*m1);
  VERIFY_IS_APPROX(m1+m2-m1,                m2);
  VERIFY_IS_APPROX(-m2+m1+m2,               m1);
  VERIFY_IS_APPROX(m1*s1,                   s1*m1);
  VERIFY_IS_APPROX((m1+m2)*s1,              s1*m1+s1*m2);
  VERIFY_IS_APPROX((s1+s2)*m1,              m1*s1+m1*s2);
  VERIFY_IS_APPROX((m1-m2)*s1,              s1*m1-s1*m2);
  VERIFY_IS_APPROX((s1-s2)*m1,              m1*s1-m1*s2);
  VERIFY_IS_APPROX((-m1+m2)*s1,             -s1*m1+s1*m2);
  VERIFY_IS_APPROX((-s1+s2)*m1,             -m1*s1+m1*s2);
  m3 = m2; m3 += m1;
  VERIFY_IS_APPROX(m3,                      m1+m2);
  m3 = m2; m3 -= m1;
  VERIFY_IS_APPROX(m3,                      m2-m1);
  m3 = m2; m3 *= s1;
  VERIFY_IS_APPROX(m3,                      s1*m2);
  if(NumTraits<Scalar>::HasFloatingPoint)
  {
    m3 = m2; m3 /= s1;
    VERIFY_IS_APPROX(m3,                    m2/s1);
  }
  
  // again, test operator() to check const-qualification
  VERIFY_IS_APPROX((-m1)(r,c), -(m1(r,c)));
  VERIFY_IS_APPROX((m1-m2)(r,c), (m1(r,c))-(m2(r,c)));
  VERIFY_IS_APPROX((m1+m2)(r,c), (m1(r,c))+(m2(r,c)));
  VERIFY_IS_APPROX((s1*m1)(r,c), s1*(m1(r,c)));
  VERIFY_IS_APPROX((m1*s1)(r,c), (m1(r,c))*s1);
  if(NumTraits<Scalar>::HasFloatingPoint)
    VERIFY_IS_APPROX((m1/s1)(r,c), (m1(r,c))/s1);
}

void EigenTest::testLinearStructure()
{
  for(int i = 0; i < m_repeat; i++) {
    linearStructure(Matrix<float, 1, 1>());
    linearStructure(Matrix4d());
    linearStructure(MatrixXcf(3, 3));
    linearStructure(MatrixXi(8, 12));
    linearStructure(MatrixXcd(20, 20));
  }
}

} // namespace Eigen
