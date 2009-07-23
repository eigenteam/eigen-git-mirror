// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@gmail.com>
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

#include "main.h"

template<typename MatrixType> void symm(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, Dynamic> Rhs1;
  typedef Matrix<Scalar, Dynamic, MatrixType::RowsAtCompileTime> Rhs2;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, Dynamic,RowMajor> Rhs3;

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols);

  m1 = (m1+m1.adjoint()).eval();

  Rhs1 rhs1 = Rhs1::Random(cols, ei_random<int>(1,320)), rhs12, rhs13;
  Rhs2 rhs2 = Rhs2::Random(ei_random<int>(1,320), rows), rhs22, rhs23;
  Rhs3 rhs3 = Rhs3::Random(cols, ei_random<int>(1,320)), rhs32, rhs33;

  Scalar s1 = ei_random<Scalar>(),
         s2 = ei_random<Scalar>();

  m2 = m1.template triangularView<LowerTriangular>();
  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<LowerTriangular>() * (s2*rhs1),
                   rhs13 = (s1*m1) * (s2*rhs1));

  m2 = m1.template triangularView<UpperTriangular>();
  VERIFY_IS_APPROX(rhs12 = (s1*m2).template selfadjointView<UpperTriangular>() * (s2*rhs1),
                   rhs13 = (s1*m1) * (s2*rhs1));

  m2 = m1.template triangularView<LowerTriangular>();
  VERIFY_IS_APPROX(rhs22 = (s1*m2).template selfadjointView<LowerTriangular>() * (s2*rhs2.adjoint()),
                   rhs23 = (s1*m1) * (s2*rhs2.adjoint()));

  m2 = m1.template triangularView<UpperTriangular>();
  VERIFY_IS_APPROX(rhs22 = (s1*m2).template selfadjointView<UpperTriangular>() * (s2*rhs2.adjoint()),
                   rhs23 = (s1*m1) * (s2*rhs2.adjoint()));

  m2 = m1.template triangularView<UpperTriangular>();
  VERIFY_IS_APPROX(rhs22 = (s1*m2.adjoint()).template selfadjointView<LowerTriangular>() * (s2*rhs2.adjoint()),
                   rhs23 = (s1*m1.adjoint()) * (s2*rhs2.adjoint()));

  // test row major = <...>
  m2 = m1.template triangularView<LowerTriangular>();
  VERIFY_IS_APPROX(rhs32 = (s1*m2).template selfadjointView<LowerTriangular>() * (s2*rhs3),
                   rhs33 = (s1*m1) * (s2 * rhs3));

  m2 = m1.template triangularView<UpperTriangular>();
  VERIFY_IS_APPROX(rhs32 = (s1*m2.adjoint()).template selfadjointView<LowerTriangular>() * (s2*rhs3).conjugate(),
                   rhs33 = (s1*m1.adjoint()) * (s2*rhs3).conjugate());

  // test matrix * selfadjoint
  m2 = m1.template triangularView<LowerTriangular>();
  VERIFY_IS_APPROX(rhs22 = (rhs2) * (m2).template selfadjointView<LowerTriangular>(),
                   rhs23 = (rhs2) * (m1));
  VERIFY_IS_APPROX(rhs22 = (s2*rhs2) * (s1*m2).template selfadjointView<LowerTriangular>(),
                   rhs23 = (s2*rhs2) * (s1*m1));
}
void test_product_symm()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    int s;
    s = ei_random<int>(10,320);
    CALL_SUBTEST( symm(MatrixXf(s, s)) );
    s = ei_random<int>(10,320);
    CALL_SUBTEST( symm(MatrixXcd(s, s)) );
  }
}
