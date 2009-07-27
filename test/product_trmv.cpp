// This file is triangularView of Eigen, a lightweight C++ template library
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

template<typename MatrixType> void trmv(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  RealScalar largerEps = 10*test_precision<RealScalar>();

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m3(rows, cols);
  VectorType v1 = VectorType::Random(rows);

  Scalar s1 = ei_random<Scalar>();

  m1 = MatrixType::Random(rows, cols);

  // check with a column-major matrix
  m3 = m1.template triangularView<Eigen::LowerTriangular>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::LowerTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UpperTriangular>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::UpperTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitLowerTriangular>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::UnitLowerTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitUpperTriangular>();
  VERIFY((m3 * v1).isApprox(m1.template triangularView<Eigen::UnitUpperTriangular>() * v1, largerEps));

  // check conjugated and scalar multiple expressions (col-major)
  m3 = m1.template triangularView<Eigen::LowerTriangular>();
  VERIFY(((s1*m3).conjugate() * v1).isApprox((s1*m1).conjugate().template triangularView<Eigen::LowerTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UpperTriangular>();
  VERIFY((m3.conjugate() * v1.conjugate()).isApprox(m1.conjugate().template triangularView<Eigen::UpperTriangular>() * v1.conjugate(), largerEps));

  // check with a row-major matrix
  m3 = m1.template triangularView<Eigen::UpperTriangular>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::LowerTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::LowerTriangular>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::UpperTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitUpperTriangular>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::UnitLowerTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::UnitLowerTriangular>();
  VERIFY((m3.transpose() * v1).isApprox(m1.transpose().template triangularView<Eigen::UnitUpperTriangular>() * v1, largerEps));

  // check conjugated and scalar multiple expressions (row-major)
  m3 = m1.template triangularView<Eigen::UpperTriangular>();
  VERIFY((m3.adjoint() * v1).isApprox(m1.adjoint().template triangularView<Eigen::LowerTriangular>() * v1, largerEps));
  m3 = m1.template triangularView<Eigen::LowerTriangular>();
  VERIFY((m3.adjoint() * (s1*v1.conjugate())).isApprox(m1.adjoint().template triangularView<Eigen::UpperTriangular>() * (s1*v1.conjugate()), largerEps));
  m3 = m1.template triangularView<Eigen::UnitUpperTriangular>();

  // TODO check with sub-matrices
}

void test_product_trmv()
{
  for(int i = 0; i < g_repeat ; i++) {
    CALL_SUBTEST( trmv(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( trmv(Matrix<float, 2, 2>()) );
    CALL_SUBTEST( trmv(Matrix3d()) );
    CALL_SUBTEST( trmv(Matrix<std::complex<float>,23, 23>()) );
    CALL_SUBTEST( trmv(MatrixXcd(17,17)) );
    CALL_SUBTEST( trmv(Matrix<float,Dynamic,Dynamic,RowMajor>(19, 19)) );
  }
}
