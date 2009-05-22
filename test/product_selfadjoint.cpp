// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@gmail.com>
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

template<typename MatrixType> void product_selfadjoint(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols);
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows);
  
  m1 = m1.adjoint()*m1;
  
  // col-lower
  m2.setZero();
  m2.template part<LowerTriangular>() = m1;
  ei_product_selfadjoint_vector<Scalar,MatrixType::Flags&RowMajorBit,LowerTriangularBit>
    (cols,m2.data(),cols, v1.data(), v2.data());
  VERIFY_IS_APPROX(v2, m1 * v1);

  // col-upper
  m2.setZero();
  m2.template part<UpperTriangular>() = m1;
  ei_product_selfadjoint_vector<Scalar,MatrixType::Flags&RowMajorBit,UpperTriangularBit>(cols,m2.data(),cols, v1.data(), v2.data());
  VERIFY_IS_APPROX(v2, m1 * v1);

}

void test_product_selfadjoint()
{
  for(int i = 0; i < g_repeat ; i++) {
    CALL_SUBTEST( product_selfadjoint(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( product_selfadjoint(Matrix<float, 2, 2>()) );
    CALL_SUBTEST( product_selfadjoint(Matrix3d()) );
    CALL_SUBTEST( product_selfadjoint(MatrixXcf(4, 4)) );
    CALL_SUBTEST( product_selfadjoint(MatrixXcd(21,21)) );
    CALL_SUBTEST( product_selfadjoint(MatrixXd(17,17)) );
    CALL_SUBTEST( product_selfadjoint(Matrix<float,Dynamic,Dynamic,RowMajor>(18,18)) );
    CALL_SUBTEST( product_selfadjoint(Matrix<std::complex<double>,Dynamic,Dynamic,RowMajor>(19, 19)) );
  }
}
