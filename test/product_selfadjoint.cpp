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

template<typename MatrixType> void product_selfadjoint(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, 1, MatrixType::RowsAtCompileTime> RowVectorType;

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3;
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows);

  RowVectorType r1 = RowVectorType::Random(rows),
                r2 = RowVectorType::Random(rows);

  Scalar s1 = ei_random<Scalar>(),
         s2 = ei_random<Scalar>(),
         s3 = ei_random<Scalar>();
  
  m1 = m1.adjoint()*m1;
  
  // lower
  m2.setZero();
  m2.template triangularView<LowerTriangular>() = m1;
  ei_product_selfadjoint_vector<Scalar,MatrixType::Flags&RowMajorBit,LowerTriangularBit>
    (cols,m2.data(),cols, v1.data(), v2.data());
  VERIFY_IS_APPROX(v2, m1 * v1);
  VERIFY_IS_APPROX((m2.template selfadjointView<LowerTriangular>() * v1).eval(), m1 * v1);

  // upper
  m2.setZero();
  m2.template triangularView<UpperTriangular>() = m1;
  ei_product_selfadjoint_vector<Scalar,MatrixType::Flags&RowMajorBit,UpperTriangularBit>(cols,m2.data(),cols, v1.data(), v2.data());
  VERIFY_IS_APPROX(v2, m1 * v1);
  VERIFY_IS_APPROX((m2.template selfadjointView<UpperTriangular>() * v1).eval(), m1 * v1);

  // rank2 update
  m2 = m1.template triangularView<LowerTriangular>();
  m2.template selfadjointView<LowerTriangular>().rank2update(v1,v2);
  VERIFY_IS_APPROX(m2, (m1 + v1 * v2.adjoint()+ v2 * v1.adjoint()).template triangularView<LowerTriangular>().toDense());
  
  m2 = m1.template triangularView<UpperTriangular>();
  m2.template selfadjointView<UpperTriangular>().rank2update(-v1,s2*v2,s3);
  VERIFY_IS_APPROX(m2, (m1 + (-s2*s3) * (v1 * v2.adjoint()+ v2 * v1.adjoint())).template triangularView<UpperTriangular>().toDense());

  m2 = m1.template triangularView<UpperTriangular>();
  m2.template selfadjointView<UpperTriangular>().rank2update(-r1.adjoint(),r2.adjoint()*s3,s1);
  VERIFY_IS_APPROX(m2, (m1 + (-s3*s1) * (r1.adjoint() * r2 + r2.adjoint() * r1)).template triangularView<UpperTriangular>().toDense());

  if (rows>1)
  {
    m2 = m1.template triangularView<LowerTriangular>();
    m2.block(1,1,rows-1,cols-1).template selfadjointView<LowerTriangular>().rank2update(v1.end(rows-1),v2.start(cols-1));
    m3 = m1;
    m3.block(1,1,rows-1,cols-1) += v1.end(rows-1) * v2.start(cols-1).adjoint()+ v2.start(cols-1) * v1.end(rows-1).adjoint();
    VERIFY_IS_APPROX(m2, m3.template triangularView<LowerTriangular>().toDense());
  }
}

void test_product_selfadjoint()
{
  for(int i = 0; i < g_repeat ; i++) {
    CALL_SUBTEST( product_selfadjoint(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( product_selfadjoint(Matrix<float, 2, 2>()) );
    CALL_SUBTEST( product_selfadjoint(Matrix3d()) );
    CALL_SUBTEST( product_selfadjoint(MatrixXcf(4, 4)) );
    CALL_SUBTEST( product_selfadjoint(MatrixXcd(21,21)) );
    CALL_SUBTEST( product_selfadjoint(MatrixXd(14,14)) );
    CALL_SUBTEST( product_selfadjoint(Matrix<float,Dynamic,Dynamic,RowMajor>(17,17)) );
    CALL_SUBTEST( product_selfadjoint(Matrix<std::complex<double>,Dynamic,Dynamic,RowMajor>(19, 19)) );
  }
}
