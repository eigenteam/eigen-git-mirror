// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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
#include <Eigen/SVD>
#include <Eigen/LU>

template<typename MatrixType, unsigned int Options> void svd(const MatrixType& m = MatrixType(), bool pickrandom = true)
{
  int rows = m.rows();
  int cols = m.cols();

  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime
  };
    
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> MatrixUType;
  typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime> MatrixVType;
  typedef Matrix<Scalar, RowsAtCompileTime, 1> ColVectorType;
  typedef Matrix<Scalar, ColsAtCompileTime, 1> InputVectorType;
  
  MatrixType a;
  if(pickrandom) a = MatrixType::Random(rows,cols);
  else a = m;

  JacobiSVD<MatrixType,Options> svd(a);
  MatrixType sigma = MatrixType::Zero(rows,cols);
  sigma.diagonal() = svd.singularValues().template cast<Scalar>();
  MatrixUType u = svd.matrixU();
  MatrixVType v = svd.matrixV();
  
  VERIFY_IS_APPROX(a, u * sigma * v.adjoint());
  VERIFY_IS_UNITARY(u);
  VERIFY_IS_UNITARY(v);
}

template<typename MatrixType> void svd_verify_assert()
{
  MatrixType tmp;

  SVD<MatrixType> svd;
  //VERIFY_RAISES_ASSERT(svd.solve(tmp, &tmp))
  VERIFY_RAISES_ASSERT(svd.matrixU())
  VERIFY_RAISES_ASSERT(svd.singularValues())
  VERIFY_RAISES_ASSERT(svd.matrixV())
  /*VERIFY_RAISES_ASSERT(svd.computeUnitaryPositive(&tmp,&tmp))
  VERIFY_RAISES_ASSERT(svd.computePositiveUnitary(&tmp,&tmp))
  VERIFY_RAISES_ASSERT(svd.computeRotationScaling(&tmp,&tmp))
  VERIFY_RAISES_ASSERT(svd.computeScalingRotation(&tmp,&tmp))*/
}

void test_jacobisvd()
{
  for(int i = 0; i < g_repeat; i++) {
    Matrix2cd m;
    m << 0, 1,
         0, 1;
    CALL_SUBTEST(( svd<Matrix2cd,0>(m, false) ));
    m << 1, 0,
         1, 0;
    CALL_SUBTEST(( svd<Matrix2cd,0>(m, false) ));
    Matrix2d n;
    n << 1, 1,
         1, -1;
    CALL_SUBTEST(( svd<Matrix2d,0>(n, false) ));
    CALL_SUBTEST(( svd<Matrix3f,0>() ));
    CALL_SUBTEST(( svd<Matrix4d,Square>() ));
    CALL_SUBTEST(( svd<Matrix<float,3,5> , AtLeastAsManyColsAsRows>() ));
    CALL_SUBTEST(( svd<Matrix<double,Dynamic,2> , AtLeastAsManyRowsAsCols>(Matrix<double,Dynamic,2>(10,2)) ));

    CALL_SUBTEST(( svd<MatrixXf,Square>(MatrixXf(50,50)) ));
    CALL_SUBTEST(( svd<MatrixXcd,AtLeastAsManyRowsAsCols>(MatrixXcd(14,7)) ));
  }
  CALL_SUBTEST(( svd<MatrixXf,0>(MatrixXf(300,200)) ));
  CALL_SUBTEST(( svd<MatrixXcd,AtLeastAsManyColsAsRows>(MatrixXcd(100,150)) ));
  
  CALL_SUBTEST(( svd_verify_assert<Matrix3f>() ));
  CALL_SUBTEST(( svd_verify_assert<Matrix3d>() ));
  CALL_SUBTEST(( svd_verify_assert<MatrixXf>() ));
  CALL_SUBTEST(( svd_verify_assert<MatrixXd>() ));
}
