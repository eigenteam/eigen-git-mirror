// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
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

template<typename MatrixType, int QRPreconditioner>
void jacobisvd_check_full(const MatrixType& m, const JacobiSVD<MatrixType, QRPreconditioner>& svd)
{
  typedef typename MatrixType::Index Index;
  Index rows = m.rows();
  Index cols = m.cols();

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

  MatrixType sigma = MatrixType::Zero(rows,cols);
  sigma.diagonal() = svd.singularValues().template cast<Scalar>();
  MatrixUType u = svd.matrixU();
  MatrixVType v = svd.matrixV();

  VERIFY_IS_APPROX(m, u * sigma * v.adjoint());
  VERIFY_IS_UNITARY(u);
  VERIFY_IS_UNITARY(v);
}

template<typename MatrixType, int QRPreconditioner>
void jacobisvd_compare_to_full(const MatrixType& m,
                               unsigned int computationOptions,
                               const JacobiSVD<MatrixType, QRPreconditioner>& referenceSvd)
{
  typedef typename MatrixType::Index Index;
  Index rows = m.rows();
  Index cols = m.cols();
  Index diagSize = std::min(rows, cols);

  JacobiSVD<MatrixType, QRPreconditioner> svd(m, computationOptions);

  VERIFY_IS_EQUAL(svd.singularValues(), referenceSvd.singularValues());
  if(computationOptions & ComputeFullU)
    VERIFY_IS_EQUAL(svd.matrixU(), referenceSvd.matrixU());
  if(computationOptions & ComputeThinU)
    VERIFY_IS_EQUAL(svd.matrixU(), referenceSvd.matrixU().leftCols(diagSize));
  if(computationOptions & ComputeFullV)
    VERIFY_IS_EQUAL(svd.matrixV(), referenceSvd.matrixV());
  if(computationOptions & ComputeThinV)
    VERIFY_IS_EQUAL(svd.matrixV(), referenceSvd.matrixV().leftCols(diagSize));
}

template<typename MatrixType, int QRPreconditioner>
void jacobisvd_test_all_computation_options(const MatrixType& m)
{
  if (QRPreconditioner == NoQRPreconditioner && m.rows() != m.cols())
    return;
  JacobiSVD<MatrixType, QRPreconditioner> fullSvd(m, ComputeFullU|ComputeFullV);

  jacobisvd_check_full(m, fullSvd);

  if(QRPreconditioner == FullPivHouseholderQRPreconditioner)
    return;

  jacobisvd_compare_to_full(m, ComputeFullU, fullSvd);
  jacobisvd_compare_to_full(m, ComputeFullV, fullSvd);
  jacobisvd_compare_to_full(m, 0, fullSvd);

  if (MatrixType::ColsAtCompileTime == Dynamic) {
    // thin U/V are only available with dynamic number of columns
    jacobisvd_compare_to_full(m, ComputeFullU|ComputeThinV, fullSvd);
    jacobisvd_compare_to_full(m,              ComputeThinV, fullSvd);
    jacobisvd_compare_to_full(m, ComputeThinU|ComputeFullV, fullSvd);
    jacobisvd_compare_to_full(m, ComputeThinU             , fullSvd);
    jacobisvd_compare_to_full(m, ComputeThinU|ComputeThinV, fullSvd);
  }
}

template<typename MatrixType>
void jacobisvd(const MatrixType& a = MatrixType(), bool pickrandom = true)
{
  MatrixType m = pickrandom ? MatrixType::Random(a.rows(), a.cols()) : a;
  jacobisvd_test_all_computation_options<MatrixType, FullPivHouseholderQRPreconditioner>(m);
  jacobisvd_test_all_computation_options<MatrixType, ColPivHouseholderQRPreconditioner>(m);
  jacobisvd_test_all_computation_options<MatrixType, HouseholderQRPreconditioner>(m);
  jacobisvd_test_all_computation_options<MatrixType, NoQRPreconditioner>(m);
}

template<typename MatrixType> void jacobisvd_verify_assert()
{
  MatrixType tmp;

  JacobiSVD<MatrixType> svd;
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
    CALL_SUBTEST_1(( jacobisvd(m, false) ));
    m << 1, 0,
         1, 0;
    CALL_SUBTEST_1(( jacobisvd(m, false) ));
    Matrix2d n;
    n << 1, 1,
         1, -1;
    CALL_SUBTEST_2(( jacobisvd(n, false) ));
    CALL_SUBTEST_3(( jacobisvd<Matrix3f>() ));
    CALL_SUBTEST_4(( jacobisvd<Matrix4d>() ));
    CALL_SUBTEST_5(( jacobisvd<Matrix<float,3,5> >() ));
    CALL_SUBTEST_6(( jacobisvd<Matrix<double,Dynamic,2> >(Matrix<double,Dynamic,2>(10,2)) ));

    CALL_SUBTEST_7(( jacobisvd<MatrixXf>(MatrixXf(50,50)) ));
    CALL_SUBTEST_8(( jacobisvd<MatrixXcd>(MatrixXcd(14,7)) ));
  }
  CALL_SUBTEST_9(( jacobisvd<MatrixXf>(MatrixXf(300,200)) ));
  CALL_SUBTEST_10(( jacobisvd<MatrixXcd>(MatrixXcd(100,150)) ));

  CALL_SUBTEST_3(( jacobisvd_verify_assert<Matrix3f>() ));
  CALL_SUBTEST_3(( jacobisvd_verify_assert<Matrix3d>() ));
  CALL_SUBTEST_9(( jacobisvd_verify_assert<MatrixXf>() ));
  CALL_SUBTEST_11(( jacobisvd_verify_assert<MatrixXd>() ));

  // Test problem size constructors
  CALL_SUBTEST_12( JacobiSVD<MatrixXf>(10, 20) );
}
