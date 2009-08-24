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
#include <Eigen/QR>

template<typename MatrixType> void qr()
{
  int rows = ei_random<int>(20,200), cols = ei_random<int>(20,200), cols2 = ei_random<int>(20,200);
  int rank = ei_random<int>(1, std::min(rows, cols)-1);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;
  MatrixType m1;
  createRandomMatrixOfRank(rank,rows,cols,m1);
  FullPivotingHouseholderQR<MatrixType> qr(m1);
  VERIFY_IS_APPROX(rank, qr.rank());
  VERIFY(cols - qr.rank() == qr.dimensionOfKernel());
  VERIFY(!qr.isInjective());
  VERIFY(!qr.isInvertible());
  VERIFY(!qr.isSurjective());

  MatrixType r = qr.matrixQR();
  // FIXME need better way to construct trapezoid
  for(int i = 0; i < rows; i++) for(int j = 0; j < cols; j++) if(i>j) r(i,j) = Scalar(0);
  
  MatrixType b = qr.matrixQ() * r;

  MatrixType c = MatrixType::Zero(rows,cols);
  
  for(int i = 0; i < cols; ++i) c.col(qr.colsPermutation().coeff(i)) = b.col(i);
  VERIFY_IS_APPROX(m1, c);
  
  MatrixType m2 = MatrixType::Random(cols,cols2);
  MatrixType m3 = m1*m2;
  m2 = MatrixType::Random(cols,cols2);
  VERIFY(qr.solve(m3, &m2));
  VERIFY_IS_APPROX(m3, m1*m2);
  m3 = MatrixType::Random(rows,cols2);
  VERIFY(!qr.solve(m3, &m2));
}

template<typename MatrixType> void qr_invertible()
{
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Scalar Scalar;

  int size = ei_random<int>(10,50);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size,size);

  if (ei_is_same_type<RealScalar,float>::ret)
  {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size,size*2);
    m1 += a * a.adjoint();
  }

  FullPivotingHouseholderQR<MatrixType> qr(m1);
  VERIFY(qr.isInjective());
  VERIFY(qr.isInvertible());
  VERIFY(qr.isSurjective());

  m3 = MatrixType::Random(size,size);
  VERIFY(qr.solve(m3, &m2));
  VERIFY_IS_APPROX(m3, m1*m2);
  
  // now construct a matrix with prescribed determinant
  m1.setZero();
  for(int i = 0; i < size; i++) m1(i,i) = ei_random<Scalar>();
  RealScalar absdet = ei_abs(m1.diagonal().prod());
  m3 = qr.matrixQ(); // get a unitary
  m1 = m3 * m1 * m3;
  qr.compute(m1);
  VERIFY_IS_APPROX(absdet, qr.absDeterminant());
  VERIFY_IS_APPROX(ei_log(absdet), qr.logAbsDeterminant());
}

template<typename MatrixType> void qr_verify_assert()
{
  MatrixType tmp;

  FullPivotingHouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp,&tmp))
  VERIFY_RAISES_ASSERT(qr.matrixQ())
  VERIFY_RAISES_ASSERT(qr.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(qr.isInjective())
  VERIFY_RAISES_ASSERT(qr.isSurjective())
  VERIFY_RAISES_ASSERT(qr.isInvertible())
  VERIFY_RAISES_ASSERT(qr.computeInverse(&tmp))
  VERIFY_RAISES_ASSERT(qr.inverse())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.logAbsDeterminant())
}

void test_qr_fullpivoting()
{
 for(int i = 0; i < 1; i++) {
    // FIXME : very weird bug here
//     CALL_SUBTEST( qr(Matrix2f()) );
    CALL_SUBTEST( qr<MatrixXf>() );
    CALL_SUBTEST( qr<MatrixXd>() );
    CALL_SUBTEST( qr<MatrixXcd>() );
  }

  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( qr_invertible<MatrixXf>() );
    CALL_SUBTEST( qr_invertible<MatrixXd>() );
    CALL_SUBTEST( qr_invertible<MatrixXcf>() );
    CALL_SUBTEST( qr_invertible<MatrixXcd>() );
  }

  CALL_SUBTEST(qr_verify_assert<Matrix3f>());
  CALL_SUBTEST(qr_verify_assert<Matrix3d>());
  CALL_SUBTEST(qr_verify_assert<MatrixXf>());
  CALL_SUBTEST(qr_verify_assert<MatrixXd>());
  CALL_SUBTEST(qr_verify_assert<MatrixXcf>());
  CALL_SUBTEST(qr_verify_assert<MatrixXcd>());
}
