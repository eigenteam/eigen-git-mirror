// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
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
#include <Eigen/Householder>

template<typename MatrixType> void householder(const MatrixType& m)
{
  /* this test covers the following files:
     Householder.h
  */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, ei_decrement_size<MatrixType::RowsAtCompileTime>::ret, 1> EssentialVectorType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;
  
  RealScalar beta;
  EssentialVectorType essential;

  VectorType v1 = VectorType::Random(rows), v2;
  v2 = v1;
  v1.makeHouseholder(&essential, &beta);
  v1.applyHouseholderOnTheLeft(essential,beta);
  
  VERIFY_IS_APPROX(v1.norm(), v2.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(v1.end(rows-1).norm(), v1.norm());
  v1 = VectorType::Random(rows);
  v2 = v1;
  v1.applyHouseholderOnTheLeft(essential,beta);
  VERIFY_IS_APPROX(v1.norm(), v2.norm());
  
  MatrixType m1(rows, cols),
             m2(rows, cols);

  v1 = VectorType::Random(rows);
  m1.colwise() = v1;
  m2 = m1;
  m1.col(0).makeHouseholder(&essential, &beta);
  m1.applyHouseholderOnTheLeft(essential,beta);
  VERIFY_IS_APPROX(m1.norm(), m2.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(m1.block(1,0,rows-1,cols).norm(), m1.norm());
  
  v1 = VectorType::Random(rows);
  SquareMatrixType m3(rows,rows), m4(rows,rows);
  m3.rowwise() = v1.transpose();
  m4 = m3;
  m3.row(0).makeHouseholder(&essential, &beta);
  m3.applyHouseholderOnTheRight(essential,beta);
  VERIFY_IS_APPROX(m3.norm(), m4.norm());
  VERIFY_IS_MUCH_SMALLER_THAN(m3.block(0,1,rows,rows-1).norm(), m3.norm());
}

void test_householder()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( householder(Matrix<double,2,2>()) );
    CALL_SUBTEST( householder(Matrix<float,2,3>()) );
    CALL_SUBTEST( householder(Matrix<double,3,5>()) );
    CALL_SUBTEST( householder(Matrix<float,4,4>()) );
    CALL_SUBTEST( householder(MatrixXd(10,12)) );
    CALL_SUBTEST( householder(MatrixXcf(16,17)) );
  }

}
