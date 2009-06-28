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

template<typename MatrixType> void diagonalmatrices(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime };
  typedef Matrix<Scalar, Rows, 1> VectorType;
  typedef Matrix<Scalar, 1, Cols> RowVectorType;
  typedef Matrix<Scalar, Rows, Rows> SquareMatrixType;
  typedef DiagonalMatrix<Scalar, Rows> LeftDiagonalMatrix;
  typedef DiagonalMatrix<Scalar, Cols> RightDiagonalMatrix;
  
  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols);
  VectorType v1 = VectorType::Random(rows),
             v2 = VectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols),
             rv2 = RowVectorType::Random(cols);
  LeftDiagonalMatrix ldm1(v1), ldm2(v2);
  RightDiagonalMatrix rdm1(rv1), rdm2(rv2);
  
  int i = ei_random<int>(0, rows-1);
  int j = ei_random<int>(0, cols-1);
  
  VERIFY_IS_APPROX( ((ldm1 * m1)(i,j))  , ldm1.diagonal()(i) * m1(i,j) );
  VERIFY_IS_APPROX( ((ldm1 * (m1+m2))(i,j))  , ldm1.diagonal()(i) * (m1+m2)(i,j) );
  VERIFY_IS_APPROX( ((m1 * rdm1)(i,j))  , rdm1.diagonal()(j) * m1(i,j) );
  VERIFY_IS_APPROX( ((v1.asDiagonal() * m1)(i,j))  , v1(i) * m1(i,j) );
  VERIFY_IS_APPROX( ((m1 * rv1.asDiagonal())(i,j))  , rv1(j) * m1(i,j) );
  VERIFY_IS_APPROX( (((v1+v2).asDiagonal() * m1)(i,j))  , (v1+v2)(i) * m1(i,j) );
  VERIFY_IS_APPROX( (((v1+v2).asDiagonal() * (m1+m2))(i,j))  , (v1+v2)(i) * (m1+m2)(i,j) );
  VERIFY_IS_APPROX( ((m1 * (rv1+rv2).asDiagonal())(i,j))  , (rv1+rv2)(j) * m1(i,j) );
  VERIFY_IS_APPROX( (((m1+m2) * (rv1+rv2).asDiagonal())(i,j))  , (rv1+rv2)(j) * (m1+m2)(i,j) );
  
  SquareMatrixType sq_m1 (v1.asDiagonal());
  VERIFY_IS_APPROX(sq_m1, v1.asDiagonal().toDenseMatrix());
  sq_m1 = v1.asDiagonal();
  VERIFY_IS_APPROX(sq_m1, v1.asDiagonal().toDenseMatrix());
  SquareMatrixType sq_m2 = v1.asDiagonal();
  VERIFY_IS_APPROX(sq_m1, sq_m2);
  
  ldm1 = v1.asDiagonal();
  LeftDiagonalMatrix ldm3(v1);
  VERIFY_IS_APPROX(ldm1.diagonal(), ldm3.diagonal());
  LeftDiagonalMatrix ldm4 = v1.asDiagonal();
  VERIFY_IS_APPROX(ldm1.diagonal(), ldm4.diagonal());
  
  sq_m1.block(0,0,rows,rows) = ldm1;
  VERIFY_IS_APPROX(sq_m1, ldm1.toDenseMatrix());
  sq_m1.transpose() = ldm1;
  VERIFY_IS_APPROX(sq_m1, ldm1.toDenseMatrix());
}

void test_diagonalmatrices()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( diagonalmatrices(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( diagonalmatrices(Matrix3f()) );
    CALL_SUBTEST( diagonalmatrices(Matrix<double,3,3,RowMajor>()) );
    CALL_SUBTEST( diagonalmatrices(Matrix4d()) );
    CALL_SUBTEST( diagonalmatrices(Matrix<float,4,4,RowMajor>()) );
    CALL_SUBTEST( diagonalmatrices(MatrixXcf(3, 5)) );
    CALL_SUBTEST( diagonalmatrices(MatrixXi(10, 8)) );
    CALL_SUBTEST( diagonalmatrices(Matrix<double,Dynamic,Dynamic,RowMajor>(20, 20)) );
    CALL_SUBTEST( diagonalmatrices(MatrixXf(21, 24)) );
  }
}
