// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

template<typename MatrixType> void matrixProd(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;

  int rows = m.rows();
  int cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols);

  VERIFY_IS_MUCH_SMALLER_THAN(MatrixType::Zero(rows, cols).prod(), Scalar(1));
  VERIFY_IS_APPROX(MatrixType::Ones(rows, cols).prod(), Scalar(1));
  Scalar x = Scalar(1);
  for(int i = 0; i < rows; i++) for(int j = 0; j < cols; j++) x *= m1(i,j);
  VERIFY_IS_APPROX(m1.prod(), x);
}

template<typename VectorType> void vectorProd(const VectorType& w)
{
  typedef typename VectorType::Scalar Scalar;
  int size = w.size();

  VectorType v = VectorType::Random(size);
  for(int i = 1; i < size; i++)
  {
    Scalar s = Scalar(1);
    for(int j = 0; j < i; j++) s *= v[j];
    VERIFY_IS_APPROX(s, v.start(i).prod());
  }

  for(int i = 0; i < size-1; i++)
  {
    Scalar s = Scalar(1);
    for(int j = i; j < size; j++) s *= v[j];
    VERIFY_IS_APPROX(s, v.end(size-i).prod());
  }

  for(int i = 0; i < size/2; i++)
  {
    Scalar s = Scalar(1);
    for(int j = i; j < size-i; j++) s *= v[j];
    VERIFY_IS_APPROX(s, v.segment(i, size-2*i).prod());
  }
}

void test_prod()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( matrixProd(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( matrixProd(Matrix2f()) );
    CALL_SUBTEST( matrixProd(Matrix4d()) );
    CALL_SUBTEST( matrixProd(MatrixXcf(3, 3)) );
    CALL_SUBTEST( matrixProd(MatrixXf(8, 12)) );
    CALL_SUBTEST( matrixProd(MatrixXi(8, 12)) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( vectorProd(VectorXf(5)) );
    CALL_SUBTEST( vectorProd(VectorXd(10)) );
    CALL_SUBTEST( vectorProd(VectorXf(33)) );
  }
}
