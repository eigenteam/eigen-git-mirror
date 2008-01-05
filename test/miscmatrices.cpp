// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#include "main.h"

namespace Eigen {

template<typename MatrixType> void miscMatrices(const MatrixType& m)
{
  /* this test covers the following files:
     DiagonalMatrix.h Ones.h
  */

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime> RowVectorType;
  int rows = m.rows();
  int cols = m.cols();
  
  int r = random<int>(0, rows-1), r2 = random<int>(0, rows-1), c = random<int>(0, cols-1);
  VERIFY_IS_APPROX(MatrixType::ones(rows,cols)(r,c), static_cast<Scalar>(1));
  MatrixType m1 = MatrixType::ones(rows,cols);
  VERIFY_IS_APPROX(m1(r,c), static_cast<Scalar>(1));
  VectorType v1 = VectorType::random(rows);
  v1[0];
  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
  square = v1.asDiagonal();
  if(r==r2) VERIFY_IS_APPROX(square(r,r2), v1[r]);
  else VERIFY_IS_MUCH_SMALLER_THAN(square(r,r2), static_cast<Scalar>(1));
  square = MatrixType::zero(rows, rows);
  square.diagonal() = VectorType::ones(rows);
  VERIFY_IS_APPROX(square, MatrixType::identity(rows));
}

void EigenTest::testMiscMatrices()
{
  for(int i = 0; i < m_repeat; i++) {
    miscMatrices(Matrix<float, 1, 1>());
    miscMatrices(Matrix4d());
    miscMatrices(MatrixXcf(3, 3));
    miscMatrices(MatrixXi(8, 12));
    miscMatrices(MatrixXcd(20, 20));
  }
}

} // namespace Eigen
