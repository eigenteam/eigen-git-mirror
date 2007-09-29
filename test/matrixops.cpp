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

template<typename MatrixType1,
         typename MatrixType2> void matrixOps(const MatrixType1& m1, const MatrixType2& m2)
{
  typedef typename MatrixType1::Scalar Scalar;
  int rows1 = m1.rows(), cols1 = m1.cols();
  int rows2 = m2.rows(), cols2 = m2.cols();
  
  MatrixType1 a(rows1, cols1), b(rows1, cols1), c(b);
  Scalar s;
  a * s;
  s * a;
  a + b;
  a - b;
  (a + b) * s;
  s * (a + b);
  a + b + c;
  a = b;
  a = b + c;
  a = s * (b - c);
  a = (a + b).eval();
  a += b;
  a -= b + b;
  a *= s;
  b /= s;
  if(rows1 == cols1)
  {
    a *= b;
    a.lazyMul(b);
  }
  
  MatrixType1 d(rows1, cols1);
  MatrixType2 e(rows2, cols2);
  QVERIFY( (d * e).rows() == rows1 && (d * e).cols() == cols2 );
}

void EigenTest::testMatrixOps()
{
  matrixOps(EiMatrix<float, 1, 1>(), EiMatrix<float, 1, 1>());
  matrixOps(EiMatrix<int, 2, 3>(), EiMatrix<int, 3, 1>());
  matrixOps(EiMatrix<double, 3, 3>(), EiMatrix<double, 3, 3>());
  matrixOps(EiMatrix<complex<float>, 4,3>(), EiMatrix<complex<float>, 3,4>());
  matrixOps(EiMatrixXf(1, 1), EiMatrixXf(1, 3));
  matrixOps(EiMatrixXi(2, 2), EiMatrixXi(2, 2));
  matrixOps(EiMatrixXd(3, 5), EiMatrixXd(5, 1));
  matrixOps(EiMatrixXcf(4, 4), EiMatrixXcf(4, 4));
  matrixOps(EiMatrixXd(3, 5), EiMatrix<double, 5, 1>());
  matrixOps(EiMatrix4cf(), EiMatrixXcf(4, 4));
}
