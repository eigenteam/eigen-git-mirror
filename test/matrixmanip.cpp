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

template<typename MatrixType> void matrixManip(const MatrixType& m)
{
  int rows = m.rows(), cols = m.cols();
  int i = rand()%rows, j = rand()%cols;

  MatrixType a(rows, cols), b(rows, cols);
  a.row(i);
  a.col(j);
  a.minor(i, j);
  a.block(1, rows-1, 1, cols-1);
  a.row(i) = b.row(i);
  a.row(i) += b.row(i);
  a.col(j) *= 2;
  a.minor(i, j) = b.block(1, rows-1, 1, cols-1);
  a.minor(i, j) -= eval(a.block(1, rows-1, 1, cols-1));
}

void EigenTest::testMatrixManip()
{
  matrixManip(Matrix<int, 2, 3>());
  matrixManip(Matrix<double, 3, 3>());
  matrixManip(Matrix<complex<float>, 4,3>());
  matrixManip(MatrixXi(2, 2));
  matrixManip(MatrixXd(3, 5));
  matrixManip(MatrixXcf(4, 4));
}
