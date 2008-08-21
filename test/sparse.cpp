// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Daniel Gomez Ferro <dgomezferro@gmail.com>
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
#include <Eigen/Sparse>

void test_sparse()
{
  int rows = 4, cols = 4;
  SparseMatrix<double> m(rows, cols);

  m.startFill(rows);
  m.fill(0, 2) = 2;
  m.fill(1, 2) = 1;
  m.fill(0, 3) = 5;
  m.endFill();

  m.coeffRef(0, 2) = 3;
  VERIFY_RAISES_ASSERT( m.coeffRef(0, 0) = 5 );
  VERIFY_IS_MUCH_SMALLER_THAN( m.coeff(0, 0), 0.000001 );
  VERIFY_IS_MUCH_SMALLER_THAN( m.coeff(0, 1), 0.000001 );
  VERIFY_IS_MUCH_SMALLER_THAN( m.coeff(2, 1), 0.000001 );
  VERIFY_IS_APPROX( m.coeff(0, 2), 3.0 );
  VERIFY_IS_APPROX( m.coeff(1, 2), 1.0 );
  VERIFY_IS_APPROX( m.coeff(0, 3), 5.0 );

  Matrix4d dm;
  double r;
  m.startFill(rows*cols);
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      r = rand();
      m.fill(j, i) = r;
      dm(j, i) = r;
    }
  }
  m.endFill();

  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      VERIFY_IS_APPROX( m.coeff(j, i), dm(j, i) );
    }
  }
}
