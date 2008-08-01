// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#include "product.h"

void test_product_large()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( product(MatrixXf(ei_random<int>(1,320), ei_random<int>(1,320))) );
    CALL_SUBTEST( product(MatrixXd(ei_random<int>(1,320), ei_random<int>(1,320))) );
    CALL_SUBTEST( product(MatrixXi(ei_random<int>(1,320), ei_random<int>(1,320))) );
    CALL_SUBTEST( product(MatrixXcf(ei_random<int>(1,50), ei_random<int>(1,50))) );
    #ifndef EIGEN_DEFAULT_TO_ROW_MAJOR
    CALL_SUBTEST( product(Matrix<float,Dynamic,Dynamic,Dynamic,Dynamic,RowMajorBit>(ei_random<int>(1,320), ei_random<int>(1,320))) );
    #else
    CALL_SUBTEST( product(Matrix<float,Dynamic,Dynamic,Dynamic,Dynamic,0>(ei_random<int>(1,320), ei_random<int>(1,320))) );
    #endif
  }
}
