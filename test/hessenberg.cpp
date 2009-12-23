// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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
#include <Eigen/Eigenvalues>

template<typename Scalar,int Size> void hessenberg(int size = Size)
{
  typedef Matrix<Scalar,Size,Size> MatrixType;
  MatrixType m = MatrixType::Random(size,size);
  HessenbergDecomposition<MatrixType> hess(m);

  VERIFY_IS_APPROX(m, hess.matrixQ() * hess.matrixH() * hess.matrixQ().adjoint());
}

void test_hessenberg()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(( hessenberg<std::complex<double>,1>() ));
    CALL_SUBTEST_2(( hessenberg<std::complex<double>,2>() ));
    CALL_SUBTEST_3(( hessenberg<std::complex<float>,4>() ));
    CALL_SUBTEST_4(( hessenberg<float,Dynamic>(ei_random<int>(1,320)) ));
    CALL_SUBTEST_5(( hessenberg<std::complex<double>,Dynamic>(ei_random<int>(1,320)) ));
  }
}
