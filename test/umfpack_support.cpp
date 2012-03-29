// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
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

#include "sparse_solver.h"

#include <Eigen/UmfPackSupport>

void test_umfpack_support()
{
  UmfPackLU<SparseMatrix<double, ColMajor> > umfpack_double_colmajor;
  UmfPackLU<SparseMatrix<std::complex<double> > > umfpack_cplxdouble_colmajor;
  CALL_SUBTEST_1(check_sparse_square_solving(umfpack_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_solving(umfpack_cplxdouble_colmajor));
  CALL_SUBTEST_1(check_sparse_square_determinant(umfpack_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_determinant(umfpack_cplxdouble_colmajor));
}
