// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@gmail.com>
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

template<typename MatrixType> void bandmatrix(MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrixType;

  int size = m.rows();

  DenseMatrixType dm1(size,size);
  dm1.setZero();

  m.diagonal().setConstant(123);
  dm1.diagonal().setConstant(123);
  for (int i=1; i<=m.supers();++i)
  {
    m.diagonal(i).setConstant(i);
    dm1.diagonal(i).setConstant(i);
  }
  for (int i=1; i<=m.subs();++i)
  {
    m.diagonal(-i).setConstant(-i);
    dm1.diagonal(-i).setConstant(-i);
  }
  std::cerr << m.toDense() << "\n\n" << dm1 << "\n\n";
  VERIFY_IS_APPROX(dm1,m.toDense());

}

void test_bandmatrix()
{
  for(int i = 0; i < g_repeat ; i++) {
    BandMatrix<float,Dynamic,Dynamic,Dynamic> m(6,6,3,2);
    CALL_SUBTEST( bandmatrix(m) );
  }
}
