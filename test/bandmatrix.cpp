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

template<typename MatrixType> void bandmatrix(const MatrixType& _m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrixType;

  int rows = _m.rows();
  int cols = _m.cols();
  int supers = _m.supers();
  int subs = _m.subs();

  MatrixType m(rows,cols,supers,subs);

  DenseMatrixType dm1(rows,cols);
  dm1.setZero();

  m.diagonal().setConstant(123);
  dm1.diagonal().setConstant(123);
  for (int i=1; i<=m.supers();++i)
  {
    m.diagonal(i).setConstant(static_cast<RealScalar>(i));
    dm1.diagonal(i).setConstant(static_cast<RealScalar>(i));
  }
  for (int i=1; i<=m.subs();++i)
  {
    m.diagonal(-i).setConstant(-static_cast<RealScalar>(i));
    dm1.diagonal(-i).setConstant(-static_cast<RealScalar>(i));
  }
  //std::cerr << m.m_data << "\n\n" << m.toDense() << "\n\n" << dm1 << "\n\n\n\n";
  VERIFY_IS_APPROX(dm1,m.toDense());

  for (int i=0; i<cols; ++i)
  {
    m.col(i).setConstant(static_cast<RealScalar>(i+1));
    dm1.col(i).setConstant(static_cast<RealScalar>(i+1));
  }
  int d = std::min(rows,cols);
  int a = std::max(0,cols-d-supers);
  int b = std::max(0,rows-d-subs);
  if(a>0) dm1.block(0,d+supers,rows,a).setZero();
  dm1.block(0,supers+1,cols-supers-1-a,cols-supers-1-a).template triangularView<UpperTriangular>().setZero();
  dm1.block(subs+1,0,rows-subs-1-b,rows-subs-1-b).template triangularView<LowerTriangular>().setZero();
  if(b>0) dm1.block(d+subs,0,b,cols).setZero();
  //std::cerr << m.m_data << "\n\n" << m.toDense() << "\n\n" << dm1 << "\n\n";
  VERIFY_IS_APPROX(dm1,m.toDense());

}

void test_bandmatrix()
{
  for(int i = 0; i < 10*g_repeat ; i++) {
    int rows = ei_random<int>(1,10);
    int cols = ei_random<int>(1,10);
    int sups = ei_random<int>(0,cols-1);
    int subs = ei_random<int>(0,rows-1);
    CALL_SUBTEST(bandmatrix(BandMatrix<float>(rows,cols,sups,subs)) );
  }
}
