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
#include <Eigen/LU>

template<typename MatrixType> void lu_non_invertible()
{
  /* this test covers the following files:
     LU.h
  */
  int rows = ei_random<int>(20,200), cols = ei_random<int>(20,200), cols2 = ei_random<int>(20,200);
  int rank = ei_random<int>(1, std::min(rows, cols)-1);

  MatrixType m1(rows, cols), m2(cols, cols2), m3(rows, cols2), k(1,1);
  createRandomMatrixOfRank(rank, rows, cols, m1);

  LU<MatrixType> lu(m1);
  typename LU<MatrixType>::KernelResultType m1kernel = lu.kernel();
  typename LU<MatrixType>::ImageResultType m1image = lu.image();

  VERIFY(rank == lu.rank());
  VERIFY(cols - lu.rank() == lu.dimensionOfKernel());
  VERIFY(!lu.isInjective());
  VERIFY(!lu.isInvertible());
  VERIFY(lu.isSurjective() == (lu.rank() == rows));
  VERIFY((m1 * m1kernel).isMuchSmallerThan(m1));
  VERIFY(m1image.lu().rank() == rank);
  MatrixType sidebyside(m1.rows(), m1.cols() + m1image.cols());
  sidebyside << m1, m1image;
  VERIFY(sidebyside.lu().rank() == rank);
  m2 = MatrixType::Random(cols,cols2);
  m3 = m1*m2;
  m2 = MatrixType::Random(cols,cols2);
  lu.solve(m3, &m2);
  VERIFY_IS_APPROX(m3, m1*m2);
  m3 = MatrixType::Random(rows,cols2);
  VERIFY(!lu.solve(m3, &m2));
}

template<typename MatrixType> void lu_invertible()
{
  /* this test covers the following files:
     LU.h
  */
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  int size = ei_random<int>(10,200);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size,size);

  if (ei_is_same_type<RealScalar,float>::ret)
  {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size,size*2);
    m1 += a * a.adjoint();
  }

  LU<MatrixType> lu(m1);
  VERIFY(0 == lu.dimensionOfKernel());
  VERIFY(size == lu.rank());
  VERIFY(lu.isInjective());
  VERIFY(lu.isSurjective());
  VERIFY(lu.isInvertible());
  VERIFY(lu.image().lu().isInvertible());
  m3 = MatrixType::Random(size,size);
  lu.solve(m3, &m2);
  VERIFY_IS_APPROX(m3, m1*m2);
  VERIFY_IS_APPROX(m2, lu.inverse()*m3);
  m3 = MatrixType::Random(size,size);
  VERIFY(lu.solve(m3, &m2));
}

void test_lu()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( lu_non_invertible<MatrixXf>() );
    CALL_SUBTEST( lu_non_invertible<MatrixXd>() );
    CALL_SUBTEST( lu_non_invertible<MatrixXcf>() );
    CALL_SUBTEST( lu_non_invertible<MatrixXcd>() );
    CALL_SUBTEST( lu_invertible<MatrixXf>() );
    CALL_SUBTEST( lu_invertible<MatrixXd>() );
    CALL_SUBTEST( lu_invertible<MatrixXcf>() );
    CALL_SUBTEST( lu_invertible<MatrixXcd>() );
  }
}
