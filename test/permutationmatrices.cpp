// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

template<typename PermutationVectorType>
void randomPermutationVector(PermutationVectorType& v, int size)
{
  typedef typename PermutationVectorType::Scalar Scalar;
  v.resize(size);
  for(int i = 0; i < size; ++i) v(i) = Scalar(i);
  if(size == 1) return;
  for(int n = 0; n < 3 * size; ++n)
  {
    int i = ei_random<int>(0, size-1);
    int j;
    do j = ei_random<int>(0, size-1); while(j==i);
    std::swap(v(i), v(j));
  }
}

using namespace std;
template<typename MatrixType> void permutationmatrices(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  enum { Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime,
         Options = MatrixType::Options };
  typedef PermutationMatrix<Rows> LeftPermutationType;
  typedef Matrix<int, Rows, 1> LeftPermutationVectorType;
  typedef PermutationMatrix<Cols> RightPermutationType;
  typedef Matrix<int, Cols, 1> RightPermutationVectorType;
  
  int rows = m.rows();
  int cols = m.cols();

  MatrixType m_original = MatrixType::Random(rows,cols);
  LeftPermutationVectorType lv;
  randomPermutationVector(lv, rows);
  LeftPermutationType lp(lv);
  RightPermutationVectorType rv;
  randomPermutationVector(rv, cols);
  RightPermutationType rp(rv);
  MatrixType m_permuted = lp * m_original * rp;

  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
        VERIFY_IS_APPROX(m_original(lv(i),j), m_permuted(i,rv(j)));

  Matrix<Scalar,Rows,Rows> lm(lp);
  Matrix<Scalar,Cols,Cols> rm(rp);

  VERIFY_IS_APPROX(m_permuted, lm*m_original*rm);
}

void test_permutationmatrices()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( permutationmatrices(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( permutationmatrices(Matrix3f()) );
    CALL_SUBTEST_3( permutationmatrices(Matrix<double,3,3,RowMajor>()) );
    CALL_SUBTEST_4( permutationmatrices(Matrix4d()) );
    CALL_SUBTEST_5( permutationmatrices(Matrix<double,4,6>()) );
    CALL_SUBTEST_6( permutationmatrices(Matrix<double,Dynamic,Dynamic,RowMajor>(20, 30)) );
    CALL_SUBTEST_7( permutationmatrices(MatrixXcf(15, 10)) );
  }
}
