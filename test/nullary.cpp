// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
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

template<typename MatrixType>
bool equalsIdentity(const MatrixType& A)
{
  typedef typename MatrixType::Scalar Scalar;
  Scalar zero = static_cast<Scalar>(0);

  bool offDiagOK = true;
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = i+1; j < A.cols(); ++j) {
      offDiagOK = offDiagOK && (A(i,j) == zero);
    }
  }
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < i; ++j) {
      offDiagOK = offDiagOK && (A(i,j) == zero);
    }
  }

  bool diagOK = (A.diagonal().array() == 1).all();
  return offDiagOK && diagOK;
}

template<typename VectorType>
void testVectorType(const VectorType& base)
{
  typedef typename ei_traits<VectorType>::Index Index;
  typedef typename ei_traits<VectorType>::Scalar Scalar;
  Scalar low = ei_random<Scalar>(-500,500);
  Scalar high = ei_random<Scalar>(-500,500);
  if (low>high) std::swap(low,high);
  const Index size = base.size();
  const Scalar step = (high-low)/(size-1);

  // check whether the result yields what we expect it to do
  VectorType m(base);
  m.setLinSpaced(size,low,high);

  VectorType n(size);
  for (int i=0; i<size; ++i)
    n(i) = low+i*step;

  VERIFY( (m-n).norm() < std::numeric_limits<Scalar>::epsilon()*10e3 );

  // random access version
  m = VectorType::LinSpaced(size,low,high);
  VERIFY( (m-n).norm() < std::numeric_limits<Scalar>::epsilon()*10e3 );

  // Assignment of a RowVectorXd to a MatrixXd (regression test for bug #79).
  VERIFY( (MatrixXd(RowVectorXd::LinSpaced(3, 0, 1)) - RowVector3d(0, 0.5, 1)).norm() < std::numeric_limits<Scalar>::epsilon() );

  // These guys sometimes fail! This is not good. Any ideas how to fix them!?
  //VERIFY( m(m.size()-1) == high );
  //VERIFY( m(0) == low );

  // sequential access version
  m = VectorType::LinSpaced(Sequential,size,low,high);
  VERIFY( (m-n).norm() < std::numeric_limits<Scalar>::epsilon()*10e3 );

  // These guys sometimes fail! This is not good. Any ideas how to fix them!?
  //VERIFY( m(m.size()-1) == high );
  //VERIFY( m(0) == low );

  // check whether everything works with row and col major vectors
  Matrix<Scalar,Dynamic,1> row_vector(size);
  Matrix<Scalar,1,Dynamic> col_vector(size);
  row_vector.setLinSpaced(size,low,high);
  col_vector.setLinSpaced(size,low,high);
  VERIFY( row_vector.isApprox(col_vector.transpose(), NumTraits<Scalar>::epsilon()));

  Matrix<Scalar,Dynamic,1> size_changer(size+50);
  size_changer.setLinSpaced(size,low,high);
  VERIFY( size_changer.size() == size );
}

template<typename MatrixType>
void testMatrixType(const MatrixType& m)
{
  typedef typename MatrixType::Index Index;
  const Index rows = m.rows();
  const Index cols = m.cols();

  MatrixType A;
  A.setIdentity(rows, cols);
  VERIFY(equalsIdentity(A));
  VERIFY(equalsIdentity(MatrixType::Identity(rows, cols)));
}

void test_nullary()
{
  CALL_SUBTEST_1( testMatrixType(Matrix2d()) );
  CALL_SUBTEST_2( testMatrixType(MatrixXcf(50,50)) );
  CALL_SUBTEST_3( testMatrixType(MatrixXf(5,7)) );
  CALL_SUBTEST_4( testVectorType(VectorXd(51)) );
  CALL_SUBTEST_5( testVectorType(VectorXd(41)) );
  CALL_SUBTEST_6( testVectorType(Vector3d()) );
  CALL_SUBTEST_7( testVectorType(VectorXf(51)) );
  CALL_SUBTEST_8( testVectorType(VectorXf(41)) );
  CALL_SUBTEST_9( testVectorType(Vector3f()) );
}
