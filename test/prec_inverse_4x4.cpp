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
#include <Eigen/LU>
#include <algorithm>

Matrix4f inverse(const Matrix4f& m)
{
  Matrix4f r;
  r(0,0) = m.minor(0,0).determinant();
  r(1,0) = -m.minor(0,1).determinant();
  r(2,0) = m.minor(0,2).determinant();
  r(3,0) = -m.minor(0,3).determinant();
  r(0,2) = m.minor(2,0).determinant();
  r(1,2) = -m.minor(2,1).determinant();
  r(2,2) = m.minor(2,2).determinant();
  r(3,2) = -m.minor(2,3).determinant();
  r(0,1) = -m.minor(1,0).determinant();
  r(1,1) = m.minor(1,1).determinant();
  r(2,1) = -m.minor(1,2).determinant();
  r(3,1) = m.minor(1,3).determinant();
  r(0,3) = -m.minor(3,0).determinant();
  r(1,3) = m.minor(3,1).determinant();
  r(2,3) = -m.minor(3,2).determinant();
  r(3,3) = m.minor(3,3).determinant();
  return r / (m(0,0)*r(0,0) + m(1,0)*r(0,1) + m(2,0)*r(0,2) + m(3,0)*r(0,3));
}

template<typename MatrixType> void inverse_permutation_4x4()
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  double error_max = 0.;
  Vector4i indices(0,1,2,3);
  for(int i = 0; i < 24; ++i)
  {
    MatrixType m = PermutationMatrix<4>(indices);
    MatrixType inv = m.inverse();
    double error = double( (m*inv-MatrixType::Identity()).norm() / epsilon<Scalar>() );
    error_max = std::max(error_max, error);
    std::next_permutation(indices.data(),indices.data()+4);
  }
  std::cerr << "inverse_permutation_4x4, Scalar = " << type_name<Scalar>() << std::endl;
  EIGEN_DEBUG_VAR(error_max);
  VERIFY(error_max < 1. );
}

template<typename MatrixType> void inverse_general_4x4(int repeat)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  double error_sum = 0., error_max = 0.;
  for(int i = 0; i < repeat; ++i)
  {
    MatrixType m;
    RealScalar absdet;
    do {
      m = MatrixType::Random();
      absdet = ei_abs(m.determinant());
    } while(absdet < 2*epsilon<Scalar>() );
    MatrixType inv = m.inverse();
    double error = double( (m*inv-MatrixType::Identity()).norm() * absdet / epsilon<Scalar>() );
    error_sum += error;
    error_max = std::max(error_max, error);
  }
  std::cerr << "inverse_general_4x4, Scalar = " << type_name<Scalar>() << std::endl;
  double error_avg = error_sum / repeat;
  EIGEN_DEBUG_VAR(error_avg);
  EIGEN_DEBUG_VAR(error_max);
  VERIFY(error_avg < (NumTraits<Scalar>::IsComplex ? 8.4 : 1.4) );
  VERIFY(error_max < (NumTraits<Scalar>::IsComplex ? 150.0 : 75.) );
}

void test_prec_inverse_4x4()
{
  CALL_SUBTEST_1((inverse_permutation_4x4<Matrix4f>()));
  CALL_SUBTEST_1(( inverse_general_4x4<Matrix4f>(200000 * g_repeat) ));

  CALL_SUBTEST_2((inverse_permutation_4x4<Matrix<double,4,4,RowMajor> >()));
  CALL_SUBTEST_2(( inverse_general_4x4<Matrix<double,4,4,RowMajor> >(200000 * g_repeat) ));

  CALL_SUBTEST_3((inverse_permutation_4x4<Matrix4cf>()));
  CALL_SUBTEST_3((inverse_general_4x4<Matrix4cf>(50000 * g_repeat)));
}
