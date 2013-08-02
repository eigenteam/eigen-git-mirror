// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
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

#include <Eigen/Core>

template <typename MatrixType>
void rvalue_copyassign(const MatrixType& m)
{  
#ifdef EIGEN_HAVE_RVALUE_REFERENCES
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  
  // create a temporary which we are about to destroy by moving
  MatrixType tmp = m;
  long src_address = reinterpret_cast<long>(tmp.data());
  
  // move the temporary to n
  MatrixType n = std::move(tmp);
  long dst_address = reinterpret_cast<long>(n.data());

  if (MatrixType::RowsAtCompileTime==Dynamic|| MatrixType::ColsAtCompileTime==Dynamic)
  {
    // verify that we actually moved the guts
    VERIFY_IS_EQUAL(src_address, dst_address);
  }

  // verify that the content did not change
  Scalar abs_diff = (m-n).array().abs().sum();
  VERIFY_IS_EQUAL(abs_diff, Scalar(0));
#endif
}

void test_rvalue_types()
{
  CALL_SUBTEST_1(rvalue_copyassign( MatrixXf::Random(50,50).eval() ));
  CALL_SUBTEST_1(rvalue_copyassign( ArrayXXf::Random(50,50).eval() ));

  CALL_SUBTEST_1(rvalue_copyassign( Matrix<float,1,Dynamic>::Random(50).eval() ));
  CALL_SUBTEST_1(rvalue_copyassign( Array<float,1,Dynamic>::Random(50).eval() ));

  CALL_SUBTEST_1(rvalue_copyassign( Matrix<float,Dynamic,1>::Random(50).eval() ));
  CALL_SUBTEST_1(rvalue_copyassign( Array<float,Dynamic,1>::Random(50).eval() ));
  
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,2,1>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,3,1>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,4,1>::Random().eval() ));

  CALL_SUBTEST_2(rvalue_copyassign( Array<float,2,2>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,3,3>::Random().eval() ));
  CALL_SUBTEST_2(rvalue_copyassign( Array<float,4,4>::Random().eval() ));
}
