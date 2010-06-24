// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#define VERIFY_TRSM(TRI,XB) { \
    (XB).setRandom(); ref = (XB); \
    (TRI).solveInPlace(XB); \
    VERIFY_IS_APPROX((TRI).toDenseMatrix() * (XB), ref); \
  }

#define VERIFY_TRSM_ONTHERIGHT(TRI,XB) { \
    (XB).setRandom(); ref = (XB); \
    (TRI).transpose().template solveInPlace<OnTheRight>(XB.transpose()); \
    VERIFY_IS_APPROX((XB).transpose() * (TRI).transpose().toDenseMatrix(), ref.transpose()); \
  }

template<typename Scalar,int Size, int Cols> void trsolve(int size=Size,int cols=Cols)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Matrix<Scalar,Size,Size,ColMajor> cmLhs(size,size);
  Matrix<Scalar,Size,Size,RowMajor> rmLhs(size,size);

  enum { order = Size==1 ? RowMajor : ColMajor };
  Matrix<Scalar,Size,Cols,order> cmRhs(size,cols), ref(size,cols);
  Matrix<Scalar,Size,Cols,order> rmRhs(size,cols);

  cmLhs.setRandom(); cmLhs *= static_cast<RealScalar>(0.1); cmLhs.diagonal().array() += static_cast<RealScalar>(1);
  rmLhs.setRandom(); rmLhs *= static_cast<RealScalar>(0.1); rmLhs.diagonal().array() += static_cast<RealScalar>(1);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<Upper>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<Lower>(), rmRhs);
  VERIFY_TRSM(cmLhs.conjugate().template triangularView<Upper>(), rmRhs);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<UnitLower>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<UnitUpper>(), rmRhs);

  VERIFY_TRSM(rmLhs            .template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM(rmLhs.conjugate().template triangularView<UnitUpper>(), rmRhs);


  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs            .template triangularView<Upper>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs            .template triangularView<Lower>(), rmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<Upper>(), rmRhs);

  VERIFY_TRSM_ONTHERIGHT(cmLhs.conjugate().template triangularView<UnitLower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(cmLhs            .template triangularView<UnitUpper>(), rmRhs);

  VERIFY_TRSM_ONTHERIGHT(rmLhs            .template triangularView<Lower>(), cmRhs);
  VERIFY_TRSM_ONTHERIGHT(rmLhs.conjugate().template triangularView<UnitUpper>(), rmRhs);
}

void test_product_trsolve()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    // matrices
    CALL_SUBTEST_1((trsolve<float,Dynamic,Dynamic>(ei_random<int>(1,320),ei_random<int>(1,320))));
    CALL_SUBTEST_2((trsolve<double,Dynamic,Dynamic>(ei_random<int>(1,320),ei_random<int>(1,320))));
    CALL_SUBTEST_3((trsolve<std::complex<double>,Dynamic,Dynamic>(ei_random<int>(1,320),ei_random<int>(1,320))));

    // vectors
    CALL_SUBTEST_4((trsolve<std::complex<double>,Dynamic,1>(ei_random<int>(1,320))));
    CALL_SUBTEST_5((trsolve<float,1,1>()));
    CALL_SUBTEST_6((trsolve<float,1,2>()));
    CALL_SUBTEST_7((trsolve<std::complex<float>,4,1>()));
  }
}
