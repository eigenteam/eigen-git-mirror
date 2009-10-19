// This file is part of Eigen, a lightweight C++ template library
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

#define VERIFY_TRSM(TRI,XB) { \
    (XB).setRandom(); ref = (XB); \
    (TRI).solveInPlace(XB); \
    VERIFY_IS_APPROX((TRI).toDense() * (XB), ref); \
  }

template<typename Scalar> void trsm(int size,int cols)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Matrix<Scalar,Dynamic,Dynamic,ColMajor> cmLhs(size,size);
  Matrix<Scalar,Dynamic,Dynamic,RowMajor> rmLhs(size,size);

  Matrix<Scalar,Dynamic,Dynamic,ColMajor> cmRhs(size,cols), ref(size,cols);
  Matrix<Scalar,Dynamic,Dynamic,RowMajor> rmRhs(size,cols);

  cmLhs.setRandom(); cmLhs *= static_cast<RealScalar>(0.1); cmLhs.diagonal().cwise() += static_cast<RealScalar>(1);
  rmLhs.setRandom(); rmLhs *= static_cast<RealScalar>(0.1); rmLhs.diagonal().cwise() += static_cast<RealScalar>(1);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<LowerTriangular>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<UpperTriangular>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<LowerTriangular>(), rmRhs);
  VERIFY_TRSM(cmLhs.conjugate().template triangularView<UpperTriangular>(), rmRhs);

  VERIFY_TRSM(cmLhs.conjugate().template triangularView<UnitLowerTriangular>(), cmRhs);
  VERIFY_TRSM(cmLhs            .template triangularView<UnitUpperTriangular>(), rmRhs);

  VERIFY_TRSM(rmLhs            .template triangularView<LowerTriangular>(), cmRhs);
  VERIFY_TRSM(rmLhs.conjugate().template triangularView<UnitUpperTriangular>(), rmRhs);
}

void test_product_trsm()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    CALL_SUBTEST1((trsm<float>(ei_random<int>(1,320),ei_random<int>(1,320))));
    CALL_SUBTEST2((trsm<std::complex<double> >(ei_random<int>(1,320),ei_random<int>(1,320))));
  }
}
