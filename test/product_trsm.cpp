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

template<typename Lhs, typename Rhs>
void solve_ref(const Lhs& lhs, Rhs& rhs)
{
  for (int j=0; j<rhs.cols(); ++j)
    lhs.solveInPlace(rhs.col(j));
}

template<typename Scalar> void trsm(int size,int cols)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Matrix<Scalar,Dynamic,Dynamic,ColMajor> cmLhs(size,size);
  Matrix<Scalar,Dynamic,Dynamic,RowMajor> rmLhs(size,size);
  
  Matrix<Scalar,Dynamic,Dynamic,ColMajor> cmRef(size,cols), cmRhs(size,cols);
  Matrix<Scalar,Dynamic,Dynamic,RowMajor> rmRef(size,cols), rmRhs(size,cols);

  cmLhs.setRandom(); cmLhs.diagonal().cwise() += 10;
  rmLhs.setRandom(); rmLhs.diagonal().cwise() += 10;

  cmRhs.setRandom(); cmRef = cmRhs;
  cmLhs.conjugate().template triangularView<LowerTriangular>().solveInPlace(cmRhs);
  solve_ref(cmLhs.conjugate().template triangularView<LowerTriangular>(),cmRef);
  VERIFY_IS_APPROX(cmRhs, cmRef);

  cmRhs.setRandom(); cmRef = cmRhs;
  cmLhs.conjugate().template triangularView<UpperTriangular>().solveInPlace(cmRhs);
  solve_ref(cmLhs.conjugate().template triangularView<UpperTriangular>(),cmRef);
  VERIFY_IS_APPROX(cmRhs, cmRef);
  
  rmRhs.setRandom(); rmRef = rmRhs;
  cmLhs.template triangularView<LowerTriangular>().solveInPlace(rmRhs);
  solve_ref(cmLhs.template triangularView<LowerTriangular>(),rmRef);
  VERIFY_IS_APPROX(rmRhs, rmRef);

  rmRhs.setRandom(); rmRef = rmRhs;
  cmLhs.template triangularView<UpperTriangular>().solveInPlace(rmRhs);
  solve_ref(cmLhs.template triangularView<UpperTriangular>(),rmRef);
  VERIFY_IS_APPROX(rmRhs, rmRef);


  cmRhs.setRandom(); cmRef = cmRhs;
  rmLhs.template triangularView<UnitLowerTriangular>().solveInPlace(cmRhs);
  solve_ref(rmLhs.template triangularView<UnitLowerTriangular>(),cmRef);
  VERIFY_IS_APPROX(cmRhs, cmRef);

  cmRhs.setRandom(); cmRef = cmRhs;
  rmLhs.template triangularView<UnitUpperTriangular>().solveInPlace(cmRhs);
  solve_ref(rmLhs.template triangularView<UnitUpperTriangular>(),cmRef);
  VERIFY_IS_APPROX(cmRhs, cmRef);

  rmRhs.setRandom(); rmRef = rmRhs;
  rmLhs.template triangularView<LowerTriangular>().solveInPlace(rmRhs);
  solve_ref(rmLhs.template triangularView<LowerTriangular>(),rmRef);
  VERIFY_IS_APPROX(rmRhs, rmRef);

  rmRhs.setRandom(); rmRef = rmRhs;
  rmLhs.template triangularView<UpperTriangular>().solveInPlace(rmRhs);
  solve_ref(rmLhs.template triangularView<UpperTriangular>(),rmRef);
  VERIFY_IS_APPROX(rmRhs, rmRef);
}
void test_product_trsm()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    trsm<float>(ei_random<int>(1,320),ei_random<int>(1,320));
    trsm<std::complex<double> >(ei_random<int>(1,320),ei_random<int>(1,320));
  }
}
