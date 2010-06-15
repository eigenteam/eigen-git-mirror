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

template<typename Scalar> void trmm(int size,int othersize)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;

  typedef Matrix<Scalar,Dynamic,Dynamic,ColMajor> MatrixType;

  MatrixType tri(size,size), upTri(size,size), loTri(size,size),
             unitUpTri(size,size), unitLoTri(size,size);
  MatrixType ge1(size,othersize), ge2(10,size), ge3;
  Matrix<Scalar,Dynamic,Dynamic,RowMajor> rge3;

  Scalar s1 = ei_random<Scalar>(),
         s2 = ei_random<Scalar>();

  tri.setRandom();
  loTri = tri.template triangularView<Lower>();
  upTri = tri.template triangularView<Upper>();
  unitLoTri = tri.template triangularView<UnitLower>();
  unitUpTri = tri.template triangularView<UnitUpper>();
  ge1.setRandom();
  ge2.setRandom();

  VERIFY_IS_APPROX( ge3 = tri.template triangularView<Lower>() * ge1, loTri * ge1);
  VERIFY_IS_APPROX(rge3 = tri.template triangularView<Lower>() * ge1, loTri * ge1);
  VERIFY_IS_APPROX( ge3 = tri.template triangularView<Upper>() * ge1, upTri * ge1);
  VERIFY_IS_APPROX(rge3 = tri.template triangularView<Upper>() * ge1, upTri * ge1);
  VERIFY_IS_APPROX( ge3 = (s1*tri.adjoint()).template triangularView<Upper>() * (s2*ge1), s1*loTri.adjoint() * (s2*ge1));
  VERIFY_IS_APPROX(rge3 = tri.adjoint().template triangularView<Upper>() * ge1, loTri.adjoint() * ge1);
  VERIFY_IS_APPROX( ge3 = tri.adjoint().template triangularView<Lower>() * ge1, upTri.adjoint() * ge1);
  VERIFY_IS_APPROX(rge3 = tri.adjoint().template triangularView<Lower>() * ge1, upTri.adjoint() * ge1);
  VERIFY_IS_APPROX( ge3 = tri.template triangularView<Lower>() * ge2.adjoint(), loTri * ge2.adjoint());
  VERIFY_IS_APPROX(rge3 = tri.template triangularView<Lower>() * ge2.adjoint(), loTri * ge2.adjoint());
  VERIFY_IS_APPROX( ge3 = tri.template triangularView<Upper>() * ge2.adjoint(), upTri * ge2.adjoint());
  VERIFY_IS_APPROX(rge3 = tri.template triangularView<Upper>() * ge2.adjoint(), upTri * ge2.adjoint());
  VERIFY_IS_APPROX( ge3 = (s1*tri).adjoint().template triangularView<Upper>() * ge2.adjoint(), ei_conj(s1) * loTri.adjoint() * ge2.adjoint());
  VERIFY_IS_APPROX(rge3 = tri.adjoint().template triangularView<Upper>() * ge2.adjoint(), loTri.adjoint() * ge2.adjoint());
  VERIFY_IS_APPROX( ge3 = tri.adjoint().template triangularView<Lower>() * ge2.adjoint(), upTri.adjoint() * ge2.adjoint());
  VERIFY_IS_APPROX(rge3 = tri.adjoint().template triangularView<Lower>() * ge2.adjoint(), upTri.adjoint() * ge2.adjoint());

  VERIFY_IS_APPROX( ge3 = tri.template triangularView<UnitLower>() * ge1, unitLoTri * ge1);
  VERIFY_IS_APPROX(rge3 = tri.template triangularView<UnitLower>() * ge1, unitLoTri * ge1);
  VERIFY_IS_APPROX( ge3 = (s1*tri).adjoint().template triangularView<UnitUpper>() * ge2.adjoint(), ei_conj(s1) * unitLoTri.adjoint() * ge2.adjoint());
}

void test_product_trmm()
{
  for(int i = 0; i < g_repeat ; i++)
  {
    CALL_SUBTEST_1((trmm<float>(ei_random<int>(1,320),ei_random<int>(1,320))));
    CALL_SUBTEST_2((trmm<std::complex<double> >(ei_random<int>(1,320),ei_random<int>(1,320))));
  }
}
