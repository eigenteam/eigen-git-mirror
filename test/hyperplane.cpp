// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
#include <Eigen/Geometry>
#include <Eigen/LU>

template<typename PlaneType> void hyperplane(const PlaneType& _plane)
{
  /* this test covers the following files:
     HyperPlane.h
  */

  const int dim = _plane.dim();
  typedef typename PlaneType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, PlaneType::DimAtCompileTime, 1> VectorType;

  VectorType p0 = VectorType::Random(dim);
  VectorType p1 = VectorType::Random(dim);

  VectorType n0 = VectorType::Random(dim).normalized();
  VectorType n1 = VectorType::Random(dim).normalized();
  
  PlaneType pl0(n0, p0);
  PlaneType pl1(n1, p1);

  Scalar s0 = ei_random<Scalar>();
  Scalar s1 = ei_random<Scalar>();

  VERIFY_IS_APPROX( n1.dot(n1), Scalar(1) );
  VERIFY_IS_APPROX( n1.dot(n1), Scalar(1) );
  
  VERIFY_IS_MUCH_SMALLER_THAN( pl0.distanceTo(p0), Scalar(1) );
  VERIFY_IS_APPROX( pl1.distanceTo(p1 + n1 * s0), s0 );
  VERIFY_IS_MUCH_SMALLER_THAN( pl1.distanceTo(pl1.project(p0)), Scalar(1) );
  VERIFY_IS_MUCH_SMALLER_THAN( pl1.distanceTo(p1 +  pl1.normal().unitOrthogonal() * s1), Scalar(1) );

}

void test_hyperplane()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( hyperplane(HyperPlane<float,2>()) );
    CALL_SUBTEST( hyperplane(HyperPlane<float,3>()) );
    CALL_SUBTEST( hyperplane(HyperPlane<double,4>()) );
    CALL_SUBTEST( hyperplane(HyperPlane<std::complex<double>,5>()) );
    CALL_SUBTEST( hyperplane(HyperPlane<double,Dynamic>(13)) );
  }
}
