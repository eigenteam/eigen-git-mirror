// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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
#include <Eigen/SVD>

/* this test covers the following files:
   Geometry/OrthoMethods.h
*/

template<typename Scalar> void orthomethods_3()
{
  typedef Matrix<Scalar,3,3> Matrix3;
  typedef Matrix<Scalar,3,1> Vector3;

  Vector3 v0 = Vector3::Random(),
          v1 = Vector3::Random(),
          v2 = Vector3::Random();

  // cross product
  VERIFY_IS_MUCH_SMALLER_THAN(v1.cross(v2).dot(v1), Scalar(1));
  Matrix3 mat3;
  mat3 << v0.normalized(),
         (v0.cross(v1)).normalized(),
         (v0.cross(v1).cross(v0)).normalized();
  VERIFY(mat3.isUnitary());


  // colwise/rowwise cross product
  mat3.setRandom();
  Vector3 vec3 = Vector3::Random();
  Matrix3 mcross;
  int i = ei_random<int>(0,2);
  mcross = mat3.colwise().cross(vec3);
  VERIFY_IS_APPROX(mcross.col(i), mat3.col(i).cross(vec3));
  mcross = mat3.rowwise().cross(vec3);
  VERIFY_IS_APPROX(mcross.row(i), mat3.row(i).cross(vec3));

}

template<typename Scalar, int Size> void orthomethods(int size=Size)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar,Size,1> VectorType;
  typedef Matrix<Scalar,3,Size> Matrix3N;
  typedef Matrix<Scalar,Size,3> MatrixN3;
  typedef Matrix<Scalar,3,1> Vector3;

  VectorType v0 = VectorType::Random(size),
             v1 = VectorType::Random(size),
             v2 = VectorType::Random(size);

  // unitOrthogonal
  VERIFY_IS_MUCH_SMALLER_THAN(v0.unitOrthogonal().dot(v0), Scalar(1));
  VERIFY_IS_APPROX(v0.unitOrthogonal().norm(), RealScalar(1));

  if (size>3)
  {
    v0.template start<3>().setZero();
    v0.end(size-3).setRandom();

    VERIFY_IS_MUCH_SMALLER_THAN(v0.unitOrthogonal().dot(v0), Scalar(1));
    VERIFY_IS_APPROX(v0.unitOrthogonal().norm(), RealScalar(1));
  }

  // colwise/rowwise cross product
  Vector3 vec3 = Vector3::Random();
  int i = ei_random<int>(0,size-1);

  Matrix3N mat3N(3,size), mcross3N(3,size);
  mat3N.setRandom();
  mcross3N = mat3N.colwise().cross(vec3);
  VERIFY_IS_APPROX(mcross3N.col(i), mat3N.col(i).cross(vec3));

  MatrixN3 matN3(size,3), mcrossN3(size,3);
  matN3.setRandom();
  mcrossN3 = matN3.rowwise().cross(vec3);
  VERIFY_IS_APPROX(mcrossN3.row(i), matN3.row(i).cross(vec3));
}

void test_geo_orthomethods()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( orthomethods_3<float>() );
    CALL_SUBTEST( orthomethods_3<double>() );
    CALL_SUBTEST( (orthomethods<float,2>()) );
    CALL_SUBTEST( (orthomethods<double,2>()) );
    CALL_SUBTEST( (orthomethods<float,3>()) );
    CALL_SUBTEST( (orthomethods<double,3>()) );
    CALL_SUBTEST( (orthomethods<float,7>()) );
    CALL_SUBTEST( (orthomethods<std::complex<double>,8>()) );
    CALL_SUBTEST( (orthomethods<float,Dynamic>(36)) );
    CALL_SUBTEST( (orthomethods<double,Dynamic>(35)) );
  }
}
