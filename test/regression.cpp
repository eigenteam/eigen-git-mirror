// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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
#include <Eigen/Regression>

template<typename VectorType,
         typename BigVecType>
void makeNoisyCohyperplanarPoints(int numPoints,
                                  VectorType **points,
                                  BigVecType *coeffs,
                                  typename VectorType::Scalar noiseAmplitude )
{
  typedef typename VectorType::Scalar Scalar;
  const int size = points[0]->size();
  // pick a random hyperplane, store the coefficients of its equation
  coeffs->resize(size + 1);
  for(int j = 0; j < size + 1; j++)
  {
    do {
      coeffs->coeffRef(j) = ei_random<Scalar>();
    } while(ei_abs(coeffs->coeffRef(j)) < 0.5);
  }

  // now pick numPoints random points on this hyperplane
  for(int i = 0; i < numPoints; i++)
  {
    VectorType& cur_point = *(points[i]);
    do
    {
      cur_point = VectorType::Random(size)/*.normalized()*/;
      // project cur_point onto the hyperplane
      Scalar x = - (coeffs->start(size).cwise()*cur_point).sum();
      cur_point *= coeffs->coeff(size) / x;
    } while( ei_abs(cur_point.norm()) < 0.5
          || ei_abs(cur_point.norm()) > 2.0 );
  }

  // add some noise to these points
  for(int i = 0; i < numPoints; i++ )
    *(points[i]) += noiseAmplitude * VectorType::Random(size);
}

template<typename VectorType,
         typename BigVecType>
void check_fitHyperplane(int numPoints,
                         VectorType **points,
                         BigVecType *coeffs,
                         typename VectorType::Scalar tolerance)
{
  int size = points[0]->size();
  BigVecType result(size + 1);
  fitHyperplane(numPoints, points, &result);
  result /= result.coeff(size);
  result *= coeffs->coeff(size);
  typename VectorType::Scalar error = (result - *coeffs).norm() / coeffs->norm();
  VERIFY(ei_abs(error) < ei_abs(tolerance));
}

void test_regression()
{
  for(int i = 0; i < g_repeat; i++)
  {
    {
      Vector2f points2f [1000];
      Vector2f *points2f_ptrs [1000];
      for(int i = 0; i < 1000; i++) points2f_ptrs[i] = &(points2f[i]);
      Vector3f coeffs3f;
      makeNoisyCohyperplanarPoints(1000, points2f_ptrs, &coeffs3f, 0.01f);
      CALL_SUBTEST(check_fitHyperplane(10, points2f_ptrs, &coeffs3f, 0.05f));
      CALL_SUBTEST(check_fitHyperplane(100, points2f_ptrs, &coeffs3f, 0.01f));
      CALL_SUBTEST(check_fitHyperplane(1000, points2f_ptrs, &coeffs3f, 0.002f));
    }

    {
      Vector4d points4d [1000];
      Vector4d *points4d_ptrs [1000];
      for(int i = 0; i < 1000; i++) points4d_ptrs[i] = &(points4d[i]);
      Matrix<double,5,1> coeffs5d;
      makeNoisyCohyperplanarPoints(1000, points4d_ptrs, &coeffs5d, 0.01);
      CALL_SUBTEST(check_fitHyperplane(10, points4d_ptrs, &coeffs5d, 0.05));
      CALL_SUBTEST(check_fitHyperplane(100, points4d_ptrs, &coeffs5d, 0.01));
      CALL_SUBTEST(check_fitHyperplane(1000, points4d_ptrs, &coeffs5d, 0.002));
    }

    {
      VectorXcd *points11cd_ptrs[1000];
      for(int i = 0; i < 1000; i++) points11cd_ptrs[i] = new VectorXcd(11);
      VectorXcd *coeffs12cd = new VectorXcd(12);
      makeNoisyCohyperplanarPoints(1000, points11cd_ptrs, coeffs12cd, 0.01);
      CALL_SUBTEST(check_fitHyperplane(100, points11cd_ptrs, coeffs12cd, 0.025));
      CALL_SUBTEST(check_fitHyperplane(1000, points11cd_ptrs, coeffs12cd, 0.006));
    }
  }
}
