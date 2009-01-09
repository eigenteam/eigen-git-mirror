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
#include <Eigen/StdVector>

template<typename MatrixType>
void check_stdvector_fixedsize()
{
  MatrixType x = MatrixType::Random(), y = MatrixType::Random();
  std::vector<MatrixType> v(10), w(20, y);
  v[5] = x;
  w[6] = v[5];
  VERIFY_IS_APPROX(w[6], v[5]);
  v = w;
  for(int i = 0; i < 20; i++)
  {
    VERIFY_IS_APPROX(w[i], v[i]);
  }
  v.resize(21);
  v[20] = x;
  VERIFY_IS_APPROX(v[20], x);
  v.resize(22,y);
  VERIFY_IS_APPROX(v[21], y);
  v.push_back(x);
  VERIFY_IS_APPROX(v[22], x);
}


void test_stdvector()
{
  // some non vectorizable fixed sizes
  CALL_SUBTEST(check_stdvector_fixedsize<Vector2f>());
  CALL_SUBTEST(check_stdvector_fixedsize<Matrix3f>());
  CALL_SUBTEST(check_stdvector_fixedsize<Matrix3d>());

  // some vectorizable fixed sizes
  CALL_SUBTEST(check_stdvector_fixedsize<Vector2d>());
  CALL_SUBTEST(check_stdvector_fixedsize<Vector4f>());
  CALL_SUBTEST(check_stdvector_fixedsize<Matrix4f>());
  CALL_SUBTEST(check_stdvector_fixedsize<Matrix4d>());
  
}
