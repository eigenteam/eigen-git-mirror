// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

template<typename Scalar,int Size> void homogeneous(void)
{
  /* this test covers the following files:
     Homogeneous.h
  */

  typedef Matrix<Scalar,Size,Size> MatrixType;
  typedef Matrix<Scalar,Size,1> VectorType;

  typedef Matrix<Scalar,Size+1,Size> HMatrixType;
  typedef Matrix<Scalar,Size+1,1> HVectorType;

  typedef Matrix<Scalar,Size,Size+1>   T1MatrixType;
  typedef Matrix<Scalar,Size+1,Size+1> T2MatrixType;
  typedef Matrix<Scalar,Size+1,Size> T3MatrixType;

  Scalar largeEps = test_precision<Scalar>();
  if (ei_is_same_type<Scalar,float>::ret)
    largeEps = 1e-3f;

  VectorType v0 = VectorType::Random(),
             v1 = VectorType::Random(),
             ones = VectorType::Ones();

  HVectorType hv0 = HVectorType::Random(),
              hv1 = HVectorType::Random();

  MatrixType m0 = MatrixType::Random(),
             m1 = MatrixType::Random();

  HMatrixType hm0 = HMatrixType::Random(),
              hm1 = HMatrixType::Random();

  hv0 << v0, 1;
  VERIFY_IS_APPROX(v0.homogeneous(), hv0);
  VERIFY_IS_APPROX(v0, hv0.hnormalized());

  hm0 << m0, ones.transpose();
  VERIFY_IS_APPROX(m0.colwise().homogeneous(), hm0);
  VERIFY_IS_APPROX(m0, hm0.colwise().hnormalized());
  hm0.row(Size-1).setRandom();
  for(int j=0; j<Size; ++j)
    m0.col(j) = hm0.col(j).start(Size) / hm0(Size,j);
  VERIFY_IS_APPROX(m0, hm0.colwise().hnormalized());

  T1MatrixType t1 = T1MatrixType::Random();
  VERIFY_IS_APPROX(t1 * (v0.homogeneous().eval()), t1 * v0.homogeneous());
  VERIFY_IS_APPROX(t1 * (m0.colwise().homogeneous().eval()), t1 * m0.colwise().homogeneous());

  T2MatrixType t2 = T2MatrixType::Random();
  VERIFY_IS_APPROX(t2 * (v0.homogeneous().eval()), t2 * v0.homogeneous());
  VERIFY_IS_APPROX(t2 * (m0.colwise().homogeneous().eval()), t2 * m0.colwise().homogeneous());

  VERIFY_IS_APPROX((v0.transpose().rowwise().homogeneous().eval()) * t2,
                    v0.transpose().rowwise().homogeneous() * t2);
  VERIFY_IS_APPROX((m0.transpose().rowwise().homogeneous().eval()) * t2,
                    m0.transpose().rowwise().homogeneous() * t2);

  T3MatrixType t3 = T3MatrixType::Random();
  VERIFY_IS_APPROX((v0.transpose().rowwise().homogeneous().eval()) * t3,
                    v0.transpose().rowwise().homogeneous() * t3);
  VERIFY_IS_APPROX((m0.transpose().rowwise().homogeneous().eval()) * t3,
                    m0.transpose().rowwise().homogeneous() * t3);

  // test product with a Transform object
  Transform<Scalar, Size, Affine> Rt;
  Matrix<Scalar, Size, Dynamic> pts, Rt_pts1;

  Rt.setIdentity();
  pts.setRandom(Size,5);

  Rt_pts1 = Rt * pts.colwise().homogeneous();
  std::cerr << (Rt_pts1 - pts).sum() << "\n";
  VERIFY_IS_MUCH_SMALLER_THAN( (Rt_pts1 - pts).sum(), Scalar(1));
}

void test_geo_homogeneous()
{
  for(int i = 0; i < g_repeat; i++) {
//     CALL_SUBTEST(( homogeneous<float,1>() ));
    CALL_SUBTEST(( homogeneous<double,3>() ));
//     CALL_SUBTEST(( homogeneous<double,8>() ));
  }
}
