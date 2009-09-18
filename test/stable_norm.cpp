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

template<typename MatrixType> void stable_norm(const MatrixType& m)
{
  /* this test covers the following files:
     StableNorm.h
  */

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  int rows = m.rows();
  int cols = m.cols();

  Scalar big = ei_abs(ei_random<Scalar>()) * (std::numeric_limits<RealScalar>::max() * RealScalar(1e-4));
  Scalar small = static_cast<RealScalar>(1)/big;

  MatrixType  vzero = MatrixType::Zero(rows, cols),
              vrand = MatrixType::Random(rows, cols),
              vbig(rows, cols),
              vsmall(rows,cols);

  vbig.fill(big);
  vsmall.fill(small);

  VERIFY_IS_MUCH_SMALLER_THAN(vzero.norm(), static_cast<RealScalar>(1));
  VERIFY_IS_APPROX(vrand.stableNorm(),      vrand.norm());
  VERIFY_IS_APPROX(vrand.blueNorm(),        vrand.norm());
  VERIFY_IS_APPROX(vrand.hypotNorm(),       vrand.norm());

  RealScalar size = static_cast<RealScalar>(m.size());

  // test overflow
  VERIFY_IS_NOT_APPROX(static_cast<Scalar>(vbig.norm()),   ei_sqrt(size)*big); // here the default norm must fail
  VERIFY_IS_APPROX(static_cast<Scalar>(vbig.stableNorm()), ei_sqrt(size)*big);
  VERIFY_IS_APPROX(static_cast<Scalar>(vbig.blueNorm()),   ei_sqrt(size)*big);
  VERIFY_IS_APPROX(static_cast<Scalar>(vbig.hypotNorm()),  ei_sqrt(size)*big);

  // test underflow
  VERIFY_IS_NOT_APPROX(static_cast<Scalar>(vsmall.norm()),   ei_sqrt(size)*small); // here the default norm must fail
  VERIFY_IS_APPROX(static_cast<Scalar>(vsmall.stableNorm()), ei_sqrt(size)*small);
  VERIFY_IS_APPROX(static_cast<Scalar>(vsmall.blueNorm()),   ei_sqrt(size)*small);
  VERIFY_IS_APPROX(static_cast<Scalar>(vsmall.hypotNorm()),  ei_sqrt(size)*small);
}

void test_stable_norm()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( stable_norm(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( stable_norm(Vector4d()) );
    CALL_SUBTEST( stable_norm(VectorXd(ei_random<int>(10,2000))) );
    CALL_SUBTEST( stable_norm(VectorXf(ei_random<int>(10,2000))) );
    CALL_SUBTEST( stable_norm(VectorXcd(ei_random<int>(10,2000))) );
  }
}
