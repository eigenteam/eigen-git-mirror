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

template<typename T> bool isFinite(const T& x)
{
  return x==x && x>=NumTraits<T>::lowest() && x<=NumTraits<T>::highest();
}

template<typename MatrixType> void stable_norm(const MatrixType& m)
{
  /* this test covers the following files:
     StableNorm.h
  */

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  // Check the basic machine-dependent constants.
  {
    int ibeta, it, iemin, iemax;

    ibeta = std::numeric_limits<RealScalar>::radix;         // base for floating-point numbers
    it    = std::numeric_limits<RealScalar>::digits;        // number of base-beta digits in mantissa
    iemin = std::numeric_limits<RealScalar>::min_exponent;  // minimum exponent
    iemax = std::numeric_limits<RealScalar>::max_exponent;  // maximum exponent

    VERIFY( (!(iemin > 1 - 2*it || 1+it>iemax || (it==2 && ibeta<5) || (it<=4 && ibeta <= 3 ) || it<2))
           && "the stable norm algorithm cannot be guaranteed on this computer");
  }


  int rows = m.rows();
  int cols = m.cols();

  Scalar big = ei_random<Scalar>() * (std::numeric_limits<RealScalar>::max() * RealScalar(1e-4));
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

  // test isFinite
  VERIFY(!isFinite( ei_abs(big)/RealScalar(0)));
  VERIFY(!isFinite(-ei_abs(big)/RealScalar(0)));
  VERIFY(!isFinite(ei_sqrt(-ei_abs(big))));

  // test overflow
  VERIFY(isFinite(ei_sqrt(size)*ei_abs(big)));
  #ifdef EIGEN_VECTORIZE_SSE
  // since x87 FPU uses 80bits of precision overflow is not detected
  if(ei_packet_traits<Scalar>::size>1)
  {
    VERIFY_IS_NOT_APPROX(static_cast<Scalar>(vbig.norm()),   ei_sqrt(size)*big); // here the default norm must fail
  }
  #endif
  VERIFY_IS_APPROX(vbig.stableNorm(), ei_sqrt(size)*ei_abs(big));
  VERIFY_IS_APPROX(vbig.blueNorm(),   ei_sqrt(size)*ei_abs(big));
  VERIFY_IS_APPROX(vbig.hypotNorm(),  ei_sqrt(size)*ei_abs(big));

  // test underflow
  VERIFY(isFinite(ei_sqrt(size)*ei_abs(small)));
  #ifdef EIGEN_VECTORIZE_SSE
  // since x87 FPU uses 80bits of precision underflow is not detected
  if(ei_packet_traits<Scalar>::size>1)
  {
    VERIFY_IS_NOT_APPROX(static_cast<Scalar>(vsmall.norm()),   ei_sqrt(size)*small); // here the default norm must fail
  }
  #endif
  VERIFY_IS_APPROX(vsmall.stableNorm(), ei_sqrt(size)*ei_abs(small));
  VERIFY_IS_APPROX(vsmall.blueNorm(),   ei_sqrt(size)*ei_abs(small));
  VERIFY_IS_APPROX(vsmall.hypotNorm(),  ei_sqrt(size)*ei_abs(small));

// Test compilation of cwise() version
  VERIFY_IS_APPROX(vrand.colwise().stableNorm(),      vrand.colwise().norm());
  VERIFY_IS_APPROX(vrand.colwise().blueNorm(),        vrand.colwise().norm());
  VERIFY_IS_APPROX(vrand.colwise().hypotNorm(),       vrand.colwise().norm());
  VERIFY_IS_APPROX(vrand.rowwise().stableNorm(),      vrand.rowwise().norm());
  VERIFY_IS_APPROX(vrand.rowwise().blueNorm(),        vrand.rowwise().norm());
  VERIFY_IS_APPROX(vrand.rowwise().hypotNorm(),       vrand.rowwise().norm());
}

void test_stable_norm()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( stable_norm(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( stable_norm(Vector4d()) );
    CALL_SUBTEST_3( stable_norm(VectorXd(ei_random<int>(10,2000))) );
    CALL_SUBTEST_4( stable_norm(VectorXf(ei_random<int>(10,2000))) );
    CALL_SUBTEST_5( stable_norm(VectorXcd(ei_random<int>(10,2000))) );
  }
}
