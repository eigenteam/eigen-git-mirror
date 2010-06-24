// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

// work around "uninitialized" warnings and give that option some testing
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#ifndef EIGEN_NO_STATIC_ASSERT
#define EIGEN_NO_STATIC_ASSERT // turn static asserts into runtime asserts in order to check them
#endif

#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE // SSE intrinsics aren't designed to allow mixing types
#endif

#include "main.h"

using namespace std;

template<int SizeAtCompileType> void mixingtypes(int size = SizeAtCompileType)
{
  typedef Matrix<float, SizeAtCompileType, SizeAtCompileType> Mat_f;
  typedef Matrix<double, SizeAtCompileType, SizeAtCompileType> Mat_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, SizeAtCompileType> Mat_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, SizeAtCompileType> Mat_cd;
  typedef Matrix<float, SizeAtCompileType, 1> Vec_f;
  typedef Matrix<double, SizeAtCompileType, 1> Vec_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, 1> Vec_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, 1> Vec_cd;

  Mat_f mf = Mat_f::Random(size,size);
  Mat_d md = mf.template cast<double>();
  Mat_cf mcf = Mat_cf::Random(size,size);
  Mat_cd mcd = mcf.template cast<complex<double> >();
  Vec_f vf = Vec_f::Random(size,1);
  Vec_d vd = vf.template cast<double>();
  Vec_cf vcf = Vec_cf::Random(size,1);
  Vec_cd vcd = vcf.template cast<complex<double> >();
  float           sf  = ei_random<float>();
  double          sd  = ei_random<double>();
  complex<float>  scf = ei_random<complex<float> >();
  complex<double> scd = ei_random<complex<double> >();


  mf+mf;
  VERIFY_RAISES_ASSERT(mf+md);
  VERIFY_RAISES_ASSERT(mf+mcf);
  VERIFY_RAISES_ASSERT(vf=vd);
  VERIFY_RAISES_ASSERT(vf+=vd);
  VERIFY_RAISES_ASSERT(mcd=md);

  // check scalar products
  VERIFY_IS_APPROX(vcf * sf , vcf * complex<float>(sf));
  VERIFY_IS_APPROX(sd * vcd, complex<double>(sd) * vcd);
  VERIFY_IS_APPROX(vf * scf , vf.template cast<complex<float> >() * scf);
  VERIFY_IS_APPROX(scd * vd, scd * vd.template cast<complex<double> >());

  // check dot product
  vf.dot(vf);
#if 0 // we get other compilation errors here than just static asserts
  VERIFY_RAISES_ASSERT(vd.dot(vf));
#endif
  VERIFY_RAISES_ASSERT(vcf.dot(vf)); // yeah eventually we should allow this but i'm too lazy to make that change now in Dot.h
                                     // especially as that might be rewritten as cwise product .sum() which would make that automatic.

  // check diagonal product
  VERIFY_IS_APPROX(vf.asDiagonal() * mcf, vf.template cast<complex<float> >().asDiagonal() * mcf);
  VERIFY_IS_APPROX(vcd.asDiagonal() * md, vcd.asDiagonal() * md.template cast<complex<double> >());
  VERIFY_IS_APPROX(mcf * vf.asDiagonal(), mcf * vf.template cast<complex<float> >().asDiagonal());
  VERIFY_IS_APPROX(md * vcd.asDiagonal(), md.template cast<complex<double> >() * vcd.asDiagonal());
//   vd.asDiagonal() * mf;    // does not even compile
//   vcd.asDiagonal() * mf;   // does not even compile

  // check inner product
  VERIFY_IS_APPROX((vf.transpose() * vcf).value(), (vf.template cast<complex<float> >().transpose() * vcf).value());

  // check outer product
  VERIFY_IS_APPROX((vf * vcf.transpose()).eval(), (vf.template cast<complex<float> >() * vcf.transpose()).eval());

  // coeff wise product

  VERIFY_IS_APPROX((vf * vcf.transpose()).eval(), (vf.template cast<complex<float> >() * vcf.transpose()).eval());

  Mat_cd mcd2 = mcd;
  VERIFY_IS_APPROX(mcd.array() *= md.array(), mcd2.array() *= md.array().template cast<std::complex<double> >());
}


void mixingtypes_large(int size)
{
  static const int SizeAtCompileType = Dynamic;
  typedef Matrix<float, SizeAtCompileType, SizeAtCompileType> Mat_f;
  typedef Matrix<double, SizeAtCompileType, SizeAtCompileType> Mat_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, SizeAtCompileType> Mat_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, SizeAtCompileType> Mat_cd;
  typedef Matrix<float, SizeAtCompileType, 1> Vec_f;
  typedef Matrix<double, SizeAtCompileType, 1> Vec_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, 1> Vec_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, 1> Vec_cd;

  Mat_f mf(size,size);
  Mat_d md(size,size);
  Mat_cf mcf(size,size);
  Mat_cd mcd(size,size);
  Vec_f vf(size,1);
  Vec_d vd(size,1);
  Vec_cf vcf(size,1);
  Vec_cd vcd(size,1);

  mf*mf;
  // FIXME large products does not allow mixing types
  VERIFY_RAISES_ASSERT(md*mcd);
  VERIFY_RAISES_ASSERT(mcd*md);
  VERIFY_RAISES_ASSERT(mf*vcf);
  VERIFY_RAISES_ASSERT(mcf*vf);
//   VERIFY_RAISES_ASSERT(mcf *= mf); // does not even compile
//   VERIFY_RAISES_ASSERT(vcd = md*vcd); // does not even compile (cannot convert complex to double)
//   VERIFY_RAISES_ASSERT(vcf = mcf*vf);

//   VERIFY_RAISES_ASSERT(mf*md);       // does not even compile
//   VERIFY_RAISES_ASSERT(mcf*mcd);     // does not even compile
//   VERIFY_RAISES_ASSERT(mcf*vcd);     // does not even compile
//   VERIFY_RAISES_ASSERT(vcf = mf*vf);
}

template<int SizeAtCompileType> void mixingtypes_small()
{
  int size = SizeAtCompileType;
  typedef Matrix<float, SizeAtCompileType, SizeAtCompileType> Mat_f;
  typedef Matrix<double, SizeAtCompileType, SizeAtCompileType> Mat_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, SizeAtCompileType> Mat_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, SizeAtCompileType> Mat_cd;
  typedef Matrix<float, SizeAtCompileType, 1> Vec_f;
  typedef Matrix<double, SizeAtCompileType, 1> Vec_d;
  typedef Matrix<std::complex<float>, SizeAtCompileType, 1> Vec_cf;
  typedef Matrix<std::complex<double>, SizeAtCompileType, 1> Vec_cd;

  Mat_f mf(size,size);
  Mat_d md(size,size);
  Mat_cf mcf(size,size);
  Mat_cd mcd(size,size);
  Vec_f vf(size,1);
  Vec_d vd(size,1);
  Vec_cf vcf(size,1);
  Vec_cd vcd(size,1);


  mf*mf;
  // FIXME shall we discard those products ?
  // 1) currently they work only if SizeAtCompileType is small enough
  // 2) in case we vectorize complexes this might be difficult to still allow that
  md*mcd;
  mcd*md;
  mf*vcf;
  mcf*vf;
  mcf *= mf;
  vcd = md*vcd;
  vcf = mcf*vf;
//   VERIFY_RAISES_ASSERT(mf*md);   // does not even compile
//   VERIFY_RAISES_ASSERT(mcf*mcd); // does not even compile
//   VERIFY_RAISES_ASSERT(mcf*vcd); // does not even compile
  VERIFY_RAISES_ASSERT(vcf = mf*vf);
}

void test_mixingtypes()
{
  // check that our operator new is indeed called:
  CALL_SUBTEST_1(mixingtypes<3>());
  CALL_SUBTEST_2(mixingtypes<4>());
  CALL_SUBTEST_3(mixingtypes<Dynamic>(20));

  CALL_SUBTEST_4(mixingtypes_small<4>());
  CALL_SUBTEST_5(mixingtypes_large(20));
}
