// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_float16
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU


#include "main.h"
#include <Eigen/src/Core/arch/CUDA/Half.h>

using Eigen::half;

void test_conversion()
{
  // Conversion from float.
  VERIFY_IS_EQUAL(Eigen::half(1.0f).x, 0x3c00);
  VERIFY_IS_EQUAL(Eigen::half(0.5f).x, 0x3800);
  VERIFY_IS_EQUAL(Eigen::half(0.33333f).x, 0x3555);
  VERIFY_IS_EQUAL(Eigen::half(0.0f).x, 0x0000);
  VERIFY_IS_EQUAL(Eigen::half(-0.0f).x, 0x8000);
  VERIFY_IS_EQUAL(Eigen::half(65504.0f).x, 0x7bff);
  VERIFY_IS_EQUAL(Eigen::half(65536.0f).x, 0x7c00);  // Becomes infinity.

  // Denormals.
  VERIFY_IS_EQUAL(Eigen::half(-5.96046e-08f).x, 0x8001);
  VERIFY_IS_EQUAL(Eigen::half(5.96046e-08f).x, 0x0001);
  VERIFY_IS_EQUAL(Eigen::half(1.19209e-07f).x, 0x0002);

  // Verify round-to-nearest-even behavior.
  float val1 = float(Eigen::half(half_impl::__half{0x3c00}));
  float val2 = float(Eigen::half(half_impl::__half{0x3c01}));
  float val3 = float(Eigen::half(half_impl::__half{0x3c02}));
  VERIFY_IS_EQUAL(Eigen::half(0.5 * (val1 + val2)).x, 0x3c00);
  VERIFY_IS_EQUAL(Eigen::half(0.5 * (val2 + val3)).x, 0x3c02);

  // Conversion from int.
  VERIFY_IS_EQUAL(Eigen::half(-1).x, 0xbc00);
  VERIFY_IS_EQUAL(Eigen::half(0).x, 0x0000);
  VERIFY_IS_EQUAL(Eigen::half(1).x, 0x3c00);
  VERIFY_IS_EQUAL(Eigen::half(2).x, 0x4000);
  VERIFY_IS_EQUAL(Eigen::half(3).x, 0x4200);

  // Conversion from bool.
  VERIFY_IS_EQUAL(Eigen::half(false).x, 0x0000);
  VERIFY_IS_EQUAL(Eigen::half(true).x, 0x3c00);

  // Conversion to float.
  VERIFY_IS_EQUAL(float(Eigen::half(half_impl::__half{0x0000})), 0.0f);
  VERIFY_IS_EQUAL(float(Eigen::half(half_impl::__half{0x3c00})), 1.0f);

  // Denormals.
  VERIFY_IS_APPROX(float(Eigen::half(half_impl::__half{0x8001})), -5.96046e-08f);
  VERIFY_IS_APPROX(float(Eigen::half(half_impl::__half{0x0001})), 5.96046e-08f);
  VERIFY_IS_APPROX(float(Eigen::half(half_impl::__half{0x0002})), 1.19209e-07f);

  // NaNs and infinities.
  VERIFY(!isinf(float(Eigen::half(65504.0f))));  // Largest finite number.
  VERIFY(!isnan(float(Eigen::half(0.0f))));
  VERIFY(isinf(float(Eigen::half(half_impl::__half{0xfc00}))));
  VERIFY(isnan(float(Eigen::half(half_impl::__half{0xfc01}))));
  VERIFY(isinf(float(Eigen::half(half_impl::__half{0x7c00}))));
  VERIFY(isnan(float(Eigen::half(half_impl::__half{0x7c01}))));
  VERIFY(isnan(float(Eigen::half(0.0 / 0.0))));
  VERIFY(isinf(float(Eigen::half(1.0 / 0.0))));
  VERIFY(isinf(float(Eigen::half(-1.0 / 0.0))));

  // Exactly same checks as above, just directly on the half representation.
  VERIFY(!numext::isinf(Eigen::half(half_impl::__half{0x7bff})));
  VERIFY(!numext::isnan(Eigen::half(half_impl::__half{0x0000})));
  VERIFY(numext::isinf(Eigen::half(half_impl::__half{0xfc00})));
  VERIFY(numext::isnan(Eigen::half(half_impl::__half{0xfc01})));
  VERIFY(numext::isinf(Eigen::half(half_impl::__half{0x7c00})));
  VERIFY(numext::isnan(Eigen::half(half_impl::__half{0x7c01})));
  VERIFY(numext::isnan(Eigen::half(0.0 / 0.0)));
  VERIFY(numext::isinf(Eigen::half(1.0 / 0.0)));
  VERIFY(numext::isinf(Eigen::half(-1.0 / 0.0)));
}

void test_arithmetic()
{
  VERIFY_IS_EQUAL(float(Eigen::half(2) + Eigen::half(2)), 4);
  VERIFY_IS_EQUAL(float(Eigen::half(2) + Eigen::half(-2)), 0);
  VERIFY_IS_APPROX(float(Eigen::half(0.33333f) + Eigen::half(0.66667f)), 1.0f);
  VERIFY_IS_EQUAL(float(Eigen::half(2.0f) * Eigen::half(-5.5f)), -11.0f);
  VERIFY_IS_APPROX(float(Eigen::half(1.0f) / Eigen::half(3.0f)), 0.33333f);
  VERIFY_IS_EQUAL(float(-Eigen::half(4096.0f)), -4096.0f);
  VERIFY_IS_EQUAL(float(-Eigen::half(-4096.0f)), 4096.0f);
}

void test_comparison()
{
  VERIFY(Eigen::half(1.0f) > Eigen::half(0.5f));
  VERIFY(Eigen::half(0.5f) < Eigen::half(1.0f));
  VERIFY(!(Eigen::half(1.0f) < Eigen::half(0.5f)));
  VERIFY(!(Eigen::half(0.5f) > Eigen::half(1.0f)));

  VERIFY(!(Eigen::half(4.0f) > Eigen::half(4.0f)));
  VERIFY(!(Eigen::half(4.0f) < Eigen::half(4.0f)));

  VERIFY(!(Eigen::half(0.0f) < Eigen::half(-0.0f)));
  VERIFY(!(Eigen::half(-0.0f) < Eigen::half(0.0f)));
  VERIFY(!(Eigen::half(0.0f) > Eigen::half(-0.0f)));
  VERIFY(!(Eigen::half(-0.0f) > Eigen::half(0.0f)));

  VERIFY(Eigen::half(0.2f) > Eigen::half(-1.0f));
  VERIFY(Eigen::half(-1.0f) < Eigen::half(0.2f));
  VERIFY(Eigen::half(-16.0f) < Eigen::half(-15.0f));

  VERIFY(Eigen::half(1.0f) == Eigen::half(1.0f));
  VERIFY(Eigen::half(1.0f) != Eigen::half(2.0f));

  // Comparisons with NaNs and infinities.
  VERIFY(!(Eigen::half(0.0 / 0.0) == Eigen::half(0.0 / 0.0)));
  VERIFY(!(Eigen::half(0.0 / 0.0) != Eigen::half(0.0 / 0.0)));

  VERIFY(!(Eigen::half(1.0) == Eigen::half(0.0 / 0.0)));
  VERIFY(!(Eigen::half(1.0) < Eigen::half(0.0 / 0.0)));
  VERIFY(!(Eigen::half(1.0) > Eigen::half(0.0 / 0.0)));
  VERIFY(!(Eigen::half(1.0) != Eigen::half(0.0 / 0.0)));

  VERIFY(Eigen::half(1.0) < Eigen::half(1.0 / 0.0));
  VERIFY(Eigen::half(1.0) > Eigen::half(-1.0 / 0.0));
}


void test_basic_functions()
{
  VERIFY_IS_EQUAL(float(numext::abs(Eigen::half(3.5f))), 3.5f);
  VERIFY_IS_EQUAL(float(numext::abs(Eigen::half(-3.5f))), 3.5f);

  VERIFY_IS_EQUAL(float(numext::floor(Eigen::half(3.5f))), 3.0f);
  VERIFY_IS_EQUAL(float(numext::floor(Eigen::half(-3.5f))), -4.0f);

  VERIFY_IS_EQUAL(float(numext::ceil(Eigen::half(3.5f))), 4.0f);
  VERIFY_IS_EQUAL(float(numext::ceil(Eigen::half(-3.5f))), -3.0f);

  VERIFY_IS_APPROX(float(numext::sqrt(Eigen::half(0.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::sqrt(Eigen::half(4.0f))), 2.0f);

  VERIFY_IS_APPROX(float(numext::pow(Eigen::half(0.0f), Eigen::half(1.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::pow(Eigen::half(2.0f), Eigen::half(2.0f))), 4.0f);

  VERIFY_IS_EQUAL(float(numext::exp(Eigen::half(0.0f))), 1.0f);
  VERIFY_IS_APPROX(float(numext::exp(Eigen::half(EIGEN_PI))), float(20.0 + EIGEN_PI));

  VERIFY_IS_EQUAL(float(numext::log(Eigen::half(1.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::log(Eigen::half(10.0f))), 2.30273f);
}

void test_trigonometric_functions()
{
  VERIFY_IS_APPROX(numext::cos(Eigen::half(0.0f)), Eigen::half(cosf(0.0f)));
  VERIFY_IS_APPROX(numext::cos(Eigen::half(EIGEN_PI)), Eigen::half(cosf(EIGEN_PI)));
  VERIFY_IS_APPROX_OR_LESS_THAN(numext::cos(Eigen::half(EIGEN_PI/2)), NumTraits<Eigen::half>::epsilon() * Eigen::half(5));
  VERIFY_IS_APPROX_OR_LESS_THAN(numext::cos(Eigen::half(3*EIGEN_PI/2)), NumTraits<Eigen::half>::epsilon() * Eigen::half(5));
  VERIFY_IS_APPROX(numext::cos(Eigen::half(3.5f)), Eigen::half(cosf(3.5f)));

  VERIFY_IS_APPROX(numext::sin(Eigen::half(0.0f)), Eigen::half(sinf(0.0f)));
  VERIFY_IS_APPROX_OR_LESS_THAN(numext::sin(Eigen::half(EIGEN_PI)), NumTraits<Eigen::half>::epsilon() * Eigen::half(10));

  VERIFY_IS_APPROX(numext::sin(Eigen::half(EIGEN_PI/2)), Eigen::half(sinf(EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::sin(Eigen::half(3*EIGEN_PI/2)), Eigen::half(sinf(3*EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::sin(Eigen::half(3.5f)), Eigen::half(sinf(3.5f)));

  VERIFY_IS_APPROX(numext::tan(Eigen::half(0.0f)), Eigen::half(tanf(0.0f)));
  VERIFY_IS_APPROX_OR_LESS_THAN(numext::tan(Eigen::half(EIGEN_PI)), NumTraits<Eigen::half>::epsilon() * Eigen::half(10));
  VERIFY_IS_APPROX(numext::tan(Eigen::half(3.5f)), Eigen::half(tanf(3.5f)));
}

void test_cxx11_float16()
{
  CALL_SUBTEST(test_conversion());
  CALL_SUBTEST(test_arithmetic());
  CALL_SUBTEST(test_comparison());
  CALL_SUBTEST(test_basic_functions());
  CALL_SUBTEST(test_trigonometric_functions());
}
