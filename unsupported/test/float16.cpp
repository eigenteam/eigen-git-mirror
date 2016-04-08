// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC float16

#include "main.h"
#include <Eigen/src/Core/arch/CUDA/Half.h>

using Eigen::half;

void test_conversion()
{
  // Conversion from float.
  VERIFY_IS_EQUAL(half(1.0f).x, 0x3c00);
  VERIFY_IS_EQUAL(half(0.5f).x, 0x3800);
  VERIFY_IS_EQUAL(half(0.33333f).x, 0x3555);
  VERIFY_IS_EQUAL(half(0.0f).x, 0x0000);
  VERIFY_IS_EQUAL(half(-0.0f).x, 0x8000);
  VERIFY_IS_EQUAL(half(65504.0f).x, 0x7bff);
  VERIFY_IS_EQUAL(half(65536.0f).x, 0x7c00);  // Becomes infinity.

  // Denormals.
  VERIFY_IS_EQUAL(half(-5.96046e-08f).x, 0x8001);
  VERIFY_IS_EQUAL(half(5.96046e-08f).x, 0x0001);
  VERIFY_IS_EQUAL(half(1.19209e-07f).x, 0x0002);

  // Verify round-to-nearest-even behavior.
  float val1 = float(half(__half{0x3c00}));
  float val2 = float(half(__half{0x3c01}));
  float val3 = float(half(__half{0x3c02}));
  VERIFY_IS_EQUAL(half(0.5 * (val1 + val2)).x, 0x3c00);
  VERIFY_IS_EQUAL(half(0.5 * (val2 + val3)).x, 0x3c02);

  // Conversion from int.
  VERIFY_IS_EQUAL(half(-1).x, 0xbc00);
  VERIFY_IS_EQUAL(half(0).x, 0x0000);
  VERIFY_IS_EQUAL(half(1).x, 0x3c00);
  VERIFY_IS_EQUAL(half(2).x, 0x4000);
  VERIFY_IS_EQUAL(half(3).x, 0x4200);

  // Conversion from bool.
  VERIFY_IS_EQUAL(half(false).x, 0x0000);
  VERIFY_IS_EQUAL(half(true).x, 0x3c00);

  // Conversion to float.
  VERIFY_IS_EQUAL(float(half(__half{0x0000})), 0.0f);
  VERIFY_IS_EQUAL(float(half(__half{0x3c00})), 1.0f);

  // Denormals.
  VERIFY_IS_APPROX(float(half(__half{0x8001})), -5.96046e-08f);
  VERIFY_IS_APPROX(float(half(__half{0x0001})), 5.96046e-08f);
  VERIFY_IS_APPROX(float(half(__half{0x0002})), 1.19209e-07f);

  // NaNs and infinities.
  VERIFY(!(numext::isinf)(float(half(65504.0f))));  // Largest finite number.
  VERIFY(!(numext::isnan)(float(half(0.0f))));
  VERIFY((numext::isinf)(float(half(__half{0xfc00}))));
  VERIFY((numext::isnan)(float(half(__half{0xfc01}))));
  VERIFY((numext::isinf)(float(half(__half{0x7c00}))));
  VERIFY((numext::isnan)(float(half(__half{0x7c01}))));
  VERIFY((numext::isnan)(float(half(0.0 / 0.0))));
  VERIFY((numext::isinf)(float(half(1.0 / 0.0))));
  VERIFY((numext::isinf)(float(half(-1.0 / 0.0))));

  // Exactly same checks as above, just directly on the half representation.
  VERIFY(!(numext::isinf)(half(__half{0x7bff})));
  VERIFY(!(numext::isnan)(half(__half{0x0000})));
  VERIFY((numext::isinf)(half(__half{0xfc00})));
  VERIFY((numext::isnan)(half(__half{0xfc01})));
  VERIFY((numext::isinf)(half(__half{0x7c00})));
  VERIFY((numext::isnan)(half(__half{0x7c01})));
  VERIFY((numext::isnan)(half(0.0 / 0.0)));
  VERIFY((numext::isinf)(half(1.0 / 0.0)));
  VERIFY((numext::isinf)(half(-1.0 / 0.0)));
}

void test_arithmetic()
{
  VERIFY_IS_EQUAL(float(half(2) + half(2)), 4);
  VERIFY_IS_EQUAL(float(half(2) + half(-2)), 0);
  VERIFY_IS_APPROX(float(half(0.33333f) + half(0.66667f)), 1.0f);
  VERIFY_IS_EQUAL(float(half(2.0f) * half(-5.5f)), -11.0f);
  VERIFY_IS_APPROX(float(half(1.0f) / half(3.0f)), 0.33333f);
  VERIFY_IS_EQUAL(float(-half(4096.0f)), -4096.0f);
  VERIFY_IS_EQUAL(float(-half(-4096.0f)), 4096.0f);
}

void test_comparison()
{
  VERIFY(half(1.0f) > half(0.5f));
  VERIFY(half(0.5f) < half(1.0f));
  VERIFY(!(half(1.0f) < half(0.5f)));
  VERIFY(!(half(0.5f) > half(1.0f)));

  VERIFY(!(half(4.0f) > half(4.0f)));
  VERIFY(!(half(4.0f) < half(4.0f)));

  VERIFY(!(half(0.0f) < half(-0.0f)));
  VERIFY(!(half(-0.0f) < half(0.0f)));
  VERIFY(!(half(0.0f) > half(-0.0f)));
  VERIFY(!(half(-0.0f) > half(0.0f)));

  VERIFY(half(0.2f) > half(-1.0f));
  VERIFY(half(-1.0f) < half(0.2f));
  VERIFY(half(-16.0f) < half(-15.0f));

  VERIFY(half(1.0f) == half(1.0f));
  VERIFY(half(1.0f) != half(2.0f));

  // Comparisons with NaNs and infinities.
  VERIFY(!(half(0.0 / 0.0) == half(0.0 / 0.0)));
  VERIFY(half(0.0 / 0.0) != half(0.0 / 0.0));

  VERIFY(!(half(1.0) == half(0.0 / 0.0)));
  VERIFY(!(half(1.0) < half(0.0 / 0.0)));
  VERIFY(!(half(1.0) > half(0.0 / 0.0)));
  VERIFY(half(1.0) != half(0.0 / 0.0));

  VERIFY(half(1.0) < half(1.0 / 0.0));
  VERIFY(half(1.0) > half(-1.0 / 0.0));
}

void test_functions()
{
  VERIFY_IS_EQUAL(float(numext::abs(half(3.5f))), 3.5f);
  VERIFY_IS_EQUAL(float(numext::abs(half(-3.5f))), 3.5f);

  VERIFY_IS_EQUAL(float(numext::exp(half(0.0f))), 1.0f);
  VERIFY_IS_APPROX(float(numext::exp(half(EIGEN_PI))), float(20.0 + EIGEN_PI));

  VERIFY_IS_EQUAL(float(numext::log(half(1.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::log(half(10.0f))), 2.30273f);
}

void test_float16()
{
  CALL_SUBTEST(test_conversion());
  CALL_SUBTEST(test_arithmetic());
  CALL_SUBTEST(test_comparison());
  CALL_SUBTEST(test_functions());
}
