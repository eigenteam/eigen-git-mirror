// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#ifdef EIGEN_TEST_MAX_SIZE
#undef EIGEN_TEST_MAX_SIZE
#endif

#define EIGEN_TEST_MAX_SIZE 50

#ifdef EIGEN_TEST_PART_1
#include "cholesky.cpp"
#endif

#ifdef EIGEN_TEST_PART_2
#include "lu.cpp"
#endif

#ifdef EIGEN_TEST_PART_3
#include "qr.cpp"
#endif

#ifdef EIGEN_TEST_PART_4
#include "qr_colpivoting.cpp"
#endif

#ifdef EIGEN_TEST_PART_5
#include "qr_fullpivoting.cpp"
#endif

#ifdef EIGEN_TEST_PART_6
#include "eigensolver_selfadjoint.cpp"
#endif

#ifdef EIGEN_TEST_PART_7
#include "jacobisvd.cpp"
#endif

#ifdef EIGEN_TEST_PART_8
#include "bdcsvd.cpp"
#endif

#include <Eigen/Dense>

#undef min
#undef max
#undef isnan
#undef isinf
#undef isfinite

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/number.hpp>

namespace mp = boost::multiprecision;
typedef mp::number<mp::cpp_dec_float<100>, mp::et_off> Real; // swith to et_on for testing with expression templates

namespace Eigen {
  template<> struct NumTraits<Real> : GenericNumTraits<Real> {
    static inline Real dummy_precision() { return 1e-50; }
  };

  template<typename T1,typename T2,typename T3,typename T4,typename T5>
  struct NumTraits<boost::multiprecision::detail::expression<T1,T2,T3,T4,T5> > : NumTraits<Real> {};

  template<>
  Real test_precision<Real>() { return 1e-50; }
}

namespace boost {
namespace multiprecision {
  // to make ADL works as expected:
  using boost::math::isfinite;

  // some specialization for the unit tests:
  inline bool test_isMuchSmallerThan(const Real& a, const Real& b) {
    return internal::isMuchSmallerThan(a, b, test_precision<Real>());
  }

  inline bool test_isApprox(const Real& a, const Real& b) {
    return internal::isApprox(a, b, test_precision<Real>());
  }

  inline bool test_isApproxOrLessThan(const Real& a, const Real& b) {
    return internal::isApproxOrLessThan(a, b, test_precision<Real>());
  }

  Real get_test_precision(const Real&) {
    return test_precision<Real>();
  }

  Real test_relative_error(const Real &a, const Real &b) {
    return Eigen::numext::sqrt(Real(Eigen::numext::abs2(a-b))/Real((Eigen::numext::mini)(Eigen::numext::abs2(a),Eigen::numext::abs2(b))));
  }
}
}

namespace Eigen {

}

void test_boostmultiprec()
{
  typedef Matrix<Real,Dynamic,Dynamic> Mat;

  std::cout << "NumTraits<Real>::epsilon()         = " << NumTraits<Real>::epsilon() << std::endl;
  std::cout << "NumTraits<Real>::dummy_precision() = " << NumTraits<Real>::dummy_precision() << std::endl;
  std::cout << "NumTraits<Real>::lowest()          = " << NumTraits<Real>::lowest() << std::endl;
  std::cout << "NumTraits<Real>::highest()         = " << NumTraits<Real>::highest() << std::endl;

  // chekc stream output
  {
    Mat A(10,10);
    A.setRandom();
    std::stringstream ss;
    ss << A;
  }

  for(int i = 0; i < g_repeat; i++) {
    int s = internal::random<int>(1,EIGEN_TEST_MAX_SIZE);

    CALL_SUBTEST_1( cholesky(Mat(s,s)) );

    CALL_SUBTEST_2( lu_non_invertible<Mat>() );
    CALL_SUBTEST_2( lu_invertible<Mat>() );

    CALL_SUBTEST_3( qr(Mat(internal::random<int>(1,EIGEN_TEST_MAX_SIZE),internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_3( qr_invertible<Mat>() );

    CALL_SUBTEST_4( qr<Mat>() );
    CALL_SUBTEST_4( cod<Mat>() );
    CALL_SUBTEST_4( qr_invertible<Mat>() );

    CALL_SUBTEST_5( qr<Mat>() );
    CALL_SUBTEST_5( qr_invertible<Mat>() );

    CALL_SUBTEST_6( selfadjointeigensolver(Mat(s,s)) );

    TEST_SET_BUT_UNUSED_VARIABLE(s)
  }

  CALL_SUBTEST_7(( jacobisvd(Mat(internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE), internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE/2))) ));
  CALL_SUBTEST_8(( bdcsvd(Mat(internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE), internal::random<int>(EIGEN_TEST_MAX_SIZE/4, EIGEN_TEST_MAX_SIZE/2))) ));
}

