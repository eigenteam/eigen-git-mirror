// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_TEST_MAIN_H
#define EIGEN_TEST_MAIN_H

#include <QtTest/QtTest>

#include <cstdlib>
#include <ctime>
#include <iostream>

#define DEFAULT_REPEAT 10

#ifndef EIGEN_NO_ASSERTION_CHECKING

  namespace Eigen
  {
    static const bool should_raise_an_assert = false;

    // Used to avoid to raise two exceptions at the time in which
    // case the exception is not properly catched.
    // This may happen when a second exceptions is raise in a destructor.
    static bool no_more_assert = false;

    struct ei_assert_exception
    {
      ei_assert_exception(void) {}
      ~ei_assert_exception() { Eigen::no_more_assert = false; }
    };
  }

  #define EI_PP_MAKE_STRING2(S) #S
  #define EI_PP_MAKE_STRING(S) EI_PP_MAKE_STRING2(S)

  // If EIGEN_DEBUG_ASSERTS is defined and if no assertion is raised while
  // one should have been, then the list of excecuted assertions is printed out.
  //
  // EIGEN_DEBUG_ASSERTS is not enabled by default as it
  // significantly increases the compilation time
  // and might even introduce side effects that would hide
  // some memory errors.
  #ifdef EIGEN_DEBUG_ASSERTS

    namespace Eigen
    {
      static bool ei_push_assert = false;
      static std::vector<std::string> ei_assert_list;
    }

    #define ei_assert(a)                       \
      if( (!(a)) && (!no_more_assert) )     \
      {                                     \
        Eigen::no_more_assert = true;       \
        throw Eigen::ei_assert_exception(); \
      }                                     \
      else if (Eigen::ei_push_assert)       \
      {                                     \
        ei_assert_list.push_back(std::string(EI_PP_MAKE_STRING(__FILE__)" ("EI_PP_MAKE_STRING(__LINE__)") : "#a) ); \
      }

    #define VERIFY_RAISES_ASSERT(a)                                               \
      {                                                                           \
        try {                                                                     \
          Eigen::ei_assert_list.clear();                                          \
          Eigen::ei_push_assert = true;                                           \
          a;                                                                      \
          Eigen::ei_push_assert = false;                                          \
          std::cout << "One of the following asserts should have been raised:\n"; \
          for (uint ai=0 ; ai<ei_assert_list.size() ; ++ai)                       \
            std::cout << "  " << ei_assert_list[ai] << "\n";                      \
          QVERIFY(Eigen::should_raise_an_assert && # a);                          \
        } catch (Eigen::ei_assert_exception e) {                                  \
          Eigen::ei_push_assert = false; QVERIFY(true);                           \
        }                                                                         \
      }

  #else // EIGEN_DEBUG_ASSERTS

    #define ei_assert(a)                       \
      if( (!(a)) && (!no_more_assert) )     \
      {                                     \
        Eigen::no_more_assert = true;       \
        throw Eigen::ei_assert_exception(); \
      }

    #define VERIFY_RAISES_ASSERT(a)                                 \
      {                                                             \
        try { a; QVERIFY(Eigen::should_raise_an_assert && # a); }   \
        catch (Eigen::ei_assert_exception e) { QVERIFY(true); }     \
      }

  #endif // EIGEN_DEBUG_ASSERTS

  #define EIGEN_USE_CUSTOM_ASSERT

#else // EIGEN_NO_ASSERTION_CHECKING

  #define VERIFY_RAISES_ASSERT(a) {}

#endif // EIGEN_NO_ASSERTION_CHECKING


//#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER RowMajorBit
#define EIGEN_INTERNAL_DEBUGGING
#include <Eigen/Core>

#define VERIFY(a) QVERIFY(a)
#define VERIFY_IS_APPROX(a, b) QVERIFY(test_ei_isApprox(a, b))
#define VERIFY_IS_NOT_APPROX(a, b) QVERIFY(!test_ei_isApprox(a, b))
#define VERIFY_IS_MUCH_SMALLER_THAN(a, b) QVERIFY(test_ei_isMuchSmallerThan(a, b))
#define VERIFY_IS_NOT_MUCH_SMALLER_THAN(a, b) QVERIFY(!test_ei_isMuchSmallerThan(a, b))
#define VERIFY_IS_APPROX_OR_LESS_THAN(a, b) QVERIFY(test_ei_isApproxOrLessThan(a, b))
#define VERIFY_IS_NOT_APPROX_OR_LESS_THAN(a, b) QVERIFY(!test_ei_isApproxOrLessThan(a, b))

namespace Eigen {

template<typename T> inline typename NumTraits<T>::Real test_precision();
template<> inline int test_precision<int>() { return 0; }
template<> inline float test_precision<float>() { return 1e-2f; }
template<> inline double test_precision<double>() { return 1e-5; }
template<> inline float test_precision<std::complex<float> >() { return test_precision<float>(); }
template<> inline double test_precision<std::complex<double> >() { return test_precision<double>(); }

inline bool test_ei_isApprox(const int& a, const int& b)
{ return ei_isApprox(a, b, test_precision<int>()); }
inline bool test_ei_isMuchSmallerThan(const int& a, const int& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<int>()); }
inline bool test_ei_isApproxOrLessThan(const int& a, const int& b)
{ return ei_isApproxOrLessThan(a, b, test_precision<int>()); }

inline bool test_ei_isApprox(const float& a, const float& b)
{ return ei_isApprox(a, b, test_precision<float>()); }
inline bool test_ei_isMuchSmallerThan(const float& a, const float& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<float>()); }
inline bool test_ei_isApproxOrLessThan(const float& a, const float& b)
{ return ei_isApproxOrLessThan(a, b, test_precision<float>()); }

inline bool test_ei_isApprox(const double& a, const double& b)
{ return ei_isApprox(a, b, test_precision<double>()); }
inline bool test_ei_isMuchSmallerThan(const double& a, const double& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<double>()); }
inline bool test_ei_isApproxOrLessThan(const double& a, const double& b)
{ return ei_isApproxOrLessThan(a, b, test_precision<double>()); }

inline bool test_ei_isApprox(const std::complex<float>& a, const std::complex<float>& b)
{ return ei_isApprox(a, b, test_precision<std::complex<float> >()); }
inline bool test_ei_isMuchSmallerThan(const std::complex<float>& a, const std::complex<float>& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<std::complex<float> >()); }

inline bool test_ei_isApprox(const std::complex<double>& a, const std::complex<double>& b)
{ return ei_isApprox(a, b, test_precision<std::complex<double> >()); }
inline bool test_ei_isMuchSmallerThan(const std::complex<double>& a, const std::complex<double>& b)
{ return ei_isMuchSmallerThan(a, b, test_precision<std::complex<double> >()); }

template<typename Derived1, typename Derived2>
inline bool test_ei_isApprox(const MatrixBase<Derived1>& m1,
                   const MatrixBase<Derived2>& m2)
{
  return m1.isApprox(m2, test_precision<typename ei_traits<Derived1>::Scalar>());
}

template<typename Derived1, typename Derived2>
inline bool test_ei_isMuchSmallerThan(const MatrixBase<Derived1>& m1,
                                   const MatrixBase<Derived2>& m2)
{
  return m1.isMuchSmallerThan(m2, test_precision<typename ei_traits<Derived1>::Scalar>());
}

template<typename Derived>
inline bool test_ei_isMuchSmallerThan(const MatrixBase<Derived>& m,
                                   const typename NumTraits<typename ei_traits<Derived>::Scalar>::Real& s)
{
  return m.isMuchSmallerThan(s, test_precision<typename ei_traits<Derived>::Scalar>());
}

class EigenTest : public QObject
{
    Q_OBJECT

  public:
    EigenTest(int repeat) : m_repeat(repeat) {}

  private slots:
    void testBasicStuff();
    void testLinearStructure();
    void testProduct();
    void testAdjoint();
    void testSubmatrices();
    void testMiscMatrices();
    void testSmallVectors();
    void testMap();
    void testCwiseops();
    void testDeterminant();
    void testTriangular();
    void testCholesky();
  protected:
    int m_repeat;
};

} // end namespace Eigen

#endif // EIGEN_TEST_MAIN_H
