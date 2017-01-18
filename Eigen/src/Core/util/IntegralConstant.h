// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_INTEGRAL_CONSTANT_H
#define EIGEN_INTEGRAL_CONSTANT_H

namespace Eigen {

namespace internal {

template<int N> struct fix_t;
template<int N> class variable_or_fixed;

template<int N> struct fix_t {
  static const int value = N;
  operator int() const { return value; }
  fix_t() {}
  fix_t(variable_or_fixed<N> other) {
    EIGEN_ONLY_USED_FOR_DEBUG(other);
    eigen_internal_assert(int(other)==N);
  }

  fix_t<-N> operator-() const { return fix_t<-N>(); }

#if EIGEN_HAS_CXX14
  // Needed in C++14 to allow fix<N>():
  fix_t operator() () const { return *this; }

  variable_or_fixed<N> operator() (int val) const { return variable_or_fixed<N>(val); }
#else
  fix_t (fix_t<N> (*)() ) {}
#endif
};

template<int N> class variable_or_fixed {
public:
  static const int value = N;
  operator int() const { return m_value; }
  variable_or_fixed(int val) { m_value = val; }
protected:
  int m_value;
};

template<typename T, int Default=Dynamic> struct get_fixed_value {
  static const int value = Default;
};

template<int N,int Default> struct get_fixed_value<fix_t<N>,Default> {
  static const int value = N;
};

#if !EIGEN_HAS_CXX14
template<int N,int Default> struct get_fixed_value<fix_t<N> (*)(),Default> {
  static const int value = N;
};
#endif

template<int N,int Default> struct get_fixed_value<variable_or_fixed<N>,Default> {
  static const int value = N ;
};

template<typename T, int N, int Default>
struct get_fixed_value<variable_if_dynamic<T,N>,Default> {
  static const int value = N;
};

template<typename T> Index get_runtime_value(const T &x) { return x; }
#if !EIGEN_HAS_CXX14
template<int N> Index get_runtime_value(fix_t<N> (*)()) { return N; }
#endif

// Cleanup integer/fix_t/variable_or_fixed/etc types:

// By default, no cleanup:
template<typename T, int DynamicKey=Dynamic, typename EnableIf=void> struct cleanup_index_type { typedef T type; };

// Convert any integral type (e.g., short, int, unsigned int, etc.) to Eigen::Index
template<typename T, int DynamicKey> struct cleanup_index_type<T,DynamicKey,typename internal::enable_if<internal::is_integral<T>::value>::type> { typedef Index type; };

#if !EIGEN_HAS_CXX14
// In c++98/c++11, fix<N> is a pointer to function that we better cleanup to a true fix_t<N>:
template<int N, int DynamicKey> struct cleanup_index_type<fix_t<N> (*)(), DynamicKey> { typedef fix_t<N> type; };
#endif

// If variable_or_fixed does not match DynamicKey, then we turn it to a pure compile-time value:
template<int N, int DynamicKey> struct cleanup_index_type<variable_or_fixed<N>, DynamicKey> { typedef fix_t<N> type; };
// If variable_or_fixed matches DynamicKey, then we turn it to a pure runtime-value (aka Index):
template<int DynamicKey> struct cleanup_index_type<variable_or_fixed<DynamicKey>, DynamicKey> { typedef Index type; };

} // end namespace internal

#ifndef EIGEN_PARSED_BY_DOXYGEN

#if EIGEN_HAS_CXX14
template<int N>
static const internal::fix_t<N> fix{};
#else
template<int N>
inline internal::fix_t<N> fix() { return internal::fix_t<N>(); }

// The generic typename T is mandatory. Otherwise, a code like fix<N> could refer to either the function above or this next overload.
// This way a code like fix<N> can only refer to the previous function.
template<int N,typename T>
inline internal::variable_or_fixed<N> fix(T val) { return internal::variable_or_fixed<N>(val); }
#endif

#else // EIGEN_PARSED_BY_DOXYGEN

/** \var fix<N>()
  * \ingroup Core_Module
  *
  * This \em identifier permits to construct an object embedding a compile-time integer \c N.
  *
  * \tparam N the compile-time integer value
  *
  * It is typically used in conjunction with the Eigen::seq and Eigen::seqN functions to pass compile-time values to them:
  * \code
  * seqN(10,fix<4>,fix<-3>)   // <=> [10 7 4 1]
  * \endcode
  *
  * See also the function fix(int) to pass both a compile-time and runtime value.
  *
  * In c++14, it is implemented as:
  * \code
  * template<int N> static const internal::fix_t<N> fix{};
  * \endcode
  * where internal::fix_t<N> is an internal template class similar to
  * <a href="http://en.cppreference.com/w/cpp/types/integral_constant">\c std::integral_constant </a><tt> <int,N> </tt>
  * Here, \c fix<N> is thus an object of type \c internal::fix_t<N>.
  *
  * In c++98/11, it is implemented as a function:
  * \code
  * template<int N> inline internal::fix_t<N> fix();
  * \endcode
  * Here internal::fix_t<N> is thus a pointer to function.
  *
  * If for some reason you want a true object in c++98 then you can write: \code fix<N>() \endcode which is also valid in c++14.
  *
  * \sa fix<N>(int), seq, seqN
  */
template<int N>
static const auto fix();

/** \fn fix<N>(int)
  * \ingroup Core_Module
  *
  * This function returns an object embedding both a compile-time integer \c N, and a fallback runtime value \a val.
  *
  * \tparam N the compile-time integer value
  * \param  val the fallback runtime integer value
  *
  * This function is a more general version of the \ref fix identifier/function that can be used in template code
  * where the compile-time value could turn out to actually mean "undefined at compile-time". For positive integers
  * such as a size or a dimension, this case is identified by Eigen::Dynamic, whereas runtime signed integers
  * (e.g., an increment/stride) are identified as Eigen::DynamicIndex. In such a case, the runtime value \a val
  * will be used as a fallback.
  *
  * A typical use case would be:
  * \code
  * template<typename Derived> void foo(const MatrixBase<Derived> &mat) {
  *   const int N = Derived::RowsAtCompileTime==Dynamic ? Dynamic : Derived::RowsAtCompileTime/2;
  *   const int n = mat.rows()/2;
  *   ... mat( seqN(0,fix<N>(n) ) ...;
  * }
  * \endcode
  * In this example, the function Eigen::seqN knows that the second argument is expected to be a size.
  * If the passed compile-time value N equals Eigen::Dynamic, then the proxy object returned by fix will be dissmissed, and converted to an Eigen::Index of value \c n.
  * Otherwise, the runtime-value \c n will be dissmissed, and the returned ArithmeticSequence will be of the exact same type as <tt> seqN(0,fix<N>) </tt>.
  *
  * \sa fix, seqN, class ArithmeticSequence
  */
template<int N>
static const auto fix(int val);

#endif // EIGEN_PARSED_BY_DOXYGEN

} // end namespace Eigen

#endif // EIGEN_INTEGRAL_CONSTANT_H
