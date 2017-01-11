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

template<int N> struct fix_t {
  static const int value = N;
  operator int() const { return value; }
  fix_t (fix_t<N> (*)() ) {}
  fix_t() {}
  // Needed in C++14 to allow fix<N>():
  fix_t operator() () const { return *this; }
};

template<typename T, int Default=Dynamic> struct get_compile_time {
  enum { value = Default };
};

template<int N,int Default> struct get_compile_time<fix_t<N>,Default> {
  enum { value = N };
};

template<typename T, int N, int Default>
struct get_compile_time<variable_if_dynamic<T,N>,Default> {
  enum { value = N };
};


template<typename T> struct is_compile_time       { enum { value = false }; };
template<int N> struct is_compile_time<fix_t<N> > { enum { value = true }; };

} // end namespace internal

#ifndef EIGEN_PARSED_BY_DOXYGEN

#if __cplusplus > 201103L
template<int N>
static const internal::fix_t<N> fix{};
#else
template<int N>
inline internal::fix_t<N> fix() { return internal::fix_t<N>(); }
#endif

#else // EIGEN_PARSED_BY_DOXYGEN

/** \var fix
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
  */
template<int N>
static const auto fix;

#endif // EIGEN_PARSED_BY_DOXYGEN

} // end namespace Eigen

#endif // EIGEN_INTEGRAL_CONSTANT_H
