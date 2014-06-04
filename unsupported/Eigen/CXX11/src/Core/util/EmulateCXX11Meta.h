// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_EMULATE_CXX11_META_H
#define EIGEN_EMULATE_CXX11_META_H



namespace Eigen {

// The array class is only available starting with cxx11. Emulate our own here
// if needed
template <typename T, size_t n> class array {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE T& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const T& operator[] (size_t index) const { return values[index]; }

  static const std::size_t size = n;

  T values[n];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(const T& v) {
    EIGEN_STATIC_ASSERT(n==1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    values[0] = v;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(const T& v1, const T& v2) {
    EIGEN_STATIC_ASSERT(n==2, YOU_MADE_A_PROGRAMMING_MISTAKE)
    values[0] = v1;
    values[1] = v2;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(const T& v1, const T& v2, const T& v3) {
    EIGEN_STATIC_ASSERT(n==3, YOU_MADE_A_PROGRAMMING_MISTAKE)
    values[0] = v1;
    values[1] = v2;
    values[2] = v3;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(const T& v1, const T& v2, const T& v3, const T& v4) {
    EIGEN_STATIC_ASSERT(n==4, YOU_MADE_A_PROGRAMMING_MISTAKE)
    values[0] = v1;
    values[1] = v2;
    values[2] = v3;
    values[3] = v4;
  }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(const T& v1, const T& v2, const T& v3, const T& v4, const T& v5) {
    EIGEN_STATIC_ASSERT(n==5, YOU_MADE_A_PROGRAMMING_MISTAKE)
    values[0] = v1;
    values[1] = v2;
    values[2] = v3;
    values[3] = v4;
    values[4] = v5;
  }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  array(std::initializer_list<T> l) {
    std::copy(l.begin(), l.end(), values);
  }
#endif
};


namespace internal {

/** \internal
  * \file CXX11/Core/util/EmulateCXX11Meta.h
  * This file emulates a subset of the functionality provided by CXXMeta.h for
  * compilers that don't yet support cxx11 such as nvcc.
  */

struct empty_list { static const std::size_t count = 0; };

template<typename T, typename Tail=empty_list> struct type_list {
  typedef T HeadType;
  typedef Tail TailType;
  static const T head;
  static const Tail tail;
  static const std::size_t count = 1 + Tail::count;
};

struct null_type { };

template<typename T1 = null_type, typename T2 = null_type, typename T3 = null_type, typename T4 = null_type, typename T5 = null_type>
struct make_type_list {
  typedef typename make_type_list<T2, T3, T4, T5>::type tailresult;

  typedef type_list<T1, tailresult> type;
};

template<> struct make_type_list<> {
  typedef empty_list type;
};


template <std::size_t index, class TList> struct get_type;

template <class Head, class Tail>
struct get_type<0, type_list<Head, Tail> >
{
  typedef Head type;
};

template <std::size_t i, class Head, class Tail>
struct get_type<i, type_list<Head, Tail> >
{
  typedef typename get_type<i-1, Tail>::type type;
};


/* numeric list */
template <typename T, T n>
struct type2val {
  typedef T type;
  static const T value = n;
};


template<typename T, size_t n, T V> struct gen_numeric_list_repeated;

template<typename T, T V> struct gen_numeric_list_repeated<T, 1, V> {
  typedef typename make_type_list<type2val<T, V> >::type type;
};

template<typename T, T V> struct gen_numeric_list_repeated<T, 2, V> {
  typedef typename make_type_list<type2val<T, V>, type2val<T, V> >::type type;
};

template<typename T, T V> struct gen_numeric_list_repeated<T, 3, V> {
  typedef typename make_type_list<type2val<T, V>, type2val<T, V>, type2val<T, V> >::type type;
};

template<typename T, T V> struct gen_numeric_list_repeated<T, 4, V> {
  typedef typename make_type_list<type2val<T, V>, type2val<T, V>, type2val<T, V>, type2val<T, V> >::type type;
};

template<typename T, T V> struct gen_numeric_list_repeated<T, 5, V> {
  typedef typename make_type_list<type2val<T, V>, type2val<T, V>, type2val<T, V>, type2val<T, V>, type2val<T, V> >::type type;
};


template <std::size_t index, class NList> struct get;

template <class Head, class Tail>
struct get<0, type_list<Head, Tail> >
{
  typedef typename Head::type type;
  static const type value = Head::value;
};

template <std::size_t i, class Head, class Tail>
struct get<i, type_list<Head, Tail> >
{
  typedef typename get<i-1, Tail>::type type;
  static const type value = get<i-1, Tail>::value;
};

template <class NList> struct arg_prod {
  static const typename NList::HeadType::type value = get<0, NList>::value * arg_prod<typename NList::TailType>::value;
};
template <> struct arg_prod<empty_list> {
  static const int value = 1;
};

template<int n, typename t>
array<t, n> repeat(t v) {
  array<t, n> array;
  array.fill(v);
  return array;
}

template<std::size_t n, typename t>
t array_prod(const array<t, n>& a) {
  t prod = 1;
  for (size_t i = 0; i < n; ++i) { prod *= a[i]; }
  return prod;
}
template<typename t>
t array_prod(const array<t, 0>& /*a*/) {
  return 0;
}

template<std::size_t I, class T, std::size_t N> inline T& array_get(array<T,N>& a) {
  return a[I];
}
template<std::size_t I, class T, std::size_t N> inline const T& array_get(const array<T,N>& a) {
  return a[I];
}

struct sum_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a + b; }
};
struct product_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a * b; }
};

struct logical_and_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a && b; }
};
struct logical_or_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a || b; }
};

struct equal_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a == b; }
};
struct not_equal_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a != b; }
};
struct lesser_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a < b; }
};
struct lesser_equal_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a <= b; }
};

struct greater_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a > b; }
};
struct greater_equal_op {
  template<typename A, typename B> static inline bool run(A a, B b) { return a >= b; }
};

struct not_op {
  template<typename A> static inline bool run(A a) { return !a; }
};
struct negation_op {
  template<typename A> static inline bool run(A a) { return -a; }
};
struct greater_equal_zero_op {
  template<typename A> static inline bool run(A a) { return a >= 0; }
};


template<typename Reducer, typename Op, typename A, std::size_t N>
inline bool array_apply_and_reduce(const array<A, N>& a) {
  EIGEN_STATIC_ASSERT(N >= 2, YOU_MADE_A_PROGRAMMING_MISTAKE)
  bool result = Reducer::run(Op::run(a[0]), Op::run(a[1]));
  for (size_t i = 2; i < N; ++i) {
    result = Reducer::run(result, Op::run(a[i]));
  }
  return result;
}

template<typename Reducer, typename Op, typename A, typename B, std::size_t N>
inline bool array_zip_and_reduce(const array<A, N>& a, const array<B, N>& b) {
  EIGEN_STATIC_ASSERT(N >= 2, YOU_MADE_A_PROGRAMMING_MISTAKE)
      bool result = Reducer::run(Op::run(a[0], b[0]), Op::run(a[1], b[1]));
  for (size_t i = 2; i < N; ++i) {
    result = Reducer::run(result, Op::run(a[i], b[i]));
  }
  return result;
}

}  // end namespace internal

}  // end namespace Eigen



#endif  // EIGEN_EMULATE_CXX11_META_H
