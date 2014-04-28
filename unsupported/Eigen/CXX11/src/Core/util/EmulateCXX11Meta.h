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
  T& operator[] (size_t index) { return values[index]; }
  const T& operator[] (size_t index) const { return values[index]; }

  T values[n];
};


namespace internal {

/** \internal
  * \file CXX11/Core/util/EmulateCXX11Meta.h
  * This file emulates a subset of the functionality provided by CXXMeta.h for
  * compilers that don't yet support cxx11 such as nvcc.
  */

struct empty_list { static const std::size_t count = 0; };

template<typename T, typename Tail=empty_list> struct type_list {
  T head;
  Tail tail;
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



template <typename T, T n>
struct type2val {
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
