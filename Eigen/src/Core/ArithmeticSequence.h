// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ARITHMETIC_SEQUENCE_H
#define EIGEN_ARITHMETIC_SEQUENCE_H

namespace Eigen {


struct all_t { all_t() {} };
static const all_t all;

struct shifted_last {
  explicit shifted_last(int o) : offset(o) {}
  int offset;
  shifted_last operator+ (int x) const { return shifted_last(offset+x); }
  shifted_last operator- (int x) const { return shifted_last(offset-x); }
  int operator- (shifted_last x) const { return offset-x.offset; }
};

struct last_t {
  last_t() {}
  shifted_last operator- (int offset) const { return shifted_last(-offset); }
  shifted_last operator+ (int offset) const { return shifted_last(+offset); }
  int operator- (last_t) const { return 0; }
  int operator- (shifted_last x) const { return -x.offset; }
};
static const last_t last;


struct shifted_end {
  explicit shifted_end(int o) : offset(o) {}
  int offset;
  shifted_end operator+ (int x) const { return shifted_end(offset+x); }
  shifted_end operator- (int x) const { return shifted_end(offset-x); }
  int operator- (shifted_end x) const { return offset-x.offset; }
};

struct end_t {
  end_t() {}
  shifted_end operator- (int offset) const { return shifted_end (-offset); }
  shifted_end operator+ (int offset) const { return shifted_end ( offset); }
  int operator- (end_t) const { return 0; }
  int operator- (shifted_end x) const { return -x.offset; }
};
static const end_t end;

template<int N> struct Index_c {
  static const int value = N;
  operator int() const { return value; }
  Index_c (Index_c<N> (*)() ) {}
  Index_c() {}
  // Needed in C++14 to allow c<N>():
  Index_c operator() () const { return *this; }
};

//--------------------------------------------------------------------------------
// Range(first,last) and Slice(first,step,last)
//--------------------------------------------------------------------------------

template<typename FirstType=Index,typename LastType=Index,typename StepType=Index_c<1> >
struct Range_t {
  Range_t(FirstType f, LastType l) : m_first(f), m_last(l) {}
  Range_t(FirstType f, LastType l, StepType s) : m_first(f), m_last(l), m_step(s) {}

  FirstType m_first;
  LastType  m_last;
  StepType  m_step;

  enum { SizeAtCompileTime = -1 };

  Index size() const { return (m_last-m_first+m_step)/m_step; }
  Index operator[] (Index k) const { return m_first + k*m_step; }
};

template<typename T> struct cleanup_slice_type { typedef Index type; };
template<> struct cleanup_slice_type<last_t> { typedef last_t type; };
template<> struct cleanup_slice_type<shifted_last> { typedef shifted_last type; };
template<> struct cleanup_slice_type<end_t> { typedef end_t type; };
template<> struct cleanup_slice_type<shifted_end> { typedef shifted_end type; };
template<int N> struct cleanup_slice_type<Index_c<N> > { typedef Index_c<N> type; };
template<int N> struct cleanup_slice_type<Index_c<N> (*)() > { typedef Index_c<N> type; };

template<typename FirstType,typename LastType>
Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type >
range(FirstType f, LastType l)  {
  return Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type>(f,l);
}

template<typename FirstType,typename LastType,typename StepType>
Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type,typename cleanup_slice_type<StepType>::type >
range(FirstType f, LastType l, StepType s)  {
  return Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type,typename cleanup_slice_type<StepType>::type>(f,l,typename cleanup_slice_type<StepType>::type(s));
}


template<typename T, int Default=-1> struct get_compile_time {
  enum { value = Default };
};

template<int N,int Default> struct get_compile_time<Index_c<N>,Default> {
  enum { value = N };
};

template<typename T> struct is_compile_time         { enum { value = false }; };
template<int N> struct is_compile_time<Index_c<N> > { enum { value = true }; };

template<typename FirstType=Index,typename SizeType=Index,typename StepType=Index_c<1> >
struct Span_t {
  Span_t(FirstType first, SizeType size) : m_first(first), m_size(size) {}
  Span_t(FirstType first, SizeType size, StepType step) : m_first(first), m_size(size), m_step(step) {}

  FirstType m_first;
  SizeType  m_size;
  StepType  m_step;

  enum { SizeAtCompileTime = get_compile_time<SizeType>::value };

  Index size() const { return m_size; }
  Index operator[] (Index k) const { return m_first + k*m_step; }
};

template<typename FirstType,typename SizeType,typename StepType>
Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type,typename cleanup_slice_type<StepType>::type >
span(FirstType first, SizeType size, StepType step)  {
  return Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type,typename cleanup_slice_type<StepType>::type>(first,size,step);
}

template<typename FirstType,typename SizeType>
Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type >
span(FirstType first, SizeType size)  {
  return Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type>(first,size);
}

#if __cplusplus > 201103L
template<int N>
static const Index_c<N> c{};
#else
template<int N>
inline Index_c<N> c() { return Index_c<N>(); }
#endif

namespace internal {

// MakeIndexing/make_indexing turn an arbitrary object of type T into something usable by MatrixSlice
template<typename T,typename EnableIf=void>
struct MakeIndexing {
  typedef T type;
};

template<typename T>
const T& make_indexing(const T& x, Index size) { return x; }

struct IntAsArray {
  IntAsArray(Index val) : m_value(val) {}
  Index operator[](Index) const { return m_value; }
  Index size() const { return 1; }
  Index m_value;
};

// Turn a single index into something that looks like an array (i.e., that exposes a .size(), and operatro[](int) methods)
template<typename T>
struct MakeIndexing<T,typename internal::enable_if<internal::is_integral<T>::value>::type> {
  // Here we could simply use Array, but maybe it's less work for the compiler to use
  // a simpler wrapper as IntAsArray
  //typedef Eigen::Array<Index,1,1> type;
  typedef IntAsArray type;
};

// Replace symbolic last/end "keywords" by their true runtime value
Index symbolic2value(Index x, Index /* size */)   { return x; }
Index symbolic2value(last_t, Index size)          { return size-1; }
Index symbolic2value(shifted_last x, Index size)  { return size+x.offset-1; }
Index symbolic2value(end_t, Index size)           { return size; }
Index symbolic2value(shifted_end x, Index size)   { return size+x.offset; }

// Convert a symbolic range into a usable one (i.e., remove last/end "keywords")
template<typename FirstType,typename LastType,typename StepType>
struct MakeIndexing<Range_t<FirstType,LastType,StepType> > {
  typedef Range_t<Index,Index,StepType> type;
};

template<typename FirstType,typename LastType,typename StepType>
Range_t<Index,Index,StepType> make_indexing(const Range_t<FirstType,LastType,StepType>& ids, Index size) {
  return Range_t<Index,Index,StepType>(symbolic2value(ids.m_first,size),symbolic2value(ids.m_last,size),ids.m_step);
}

// Convert a symbolic span into a usable one (i.e., remove last/end "keywords")
template<typename FirstType,typename SizeType,typename StepType>
struct MakeIndexing<Span_t<FirstType,SizeType,StepType> > {
  typedef Span_t<Index,SizeType,StepType> type;
};

template<typename FirstType,typename SizeType,typename StepType>
Span_t<Index,SizeType,StepType> make_indexing(const Span_t<FirstType,SizeType,StepType>& ids, Index size) {
  return Span_t<Index,SizeType,StepType>(symbolic2value(ids.m_first,size),ids.m_size,ids.m_step);
}

// Convert a symbolic 'all' into a usable range
// Implementation-wise, it would be more efficient to not having to store m_size since
// this information is already in the nested expression. To this end, we would need a
// get_size(indices, underlying_size); function returning indices.size() by default.
struct AllRange {
  AllRange(Index size) : m_size(size) {}
  Index operator[](Index i) const { return i; }
  Index size() const { return m_size; }
  Index m_size;
};

template<>
struct MakeIndexing<all_t> {
  typedef AllRange type;
};

AllRange make_indexing(all_t , Index size) {
  return AllRange(size);
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_ARITHMETIC_SEQUENCE_H
