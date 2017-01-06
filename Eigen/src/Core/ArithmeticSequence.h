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

//--------------------------------------------------------------------------------
// Pseudo keywords: all, last, end
//--------------------------------------------------------------------------------

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

//--------------------------------------------------------------------------------
// integral constant
//--------------------------------------------------------------------------------

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

template<typename T> struct is_compile_time       { enum { value = false }; };
template<int N> struct is_compile_time<fix_t<N> > { enum { value = true }; };

#if __cplusplus > 201103L
template<int N>
static const fix_t<N> fix{};
#else
template<int N>
inline fix_t<N> fix() { return fix_t<N>(); }
#endif

//--------------------------------------------------------------------------------
// range(first,last,incr) and span(first,size,incr)
//--------------------------------------------------------------------------------

template<typename FirstType=Index,typename LastType=Index,typename IncrType=fix_t<1> >
struct Range_t {
  Range_t(FirstType f, LastType l) : m_first(f), m_last(l) {}
  Range_t(FirstType f, LastType l, IncrType s) : m_first(f), m_last(l), m_incr(s) {}

  FirstType m_first;
  LastType  m_last;
  IncrType  m_incr;

  enum { SizeAtCompileTime = -1 };

  Index size() const { return (m_last-m_first+m_incr)/m_incr; }
  Index operator[] (Index k) const { return m_first + k*m_incr; }
};

template<typename T> struct cleanup_slice_type { typedef Index type; };
template<> struct cleanup_slice_type<last_t> { typedef last_t type; };
template<> struct cleanup_slice_type<shifted_last> { typedef shifted_last type; };
template<> struct cleanup_slice_type<end_t> { typedef end_t type; };
template<> struct cleanup_slice_type<shifted_end> { typedef shifted_end type; };
template<int N> struct cleanup_slice_type<fix_t<N> > { typedef fix_t<N> type; };
template<int N> struct cleanup_slice_type<fix_t<N> (*)() > { typedef fix_t<N> type; };

template<typename FirstType,typename LastType>
Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type >
range(FirstType f, LastType l)  {
  return Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type>(f,l);
}

template<typename FirstType,typename LastType,typename IncrType>
Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type,typename cleanup_slice_type<IncrType>::type >
range(FirstType f, LastType l, IncrType s)  {
  return Range_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<LastType>::type,typename cleanup_slice_type<IncrType>::type>(f,l,typename cleanup_slice_type<IncrType>::type(s));
}




template<typename FirstType=Index,typename SizeType=Index,typename IncrType=fix_t<1> >
struct Span_t {
  Span_t(FirstType first, SizeType size) : m_first(first), m_size(size) {}
  Span_t(FirstType first, SizeType size, IncrType incr) : m_first(first), m_size(size), m_incr(incr) {}

  FirstType m_first;
  SizeType  m_size;
  IncrType  m_incr;

  enum { SizeAtCompileTime = get_compile_time<SizeType>::value };

  Index size() const { return m_size; }
  Index operator[] (Index k) const { return m_first + k*m_incr; }
};

template<typename FirstType,typename SizeType,typename IncrType>
Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type,typename cleanup_slice_type<IncrType>::type >
span(FirstType first, SizeType size, IncrType incr)  {
  return Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type,typename cleanup_slice_type<IncrType>::type>(first,size,incr);
}

template<typename FirstType,typename SizeType>
Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type >
span(FirstType first, SizeType size)  {
  return Span_t<typename cleanup_slice_type<FirstType>::type,typename cleanup_slice_type<SizeType>::type>(first,size);
}



namespace internal {

template<typename T, int XprSize, typename EnableIf = void> struct get_compile_time_size {
  enum { value = -1 };
};

template<typename T, int XprSize> struct get_compile_time_size<T,XprSize,typename internal::enable_if<((T::SizeAtCompileTime&0)==0)>::type> {
  enum { value = T::SizeAtCompileTime };
};

#ifdef EIGEN_HAS_CXX11
template<typename T, int XprSize, int N> struct get_compile_time_size<std::array<T,N>,XprSize> {
  enum { value = N };
};
#endif

// MakeIndexing/make_indexing turn an arbitrary object of type T into something usable by MatrixSlice
template<typename T,typename EnableIf=void>
struct MakeIndexing {
  typedef T type;
};

template<typename T>
const T& make_indexing(const T& x, Index size) { return x; }

struct IntAsArray {
  enum {
    SizeAtCompileTime = 1
  };
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
template<typename FirstType,typename LastType,typename IncrType>
struct MakeIndexing<Range_t<FirstType,LastType,IncrType> > {
  typedef Range_t<Index,Index,IncrType> type;
};

template<typename FirstType,typename LastType,typename IncrType>
Range_t<Index,Index,IncrType> make_indexing(const Range_t<FirstType,LastType,IncrType>& ids, Index size) {
  return Range_t<Index,Index,IncrType>(symbolic2value(ids.m_first,size),symbolic2value(ids.m_last,size),ids.m_incr);
}

// Convert a symbolic span into a usable one (i.e., remove last/end "keywords")
template<typename FirstType,typename SizeType,typename IncrType>
struct MakeIndexing<Span_t<FirstType,SizeType,IncrType> > {
  typedef Span_t<Index,SizeType,IncrType> type;
};

template<typename FirstType,typename SizeType,typename IncrType>
Span_t<Index,SizeType,IncrType> make_indexing(const Span_t<FirstType,SizeType,IncrType>& ids, Index size) {
  return Span_t<Index,SizeType,IncrType>(symbolic2value(ids.m_first,size),ids.m_size,ids.m_incr);
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

template<int XprSize> struct get_compile_time_size<AllRange,XprSize> {
  enum { value = XprSize };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_ARITHMETIC_SEQUENCE_H
