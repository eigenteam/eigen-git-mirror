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

// A simple wrapper around an Index to provide the eval method.
// We could also use a free-function symbolic_eval...
class symbolic_value_wrapper {
public:
  symbolic_value_wrapper(Index val) : m_value(val) {}
  template<typename T>
  Index eval(const T&) const { return m_value; }
protected:
  Index m_value;
};

//--------------------------------------------------------------------------------
// minimalistic symbolic scalar type
//--------------------------------------------------------------------------------

template<typename Tag> class symbolic_symbol;
template<typename Arg0> class symbolic_negate;
template<typename Arg1,typename Arg2> class symbolic_add;
template<typename Arg1,typename Arg2> class symbolic_product;
template<typename Arg1,typename Arg2> class symbolic_quotient;

template<typename Derived>
class symbolic_index_base
{
public:
  const Derived& derived() const { return *static_cast<const Derived*>(this); }

  symbolic_negate<Derived> operator-() const { return symbolic_negate<Derived>(derived()); }

  symbolic_add<Derived,symbolic_value_wrapper> operator+(Index b) const
  { return symbolic_add<Derived,symbolic_value_wrapper >(derived(),  b); }
  symbolic_add<Derived,symbolic_value_wrapper> operator-(Index a) const
  { return symbolic_add<Derived,symbolic_value_wrapper >(derived(), -a); }
  symbolic_quotient<Derived,symbolic_value_wrapper> operator/(Index a) const
  { return symbolic_quotient<Derived,symbolic_value_wrapper>(derived(),a); }

  friend symbolic_add<Derived,symbolic_value_wrapper> operator+(Index a, const symbolic_index_base& b)
  { return symbolic_add<Derived,symbolic_value_wrapper>(b.derived(), a); }
  friend symbolic_add<symbolic_negate<Derived>,symbolic_value_wrapper> operator-(Index a, const symbolic_index_base& b)
  { return symbolic_add<symbolic_negate<Derived>,symbolic_value_wrapper>(-b.derived(), a); }
  friend symbolic_add<symbolic_value_wrapper,Derived> operator/(Index a, const symbolic_index_base& b)
  { return symbolic_add<symbolic_value_wrapper,Derived>(a,b.derived()); }

  template<typename OtherDerived>
  symbolic_add<Derived,OtherDerived> operator+(const symbolic_index_base<OtherDerived> &b) const
  { return symbolic_add<Derived,OtherDerived>(derived(),  b.derived()); }

  template<typename OtherDerived>
  symbolic_add<Derived,symbolic_negate<OtherDerived> > operator-(const symbolic_index_base<OtherDerived> &b) const
  { return symbolic_add<Derived,symbolic_negate<OtherDerived> >(derived(), -b.derived()); }

  template<typename OtherDerived>
  symbolic_add<Derived,OtherDerived> operator/(const symbolic_index_base<OtherDerived> &b) const
  { return symbolic_quotient<Derived,OtherDerived>(derived(), b.derived()); }
};

template<typename T>
struct is_symbolic {
  enum { value = internal::is_convertible<T,symbolic_index_base<T> >::value };
};

template<typename Tag>
class symbolic_value_pair
{
public:
  symbolic_value_pair(Index val) : m_value(val) {}
  Index value() const { return m_value; }
protected:
  Index m_value;
};

template<typename Tag>
class symbolic_value : public symbolic_index_base<symbolic_value<Tag> >
{
public:
  symbolic_value() {}

  Index eval(const symbolic_value_pair<Tag> &values) const { return values.value(); }

  // TODO add a c++14 eval taking a tuple of symbolic_value_pair and getting the value with std::get<symbolic_value_pair<Tag> >...
};

template<typename Arg0>
class symbolic_negate : public symbolic_index_base<symbolic_negate<Arg0> >
{
public:
  symbolic_negate(const Arg0& arg0) : m_arg0(arg0) {}

  template<typename T>
  Index eval(const T& values) const { return -m_arg0.eval(values); }
protected:
  Arg0 m_arg0;
};

template<typename Arg0, typename Arg1>
class symbolic_add : public symbolic_index_base<symbolic_add<Arg0,Arg1> >
{
public:
  symbolic_add(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template<typename T>
  Index eval(const T& values) const { return m_arg0.eval(values) + m_arg1.eval(values); }
protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

template<typename Arg0, typename Arg1>
class symbolic_product : public symbolic_index_base<symbolic_product<Arg0,Arg1> >
{
public:
  symbolic_product(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template<typename T>
  Index eval(const T& values) const { return m_arg0.eval(values) * m_arg1.eval(values); }
protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

template<typename Arg0, typename Arg1>
class symbolic_quotient : public symbolic_index_base<symbolic_quotient<Arg0,Arg1> >
{
public:
  symbolic_quotient(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template<typename T>
  Index eval(const T& values) const { return m_arg0.eval(values) / m_arg1.eval(values); }
protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

struct symb_last_tag {};

static const symbolic_value<symb_last_tag> last;
static const symbolic_add<symbolic_value<symb_last_tag>,symbolic_value_wrapper> end(last+1);

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
// seq(first,last,incr) and seqN(first,size,incr)
//--------------------------------------------------------------------------------

template<typename FirstType=Index,typename SizeType=Index,typename IncrType=fix_t<1> >
class ArithemeticSequence
{

public:
  ArithemeticSequence(FirstType first, SizeType size) : m_first(first), m_size(size) {}
  ArithemeticSequence(FirstType first, SizeType size, IncrType incr) : m_first(first), m_size(size), m_incr(incr) {}

  enum {
    SizeAtCompileTime = get_compile_time<SizeType>::value,
    IncrAtCompileTime = get_compile_time<IncrType,DynamicIndex>::value
  };

  Index size()  const { return m_size; }
  Index operator[](Index i) const { return m_first + i * m_incr; }

  const FirstType& firstObject() const { return m_first; }
  const SizeType&  sizeObject()  const { return m_size; }
  const IncrType&  incrObject()  const { return m_incr; }

protected:
  FirstType m_first;
  SizeType  m_size;
  IncrType  m_incr;
};

template<typename T> struct cleanup_seq_type { typedef T type; };
template<int N> struct cleanup_seq_type<fix_t<N> > { typedef fix_t<N> type; };
template<int N> struct cleanup_seq_type<fix_t<N> (*)() > { typedef fix_t<N> type; };

template<typename FirstType,typename SizeType,typename IncrType>
ArithemeticSequence<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<SizeType>::type,typename cleanup_seq_type<IncrType>::type >
seqN(FirstType first, SizeType size, IncrType incr)  {
  return ArithemeticSequence<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<SizeType>::type,typename cleanup_seq_type<IncrType>::type>(first,size,incr);
}

template<typename FirstType,typename SizeType>
ArithemeticSequence<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<SizeType>::type >
seqN(FirstType first, SizeType size)  {
  return ArithemeticSequence<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<SizeType>::type>(first,size);
}

template<typename FirstType,typename LastType>
auto seq(FirstType f, LastType l) -> decltype(seqN(f,(l-f+1)))
{
  return seqN(f,(l-f+1));
}

template<typename FirstType,typename LastType, typename IncrType>
auto seq(FirstType f, LastType l, IncrType incr)
  -> decltype(seqN(f,(l-f+typename cleanup_seq_type<IncrType>::type(incr))/typename cleanup_seq_type<IncrType>::type(incr),typename cleanup_seq_type<IncrType>::type(incr)))
{
  typedef typename cleanup_seq_type<IncrType>::type CleanedIncrType;
  return seqN(f,(l-f+CleanedIncrType(incr))/CleanedIncrType(incr),CleanedIncrType(incr));
}

namespace internal {

template<typename T>
Index size(const T& x) { return x.size(); }

template<typename T,std::size_t N>
Index size(const T (&x) [N]) { return N; }

template<typename T, int XprSize, typename EnableIf = void> struct get_compile_time_size {
  enum { value = Dynamic };
};

template<typename T, int XprSize> struct get_compile_time_size<T,XprSize,typename internal::enable_if<((T::SizeAtCompileTime&0)==0)>::type> {
  enum { value = T::SizeAtCompileTime };
};

template<typename T, int XprSize, int N> struct get_compile_time_size<const T (&)[N],XprSize> {
  enum { value = N };
};

#ifdef EIGEN_HAS_CXX11
template<typename T, int XprSize, std::size_t N> struct get_compile_time_size<std::array<T,N>,XprSize> {
  enum { value = N };
};
#endif

template<typename T, typename EnableIf = void> struct get_compile_time_incr {
  enum { value = UndefinedIncr };
};

template<typename FirstType,typename SizeType,typename IncrType>
struct get_compile_time_incr<ArithemeticSequence<FirstType,SizeType,IncrType> > {
  enum { value = get_compile_time<IncrType,DynamicIndex>::value };
};


// MakeIndexing/make_indexing turn an arbitrary object of type T into something usable by MatrixSlice
template<typename T,typename EnableIf=void>
struct MakeIndexing {
  typedef T type;
};

template<typename T>
const T& make_indexing(const T& x, Index /*size*/) { return x; }

struct IntAsArray {
  enum {
    SizeAtCompileTime = 1
  };
  IntAsArray(Index val) : m_value(val) {}
  Index operator[](Index) const { return m_value; }
  Index size() const { return 1; }
  Index m_value;
};

template<> struct get_compile_time_incr<IntAsArray> {
  enum { value = 1 }; // 1 or 0 ??
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

template<int N>
fix_t<N> symbolic2value(fix_t<N> x, Index /*size*/)   { return x; }

template<typename Derived>
Index symbolic2value(const symbolic_index_base<Derived> &x, Index size)
{
  Index h=x.derived().eval(symbolic_value_pair<symb_last_tag>(size-1));
  return x.derived().eval(symbolic_value_pair<symb_last_tag>(size-1));
}

// Convert a symbolic span into a usable one (i.e., remove last/end "keywords")
template<typename T>
struct make_size_type {
  typedef typename internal::conditional<is_symbolic<T>::value, Index, T>::type type;
};

template<typename FirstType,typename SizeType,typename IncrType>
struct MakeIndexing<ArithemeticSequence<FirstType,SizeType,IncrType> > {
  typedef ArithemeticSequence<Index,typename make_size_type<SizeType>::type,IncrType> type;
};

template<typename FirstType,typename SizeType,typename IncrType>
ArithemeticSequence<Index,typename make_size_type<SizeType>::type,IncrType>
make_indexing(const ArithemeticSequence<FirstType,SizeType,IncrType>& ids, Index size) {
  return ArithemeticSequence<Index,typename make_size_type<SizeType>::type,IncrType>(
            symbolic2value(ids.firstObject(),size),symbolic2value(ids.sizeObject(),size),ids.incrObject());
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

template<> struct get_compile_time_incr<AllRange> {
  enum { value = 1 };
};

} // end namespace internal

//--------------------------------------------------------------------------------

namespace legacy {
// Here are some initial code that I keep here for now to compare the quality of the code generated by the compilers

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

Index symbolic2value(last_t, Index size)          { return size-1; }
Index symbolic2value(shifted_last x, Index size)  { return size+x.offset-1; }
Index symbolic2value(end_t, Index size)           { return size; }
Index symbolic2value(shifted_end x, Index size)   { return size+x.offset; }

template<typename FirstType=Index,typename LastType=Index,typename IncrType=fix_t<1> >
class ArithemeticSequenceProxyWithBounds
{
public:
  ArithemeticSequenceProxyWithBounds(FirstType f, LastType l) : m_first(f), m_last(l) {}
  ArithemeticSequenceProxyWithBounds(FirstType f, LastType l, IncrType s) : m_first(f), m_last(l), m_incr(s) {}

  enum {
    SizeAtCompileTime = -1,
    IncrAtCompileTime = get_compile_time<IncrType,DynamicIndex>::value
  };

  Index size() const { return (m_last-m_first+m_incr)/m_incr; }
  Index operator[](Index i) const { return m_first + i * m_incr; }

  const FirstType& firstObject() const { return m_first; }
  const LastType&  lastObject()  const { return m_last; }
  const IncrType&  incrObject()  const { return m_incr; }

protected:
  FirstType m_first;
  LastType  m_last;
  IncrType  m_incr;
};

template<typename FirstType,typename LastType>
ArithemeticSequenceProxyWithBounds<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<LastType>::type >
seq(FirstType f, LastType l)  {
  return ArithemeticSequenceProxyWithBounds<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<LastType>::type>(f,l);
}

template<typename FirstType,typename LastType,typename IncrType>
ArithemeticSequenceProxyWithBounds<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<LastType>::type,typename cleanup_seq_type<IncrType>::type >
seq(FirstType f, LastType l, IncrType s)  {
  return ArithemeticSequenceProxyWithBounds<typename cleanup_seq_type<FirstType>::type,typename cleanup_seq_type<LastType>::type,typename cleanup_seq_type<IncrType>::type>(f,l,typename cleanup_seq_type<IncrType>::type(s));
}

}

namespace internal {

template<typename FirstType,typename LastType,typename IncrType>
struct get_compile_time_incr<legacy::ArithemeticSequenceProxyWithBounds<FirstType,LastType,IncrType> > {
  enum { value = get_compile_time<IncrType,DynamicIndex>::value };
};

// Convert a symbolic range into a usable one (i.e., remove last/end "keywords")
template<typename FirstType,typename LastType,typename IncrType>
struct MakeIndexing<legacy::ArithemeticSequenceProxyWithBounds<FirstType,LastType,IncrType> > {
  typedef legacy::ArithemeticSequenceProxyWithBounds<Index,Index,IncrType> type;
};

template<typename FirstType,typename LastType,typename IncrType>
legacy::ArithemeticSequenceProxyWithBounds<Index,Index,IncrType>
make_indexing(const legacy::ArithemeticSequenceProxyWithBounds<FirstType,LastType,IncrType>& ids, Index size) {
  return legacy::ArithemeticSequenceProxyWithBounds<Index,Index,IncrType>(
            symbolic2value(ids.firstObject(),size),symbolic2value(ids.lastObject(),size),ids.incrObject());
}

}

} // end namespace Eigen

#endif // EIGEN_ARITHMETIC_SEQUENCE_H
