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

//--------------------------------------------------------------------------------
// minimalistic symbolic scalar type
//--------------------------------------------------------------------------------


/** This namespace defines a set of classes and functions to build and evaluate symbolic expressions of scalar type Index.
  * Here is a simple example:
  *
  * \code
  * // First step, defines symbols:
  * struct x_tag {};  static const Symbolic::SymbolExpr<x_tag> x;
  * struct y_tag {};  static const Symbolic::SymbolExpr<y_tag> y;
  * struct z_tag {};  static const Symbolic::SymbolExpr<z_tag> z;
  *
  * // Defines an expression:
  * auto expr = (x+3)/y+z;
  *
  * // And evaluate it: (c++14)
  * std::cout << expr.eval(x=6,y=3,z=-13) << "\n";
  *
  * // In c++98/11, only one symbol per expression is supported for now:
  * auto expr98 = (3-x)/2;
  * std::cout << expr98.eval(x=6) << "\n";
  *
  * It is currently only used internally to define and minipulate the placeholders::last and placeholders::end symbols in Eigen::seq and Eigen::seqN.
  *
  */
namespace Symbolic {

template<typename Tag> class Symbol;
template<typename Arg0> class NegateExpr;
template<typename Arg1,typename Arg2> class AddExpr;
template<typename Arg1,typename Arg2> class ProductExpr;
template<typename Arg1,typename Arg2> class QuotientExpr;

// A simple wrapper around an Index to provide the eval method.
// We could also use a free-function symbolic_eval...
class ValueExpr {
public:
  ValueExpr(Index val) : m_value(val) {}
  template<typename T>
  Index eval_impl(const T&) const { return m_value; }
protected:
  Index m_value;
};

/** \class BaseExpr
  * Common base class of any symbolic expressions
  */
template<typename Derived>
class BaseExpr
{
public:
  const Derived& derived() const { return *static_cast<const Derived*>(this); }

  /** Evaluate the expression given the \a values of the symbols.
    *
    * \param values defines the values of the symbols, it can either be a SymbolValue or a std::tuple of SymbolValue
    *               as constructed by SymbolExpr::operator= operator.
    *
    */
  template<typename T>
  Index eval(const T& values) const { return derived().eval_impl(values); }

#if __cplusplus > 201103L
  template<typename... Types>
  Index eval(Types&&... values) const { return derived().eval_impl(std::make_tuple(values...)); }
#endif

  NegateExpr<Derived> operator-() const { return NegateExpr<Derived>(derived()); }

  AddExpr<Derived,ValueExpr> operator+(Index b) const
  { return AddExpr<Derived,ValueExpr >(derived(),  b); }
  AddExpr<Derived,ValueExpr> operator-(Index a) const
  { return AddExpr<Derived,ValueExpr >(derived(), -a); }
  QuotientExpr<Derived,ValueExpr> operator/(Index a) const
  { return QuotientExpr<Derived,ValueExpr>(derived(),a); }

  friend AddExpr<Derived,ValueExpr> operator+(Index a, const BaseExpr& b)
  { return AddExpr<Derived,ValueExpr>(b.derived(), a); }
  friend AddExpr<NegateExpr<Derived>,ValueExpr> operator-(Index a, const BaseExpr& b)
  { return AddExpr<NegateExpr<Derived>,ValueExpr>(-b.derived(), a); }
  friend AddExpr<ValueExpr,Derived> operator/(Index a, const BaseExpr& b)
  { return AddExpr<ValueExpr,Derived>(a,b.derived()); }

  template<typename OtherDerived>
  AddExpr<Derived,OtherDerived> operator+(const BaseExpr<OtherDerived> &b) const
  { return AddExpr<Derived,OtherDerived>(derived(),  b.derived()); }

  template<typename OtherDerived>
  AddExpr<Derived,NegateExpr<OtherDerived> > operator-(const BaseExpr<OtherDerived> &b) const
  { return AddExpr<Derived,NegateExpr<OtherDerived> >(derived(), -b.derived()); }

  template<typename OtherDerived>
  QuotientExpr<Derived,OtherDerived> operator/(const BaseExpr<OtherDerived> &b) const
  { return QuotientExpr<Derived,OtherDerived>(derived(), b.derived()); }
};

template<typename T>
struct is_symbolic {
  // BaseExpr has no conversion ctor, so we only to check whether T can be staticaly cast to its base class BaseExpr<T>.
  enum { value = internal::is_convertible<T,BaseExpr<T> >::value };
};

/** Represents the actual value of a symbol identified by its tag
  *
  * It is the return type of SymbolValue::operator=, and most of the time this is only way it is used.
  */
template<typename Tag>
class SymbolValue
{
public:
  /** Default constructor from the value \a val */
  SymbolValue(Index val) : m_value(val) {}

  /** \returns the stored value of the symbol */
  Index value() const { return m_value; }
protected:
  Index m_value;
};

/** Expression of a symbol uniquely identified by the tag \tparam TagT */
template<typename TagT>
class SymbolExpr : public BaseExpr<SymbolExpr<TagT> >
{
public:
  typedef TagT Tag;
  SymbolExpr() {}

  /** Associate the value \a val to the given symbol \c *this, uniquely identified by its \c Tag.
    *
    * The returned object should be passed to ExprBase::eval() to evaluate a given expression with this specified runtime-time value.
    */
  SymbolValue<Tag> operator=(Index val) const {
    return SymbolValue<Tag>(val);
  }

  Index eval_impl(const SymbolValue<Tag> &values) const { return values.value(); }

#if __cplusplus > 201103L
  // C++14 versions suitable for multiple symbols
  template<typename... Types>
  Index eval_impl(const std::tuple<Types...>& values) const { return std::get<SymbolValue<Tag> >(values).value(); }
#endif
};

template<typename Arg0>
class NegateExpr : public BaseExpr<NegateExpr<Arg0> >
{
public:
  NegateExpr(const Arg0& arg0) : m_arg0(arg0) {}

  template<typename T>
  Index eval_impl(const T& values) const { return -m_arg0.eval_impl(values); }
protected:
  Arg0 m_arg0;
};

template<typename Arg0, typename Arg1>
class AddExpr : public BaseExpr<AddExpr<Arg0,Arg1> >
{
public:
  AddExpr(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template<typename T>
  Index eval_impl(const T& values) const { return m_arg0.eval_impl(values) + m_arg1.eval_impl(values); }
protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

template<typename Arg0, typename Arg1>
class ProductExpr : public BaseExpr<ProductExpr<Arg0,Arg1> >
{
public:
  ProductExpr(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template<typename T>
  Index eval_impl(const T& values) const { return m_arg0.eval_impl(values) * m_arg1.eval_impl(values); }
protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

template<typename Arg0, typename Arg1>
class QuotientExpr : public BaseExpr<QuotientExpr<Arg0,Arg1> >
{
public:
  QuotientExpr(const Arg0& arg0, const Arg1& arg1) : m_arg0(arg0), m_arg1(arg1) {}

  template<typename T>
  Index eval_impl(const T& values) const { return m_arg0.eval_impl(values) / m_arg1.eval_impl(values); }
protected:
  Arg0 m_arg0;
  Arg1 m_arg1;
};

} // end namespace Symbolic

namespace placeholders {

namespace internal {
struct symbolic_last_tag {};
}

static const Symbolic::SymbolExpr<internal::symbolic_last_tag> last;
static const Symbolic::AddExpr<Symbolic::SymbolExpr<internal::symbolic_last_tag>,Symbolic::ValueExpr> end(last+1);

} // end namespace placeholders

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
  Index first()  const { return m_first; }
  Index operator[](Index i) const { return m_first + i * m_incr; }

  const FirstType& firstObject() const { return m_first; }
  const SizeType&  sizeObject()  const { return m_size; }
  const IncrType&  incrObject()  const { return m_incr; }

protected:
  FirstType m_first;
  SizeType  m_size;
  IncrType  m_incr;
};

namespace internal {

template<typename T, typename EnableIf=void> struct cleanup_seq_type { typedef T type; };
template<typename T> struct cleanup_seq_type<T,typename internal::enable_if<internal::is_integral<T>::value>::type> { typedef Index type; };
template<int N> struct cleanup_seq_type<fix_t<N> > { typedef fix_t<N> type; };
template<int N> struct cleanup_seq_type<fix_t<N> (*)() > { typedef fix_t<N> type; };

}

template<typename FirstType,typename SizeType,typename IncrType>
ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type,typename internal::cleanup_seq_type<IncrType>::type >
seqN(FirstType first, SizeType size, IncrType incr)  {
  return ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type,typename internal::cleanup_seq_type<IncrType>::type>(first,size,incr);
}

template<typename FirstType,typename SizeType>
ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type >
seqN(FirstType first, SizeType size)  {
  return ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type>(first,size);
}

#if EIGEN_HAS_CXX11
template<typename FirstType,typename LastType>
auto seq(FirstType f, LastType l) -> decltype(seqN(f,(l-f+fix<1>())))
{
  return seqN(f,(l-f+fix<1>()));
}

template<typename FirstType,typename LastType, typename IncrType>
auto seq(FirstType f, LastType l, IncrType incr)
  -> decltype(seqN(f,   (l-f+typename internal::cleanup_seq_type<IncrType>::type(incr))
                      / typename internal::cleanup_seq_type<IncrType>::type(incr),typename internal::cleanup_seq_type<IncrType>::type(incr)))
{
  typedef typename internal::cleanup_seq_type<IncrType>::type CleanedIncrType;
  return seqN(f,(l-f+CleanedIncrType(incr))/CleanedIncrType(incr),CleanedIncrType(incr));
}
#else
template<typename FirstType,typename LastType>
typename internal::enable_if<!(Symbolic::is_symbolic<FirstType>::value || Symbolic::is_symbolic<LastType>::value),
                             ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,Index> >::type
seq(FirstType f, LastType l)
{
  return seqN(f,(l-f+1));
}

template<typename FirstTypeDerived,typename LastType>
typename internal::enable_if<!Symbolic::is_symbolic<LastType>::value,
    ArithemeticSequence<FirstTypeDerived, Symbolic::AddExpr<Symbolic::AddExpr<Symbolic::NegateExpr<FirstTypeDerived>,Symbolic::ValueExpr>,
                                                            Symbolic::ValueExpr> > >::type
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, LastType l)
{
  return seqN(f.derived(),(l-f.derived()+1));
}

template<typename FirstType,typename LastTypeDerived>
typename internal::enable_if<!Symbolic::is_symbolic<FirstType>::value,
    ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,
                        Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,Symbolic::ValueExpr>,Symbolic::ValueExpr> > >::type
seq(FirstType f, const Symbolic::BaseExpr<LastTypeDerived> &l)
{
  return seqN(f,(l.derived()-f+1));
}

template<typename FirstTypeDerived,typename LastTypeDerived>
ArithemeticSequence<FirstTypeDerived,
                    Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,Symbolic::NegateExpr<FirstTypeDerived> >,Symbolic::ValueExpr> >
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, const Symbolic::BaseExpr<LastTypeDerived> &l)
{
  return seqN(f.derived(),(l.derived()-f.derived()+1));
}


template<typename FirstType,typename LastType, typename IncrType>
typename internal::enable_if<!(Symbolic::is_symbolic<FirstType>::value || Symbolic::is_symbolic<LastType>::value),
    ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,Index,typename internal::cleanup_seq_type<IncrType>::type> >::type
seq(FirstType f, LastType l, IncrType incr)
{
  typedef typename internal::cleanup_seq_type<IncrType>::type CleanedIncrType;
  return seqN(f,(l-f+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}

template<typename FirstTypeDerived,typename LastType, typename IncrType>
typename internal::enable_if<!Symbolic::is_symbolic<LastType>::value,
    ArithemeticSequence<FirstTypeDerived,
                        Symbolic::QuotientExpr<Symbolic::AddExpr<Symbolic::AddExpr<Symbolic::NegateExpr<FirstTypeDerived>,
                                                                                   Symbolic::ValueExpr>,
                                                                 Symbolic::ValueExpr>,
                                              Symbolic::ValueExpr>,
                        typename internal::cleanup_seq_type<IncrType>::type> >::type
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, LastType l, IncrType incr)
{
  typedef typename internal::cleanup_seq_type<IncrType>::type CleanedIncrType;
  return seqN(f.derived(),(l-f.derived()+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}

template<typename FirstType,typename LastTypeDerived, typename IncrType>
typename internal::enable_if<!Symbolic::is_symbolic<FirstType>::value,
    ArithemeticSequence<typename internal::cleanup_seq_type<FirstType>::type,
                        Symbolic::QuotientExpr<Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,Symbolic::ValueExpr>,
                                                                 Symbolic::ValueExpr>,
                                               Symbolic::ValueExpr>,
      typename internal::cleanup_seq_type<IncrType>::type> >::type
seq(FirstType f, const Symbolic::BaseExpr<LastTypeDerived> &l, IncrType incr)
{
  typedef typename internal::cleanup_seq_type<IncrType>::type CleanedIncrType;
  return seqN(f,(l.derived()-f+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}

template<typename FirstTypeDerived,typename LastTypeDerived, typename IncrType>
ArithemeticSequence<FirstTypeDerived,
                    Symbolic::QuotientExpr<Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,
                                                                               Symbolic::NegateExpr<FirstTypeDerived> >,
                                                             Symbolic::ValueExpr>,
                                          Symbolic::ValueExpr>,
  typename internal::cleanup_seq_type<IncrType>::type>
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, const Symbolic::BaseExpr<LastTypeDerived> &l, IncrType incr)
{
  typedef typename internal::cleanup_seq_type<IncrType>::type CleanedIncrType;
  return seqN(f.derived(),(l.derived()-f.derived()+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}
#endif

namespace internal {

template<typename T>
Index size(const T& x) { return x.size(); }

template<typename T,std::size_t N>
Index size(const T (&) [N]) { return N; }

template<typename T>
Index first(const T& x) { return x.first(); }

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
  Index first() const { return m_value; }
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
inline Index eval_expr_given_size(Index x, Index /* size */)   { return x; }

template<int N>
fix_t<N> eval_expr_given_size(fix_t<N> x, Index /*size*/)   { return x; }

template<typename Derived>
Index eval_expr_given_size(const Symbolic::BaseExpr<Derived> &x, Index size)
{
  return x.derived().eval(placeholders::last=size-1);
}

// Convert a symbolic span into a usable one (i.e., remove last/end "keywords")
template<typename T>
struct make_size_type {
  typedef typename internal::conditional<Symbolic::is_symbolic<T>::value, Index, T>::type type;
};

template<typename FirstType,typename SizeType,typename IncrType>
struct MakeIndexing<ArithemeticSequence<FirstType,SizeType,IncrType> > {
  typedef ArithemeticSequence<Index,typename make_size_type<SizeType>::type,IncrType> type;
};

template<typename FirstType,typename SizeType,typename IncrType>
ArithemeticSequence<Index,typename make_size_type<SizeType>::type,IncrType>
make_indexing(const ArithemeticSequence<FirstType,SizeType,IncrType>& ids, Index size) {
  return ArithemeticSequence<Index,typename make_size_type<SizeType>::type,IncrType>(
            eval_expr_given_size(ids.firstObject(),size),eval_expr_given_size(ids.sizeObject(),size),ids.incrObject());
}

// Convert a symbolic 'all' into a usable range
// Implementation-wise, it would be more efficient to not having to store m_size since
// this information is already in the nested expression. To this end, we would need a
// get_size(indices, underlying_size); function returning indices.size() by default.
struct AllRange {
  AllRange(Index size) : m_size(size) {}
  Index operator[](Index i) const { return i; }
  Index size() const { return m_size; }
  Index first() const { return 0; }
  Index m_size;
};

template<>
struct MakeIndexing<all_t> {
  typedef AllRange type;
};

inline AllRange make_indexing(all_t , Index size) {
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

inline Index eval_expr_given_size(last_t, Index size)          { return size-1; }
inline Index eval_expr_given_size(shifted_last x, Index size)  { return size+x.offset-1; }
inline Index eval_expr_given_size(end_t, Index size)           { return size; }
inline Index eval_expr_given_size(shifted_end x, Index size)   { return size+x.offset; }

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
ArithemeticSequenceProxyWithBounds<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<LastType>::type >
seq(FirstType f, LastType l)  {
  return ArithemeticSequenceProxyWithBounds<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<LastType>::type>(f,l);
}

template<typename FirstType,typename LastType,typename IncrType>
ArithemeticSequenceProxyWithBounds< typename internal::cleanup_seq_type<FirstType>::type,
                                    typename internal::cleanup_seq_type<LastType>::type,
                                    typename internal::cleanup_seq_type<IncrType>::type >
seq(FirstType f, LastType l, IncrType s)
{
  return ArithemeticSequenceProxyWithBounds<typename internal::cleanup_seq_type<FirstType>::type,
                                            typename internal::cleanup_seq_type<LastType>::type,
                                            typename internal::cleanup_seq_type<IncrType>::type>
                                           (f,l,typename internal::cleanup_seq_type<IncrType>::type(s));
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
            eval_expr_given_size(ids.firstObject(),size),eval_expr_given_size(ids.lastObject(),size),ids.incrObject());
}

}

} // end namespace Eigen

#endif // EIGEN_ARITHMETIC_SEQUENCE_H
