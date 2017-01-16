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

/** \namespace Eigen::placeholders
  * \ingroup Core_Module
  *
  * Namespace containing symbolic placeholder and identifiers
  */
namespace placeholders {

namespace internal {
struct symbolic_last_tag {};
}

/** \var last
  * \ingroup Core_Module
  *
  * Can be used as a parameter to Eigen::seq and Eigen::seqN functions to symbolically reference the last element/row/columns
  * of the underlying vector or matrix once passed to DenseBase::operator()(const RowIndices&, const ColIndices&).
  *
  * This symbolic placeholder support standard arithmetic operation.
  *
  * A typical usage example would be:
  * \code
  * using namespace Eigen;
  * using Eigen::placeholders::last;
  * VectorXd v(n);
  * v(seq(2,last-2)).setOnes();
  * \endcode
  *
  * \sa end
  */
static const Symbolic::SymbolExpr<internal::symbolic_last_tag> last;

/** \var end
  * \ingroup Core_Module
  *
  * Can be used as a parameter to Eigen::seq and Eigen::seqN functions to symbolically reference the last+1 element/row/columns
  * of the underlying vector or matrix once passed to DenseBase::operator()(const RowIndices&, const ColIndices&).
  *
  * This symbolic placeholder support standard arithmetic operation.
  * It is essentially an alias to last+1
  *
  * \sa last
  */
#ifdef EIGEN_PARSED_BY_DOXYGEN
static const auto end = last+1;
#else
static const Symbolic::AddExpr<Symbolic::SymbolExpr<internal::symbolic_last_tag>,Symbolic::ValueExpr> end(last+1);
#endif

} // end namespace placeholders

//--------------------------------------------------------------------------------
// seq(first,last,incr) and seqN(first,size,incr)
//--------------------------------------------------------------------------------

/** \class ArithmeticSequence
  * \ingroup Core_Module
  *
  * This class represents an arithmetic progression \f$ a_0, a_1, a_2, ..., a_{n-1}\f$ defined by
  * its \em first value \f$ a_0 \f$, its \em size (aka length) \em n, and the \em increment (aka stride)
  * that is equal to \f$ a_{i+1}-a_{i}\f$ for any \em i.
  *
  * It is internally used as the return type of the Eigen::seq and Eigen::seqN functions, and as the input arguments
  * of DenseBase::operator()(const RowIndices&, const ColIndices&), and most of the time this is the
  * only way it is used.
  *
  * \tparam FirstType type of the first element, usually an Index,
  *                   but internally it can be a symbolic expression
  * \tparam SizeType type representing the size of the sequence, usually an Index
  *                  or a compile time integral constant. Internally, it can also be a symbolic expression
  * \tparam IncrType type of the increment, can be a runtime Index, or a compile time integral constant (default is compile-time 1)
  *
  * \sa Eigen::seq, Eigen::seqN, DenseBase::operator()(const RowIndices&, const ColIndices&), class IndexedView
  */
template<typename FirstType=Index,typename SizeType=Index,typename IncrType=internal::fix_t<1> >
class ArithmeticSequence
{

public:
  ArithmeticSequence(FirstType first, SizeType size) : m_first(first), m_size(size) {}
  ArithmeticSequence(FirstType first, SizeType size, IncrType incr) : m_first(first), m_size(size), m_incr(incr) {}

  enum {
    SizeAtCompileTime = internal::get_compile_time<SizeType>::value,
    IncrAtCompileTime = internal::get_compile_time<IncrType,DynamicIndex>::value
  };

  /** \returns the size, i.e., number of elements, of the sequence */
  Index size()  const { return m_size; }

  /** \returns the first element \f$ a_0 \f$ in the sequence */
  Index first()  const { return m_first; }

  /** \returns the value \f$ a_i \f$ at index \a i in the sequence. */
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

// Cleanup return types:

// By default, no change:
template<typename T, int DynamicKey=Dynamic, typename EnableIf=void> struct cleanup_seq_type { typedef T type; };

// Convert short, int, unsigned int, etc. to Eigen::Index
template<typename T, int DynamicKey> struct cleanup_seq_type<T,DynamicKey,typename internal::enable_if<internal::is_integral<T>::value>::type> { typedef Index type; };

// In c++98/c++11, fix<N> is a pointer to function that we better cleanup to a true fix_t<N>:
template<int N, int DynamicKey> struct cleanup_seq_type<fix_t<N> (*)(), DynamicKey> { typedef fix_t<N> type; };

// If variable_or_fixed does not match DynamicKey, then we turn it to a pure compile-time value:
template<int N, int DynamicKey> struct cleanup_seq_type<variable_or_fixed<N>, DynamicKey> { typedef fix_t<N> type; };
// If variable_or_fixed matches DynamicKey, then we turn it to a pure runtime-value (aka Index):
template<int DynamicKey> struct cleanup_seq_type<variable_or_fixed<DynamicKey>, DynamicKey> { typedef Index type; };

// Helper to cleanup the type of the increment:
template<typename T> struct cleanup_seq_incr {
  typedef typename cleanup_seq_type<T,DynamicIndex>::type type;
};

}

/** \returns an ArithmeticSequence starting at \a first, of length \a size, and increment \a incr
  *
  * \sa seqN(FirstType,SizeType), seq(FirstType,LastType,IncrType) */
template<typename FirstType,typename SizeType,typename IncrType>
ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type,typename internal::cleanup_seq_incr<IncrType>::type >
seqN(FirstType first, SizeType size, IncrType incr)  {
  return ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type,typename internal::cleanup_seq_incr<IncrType>::type>(first,size,incr);
}

/** \returns an ArithmeticSequence starting at \a first, of length \a size, and unit increment
  *
  * \sa seqN(FirstType,SizeType,IncrType), seq(FirstType,LastType) */
template<typename FirstType,typename SizeType>
ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type >
seqN(FirstType first, SizeType size)  {
  return ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<SizeType>::type>(first,size);
}

#ifdef EIGEN_PARSED_BY_DOXYGEN

/** \returns an ArithmeticSequence starting at \a f, up (or down) to \a l, and with positive (or negative) increment \a incr
  *
  * It is essentially an alias to:
  * \code
  * seqN(f, (l-f+incr)/incr, incr);
  * \endcode
  *
  * \sa seqN(FirstType,SizeType,IncrType), seq(FirstType,LastType)
  */
template<typename FirstType,typename LastType, typename IncrType>
auto seq(FirstType f, LastType l, IncrType incr);

/** \returns an ArithmeticSequence starting at \a f, up (or down) to \a l, and unit increment
  *
  * It is essentially an alias to:
  * \code
  * seqN(f,l-f+1);
  * \endcode
  *
  * \sa seqN(FirstType,SizeType), seq(FirstType,LastType,IncrType)
  */
template<typename FirstType,typename LastType>
auto seq(FirstType f, LastType l);

#else // EIGEN_PARSED_BY_DOXYGEN

#if EIGEN_HAS_CXX11
template<typename FirstType,typename LastType>
auto seq(FirstType f, LastType l) -> decltype(seqN(f,(l-f+fix<1>())))
{
  return seqN(f,(l-f+fix<1>()));
}

template<typename FirstType,typename LastType, typename IncrType>
auto seq(FirstType f, LastType l, IncrType incr)
  -> decltype(seqN(f,   (l-f+typename internal::cleanup_seq_incr<IncrType>::type(incr))
                      / typename internal::cleanup_seq_incr<IncrType>::type(incr),typename internal::cleanup_seq_incr<IncrType>::type(incr)))
{
  typedef typename internal::cleanup_seq_incr<IncrType>::type CleanedIncrType;
  return seqN(f,(l-f+CleanedIncrType(incr))/CleanedIncrType(incr),CleanedIncrType(incr));
}
#else
template<typename FirstType,typename LastType>
typename internal::enable_if<!(Symbolic::is_symbolic<FirstType>::value || Symbolic::is_symbolic<LastType>::value),
                             ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,Index> >::type
seq(FirstType f, LastType l)
{
  return seqN(f,(l-f+1));
}

template<typename FirstTypeDerived,typename LastType>
typename internal::enable_if<!Symbolic::is_symbolic<LastType>::value,
    ArithmeticSequence<FirstTypeDerived, Symbolic::AddExpr<Symbolic::AddExpr<Symbolic::NegateExpr<FirstTypeDerived>,Symbolic::ValueExpr>,
                                                            Symbolic::ValueExpr> > >::type
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, LastType l)
{
  return seqN(f.derived(),(l-f.derived()+1));
}

template<typename FirstType,typename LastTypeDerived>
typename internal::enable_if<!Symbolic::is_symbolic<FirstType>::value,
    ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,
                        Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,Symbolic::ValueExpr>,Symbolic::ValueExpr> > >::type
seq(FirstType f, const Symbolic::BaseExpr<LastTypeDerived> &l)
{
  return seqN(f,(l.derived()-f+1));
}

template<typename FirstTypeDerived,typename LastTypeDerived>
ArithmeticSequence<FirstTypeDerived,
                    Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,Symbolic::NegateExpr<FirstTypeDerived> >,Symbolic::ValueExpr> >
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, const Symbolic::BaseExpr<LastTypeDerived> &l)
{
  return seqN(f.derived(),(l.derived()-f.derived()+1));
}


template<typename FirstType,typename LastType, typename IncrType>
typename internal::enable_if<!(Symbolic::is_symbolic<FirstType>::value || Symbolic::is_symbolic<LastType>::value),
    ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,Index,typename internal::cleanup_seq_incr<IncrType>::type> >::type
seq(FirstType f, LastType l, IncrType incr)
{
  typedef typename internal::cleanup_seq_incr<IncrType>::type CleanedIncrType;
  return seqN(f,(l-f+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}

template<typename FirstTypeDerived,typename LastType, typename IncrType>
typename internal::enable_if<!Symbolic::is_symbolic<LastType>::value,
    ArithmeticSequence<FirstTypeDerived,
                        Symbolic::QuotientExpr<Symbolic::AddExpr<Symbolic::AddExpr<Symbolic::NegateExpr<FirstTypeDerived>,
                                                                                   Symbolic::ValueExpr>,
                                                                 Symbolic::ValueExpr>,
                                              Symbolic::ValueExpr>,
                        typename internal::cleanup_seq_incr<IncrType>::type> >::type
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, LastType l, IncrType incr)
{
  typedef typename internal::cleanup_seq_incr<IncrType>::type CleanedIncrType;
  return seqN(f.derived(),(l-f.derived()+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}

template<typename FirstType,typename LastTypeDerived, typename IncrType>
typename internal::enable_if<!Symbolic::is_symbolic<FirstType>::value,
    ArithmeticSequence<typename internal::cleanup_seq_type<FirstType>::type,
                        Symbolic::QuotientExpr<Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,Symbolic::ValueExpr>,
                                                                 Symbolic::ValueExpr>,
                                               Symbolic::ValueExpr>,
                        typename internal::cleanup_seq_incr<IncrType>::type> >::type
seq(FirstType f, const Symbolic::BaseExpr<LastTypeDerived> &l, IncrType incr)
{
  typedef typename internal::cleanup_seq_incr<IncrType>::type CleanedIncrType;
  return seqN(f,(l.derived()-f+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}

template<typename FirstTypeDerived,typename LastTypeDerived, typename IncrType>
ArithmeticSequence<FirstTypeDerived,
                    Symbolic::QuotientExpr<Symbolic::AddExpr<Symbolic::AddExpr<LastTypeDerived,
                                                                               Symbolic::NegateExpr<FirstTypeDerived> >,
                                                             Symbolic::ValueExpr>,
                                          Symbolic::ValueExpr>,
                    typename internal::cleanup_seq_incr<IncrType>::type>
seq(const Symbolic::BaseExpr<FirstTypeDerived> &f, const Symbolic::BaseExpr<LastTypeDerived> &l, IncrType incr)
{
  typedef typename internal::cleanup_seq_incr<IncrType>::type CleanedIncrType;
  return seqN(f.derived(),(l.derived()-f.derived()+CleanedIncrType(incr))/CleanedIncrType(incr), incr);
}
#endif

#endif // EIGEN_PARSED_BY_DOXYGEN

namespace internal {

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

template<typename FirstType,typename SizeType,typename IncrType,int XprSize>
struct IndexedViewCompatibleType<ArithmeticSequence<FirstType,SizeType,IncrType>, XprSize> {
  typedef ArithmeticSequence<Index,typename make_size_type<SizeType>::type,IncrType> type;
};

template<typename FirstType,typename SizeType,typename IncrType>
ArithmeticSequence<Index,typename make_size_type<SizeType>::type,IncrType>
makeIndexedViewCompatible(const ArithmeticSequence<FirstType,SizeType,IncrType>& ids, Index size) {
  return ArithmeticSequence<Index,typename make_size_type<SizeType>::type,IncrType>(
            eval_expr_given_size(ids.firstObject(),size),eval_expr_given_size(ids.sizeObject(),size),ids.incrObject());
}

template<typename FirstType,typename SizeType,typename IncrType>
struct get_compile_time_incr<ArithmeticSequence<FirstType,SizeType,IncrType> > {
  enum { value = get_compile_time<IncrType,DynamicIndex>::value };
};

} // end namespace internal

//--------------------------------------------------------------------------------

namespace legacy {
// Here are some initial code that I keep here for now to compare the quality of the code generated by the compilers
// This part will be removed once we have checked everything is right.

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

template<typename FirstType=Index,typename LastType=Index,typename IncrType=internal::fix_t<1> >
class ArithmeticSequenceProxyWithBounds
{
public:
  ArithmeticSequenceProxyWithBounds(FirstType f, LastType l) : m_first(f), m_last(l) {}
  ArithmeticSequenceProxyWithBounds(FirstType f, LastType l, IncrType s) : m_first(f), m_last(l), m_incr(s) {}

  enum {
    SizeAtCompileTime = -1,
    IncrAtCompileTime = internal::get_compile_time<IncrType,DynamicIndex>::value
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
ArithmeticSequenceProxyWithBounds<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<LastType>::type >
seq(FirstType f, LastType l)  {
  return ArithmeticSequenceProxyWithBounds<typename internal::cleanup_seq_type<FirstType>::type,typename internal::cleanup_seq_type<LastType>::type>(f,l);
}

template<typename FirstType,typename LastType,typename IncrType>
ArithmeticSequenceProxyWithBounds< typename internal::cleanup_seq_type<FirstType>::type,
                                    typename internal::cleanup_seq_type<LastType>::type,
                                    typename internal::cleanup_seq_incr<IncrType>::type >
seq(FirstType f, LastType l, IncrType s)
{
  return ArithmeticSequenceProxyWithBounds<typename internal::cleanup_seq_type<FirstType>::type,
                                            typename internal::cleanup_seq_type<LastType>::type,
                                            typename internal::cleanup_seq_incr<IncrType>::type>
                                           (f,l,typename internal::cleanup_seq_incr<IncrType>::type(s));
}

}

namespace internal {

template<typename FirstType,typename LastType,typename IncrType>
struct get_compile_time_incr<legacy::ArithmeticSequenceProxyWithBounds<FirstType,LastType,IncrType> > {
  enum { value = get_compile_time<IncrType,DynamicIndex>::value };
};

// Convert a symbolic range into a usable one (i.e., remove last/end "keywords")
template<typename FirstType,typename LastType,typename IncrType,int XprSize>
struct IndexedViewCompatibleType<legacy::ArithmeticSequenceProxyWithBounds<FirstType,LastType,IncrType>,XprSize> {
  typedef legacy::ArithmeticSequenceProxyWithBounds<Index,Index,IncrType> type;
};

template<typename FirstType,typename LastType,typename IncrType>
legacy::ArithmeticSequenceProxyWithBounds<Index,Index,IncrType>
makeIndexedViewCompatible(const legacy::ArithmeticSequenceProxyWithBounds<FirstType,LastType,IncrType>& ids, Index size) {
  return legacy::ArithmeticSequenceProxyWithBounds<Index,Index,IncrType>(
            eval_expr_given_size(ids.firstObject(),size),eval_expr_given_size(ids.lastObject(),size),ids.incrObject());
}

}

} // end namespace Eigen

#endif // EIGEN_ARITHMETIC_SEQUENCE_H
