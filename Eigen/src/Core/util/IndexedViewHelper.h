// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_INDEXED_VIEW_HELPER_H
#define EIGEN_INDEXED_VIEW_HELPER_H

namespace Eigen {

namespace internal {

// Extract increment/step at compile time
template<typename T, typename EnableIf = void> struct get_compile_time_incr {
  enum { value = UndefinedIncr };
};

// Analogue of std::get<0>(x), but tailored for our needs.
template<typename T>
Index first(const T& x) { return x.first(); }

// IndexedViewCompatibleType/makeIndexedViewCompatible turn an arbitrary object of type T into something usable by MatrixSlice
// The generic implementation is a no-op
template<typename T,int XprSize,typename EnableIf=void>
struct IndexedViewCompatibleType {
  typedef T type;
};

template<typename T>
const T& makeIndexedViewCompatible(const T& x, Index /*size*/) { return x; }

//--------------------------------------------------------------------------------
// Handling of a single Index
//--------------------------------------------------------------------------------

struct SingleRange {
  enum {
    SizeAtCompileTime = 1
  };
  SingleRange(Index val) : m_value(val) {}
  Index operator[](Index) const { return m_value; }
  Index size() const { return 1; }
  Index first() const { return m_value; }
  Index m_value;
};

template<> struct get_compile_time_incr<SingleRange> {
  enum { value = 1 }; // 1 or 0 ??
};

// Turn a single index into something that looks like an array (i.e., that exposes a .size(), and operatro[](int) methods)
template<typename T, int XprSize>
struct IndexedViewCompatibleType<T,XprSize,typename internal::enable_if<internal::is_integral<T>::value>::type> {
  // Here we could simply use Array, but maybe it's less work for the compiler to use
  // a simpler wrapper as SingleRange
  //typedef Eigen::Array<Index,1,1> type;
  typedef SingleRange type;
};

//--------------------------------------------------------------------------------
// Handling of all
//--------------------------------------------------------------------------------

struct all_t { all_t() {} };

// Convert a symbolic 'all' into a usable range type
template<int XprSize>
struct AllRange {
  enum { SizeAtCompileTime = XprSize };
  AllRange(Index size = XprSize) : m_size(size) {}
  Index operator[](Index i) const { return i; }
  Index size() const { return m_size.value(); }
  Index first() const { return 0; }
  variable_if_dynamic<Index,XprSize> m_size;
};

template<int XprSize>
struct IndexedViewCompatibleType<all_t,XprSize> {
  typedef AllRange<XprSize> type;
};

template<typename XprSizeType>
inline AllRange<get_compile_time<XprSizeType>::value> makeIndexedViewCompatible(all_t , XprSizeType size) {
  return AllRange<get_compile_time<XprSizeType>::value>(size);
}

template<int Size> struct get_compile_time_incr<AllRange<Size> > {
  enum { value = 1 };
};

} // end namespace internal


namespace placeholders {

/** \var all
  * \ingroup Core_Module
  * Can be used as a parameter to DenseBase::operator()(const RowIndices&, const ColIndices&) to index all rows or columns
  */
static const Eigen::internal::all_t all;

}

} // end namespace Eigen

#endif // EIGEN_INDEXED_VIEW_HELPER_H
