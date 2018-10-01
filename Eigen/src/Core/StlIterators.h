// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

namespace Eigen {

template<typename XprType,typename Derived>
class DenseStlIteratorBase
{
public:
  typedef std::ptrdiff_t difference_type;
  typedef std::random_access_iterator_tag iterator_category;

  DenseStlIteratorBase() : mp_xpr(0), m_index(0) {}
  DenseStlIteratorBase(XprType& xpr, Index index) : mp_xpr(&xpr), m_index(index) {}

  void swap(DenseStlIteratorBase& other) {
    std::swap(mp_xpr,other.mp_xpr);
    std::swap(m_index,other.m_index);
  }

  Derived& operator++() { ++m_index; return derived(); }
  Derived& operator--() { --m_index; return derived(); }

  Derived operator++(int) { Derived prev(derived()); operator++(); return prev;}
  Derived operator--(int) { Derived prev(derived()); operator--(); return prev;}

  friend Derived operator+(const DenseStlIteratorBase& a, int b) { Derived ret(a.derived()); ret += b; return ret; }
  friend Derived operator-(const DenseStlIteratorBase& a, int b) { Derived ret(a.derived()); ret -= b; return ret; }
  friend Derived operator+(int a, const DenseStlIteratorBase& b) { Derived ret(b.derived()); ret += a; return ret; }
  friend Derived operator-(int a, const DenseStlIteratorBase& b) { Derived ret(b.derived()); ret -= a; return ret; }
  
  Derived& operator+=(int b) { m_index += b; return derived(); }
  Derived& operator-=(int b) { m_index -= b; return derived(); }

  difference_type operator-(const DenseStlIteratorBase& other) const { eigen_assert(mp_xpr == other.mp_xpr);return m_index - other.m_index; }

  bool operator==(const DenseStlIteratorBase& other) { eigen_assert(mp_xpr == other.mp_xpr); return m_index == other.m_index; }
  bool operator!=(const DenseStlIteratorBase& other) { eigen_assert(mp_xpr == other.mp_xpr); return m_index != other.m_index; }
  bool operator< (const DenseStlIteratorBase& other) { eigen_assert(mp_xpr == other.mp_xpr); return m_index <  other.m_index; }
  bool operator<=(const DenseStlIteratorBase& other) { eigen_assert(mp_xpr == other.mp_xpr); return m_index <= other.m_index; }
  bool operator> (const DenseStlIteratorBase& other) { eigen_assert(mp_xpr == other.mp_xpr); return m_index >  other.m_index; }
  bool operator>=(const DenseStlIteratorBase& other) { eigen_assert(mp_xpr == other.mp_xpr); return m_index >= other.m_index; }

protected:

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  XprType *mp_xpr;
  Index m_index;
};

template<typename XprType>
class DenseStlIterator : public DenseStlIteratorBase<XprType, DenseStlIterator<XprType> >
{
public:
  typedef typename XprType::Scalar value_type;

protected:

  enum {
    has_direct_access = (internal::traits<XprType>::Flags & DirectAccessBit) ? 1 : 0,
    has_write_access  = internal::is_lvalue<XprType>::value
  };

  typedef DenseStlIteratorBase<XprType,DenseStlIterator> Base;
  using Base::m_index;
  using Base::mp_xpr;

  typedef typename internal::conditional<bool(has_direct_access), const value_type&, const value_type>::type read_only_ref_t;

public:
  
  typedef typename internal::conditional<bool(has_write_access), value_type *, const value_type *>::type pointer;
  typedef typename internal::conditional<bool(has_write_access), value_type&, read_only_ref_t>::type reference;
  

  DenseStlIterator() : Base() {}
  DenseStlIterator(XprType& xpr, Index index) : Base(xpr,index) {}

  reference operator*()       const { return (*mp_xpr)(m_index); }
  reference operator[](int i) const { return (*mp_xpr)(i);       }
  
  pointer   operator->() const { return &((*mp_xpr)(m_index)); }
};

template<typename XprType,typename Derived>
void swap(DenseStlIteratorBase<XprType,Derived>& a, DenseStlIteratorBase<XprType,Derived>& b) {
  a.swap(b);
}

template<typename Derived>
inline DenseStlIterator<Derived> DenseBase<Derived>::begin()
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return DenseStlIterator<Derived>(derived(), 0);
}

template<typename Derived>
inline DenseStlIterator<const Derived> DenseBase<Derived>::begin() const
{
  return cbegin();
}

template<typename Derived>
inline DenseStlIterator<const Derived> DenseBase<Derived>::cbegin() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return DenseStlIterator<const Derived>(derived(), 0);
}

template<typename Derived>
inline DenseStlIterator<Derived> DenseBase<Derived>::end()
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return DenseStlIterator<Derived>(derived(), size());
}

template<typename Derived>
inline DenseStlIterator<const Derived> DenseBase<Derived>::end() const
{
  return cend();
}

template<typename Derived>
inline DenseStlIterator<const Derived> DenseBase<Derived>::cend() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return DenseStlIterator<const Derived>(derived(), size());
}

template<typename XprType>
class DenseColStlIterator : public DenseStlIteratorBase<XprType, DenseColStlIterator<XprType> >
{
protected:

  enum { is_lvalue  = internal::is_lvalue<XprType>::value };

  typedef DenseStlIteratorBase<XprType,DenseColStlIterator> Base;
  using Base::m_index;
  using Base::mp_xpr;

public:
  typedef typename internal::conditional<bool(is_lvalue), typename XprType::ColXpr, typename XprType::ConstColXpr>::type value_type;
  typedef value_type* pointer;
  typedef value_type  reference;
  
  DenseColStlIterator() : Base() {}
  DenseColStlIterator(XprType& xpr, Index index) : Base(xpr,index) {}

  reference operator*()       const { return (*mp_xpr).col(m_index); }
  reference operator[](int i) const { return (*mp_xpr).col(i);       }
  
  pointer   operator->() const { return &((*mp_xpr).col(m_index)); }
};

template<typename XprType>
class DenseRowStlIterator : public DenseStlIteratorBase<XprType, DenseRowStlIterator<XprType> >
{
protected:

  enum { is_lvalue  = internal::is_lvalue<XprType>::value };

  typedef DenseStlIteratorBase<XprType,DenseRowStlIterator> Base;
  using Base::m_index;
  using Base::mp_xpr;

public:
  typedef typename internal::conditional<bool(is_lvalue), typename XprType::RowXpr, typename XprType::ConstRowXpr>::type value_type;
  typedef value_type* pointer;
  typedef value_type  reference;
  
  DenseRowStlIterator() : Base() {}
  DenseRowStlIterator(XprType& xpr, Index index) : Base(xpr,index) {}

  reference operator*()       const { return (*mp_xpr).row(m_index); }
  reference operator[](int i) const { return (*mp_xpr).row(i);       }
  
  pointer   operator->() const { return &((*mp_xpr).row(m_index)); }
};


template<typename Xpr>
class ColsProxy
{
public:
  ColsProxy(Xpr& xpr) : m_xpr(xpr) {}
  DenseColStlIterator<Xpr>       begin()  const { return DenseColStlIterator<Xpr>(m_xpr, 0); }
  DenseColStlIterator<const Xpr> cbegin() const { return DenseColStlIterator<const Xpr>(m_xpr, 0); }

  DenseColStlIterator<Xpr>       end()    const { return DenseColStlIterator<Xpr>(m_xpr, m_xpr.cols()); }
  DenseColStlIterator<const Xpr> cend()   const { return DenseColStlIterator<const Xpr>(m_xpr, m_xpr.cols()); }

protected:
  Xpr& m_xpr;
};

template<typename Xpr>
class RowsProxy
{
public:
  RowsProxy(Xpr& xpr) : m_xpr(xpr) {}
  DenseRowStlIterator<Xpr>       begin()  const { return DenseRowStlIterator<Xpr>(m_xpr, 0); }
  DenseRowStlIterator<const Xpr> cbegin() const { return DenseRowStlIterator<const Xpr>(m_xpr, 0); }

  DenseRowStlIterator<Xpr>       end()    const { return DenseRowStlIterator<Xpr>(m_xpr, m_xpr.rows()); }
  DenseRowStlIterator<const Xpr> cend()   const { return DenseRowStlIterator<const Xpr>(m_xpr, m_xpr.rows()); }

protected:
  Xpr& m_xpr;
};

template<typename Derived>
ColsProxy<Derived> DenseBase<Derived>::allCols()
{ return ColsProxy<Derived>(derived()); }

template<typename Derived>
ColsProxy<const Derived> DenseBase<Derived>::allCols() const
{ return ColsProxy<const Derived>(derived()); }

template<typename Derived>
RowsProxy<Derived> DenseBase<Derived>::allRows()
{ return RowsProxy<Derived>(derived()); }

template<typename Derived>
RowsProxy<const Derived> DenseBase<Derived>::allRows() const
{ return RowsProxy<const Derived>(derived()); }

} // namespace Eigen
