// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_SPARSE_CWISE_UNARY_OP_H
#define EIGEN_SPARSE_CWISE_UNARY_OP_H

// template<typename UnaryOp, typename MatrixType>
// struct ei_traits<SparseCwiseUnaryOp<UnaryOp, MatrixType> > : ei_traits<MatrixType>
// {
//   typedef typename ei_result_of<
//                      UnaryOp(typename MatrixType::Scalar)
//                    >::type Scalar;
//   typedef typename MatrixType::Nested MatrixTypeNested;
//   typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
//   enum {
//     CoeffReadCost = _MatrixTypeNested::CoeffReadCost + ei_functor_traits<UnaryOp>::Cost
//   };
// };

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOpImp<UnaryOp,MatrixType,Sparse>
  : public SparseMatrixBase<CwiseUnaryOp<UnaryOp, MatrixType> >
{
  public:

    class InnerIterator;
//     typedef typename ei_unref<LhsNested>::type _LhsNested;

    typedef CwiseUnaryOp<UnaryOp, MatrixType> Derived;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Derived)
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>::InnerIterator
{
    typedef typename CwiseUnaryOpImpl::Scalar Scalar;
    typedef typename ei_traits<CwiseUnaryOpImpl>::_MatrixTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const SparseCwiseUnaryOp& unaryOp, int outer)
      : m_iter(unaryOp.nestedExpression(),outer), m_functor(unaryOp._functor())
    {}

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    { ++m_iter; return *this; }

    EIGEN_STRONG_INLINE Scalar value() const { return m_functor(m_iter.value()); }

    EIGEN_STRONG_INLINE int index() const { return m_iter.index(); }
    EIGEN_STRONG_INLINE int row() const { return m_iter.row(); }
    EIGEN_STRONG_INLINE int col() const { return m_iter.col(); }

    EIGEN_STRONG_INLINE operator bool() const { return m_iter; }

  protected:
    MatrixTypeIterator m_iter;
    const UnaryOp m_functor;
};

template<typename ViewOp, typename MatrixType>
class CwiseUnaryOpImp<ViewOp,MatrixType,Sparse>
  : public SparseMatrixBase<CwiseUnaryView<ViewOp, MatrixType> >
{
  public:

    class InnerIterator;
//     typedef typename ei_unref<LhsNested>::type _LhsNested;

    typedef CwiseUnaryView<ViewOp, MatrixType> Derived;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Derived)
};

template<typename ViewOp, typename MatrixType>
class CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::InnerIterator
{
    typedef typename CwiseUnaryViewImpl::Scalar Scalar;
    typedef typename ei_traits<SparseCwiseUnaryOp>::_MatrixTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const CwiseUnaryViewImpl& unaryView, int outer)
      : m_iter(unaryView.nestedExpression(),outer), m_functor(unaryView._functor())
    {}

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    { ++m_iter; return *this; }

    EIGEN_STRONG_INLINE Scalar value() const { return m_functor(m_iter.value()); }

    EIGEN_STRONG_INLINE int index() const { return m_iter.index(); }
    EIGEN_STRONG_INLINE int row() const { return m_iter.row(); }
    EIGEN_STRONG_INLINE int col() const { return m_iter.col(); }

    EIGEN_STRONG_INLINE operator bool() const { return m_iter; }

  protected:
    MatrixTypeIterator m_iter;
    const UnaryOp m_functor;
};


/*
template<typename Derived>
template<typename CustomUnaryOp>
EIGEN_STRONG_INLINE const SparseCwiseUnaryOp<CustomUnaryOp, Derived>
SparseMatrixBase<Derived>::unaryExpr(const CustomUnaryOp& func) const
{
  return SparseCwiseUnaryOp<CustomUnaryOp, Derived>(derived(), func);
}

template<typename Derived>
EIGEN_STRONG_INLINE const SparseCwiseUnaryOp<ei_scalar_opposite_op<typename ei_traits<Derived>::Scalar>,Derived>
SparseMatrixBase<Derived>::operator-() const
{
  return derived();
}

template<typename ExpressionType>
EIGEN_STRONG_INLINE const EIGEN_SPARSE_CWISE_UNOP_RETURN_TYPE(ei_scalar_abs_op)
SparseCwise<ExpressionType>::abs() const
{
  return _expression();
}

template<typename ExpressionType>
EIGEN_STRONG_INLINE const EIGEN_SPARSE_CWISE_UNOP_RETURN_TYPE(ei_scalar_abs2_op)
SparseCwise<ExpressionType>::abs2() const
{
  return _expression();
}

template<typename Derived>
EIGEN_STRONG_INLINE typename SparseMatrixBase<Derived>::ConjugateReturnType
SparseMatrixBase<Derived>::conjugate() const
{
  return ConjugateReturnType(derived());
}

template<typename Derived>
EIGEN_STRONG_INLINE const typename SparseMatrixBase<Derived>::RealReturnType
SparseMatrixBase<Derived>::real() const { return derived(); }

template<typename Derived>
EIGEN_STRONG_INLINE const typename SparseMatrixBase<Derived>::ImagReturnType
SparseMatrixBase<Derived>::imag() const { return derived(); }

template<typename Derived>
template<typename NewType>
EIGEN_STRONG_INLINE const SparseCwiseUnaryOp<ei_scalar_cast_op<typename ei_traits<Derived>::Scalar, NewType>, Derived>
SparseMatrixBase<Derived>::cast() const
{
  return derived();
}

template<typename Derived>
EIGEN_STRONG_INLINE const SparseCwiseUnaryOp<ei_scalar_multiple_op<typename ei_traits<Derived>::Scalar>, Derived>
SparseMatrixBase<Derived>::operator*(const Scalar& scalar) const
{
  return SparseCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, Derived>
    (derived(), ei_scalar_multiple_op<Scalar>(scalar));
}

template<typename Derived>
EIGEN_STRONG_INLINE const SparseCwiseUnaryOp<ei_scalar_quotient1_op<typename ei_traits<Derived>::Scalar>, Derived>
SparseMatrixBase<Derived>::operator/(const Scalar& scalar) const
{
  return SparseCwiseUnaryOp<ei_scalar_quotient1_op<Scalar>, Derived>
    (derived(), ei_scalar_quotient1_op<Scalar>(scalar));
}

template<typename Derived>
EIGEN_STRONG_INLINE Derived&
SparseMatrixBase<Derived>::operator*=(const Scalar& other)
{
  for (int j=0; j<outerSize(); ++j)
    for (typename Derived::InnerIterator i(derived(),j); i; ++i)
      i.valueRef() *= other;
  return derived();
}

template<typename Derived>
EIGEN_STRONG_INLINE Derived&
SparseMatrixBase<Derived>::operator/=(const Scalar& other)
{
  for (int j=0; j<outerSize(); ++j)
    for (typename Derived::InnerIterator i(derived(),j); i; ++i)
      i.valueRef() /= other;
  return derived();
}*/

#endif // EIGEN_SPARSE_CWISE_UNARY_OP_H
