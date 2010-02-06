// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_CWISE_UNARY_VIEW_H
#define EIGEN_CWISE_UNARY_VIEW_H

/** \class CwiseUnaryView
  *
  * \brief Generic lvalue expression of a coefficient-wise unary operator of a matrix or a vector
  *
  * \param ViewOp template functor implementing the view
  * \param MatrixType the type of the matrix we are applying the unary operator
  *
  * This class represents a lvalue expression of a generic unary view operator of a matrix or a vector.
  * It is the return type of real() and imag(), and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::unaryViewExpr(const CustomUnaryOp &) const, class CwiseUnaryOp
  */
template<typename ViewOp, typename MatrixType>
struct ei_traits<CwiseUnaryView<ViewOp, MatrixType> >
 : ei_traits<MatrixType>
{
  typedef typename ei_result_of<
                     ViewOp(typename ei_traits<MatrixType>::Scalar)
                   >::type Scalar;
  typedef typename MatrixType::Nested MatrixTypeNested;
  typedef typename ei_cleantype<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    Flags = (ei_traits<_MatrixTypeNested>::Flags & (HereditaryBits | LinearAccessBit | AlignedBit))
      | EIGEN_PROPAGATE_NESTING_BIT(ei_traits<MatrixType>::Flags), // if I am not wrong, I need to test this on MatrixType and not on the nested type
    CoeffReadCost = ei_traits<_MatrixTypeNested>::CoeffReadCost + ei_functor_traits<ViewOp>::Cost
  };
};

template<typename ViewOp, typename MatrixType, typename StorageType>
class CwiseUnaryViewImpl;

template<typename ViewOp, typename MatrixType>
class CwiseUnaryView : ei_no_assignment_operator,
  public CwiseUnaryViewImpl<ViewOp, MatrixType, typename ei_traits<MatrixType>::StorageType>
{
  public:

    typedef typename CwiseUnaryViewImpl<ViewOp, MatrixType,typename ei_traits<MatrixType>::StorageType>::Base Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE_NEW(CwiseUnaryView)

    inline CwiseUnaryView(const MatrixType& mat, const ViewOp& func = ViewOp())
      : m_matrix(mat), m_functor(func) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(CwiseUnaryView)

    EIGEN_STRONG_INLINE int rows() const { return m_matrix.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return m_matrix.cols(); }

    /** \returns the functor representing unary operation */
    const ViewOp& functor() const { return m_functor; }

    /** \returns the nested expression */
    const typename ei_cleantype<typename MatrixType::Nested>::type&
    nestedExpression() const { return m_matrix; }

    /** \returns the nested expression */
    typename ei_cleantype<typename MatrixType::Nested>::type&
    nestedExpression() { return m_matrix.const_cast_derived(); }

  protected:
    // FIXME changed from MatrixType::Nested because of a weird compilation error with sun CC
    const typename ei_nested<MatrixType>::type m_matrix;
    const ViewOp m_functor;
};

template<typename ViewOp, typename MatrixType>
class CwiseUnaryViewImpl<ViewOp,MatrixType,Dense>
  : public MatrixType::template MakeBase< CwiseUnaryView<ViewOp, MatrixType> >::Type
{
    typedef CwiseUnaryView<ViewOp, MatrixType> Derived;

  public:

    typedef typename MatrixType::template MakeBase< CwiseUnaryView<ViewOp, MatrixType> >::Type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Derived)

    EIGEN_STRONG_INLINE const Scalar coeff(int row, int col) const
    {
      return derived().functor()(derived().nestedExpression().coeff(row, col));
    }

    EIGEN_STRONG_INLINE const Scalar coeff(int index) const
    {
      return derived().functor()(derived().nestedExpression().coeff(index));
    }

    EIGEN_STRONG_INLINE Scalar& coeffRef(int row, int col)
    {
      return derived().functor()(const_cast_derived().nestedExpression().coeffRef(row, col));
    }

    EIGEN_STRONG_INLINE Scalar& coeffRef(int index)
    {
      return derived().functor()(const_cast_derived().nestedExpression().coeffRef(index));
    }
};



#endif // EIGEN_CWISE_UNARY_VIEW_H
