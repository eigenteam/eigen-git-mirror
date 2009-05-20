// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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
                     ViewOp(typename MatrixType::Scalar)
                   >::type Scalar;
  typedef typename MatrixType::Nested MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    Flags = (_MatrixTypeNested::Flags & (HereditaryBits | LinearAccessBit | AlignedBit)),
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost + ei_functor_traits<ViewOp>::Cost
  };
};

template<typename ViewOp, typename MatrixType>
class CwiseUnaryView : ei_no_assignment_operator,
  public MatrixBase<CwiseUnaryView<ViewOp, MatrixType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseUnaryView)

    inline CwiseUnaryView(const MatrixType& mat, const ViewOp& func = ViewOp())
      : m_matrix(mat), m_functor(func) {}
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(CwiseUnaryView)

    EIGEN_STRONG_INLINE int rows() const { return m_matrix.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return m_matrix.cols(); }

    EIGEN_STRONG_INLINE const Scalar coeff(int row, int col) const
    {
      return m_functor(m_matrix.coeff(row, col));
    }

    EIGEN_STRONG_INLINE const Scalar coeff(int index) const
    {
      return m_functor(m_matrix.coeff(index));
    }
    
    EIGEN_STRONG_INLINE Scalar& coeffRef(int row, int col)
    {
      return m_functor(m_matrix.const_cast_derived().coeffRef(row, col));
    }

    EIGEN_STRONG_INLINE Scalar& coeffRef(int index)
    {
      return m_functor(m_matrix.const_cast_derived().coeffRef(index));
    }

  protected:
    const typename MatrixType::Nested m_matrix;
    const ViewOp m_functor;
};

/** \returns an expression of a custom coefficient-wise unary operator \a func of *this
  *
  * The template parameter \a CustomUnaryOp is the type of the functor
  * of the custom unary operator.
  *
  * \addexample CustomCwiseUnaryFunctors \label How to use custom coeff wise unary functors
  *
  * Example:
  * \include class_CwiseUnaryOp.cpp
  * Output: \verbinclude class_CwiseUnaryOp.out
  *
  * \sa class CwiseUnaryOp, class CwiseBinarOp, MatrixBase::operator-, Cwise::abs
  */
template<typename Derived>
template<typename CustomViewOp>
EIGEN_STRONG_INLINE const CwiseUnaryView<CustomViewOp, Derived>
MatrixBase<Derived>::unaryViewExpr(const CustomViewOp& func) const
{
  return CwiseUnaryView<CustomViewOp, Derived>(derived(), func);
}

/** \returns a non const expression of the real part of \c *this.
  *
  * \sa imag() */
template<typename Derived>
EIGEN_STRONG_INLINE typename MatrixBase<Derived>::NonConstRealReturnType
MatrixBase<Derived>::real() { return derived(); }

/** \returns a non const expression of the imaginary part of \c *this.
  *
  * \sa real() */
template<typename Derived>
EIGEN_STRONG_INLINE typename MatrixBase<Derived>::NonConstImagReturnType
MatrixBase<Derived>::imag() { return derived(); }

#endif // EIGEN_CWISE_UNARY_VIEW_H
