// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
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

#ifndef EIGEN_ARRAY_CWISE_OPERATORS_H
#define EIGEN_ARRAY_CWISE_OPERATORS_H

// -- unary operators --

/** \array_module
  * 
  * \returns an expression of the coefficient-wise square root of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_sqrt_op)
Cwise<ExpressionType>::sqrt() const
{
  return _expression();
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise exponential of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_exp_op)
Cwise<ExpressionType>::exp() const
{
  return _expression();
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise logarithm of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_log_op)
Cwise<ExpressionType>::log() const
{
  return _expression();
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise cosine of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_cos_op)
Cwise<ExpressionType>::cos() const
{
  return _expression();
}


/** \array_module
  * 
  * \returns an expression of the coefficient-wise sine of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_sin_op)
Cwise<ExpressionType>::sin() const
{
  return _expression();
}


/** \array_module
  * 
  * \returns an expression of the coefficient-wise power of *this to the given exponent. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_pow_op)
Cwise<ExpressionType>::pow(const Scalar& exponent) const
{
  return EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_pow_op)(_expression(), ei_scalar_pow_op<Scalar>(exponent));
}


/** \array_module
  * 
  * \returns an expression of the coefficient-wise inverse of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_inverse_op)
Cwise<ExpressionType>::inverse() const
{
  return _expression();
}

/** \array_module
  *
  * \returns an expression of the coefficient-wise square of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_square_op)
Cwise<ExpressionType>::square() const
{
  return _expression();
}

/** \array_module
  *
  * \returns an expression of the coefficient-wise cube of *this. */
template<typename ExpressionType>
inline const EIGEN_CWISE_UNOP_RETURN_TYPE(ei_scalar_cube_op)
Cwise<ExpressionType>::cube() const
{
  return _expression();
}


// -- binary operators --

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \< operator of *this and \a other
  *
  * See MatrixBase::all() for an example.
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::less)
Cwise<ExpressionType>::operator<(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(std::less)(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \<= operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::less_equal)
Cwise<ExpressionType>::operator<=(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(std::less_equal)(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \> operator of *this and \a other
  *
  * See MatrixBase::all() for an example.
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater)
Cwise<ExpressionType>::operator>(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater)(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \>= operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater_equal)
Cwise<ExpressionType>::operator>=(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater_equal)(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise == operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::equal_to)
Cwise<ExpressionType>::operator==(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(std::equal_to)(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise != operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::not_equal_to)
Cwise<ExpressionType>::operator!=(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_CWISE_BINOP_RETURN_TYPE(std::not_equal_to)(_expression(), other.derived());
}


/** \array_module
  * \returns an expression of \c *this with each coeff incremented by the constant \a scalar */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::ScalarAddReturnType
Cwise<ExpressionType>::operator+(const Scalar& scalar) const
{
  return typename Cwise<ExpressionType>::ScalarAddReturnType(m_matrix, ei_scalar_add_op<Scalar>(scalar));
}

/** \array_module
  * \see operator+() */
template<typename ExpressionType>
inline ExpressionType& Cwise<ExpressionType>::operator+=(const Scalar& scalar)
{
  return m_matrix.const_cast_derived() = *this + scalar;
}

/** \array_module
  * \returns an expression of \c *this with each coeff decremented by the constant \a scalar */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::ScalarAddReturnType
Cwise<ExpressionType>::operator-(const Scalar& scalar) const
{
  return *this + (-scalar);
}

/** \array_module
  * \see operator- */
template<typename ExpressionType>
inline ExpressionType& Cwise<ExpressionType>::operator-=(const Scalar& scalar)
{
  return m_matrix.const_cast_derived() = *this - scalar;
}

#endif // EIGEN_ARRAY_CWISE_OPERATORS_H
