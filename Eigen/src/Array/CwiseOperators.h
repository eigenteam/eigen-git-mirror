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
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_sqrt_op>::ReturnType
Cwise<ExpressionType>::sqrt() const
{
  return _expression();
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise exponential of *this. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_exp_op>::ReturnType
Cwise<ExpressionType>::exp() const
{
  return _expression();
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise logarithm of *this. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_log_op>::ReturnType
Cwise<ExpressionType>::log() const
{
  return _expression();
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise cosine of *this. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_cos_op>::ReturnType
Cwise<ExpressionType>::cos() const
{
  return _expression();
}


/** \array_module
  * 
  * \returns an expression of the coefficient-wise sine of *this. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_sin_op>::ReturnType
Cwise<ExpressionType>::sin() const
{
  return _expression();
}


/** \array_module
  * 
  * \returns an expression of the coefficient-wise power of *this to the given exponent. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_pow_op>::ReturnType
Cwise<ExpressionType>::pow(const Scalar& exponent) const
{
  return typename UnOp<ei_scalar_pow_op>::ReturnType(_expression(), ei_scalar_pow_op<Scalar>(exponent));
}


/** \array_module
  * 
  * \returns an expression of the coefficient-wise inverse of *this. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_inverse_op>::ReturnType
Cwise<ExpressionType>::inverse() const
{
  return _expression();
}

/** \array_module
  *
  * \returns an expression of the coefficient-wise square of *this. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_square_op>::ReturnType
Cwise<ExpressionType>::square() const
{
  return _expression();
}

/** \array_module
  *
  * \returns an expression of the coefficient-wise cube of *this. */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::template UnOp<ei_scalar_cube_op>::ReturnType
Cwise<ExpressionType>::cube() const
{
  return _expression();
}


// -- binary operators --

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \< operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const typename Cwise<ExpressionType>::template BinOp<std::less, OtherDerived>::ReturnType
Cwise<ExpressionType>::operator<(const MatrixBase<OtherDerived> &other) const
{
  return typename BinOp<std::less, OtherDerived>::ReturnType(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \<= operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const typename Cwise<ExpressionType>::template BinOp<std::less_equal, OtherDerived>::ReturnType
Cwise<ExpressionType>::operator<=(const MatrixBase<OtherDerived> &other) const
{
  return typename BinOp<std::less_equal, OtherDerived>::ReturnType(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \> operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const typename Cwise<ExpressionType>::template BinOp<std::greater, OtherDerived>::ReturnType
Cwise<ExpressionType>::operator>(const MatrixBase<OtherDerived> &other) const
{
  return typename BinOp<std::greater, OtherDerived>::ReturnType(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise \>= operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const typename Cwise<ExpressionType>::template BinOp<std::greater_equal, OtherDerived>::ReturnType
Cwise<ExpressionType>::operator>=(const MatrixBase<OtherDerived> &other) const
{
  return typename BinOp<std::greater_equal, OtherDerived>::ReturnType(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise == operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const typename Cwise<ExpressionType>::template BinOp<std::equal_to, OtherDerived>::ReturnType
Cwise<ExpressionType>::operator==(const MatrixBase<OtherDerived> &other) const
{
  return typename BinOp<std::equal_to, OtherDerived>::ReturnType(_expression(), other.derived());
}

/** \array_module
  * 
  * \returns an expression of the coefficient-wise != operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename ExpressionType>
template<typename OtherDerived>
inline const typename Cwise<ExpressionType>::template BinOp<std::not_equal_to, OtherDerived>::ReturnType
Cwise<ExpressionType>::operator!=(const MatrixBase<OtherDerived> &other) const
{
  return typename BinOp<std::not_equal_to, OtherDerived>::ReturnType(_expression(), other.derived());
}


/** \returns an expression of \c *this with each coeff incremented by the constant \a scalar */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::ScalarAddReturnType
Cwise<ExpressionType>::operator+(const Scalar& scalar) const
{
  return typename Cwise<ExpressionType>::ScalarAddReturnType(m_matrix, ei_scalar_add_op<Scalar>(scalar));
}

/** \see operator+ */
template<typename ExpressionType>
inline ExpressionType& Cwise<ExpressionType>::operator+=(const Scalar& scalar)
{
  return m_matrix.const_cast_derived() = *this + scalar;
}

/** \returns an expression of \c *this with each coeff decremented by the constant \a scalar */
template<typename ExpressionType>
inline const typename Cwise<ExpressionType>::ScalarAddReturnType
Cwise<ExpressionType>::operator-(const Scalar& scalar) const
{
  return *this + (-scalar);
}

/** \see operator- */
template<typename ExpressionType>
inline ExpressionType& Cwise<ExpressionType>::operator-=(const Scalar& scalar)
{
  return m_matrix.const_cast_derived() = *this - scalar;
}

#endif // EIGEN_ARRAY_CWISE_OPERATORS_H
