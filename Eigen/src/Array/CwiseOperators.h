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

/** \returns an expression of the coefficient-wise square root of *this. */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_sqrt_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseSqrt() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise exponential of *this. */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_exp_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseExp() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise logarithm of *this. */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_log_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseLog() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise cosine of *this. */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_cos_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseCos() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise sine of *this. */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_sin_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwiseSin() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise power of *this to the given exponent. */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_pow_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::cwisePow(const Scalar& exponent) const
{
  return CwiseUnaryOp<ei_scalar_pow_op<Scalar>, Derived>
    (derived(), ei_scalar_pow_op<Scalar>(exponent));
}

// -- binary operators --

/** \returns an expression of the coefficient-wise \< operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<std::less<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
MatrixBase<Derived>::cwiseLessThan(const MatrixBase<OtherDerived> &other) const
{
  return cwise(other, std::less<Scalar>());
}

/** \returns an expression of the coefficient-wise \<= operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<std::less_equal<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
MatrixBase<Derived>::cwiseLessEqual(const MatrixBase<OtherDerived> &other) const
{
  return cwise(other, std::less_equal<Scalar>());
}

/** \returns an expression of the coefficient-wise \> operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<std::greater<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
MatrixBase<Derived>::cwiseGreaterThan(const MatrixBase<OtherDerived> &other) const
{
  return cwise(other, std::greater<Scalar>());
}

/** \returns an expression of the coefficient-wise \>= operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<std::greater_equal<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
MatrixBase<Derived>::cwiseGreaterEqual(const MatrixBase<OtherDerived> &other) const
{
  return cwise(other, std::greater_equal<Scalar>());
}

/** \returns an expression of the coefficient-wise == operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<std::equal_to<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
MatrixBase<Derived>::cwiseEqualTo(const MatrixBase<OtherDerived> &other) const
{
  return cwise(other, std::equal_to<Scalar>());
}

/** \returns an expression of the coefficient-wise != operator of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Derived>
template<typename OtherDerived>
inline const CwiseBinaryOp<std::not_equal_to<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
MatrixBase<Derived>::cwiseNotEqualTo(const MatrixBase<OtherDerived> &other) const
{
  return cwise(other, std::not_equal_to<Scalar>());
}

#endif // EIGEN_ARRAY_CWISE_OPERATORS_H
