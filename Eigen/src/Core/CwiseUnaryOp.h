// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_CWISE_UNARY_OP_H
#define EIGEN_CWISE_UNARY_OP_H

/** \class CwiseUnaryOp
  *
  * \brief Generic expression of a coefficient-wise unary operator of a matrix or a vector
  *
  * \param UnaryOp template functor implementing the operator
  * \param MatrixType the type of the matrix we are applying the unary operator
  *
  * This class represents an expression of a generic unary operator of a matrix or a vector.
  * It is the return type of the unary operator-, of a matrix or a vector, and most
  * of the time this is the only way it is used.
  *
  * \sa MatrixBase::cwise(const CustomUnaryOp &) const, class CwiseBinaryOp, class CwiseNullaryOp
  */
template<typename UnaryOp, typename MatrixType>
struct ei_traits<CwiseUnaryOp<UnaryOp, MatrixType> >
{
  typedef typename ei_result_of<
                     UnaryOp(typename MatrixType::Scalar)
                   >::type Scalar;
  typedef typename MatrixType::Nested MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    MatrixTypeCoeffReadCost = _MatrixTypeNested::CoeffReadCost,
    MatrixTypeFlags = _MatrixTypeNested::Flags,
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = (MatrixTypeFlags & (
      HereditaryBits | Like1DArrayBit
      | (ei_functor_traits<UnaryOp>::IsVectorizable ? VectorizableBit : 0))),
    CoeffReadCost = MatrixTypeCoeffReadCost + ei_functor_traits<UnaryOp>::Cost
  };
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOp : ei_no_assignment_operator,
  public MatrixBase<CwiseUnaryOp<UnaryOp, MatrixType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseUnaryOp)

    inline CwiseUnaryOp(const MatrixType& mat, const UnaryOp& func = UnaryOp())
      : m_matrix(mat), m_functor(func) {}

  private:

    inline int _rows() const { return m_matrix.rows(); }
    inline int _cols() const { return m_matrix.cols(); }

    inline const Scalar _coeff(int row, int col) const
    {
      return m_functor(m_matrix.coeff(row, col));
    }

    template<int LoadMode>
    inline PacketScalar _packetCoeff(int row, int col) const
    {
      return m_functor.packetOp(m_matrix.template packetCoeff<LoadMode>(row, col));
    }

  protected:
    const typename MatrixType::Nested m_matrix;
    const UnaryOp m_functor;
};

/** \returns an expression of a custom coefficient-wise unary operator \a func of *this
  *
  * The template parameter \a CustomUnaryOp is the type of the functor
  * of the custom unary operator.
  *
  * Here is an example:
  * \include class_CwiseUnaryOp.cpp
  * Output: \verbinclude class_CwiseUnaryOp.out
  *
  * \sa class CwiseUnaryOp, class CwiseBinarOp, MatrixBase::operator-, MatrixBase::cwiseAbs
  */
template<typename Derived>
template<typename CustomUnaryOp>
inline const CwiseUnaryOp<CustomUnaryOp, Derived>
MatrixBase<Derived>::cwise(const CustomUnaryOp& func) const
{
  return CwiseUnaryOp<CustomUnaryOp, Derived>(derived(), func);
}

/** \returns an expression of the opposite of \c *this
  */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_opposite_op<typename ei_traits<Derived>::Scalar>,Derived>
MatrixBase<Derived>::operator-() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise absolute value of \c *this
  */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_abs_op<typename ei_traits<Derived>::Scalar>,Derived>
MatrixBase<Derived>::cwiseAbs() const
{
  return derived();
}

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_abs2_op<typename ei_traits<Derived>::Scalar>,Derived>
MatrixBase<Derived>::cwiseAbs2() const
{
  return derived();
}

/** \returns an expression of the complex conjugate of *this.
  *
  * \sa adjoint() */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_conjugate_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::conjugate() const
{
  return derived();
}

/** \returns an expression of *this with the \a Scalar type casted to
  * \a NewScalar.
  *
  * The template parameter \a NewScalar is the type we are casting the scalars to.
  *
  * \sa class CwiseUnaryOp
  */
template<typename Derived>
template<typename NewType>
inline const CwiseUnaryOp<ei_scalar_cast_op<typename ei_traits<Derived>::Scalar, NewType>, Derived>
MatrixBase<Derived>::cast() const
{
  return derived();
}

/** \relates MatrixBase */
template<typename Derived>
inline const typename MatrixBase<Derived>::ScalarMultipleReturnType
MatrixBase<Derived>::operator*(const Scalar& scalar) const
{
  return CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, Derived>
    (derived(), ei_scalar_multiple_op<Scalar>(scalar));
}

/** \relates MatrixBase */
template<typename Derived>
inline const CwiseUnaryOp<ei_scalar_quotient1_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::operator/(const Scalar& scalar) const
{
  return CwiseUnaryOp<ei_scalar_quotient1_op<Scalar>, Derived>
    (derived(), ei_scalar_quotient1_op<Scalar>(scalar));
}

template<typename Derived>
inline Derived&
MatrixBase<Derived>::operator*=(const Scalar& other)
{
  return *this = *this * other;
}

template<typename Derived>
inline Derived&
MatrixBase<Derived>::operator/=(const Scalar& other)
{
  return *this = *this / other;
}

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

#endif // EIGEN_CWISE_UNARY_OP_H
