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

#ifndef EIGEN_CWISE_NULLARY_OP_H
#define EIGEN_CWISE_NULLARY_OP_H

/** \class CwiseNullaryOp
  *
  * \brief Generic expression of a matrix where all coefficients are defined by a functor
  *
  * \param NullaryOp template functor implementing the operator
  *
  * This class represents an expression of a generic nullary operator.
  * It is the return type of the ones(), zero(), constant(), identity() and random() functions,
  * and most of the time this is the only way it is used.
  *
  * However, if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * \sa class CwiseUnaryOp, class CwiseBinaryOp, MatrixBase::create()
  */
template<typename NullaryOp, typename MatrixType>
struct ei_traits<CwiseNullaryOp<NullaryOp, MatrixType> >
{
  typedef typename MatrixType::Scalar Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = (MatrixType::Flags
      & (HereditaryBits | Like1DArrayBit | (ei_functor_traits<NullaryOp>::IsVectorizable ? VectorizableBit : 0)))
      | (ei_functor_traits<NullaryOp>::IsRepeatable ? 0 : EvalBeforeNestingBit),
    CoeffReadCost = ei_functor_traits<NullaryOp>::Cost
  };
};

template<typename NullaryOp, typename MatrixType>
class CwiseNullaryOp : ei_no_assignment_operator,
  public MatrixBase<CwiseNullaryOp<NullaryOp, MatrixType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseNullaryOp)

    CwiseNullaryOp(int rows, int cols, const NullaryOp& func = NullaryOp())
      : m_rows(rows), m_cols(cols), m_functor(func)
    {
      ei_assert(rows > 0
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }

  private:

    int _rows() const { return m_rows.value(); }
    int _cols() const { return m_cols.value(); }

    const Scalar _coeff(int rows, int cols) const
    {
      return m_functor(rows, cols);
    }

    template<int LoadMode>
    PacketScalar _packetCoeff(int, int) const
    {
      return m_functor.packetOp();
    }

  protected:
    const ei_int_if_dynamic<RowsAtCompileTime> m_rows;
    const ei_int_if_dynamic<ColsAtCompileTime> m_cols;
    const NullaryOp m_functor;
};


/** \returns an expression of a matrix defined by a custom functor \a func
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so zero() should be used
  * instead.
  *
  * The template parameter \a CustomNullaryOp is the type of the functor.
  *
  * \sa class CwiseNullaryOp
  */
template<typename Derived>
template<typename CustomNullaryOp>
const CwiseNullaryOp<CustomNullaryOp, Derived>
MatrixBase<Derived>::create(int rows, int cols, const CustomNullaryOp& func)
{
  return CwiseNullaryOp<CustomNullaryOp, Derived>(rows, cols, func);
}

/** \returns an expression of a matrix defined by a custom functor \a func
  *
  * The parameter \a size is the size of the returned vector.
  * Must be compatible with this MatrixBase type.
  *
  * \only_for_vectors
  *
  * This variant is meant to be used for dynamic-size vector types. For fixed-size types,
  * it is redundant to pass \a size as argument, so zero() should be used
  * instead.
  *
  * The template parameter \a CustomNullaryOp is the type of the functor.
  *
  * \sa class CwiseNullaryOp
  */
template<typename Derived>
template<typename CustomNullaryOp>
const CwiseNullaryOp<CustomNullaryOp, Derived>
MatrixBase<Derived>::create(int size, const CustomNullaryOp& func)
{
  ei_assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1) return CwiseNullaryOp<CustomNullaryOp, Derived>(1, size, func);
  else return CwiseNullaryOp<CustomNullaryOp, Derived>(size, 1, func);
}

/** \returns an expression of a matrix defined by a custom functor \a func
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variants taking size arguments.
  *
  * The template parameter \a CustomNullaryOp is the type of the functor.
  *
  * \sa class CwiseNullaryOp
  */
template<typename Derived>
template<typename CustomNullaryOp>
const CwiseNullaryOp<CustomNullaryOp, Derived>
MatrixBase<Derived>::create(const CustomNullaryOp& func)
{
  return CwiseNullaryOp<CustomNullaryOp, Derived>(RowsAtCompileTime, ColsAtCompileTime, func);
}

/** \returns an expression of a constant matrix of value \a value
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so zero() should be used
  * instead.
  *
  * The template parameter \a CustomNullaryOp is the type of the functor.
  *
  * \sa class CwiseNullaryOp
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::constant(int rows, int cols, const Scalar& value)
{
  return create(rows, cols, ei_scalar_constant_op<Scalar>(value));
}

/** \returns an expression of a constant matrix of value \a value
  *
  * The parameter \a size is the size of the returned vector.
  * Must be compatible with this MatrixBase type.
  *
  * \only_for_vectors
  *
  * This variant is meant to be used for dynamic-size vector types. For fixed-size types,
  * it is redundant to pass \a size as argument, so zero() should be used
  * instead.
  *
  * The template parameter \a CustomNullaryOp is the type of the functor.
  *
  * \sa class CwiseNullaryOp
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::constant(int size, const Scalar& value)
{
  return create(size, ei_scalar_constant_op<Scalar>(value));
}

/** \returns an expression of a constant matrix of value \a value
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variants taking size arguments.
  *
  * The template parameter \a CustomNullaryOp is the type of the functor.
  *
  * \sa class CwiseNullaryOp
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::constant(const Scalar& value)
{
  return create(RowsAtCompileTime, ColsAtCompileTime, ei_scalar_constant_op<Scalar>(value));
}

template<typename Derived>
bool MatrixBase<Derived>::isApproxToConstant
(const Scalar& value, typename NumTraits<Scalar>::Real prec) const
{
  for(int j = 0; j < cols(); j++)
    for(int i = 0; i < rows(); i++)
      if(!ei_isApprox(coeff(i, j), value, prec))
        return false;
  return true;
}

/** Sets all coefficients in this expression to \a value.
  *
  * \sa class CwiseNullaryOp, zero(), ones()
  */
template<typename Derived>
Derived& MatrixBase<Derived>::setConstant(const Scalar& value)
{
  return *this = constant(rows(), cols(), value);
}

// zero:

/** \returns an expression of a zero matrix.
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so zero() should be used
  * instead.
  *
  * Example: \include MatrixBase_zero_int_int.cpp
  * Output: \verbinclude MatrixBase_zero_int_int.out
  *
  * \sa zero(), zero(int)
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::zero(int rows, int cols)
{
  return constant(rows, cols, Scalar(0));
}

/** \returns an expression of a zero vector.
  *
  * The parameter \a size is the size of the returned vector.
  * Must be compatible with this MatrixBase type.
  *
  * \only_for_vectors
  *
  * This variant is meant to be used for dynamic-size vector types. For fixed-size types,
  * it is redundant to pass \a size as argument, so zero() should be used
  * instead.
  *
  * Example: \include MatrixBase_zero_int.cpp
  * Output: \verbinclude MatrixBase_zero_int.out
  *
  * \sa zero(), zero(int,int)
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::zero(int size)
{
  return constant(size, Scalar(0));
}

/** \returns an expression of a fixed-size zero matrix or vector.
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variants taking size arguments.
  *
  * Example: \include MatrixBase_zero.cpp
  * Output: \verbinclude MatrixBase_zero.out
  *
  * \sa zero(int), zero(int,int)
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::zero()
{
  return constant(Scalar(0));
}

/** \returns true if *this is approximately equal to the zero matrix,
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isZero.cpp
  * Output: \verbinclude MatrixBase_isZero.out
  *
  * \sa class CwiseNullaryOp, zero()
  */
template<typename Derived>
bool MatrixBase<Derived>::isZero
(typename NumTraits<Scalar>::Real prec) const
{
  for(int j = 0; j < cols(); j++)
    for(int i = 0; i < rows(); i++)
      if(!ei_isMuchSmallerThan(coeff(i, j), static_cast<Scalar>(1), prec))
        return false;
  return true;
}

/** Sets all coefficients in this expression to zero.
  *
  * Example: \include MatrixBase_setZero.cpp
  * Output: \verbinclude MatrixBase_setZero.out
  *
  * \sa class CwiseNullaryOp, zero()
  */
template<typename Derived>
Derived& MatrixBase<Derived>::setZero()
{
  return setConstant(Scalar(0));
}

// ones:

/** \returns an expression of a matrix where all coefficients equal one.
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so ones() should be used
  * instead.
  *
  * Example: \include MatrixBase_ones_int_int.cpp
  * Output: \verbinclude MatrixBase_ones_int_int.out
  *
  * \sa ones(), ones(int), isOnes(), class Ones
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::ones(int rows, int cols)
{
  return constant(rows, cols, Scalar(1));
}

/** \returns an expression of a vector where all coefficients equal one.
  *
  * The parameter \a size is the size of the returned vector.
  * Must be compatible with this MatrixBase type.
  *
  * \only_for_vectors
  *
  * This variant is meant to be used for dynamic-size vector types. For fixed-size types,
  * it is redundant to pass \a size as argument, so ones() should be used
  * instead.
  *
  * Example: \include MatrixBase_ones_int.cpp
  * Output: \verbinclude MatrixBase_ones_int.out
  *
  * \sa ones(), ones(int,int), isOnes(), class Ones
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::ones(int size)
{
  return constant(size, Scalar(1));
}

/** \returns an expression of a fixed-size matrix or vector where all coefficients equal one.
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variants taking size arguments.
  *
  * Example: \include MatrixBase_ones.cpp
  * Output: \verbinclude MatrixBase_ones.out
  *
  * \sa ones(int), ones(int,int), isOnes(), class Ones
  */
template<typename Derived>
const typename MatrixBase<Derived>::ConstantReturnType
MatrixBase<Derived>::ones()
{
  return constant(Scalar(1));
}

/** \returns true if *this is approximately equal to the matrix where all coefficients
  *          are equal to 1, within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isOnes.cpp
  * Output: \verbinclude MatrixBase_isOnes.out
  *
  * \sa class CwiseNullaryOp, ones()
  */
template<typename Derived>
bool MatrixBase<Derived>::isOnes
(typename NumTraits<Scalar>::Real prec) const
{
  return isApproxToConstant(Scalar(1), prec);
}

/** Sets all coefficients in this expression to one.
  *
  * Example: \include MatrixBase_setOnes.cpp
  * Output: \verbinclude MatrixBase_setOnes.out
  *
  * \sa class CwiseNullaryOp, ones()
  */
template<typename Derived>
Derived& MatrixBase<Derived>::setOnes()
{
  return setConstant(Scalar(1));
}

// Identity:

/** \returns an expression of the identity matrix (not necessarily square).
  *
  * The parameters \a rows and \a cols are the number of rows and of columns of
  * the returned matrix. Must be compatible with this MatrixBase type.
  *
  * This variant is meant to be used for dynamic-size matrix types. For fixed-size types,
  * it is redundant to pass \a rows and \a cols as arguments, so identity() should be used
  * instead.
  *
  * Example: \include MatrixBase_identity_int_int.cpp
  * Output: \verbinclude MatrixBase_identity_int_int.out
  *
  * \sa identity(), setIdentity(), isIdentity()
  */
template<typename Derived>
inline const CwiseNullaryOp<ei_scalar_identity_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::identity(int rows, int cols)
{
  return create(rows, cols, ei_scalar_identity_op<Scalar>());
}

/** \returns an expression of the identity matrix (not necessarily square).
  *
  * This variant is only for fixed-size MatrixBase types. For dynamic-size types, you
  * need to use the variant taking size arguments.
  *
  * Example: \include MatrixBase_identity.cpp
  * Output: \verbinclude MatrixBase_identity.out
  *
  * \sa identity(int,int), setIdentity(), isIdentity()
  */
template<typename Derived>
inline const CwiseNullaryOp<ei_scalar_identity_op<typename ei_traits<Derived>::Scalar>, Derived>
MatrixBase<Derived>::identity()
{
  return create(RowsAtCompileTime, ColsAtCompileTime, ei_scalar_identity_op<Scalar>());
}

/** \returns true if *this is approximately equal to the identity matrix
  *          (not necessarily square),
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isIdentity.cpp
  * Output: \verbinclude MatrixBase_isIdentity.out
  *
  * \sa class CwiseNullaryOp, identity(), identity(int,int), setIdentity()
  */
template<typename Derived>
bool MatrixBase<Derived>::isIdentity
(typename NumTraits<Scalar>::Real prec) const
{
  for(int j = 0; j < cols(); j++)
  {
    for(int i = 0; i < rows(); i++)
    {
      if(i == j)
      {
        if(!ei_isApprox(coeff(i, j), static_cast<Scalar>(1), prec))
          return false;
      }
      else
      {
        if(!ei_isMuchSmallerThan(coeff(i, j), static_cast<RealScalar>(1), prec))
          return false;
      }
    }
  }
  return true;
}

/** Writes the identity expression (not necessarily square) into *this.
  *
  * Example: \include MatrixBase_setIdentity.cpp
  * Output: \verbinclude MatrixBase_setIdentity.out
  *
  * \sa class CwiseNullaryOp, identity(), identity(int,int), isIdentity()
  */
template<typename Derived>
inline Derived& MatrixBase<Derived>::setIdentity()
{
  return *this = identity(rows(), cols());
}

#endif // EIGEN_CWISE_NULLARY_OP_H
