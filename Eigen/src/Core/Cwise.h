// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_CWISE_H
#define EIGEN_CWISE_H

/** \internal
  * \array_module
  *
  * \brief Template functor to add a scalar to a fixed other one
  *
  * \sa class CwiseUnaryOp, Array::operator+
  */
template<typename Scalar, bool PacketAccess = (int(ei_packet_traits<Scalar>::size)>1?true:false) > struct ei_scalar_add_op;

/** \class Cwise
  *
  * \brief Pseudo expression providing additional coefficient-wise operations
  *
  * \param ExpressionType the type of the object on which to do coefficient-wise operations
  *
  * This class represents an expression with additional coefficient-wise features.
  * It is the return type of MatrixBase::cwise()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::cwise()
  */
template<typename ExpressionType> class Cwise
{
  public:

    typedef typename ei_traits<ExpressionType>::Scalar Scalar;
    typedef typename ei_meta_if<ei_must_nest_by_value<ExpressionType>::ret,
        ExpressionType, const ExpressionType&>::ret ExpressionTypeNested;
//     typedef NestByValue<typename ExpressionType::ConstantReturnType> ConstantReturnType;
    typedef CwiseUnaryOp<ei_scalar_add_op<Scalar>, ExpressionType> ScalarAddReturnType;

    template<template<typename _Scalar> class Functor, typename OtherDerived> struct BinOp
    {
      typedef CwiseBinaryOp<Functor<typename ei_traits<ExpressionType>::Scalar>,
                            ExpressionType,
                            OtherDerived
                           > ReturnType;
    };

    template<template<typename _Scalar> class Functor> struct UnOp
    {
      typedef CwiseUnaryOp<Functor<typename ei_traits<ExpressionType>::Scalar>,
                           ExpressionType
                          > ReturnType;
    };

    inline Cwise(const ExpressionType& matrix) : m_matrix(matrix) {}

    /** \internal */
    inline const ExpressionType& _expression() const { return m_matrix; }

    template<typename OtherDerived>
    const typename BinOp<ei_scalar_product_op, OtherDerived>::ReturnType
    operator*(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const typename BinOp<ei_scalar_quotient_op, OtherDerived>::ReturnType
    operator/(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const typename BinOp<ei_scalar_min_op, OtherDerived>::ReturnType
    min(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const typename BinOp<ei_scalar_max_op, OtherDerived>::ReturnType
    max(const MatrixBase<OtherDerived> &other) const;

    const typename UnOp<ei_scalar_abs_op>::ReturnType abs() const;
    const typename UnOp<ei_scalar_abs2_op>::ReturnType abs2() const;
    const typename UnOp<ei_scalar_square_op>::ReturnType square() const;
    const typename UnOp<ei_scalar_cube_op>::ReturnType cube() const;
    const typename UnOp<ei_scalar_inverse_op>::ReturnType inverse() const;
    const typename UnOp<ei_scalar_sqrt_op>::ReturnType sqrt() const;
    const typename UnOp<ei_scalar_exp_op>::ReturnType exp() const;
    const typename UnOp<ei_scalar_log_op>::ReturnType log() const;
    const typename UnOp<ei_scalar_cos_op>::ReturnType cos() const;
    const typename UnOp<ei_scalar_sin_op>::ReturnType sin() const;
    const typename UnOp<ei_scalar_pow_op>::ReturnType pow(const Scalar& exponent) const;


    const ScalarAddReturnType
    operator+(const Scalar& scalar) const;

    /** \relates Cwise */
    friend const ScalarAddReturnType
    operator+(const Scalar& scalar, const Cwise& mat)
    { return mat + scalar; }

    ExpressionType& operator+=(const Scalar& scalar);

    const ScalarAddReturnType
    operator-(const Scalar& scalar) const;

    ExpressionType& operator-=(const Scalar& scalar);

    template<typename OtherDerived> const typename BinOp<std::less, OtherDerived>::ReturnType
    operator<(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const typename BinOp<std::less_equal, OtherDerived>::ReturnType
    operator<=(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const typename BinOp<std::greater, OtherDerived>::ReturnType
    operator>(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const typename BinOp<std::greater_equal, OtherDerived>::ReturnType
    operator>=(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const typename BinOp<std::equal_to, OtherDerived>::ReturnType
    operator==(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> const typename BinOp<std::not_equal_to, OtherDerived>::ReturnType
    operator!=(const MatrixBase<OtherDerived>& other) const;


  protected:
    ExpressionTypeNested m_matrix;
};

/** \array_module
  *
  * \returns a Cwise expression of *this providing additional coefficient-wise operations
  *
  * \sa class Cwise
  */
template<typename Derived>
inline const Cwise<Derived>
MatrixBase<Derived>::cwise() const
{
  return derived();
}

/** \array_module
  *
  * \returns a Cwise expression of *this providing additional coefficient-wise operations
  *
  * \sa class Cwise
  */
template<typename Derived>
inline Cwise<Derived>
MatrixBase<Derived>::cwise()
{
  return derived();
}

#endif // EIGEN_CWISE_H
