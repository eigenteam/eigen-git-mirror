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

#ifndef EIGEN_ARRAY_H
#define EIGEN_ARRAY_H

/** \class Array
  *
  * \brief Pseudo expression offering additional features to an expression
  *
  * \param ExpressionType the type of the object of which we want array related features
  *
  * This class represents an expression with additional, array related, features.
  * It is the return type of MatrixBase::array()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::array()
  */
template<typename ExpressionType> class Array
{
  public:

    typedef typename ei_traits<ExpressionType>::Scalar Scalar;
    typedef typename ei_meta_if<ei_must_nest_by_value<ExpressionType>::ret,
        ExpressionType, const ExpressionType&>::ret ExpressionTypeNested;
//     typedef NestByValue<typename ExpressionType::ConstantReturnType> ConstantReturnType;
    typedef CwiseUnaryOp<ei_scalar_add_op<Scalar>, ExpressionType> ScalarAddReturnType;

    inline Array(const ExpressionType& matrix) : m_matrix(matrix) {}

    /** \internal */
    inline const ExpressionType& _expression() const { return m_matrix; }

    const ScalarAddReturnType
    operator+(const Scalar& scalar) const;

    /** \relates Array */
    friend const ScalarAddReturnType
    operator+(const Scalar& scalar, const Array& mat)
    { return mat + scalar; }

    ExpressionType& operator+=(const Scalar& scalar);

    const ScalarAddReturnType
    operator-(const Scalar& scalar) const;

    ExpressionType& operator-=(const Scalar& scalar);

    /** \returns true if each coeff of \c *this is less than its respective coeff of \a other */
    template<typename OtherDerived> bool operator<(const Array<OtherDerived>& other) const
    { return m_matrix.cwiseLessThan(other._expression()).all(); }

    /** \returns true if each coeff of \c *this is less or equal to its respective coeff of \a other */
    template<typename OtherDerived> bool operator<=(const Array<OtherDerived>& other) const
    { return m_matrix.cwiseLessEqual(other._expression()).all(); }

    /** \returns true if each coeff of \c *this is greater to its respective coeff of \a other */
    template<typename OtherDerived>
    bool operator>(const Array<OtherDerived>& other) const
    { return m_matrix.cwiseGreaterThan(other._expression()).all(); }

    /** \returns true if each coeff of \c *this is greater or equal to its respective coeff of \a other */
    template<typename OtherDerived>
    bool operator>=(const Array<OtherDerived>& other) const
    { return m_matrix.cwiseGreaterEqual(other._expression()).all(); }

  protected:
    ExpressionTypeNested m_matrix;
};

/** \returns an expression of \c *this with each coeff incremented by the constant \a scalar */
template<typename ExpressionType>
const typename Array<ExpressionType>::ScalarAddReturnType
Array<ExpressionType>::operator+(const Scalar& scalar) const
{
  return CwiseUnaryOp<ei_scalar_add_op<Scalar>, ExpressionType>(m_matrix, ei_scalar_add_op<Scalar>(scalar));
}

/** \see operator+ */
template<typename ExpressionType>
ExpressionType& Array<ExpressionType>::operator+=(const Scalar& scalar)
{
  m_matrix.const_cast_derived() = *this + scalar;
  return m_matrix.const_cast_derived();
}

/** \returns an expression of \c *this with each coeff decremented by the constant \a scalar */
template<typename ExpressionType>
const typename Array<ExpressionType>::ScalarAddReturnType
Array<ExpressionType>::operator-(const Scalar& scalar) const
{
  return *this + (-scalar);
}

/** \see operator- */
template<typename ExpressionType>
ExpressionType& Array<ExpressionType>::operator-=(const Scalar& scalar)
{
  m_matrix.const_cast_derived() = *this - scalar;
  return m_matrix.const_cast_derived();
}

/** \array_module
  *
  * \returns an Array expression of *this providing additional,
  * array related, features.
  *
  * \sa class Array
  */
template<typename Derived>
inline const Array<Derived>
MatrixBase<Derived>::array() const
{
  return derived();
}

/** \array_module
  *
  * \returns an Array expression of *this providing additional,
  * array related, features.
  *
  * \sa class Array
  */
template<typename Derived>
inline Array<Derived>
MatrixBase<Derived>::array()
{
  return derived();
}

#endif // EIGEN_FLAGGED_H
