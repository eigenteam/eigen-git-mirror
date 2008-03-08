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

#ifndef EIGEN_CWISE_BINARY_OP_H
#define EIGEN_CWISE_BINARY_OP_H

/** \class CwiseBinaryOp
  *
  * \brief Generic expression of a coefficient-wise operator between two matrices or vectors
  *
  * \param BinaryOp template functor implementing the operator
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  *
  * This class represents an expression of a generic binary operator of two matrices or vectors.
  * It is the return type of the operator+, operator-, cwiseProduct, cwiseQuotient between matrices or vectors, and most
  * of the time this is the only way it is used.
  *
  * However, if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_CwiseBinaryOp.cpp
  *
  * \sa class ScalarProductOp, class ScalarQuotientOp
  */
template<typename BinaryOp, typename Lhs, typename Rhs>
class CwiseBinaryOp : NoOperatorEquals,
  public MatrixBase<
            typename ei_result_of<BinaryOp(typename Lhs::Scalar,typename Rhs::Scalar)>::type,
            CwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
  public:
    typedef typename ei_result_of<BinaryOp(typename Lhs::Scalar,typename Rhs::Scalar)>::type Scalar;
    typedef typename Lhs::AsArg LhsRef;
    typedef typename Rhs::AsArg RhsRef;
    friend class MatrixBase<Scalar, CwiseBinaryOp>;
    friend class MatrixBase<Scalar, CwiseBinaryOp>::Traits;
    typedef MatrixBase<Scalar, CwiseBinaryOp> Base;

    CwiseBinaryOp(const LhsRef& lhs, const RhsRef& rhs, const BinaryOp& func = BinaryOp())
      : m_lhs(lhs), m_rhs(rhs), m_functor(func)
    {
      assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }

  private:
    enum {
      RowsAtCompileTime = Lhs::Traits::RowsAtCompileTime,
      ColsAtCompileTime = Lhs::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = Lhs::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Lhs::Traits::MaxColsAtCompileTime
    };

    const CwiseBinaryOp& _asArg() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_lhs.cols(); }

    Scalar _coeff(int row, int col) const
    {
      return m_functor(m_lhs.coeff(row, col), m_rhs.coeff(row, col));
    }

  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
    const BinaryOp m_functor;
};

/** \internal
  * \brief Template functor to compute the sum of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+
  */
struct ScalarSumOp EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return a + b; }
};

/** \internal
  * \brief Template functor to compute the difference of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator-
  */
struct ScalarDifferenceOp EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return a - b; }
};

/** \internal
  * \brief Template functor to compute the product of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseProduct()
  */
struct ScalarProductOp EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return a * b; }
};

/** \internal
  * \brief Template functor to compute the quotient of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseQuotient()
  */
struct ScalarQuotientOp EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return a / b; }
};

/** \relates MatrixBase
  *
  * \returns an expression of the difference of \a mat1 and \a mat2
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator-=()
  */
template<typename Scalar, typename Derived1, typename Derived2>
const CwiseBinaryOp<ScalarDifferenceOp, Derived1, Derived2>
operator-(const MatrixBase<Scalar, Derived1> &mat1, const MatrixBase<Scalar, Derived2> &mat2)
{
  return CwiseBinaryOp<ScalarDifferenceOp, Derived1, Derived2>(mat1.asArg(), mat2.asArg());
}

/** replaces \c *this by \c *this - \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
MatrixBase<Scalar, Derived>::operator-=(const MatrixBase<Scalar, OtherDerived> &other)
{
  return *this = *this - other;
}


/** \relates MatrixBase
  *
  * \returns an expression of the sum of \a mat1 and \a mat2
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+=()
  */
template<typename Scalar, typename Derived1, typename Derived2>
const CwiseBinaryOp<ScalarSumOp, Derived1, Derived2>
operator+(const MatrixBase<Scalar, Derived1> &mat1, const MatrixBase<Scalar, Derived2> &mat2)
{
  return CwiseBinaryOp<ScalarSumOp, Derived1, Derived2>(mat1.asArg(), mat2.asArg());
}

/** replaces \c *this by \c *this + \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
MatrixBase<Scalar, Derived>::operator+=(const MatrixBase<Scalar, OtherDerived>& other)
{
  return *this = *this + other;
}


/** \returns an expression of the Schur product (coefficient wise product) of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Scalar, typename Derived>
template<typename OtherDerived>
const CwiseBinaryOp<ScalarProductOp, Derived, OtherDerived>
MatrixBase<Scalar, Derived>::cwiseProduct(const MatrixBase<Scalar, OtherDerived> &other) const
{
  return CwiseBinaryOp<ScalarProductOp, Derived, OtherDerived>(asArg(), other.asArg());
}


/** \returns an expression of the coefficient-wise quotient of *this and \a other
  *
  * \sa class CwiseBinaryOp
  */
template<typename Scalar, typename Derived>
template<typename OtherDerived>
const CwiseBinaryOp<ScalarQuotientOp, Derived, OtherDerived>
MatrixBase<Scalar, Derived>::cwiseQuotient(const MatrixBase<Scalar, OtherDerived> &other) const
{
  return CwiseBinaryOp<ScalarQuotientOp, Derived, OtherDerived>(asArg(), other.asArg());
}


/** \returns an expression of a custom coefficient-wise operator \a func of *this and \a other
  *
  * The template parameter \a CustomBinaryOp is the type of the functor
  * of the custom operator (see class CwiseBinaryOp for an example)
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, MatrixBase::operator-, MatrixBase::cwiseProduct, MatrixBase::cwiseQuotient
  */
template<typename Scalar, typename Derived>
template<typename CustomBinaryOp, typename OtherDerived>
const CwiseBinaryOp<CustomBinaryOp, Derived, OtherDerived>
MatrixBase<Scalar, Derived>::cwise(const MatrixBase<Scalar, OtherDerived> &other, const CustomBinaryOp& func) const
{
  return CwiseBinaryOp<CustomBinaryOp, Derived, OtherDerived>(asArg(), other.asArg(), func);
}


#endif // EIGEN_CWISE_BINARY_OP_H
