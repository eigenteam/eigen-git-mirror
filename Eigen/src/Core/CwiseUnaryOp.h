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
  * \sa class CwiseBinaryOp
  */
template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOp : NoOperatorEquals,
  public MatrixBase<
    typename ei_result_of<UnaryOp(typename MatrixType::Scalar)>::type,
    CwiseUnaryOp<UnaryOp, MatrixType> >
{
  public:
    typedef typename ei_result_of<UnaryOp(typename MatrixType::Scalar)>::type Scalar;
    typedef typename MatrixType::AsArg MatRef;
    friend class MatrixBase<Scalar, CwiseUnaryOp>;
    friend class MatrixBase<Scalar, CwiseUnaryOp>::Traits;
    typedef MatrixBase<Scalar, CwiseUnaryOp> Base;

    CwiseUnaryOp(const MatRef& mat, const UnaryOp& func = UnaryOp()) : m_matrix(mat), m_functor(func) {}

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    const CwiseUnaryOp& _asArg() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    Scalar _coeff(int row, int col) const
    {
      return m_functor(m_matrix.coeff(row, col));
    }

  protected:
    const MatRef m_matrix;
    const UnaryOp m_functor;
};

/** \brief Template functor to compute the opposite of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator-
  */
struct ScalarOppositeOp EIGEN_EMPTY_STRUCT {
  template<typename Scalar> Scalar operator() (const Scalar& a) const { return -a; }
};

/** \brief Template functor to compute the absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseAbs
  */
struct ScalarAbsOp EIGEN_EMPTY_STRUCT {
  template<typename Scalar> Scalar operator() (const Scalar& a) const { return ei_abs(a); }
};


/** \returns an expression of the opposite of \c *this
  */
template<typename Scalar, typename Derived>
const CwiseUnaryOp<ScalarOppositeOp,Derived>
MatrixBase<Scalar, Derived>::operator-() const
{
  return CwiseUnaryOp<ScalarOppositeOp,Derived>(asArg());
}

/** \returns an expression of the opposite of \c *this
  */
template<typename Scalar, typename Derived>
const CwiseUnaryOp<ScalarAbsOp,Derived>
MatrixBase<Scalar, Derived>::cwiseAbs() const
{
  return CwiseUnaryOp<ScalarAbsOp,Derived>(asArg());
}


/** \returns an expression of a custom coefficient-wise unary operator \a func of *this
  *
  * The template parameter \a CustomUnaryOp is the type of the functor
  * of the custom unary operator.
  * 
  * Here is an example:
  * \include class_CwiseUnaryOp.cpp
  *
  * \sa class CwiseUnaryOp, class CwiseBinarOp, MatrixBase::operator-, MatrixBase::cwiseAbs
  */
template<typename Scalar, typename Derived>
template<typename CustomUnaryOp>
const CwiseUnaryOp<CustomUnaryOp, Derived>
MatrixBase<Scalar, Derived>::cwise(const CustomUnaryOp& func) const
{
  return CwiseUnaryOp<CustomUnaryOp, Derived>(asArg(), func);
}


/** \brief Template functor to compute the conjugate of a complex value
  *
  * \sa class CwiseUnaryOp, MatrixBase::conjugate()
  */
struct ScalarConjugateOp EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a) const { return ei_conj(a); }
};

/** \returns an expression of the complex conjugate of *this.
  *
  * \sa adjoint() */
template<typename Scalar, typename Derived>
const CwiseUnaryOp<ScalarConjugateOp, Derived>
MatrixBase<Scalar, Derived>::conjugate() const
{
  return CwiseUnaryOp<ScalarConjugateOp, Derived>(asArg());
}

/** \brief Template functor to cast a scalar to another
  *
  * \sa class CwiseUnaryOp, MatrixBase::cast()
  */
template<typename NewType>
struct ScalarCastOp EIGEN_EMPTY_STRUCT {
    typedef NewType result_type;
    template<typename Scalar> NewType operator() (const Scalar& a) const { return static_cast<NewType>(a); }
};

/** \returns an expression of *this with the \a Scalar type casted to
  * \a NewScalar.
  *
  * The template parameter \a NewScalar is the type we are casting the scalars to.
  *
  * Example: \include MatrixBase_cast.cpp
  * Output: \verbinclude MatrixBase_cast.out
  *
  * \sa class CwiseUnaryOp, class ScalarCastOp
  */
template<typename Scalar, typename Derived>
template<typename NewType>
const CwiseUnaryOp<ScalarCastOp<NewType>, Derived>
MatrixBase<Scalar, Derived>::cast() const
{
  return CwiseUnaryOp<ScalarCastOp<NewType>, Derived>(asArg());
}


/** \brief Template functor to multiply a scalar by a fixed another one
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator*, MatrixBase::operator/
  */
template<typename Scalar>
struct ScalarMultipleOp {
    ScalarMultipleOp(const Scalar& other) : m_other(other) {}
    Scalar operator() (const Scalar& a) const { return a * m_other; }
    const Scalar m_other;
};

/** \relates MatrixBase \sa class ScalarMultipleOp */
template<typename Scalar, typename Derived>
const CwiseUnaryOp<ScalarMultipleOp<Scalar>, Derived>
MatrixBase<Scalar, Derived>::operator*(const Scalar& scalar) const
{
  return CwiseUnaryOp<ScalarMultipleOp<Scalar>, Derived>(asArg(), ScalarMultipleOp<Scalar>(scalar));
}

/** \relates MatrixBase \sa class ScalarMultipleOp */
template<typename Scalar, typename Derived>
const CwiseUnaryOp<ScalarMultipleOp<Scalar>, Derived>
MatrixBase<Scalar, Derived>::operator/(const Scalar& scalar) const
{
  assert(NumTraits<Scalar>::HasFloatingPoint);
  return CwiseUnaryOp<ScalarMultipleOp<Scalar>, Derived>(asArg(), ScalarMultipleOp<Scalar>(static_cast<Scalar>(1) / scalar));
}

/** \sa ScalarMultipleOp */
template<typename Scalar, typename Derived>
Derived&
MatrixBase<Scalar, Derived>::operator*=(const Scalar& other)
{
  return *this = *this * other;
}

/** \sa ScalarMultipleOp */
template<typename Scalar, typename Derived>
Derived&
MatrixBase<Scalar, Derived>::operator/=(const Scalar& other)
{
  return *this = *this / other;
}

#endif // EIGEN_CWISE_UNARY_OP_H
