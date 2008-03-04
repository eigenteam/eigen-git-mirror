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
  public MatrixBase<typename MatrixType::Scalar, CwiseUnaryOp<UnaryOp, MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::AsArg MatRef;
    friend class MatrixBase<Scalar, CwiseUnaryOp>;
    friend class MatrixBase<Scalar, CwiseUnaryOp>::Traits;
    typedef MatrixBase<Scalar, CwiseUnaryOp> Base;

    CwiseUnaryOp(const MatRef& mat) : m_matrix(mat) {}

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
      return UnaryOp::template op<Scalar>(m_matrix.coeff(row, col));
    }

  protected:
    const MatRef m_matrix;
};

/** \brief Template functor to compute the opposite of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::operator-
  */
struct ScalarOppositeOp {
  template<typename Scalar> static Scalar op(const Scalar& a) { return -a; }
};

/** \brief Template functor to compute the absolute value of a scalar
  *
  * \sa class CwiseUnaryOp, MatrixBase::cwiseAbs
  */
struct ScalarAbsOp {
  template<typename Scalar> static Scalar op(const Scalar& a) { return ei_abs(a); }
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


/** \relates MatrixBase
  *
  * \returns an expression of a custom coefficient-wise unary operator of \a mat
  *
  * The template parameter \a CustomUnaryOp is the template functor
  * of the custom unary operator.
  *
  * \sa class CwiseUnaryOp, class CwiseBinarOp, MatrixBase::cwise, MatrixBase::operator-, MatrixBase::cwiseAbs
  */
template<typename CustomUnaryOp, typename Scalar, typename Derived>
const CwiseUnaryOp<CustomUnaryOp, Derived>
cwise(const MatrixBase<Scalar, Derived> &mat)
{
  return CwiseUnaryOp<CustomUnaryOp, Derived>(mat.asArg());
}

/** \returns an expression of a custom coefficient-wise unary operator of *this
  *
  * The template parameter \a CustomUnaryOp is the template functor
  * of the custom unary operator.
  *
  * \note since cwise is a templated member with a mandatory template parameter,
  * the keyword template as to be used if the matrix type is also a template parameter:
  * \code
  * template <typename MatrixType> void foo(const MatrixType& m) {
  *   m.template cwise<ScalarAbsOp>();
  * }
  * \endcode
  *
  * \sa class CwiseUnaryOp, class CwiseBinarOp, MatrixBase::operator-, MatrixBase::cwiseAbs
  */
template<typename Scalar, typename Derived>
template<typename CustomUnaryOp>
const CwiseUnaryOp<CustomUnaryOp, Derived>
MatrixBase<Scalar, Derived>::cwise() const
{
  return CwiseUnaryOp<CustomUnaryOp, Derived>(asArg());
}


/** \brief Template functor to compute the conjugate of a complex value
  *
  * \sa class CwiseUnaryOp, MatrixBase::conjugate()
  */
struct ScalarConjugateOp {
    template<typename Scalar> static Scalar op(const Scalar& a) { return ei_conj(a); }
};

/** \returns an expression of the complex conjugate of *this.
  *
  * \sa adjoint(), class Conjugate */
template<typename Scalar, typename Derived>
const CwiseUnaryOp<ScalarConjugateOp, Derived>
MatrixBase<Scalar, Derived>::conjugate() const
{
  return CwiseUnaryOp<ScalarConjugateOp, Derived>(asArg());
}



#endif // EIGEN_CWISE_UNARY_OP_H
