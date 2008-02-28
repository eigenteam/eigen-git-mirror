// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
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

#ifndef EIGEN_SCALARMULTIPLE_H
#define EIGEN_SCALARMULTIPLE_H

/** \class ScalarMultiple
  *
  * \brief Expression of the product of a matrix or vector by a scalar
  *
  * \param FactorTye the type of scalar by which to multiply
  * \param MatrixType the type of the matrix or vector to multiply
  *
  * This class represents an expression of the product of a matrix or vector by a scalar.
  * It is the return type of the operator* between a matrix or vector and a scalar, and most
  * of the time this is the only way it is used.
  */
template<typename FactorType, typename MatrixType> class ScalarMultiple : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, ScalarMultiple<FactorType, MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, ScalarMultiple>;
    typedef MatrixBase<Scalar, ScalarMultiple> Base;

    ScalarMultiple(const MatRef& matrix, FactorType factor)
      : m_matrix(matrix), m_factor(factor) {}

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    const ScalarMultiple& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    Scalar _coeff(int row, int col) const
    {
      return m_factor * m_matrix.coeff(row, col);
    }

  protected:
    const MatRef m_matrix;
    const FactorType m_factor;
};

#define EIGEN_MAKE_SCALAR_OPS(FactorType)                              \
/** \relates MatrixBase \sa class ScalarMultiple */                    \
template<typename Scalar, typename Derived>                            \
const ScalarMultiple<FactorType, Derived>                              \
operator*(const MatrixBase<Scalar, Derived>& matrix,                   \
          FactorType scalar)                                           \
{                                                                      \
  return ScalarMultiple<FactorType, Derived>(matrix.ref(), scalar);    \
}                                                                      \
                                                                       \
/** \relates MatrixBase \sa class ScalarMultiple */                    \
template<typename Scalar, typename Derived>                            \
const ScalarMultiple<FactorType, Derived>                              \
operator*(FactorType scalar,                                           \
          const MatrixBase<Scalar, Derived>& matrix)                   \
{                                                                      \
  return ScalarMultiple<FactorType, Derived>(matrix.ref(), scalar);    \
}                                                                      \
                                                                       \
/** \relates MatrixBase \sa class ScalarMultiple */                    \
template<typename Scalar, typename Derived>                            \
const ScalarMultiple<typename NumTraits<FactorType>::FloatingPoint, Derived> \
operator/(const MatrixBase<Scalar, Derived>& matrix,                   \
          FactorType scalar)                                           \
{                                                                      \
  assert(NumTraits<Scalar>::HasFloatingPoint);                         \
  return matrix * (static_cast<                                        \
                     typename NumTraits<FactorType>::FloatingPoint     \
                   >(1) / scalar);                                     \
}                                                                      \
                                                                       \
/** \sa class ScalarMultiple */                                        \
template<typename Scalar, typename Derived>                            \
Derived &                                                              \
MatrixBase<Scalar, Derived>::operator*=(const FactorType &other)       \
{                                                                      \
  return *this = *this * other;                                        \
}                                                                      \
                                                                       \
/** \sa class ScalarMultiple */                                        \
template<typename Scalar, typename Derived>                            \
Derived &                                                              \
MatrixBase<Scalar, Derived>::operator/=(const FactorType &other)       \
{                                                                      \
  return *this = *this / other;                                        \
}

EIGEN_MAKE_SCALAR_OPS(int)
EIGEN_MAKE_SCALAR_OPS(float)
EIGEN_MAKE_SCALAR_OPS(double)
EIGEN_MAKE_SCALAR_OPS(std::complex<float>)
EIGEN_MAKE_SCALAR_OPS(std::complex<double>)

#undef EIGEN_MAKE_SCALAR_OPS

#endif // EIGEN_SCALARMULTIPLE_H
