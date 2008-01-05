// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_SCALARMULTIPLE_H
#define EIGEN_SCALARMULTIPLE_H

template<typename FactorType, typename MatrixType> class ScalarMultiple : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, ScalarMultiple<FactorType, MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, ScalarMultiple<FactorType, MatrixType> >;

    ScalarMultiple(const MatRef& matrix, FactorType factor)
      : m_matrix(matrix), m_factor(factor) {}

  private:
    static const TraversalOrder _Order = MatrixType::Order;
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;

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
/** \relates MatrixBase */                                             \
template<typename Scalar, typename Derived>                            \
const ScalarMultiple<FactorType, Derived>                              \
operator*(const MatrixBase<Scalar, Derived>& matrix,                   \
          FactorType scalar)                                           \
{                                                                      \
  return ScalarMultiple<FactorType, Derived>(matrix.ref(), scalar);    \
}                                                                      \
                                                                       \
/** \relates MatrixBase */                                             \
template<typename Scalar, typename Derived>                            \
const ScalarMultiple<FactorType, Derived>                              \
operator*(FactorType scalar,                                           \
          const MatrixBase<Scalar, Derived>& matrix)                   \
{                                                                      \
  return ScalarMultiple<FactorType, Derived>(matrix.ref(), scalar);    \
}                                                                      \
                                                                       \
/** \relates MatrixBase */                                             \
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
template<typename Scalar, typename Derived>                            \
Derived &                                                              \
MatrixBase<Scalar, Derived>::operator*=(const FactorType &other)       \
{                                                                      \
  return *this = *this * other;                                        \
}                                                                      \
                                                                       \
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
