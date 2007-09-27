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

#ifndef EIGEN_SCALAROPS_H
#define EIGEN_SCALAROPS_H

namespace Eigen {

template<typename MatrixType> class ScalarProduct
  : public EigenBase<typename MatrixType::Scalar, ScalarProduct<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    typedef ScalarProduct Ref;
    friend class EigenBase<typename MatrixType::Scalar, ScalarProduct<MatrixType> >;

    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    ScalarProduct(const MatRef& matrix, Scalar scalar)
      : m_matrix(matrix), m_scalar(scalar) {}

    ScalarProduct(const ScalarProduct& other)
      : m_matrix(other.m_matrix), m_scalar(other.m_scalar) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ScalarProduct)

  private:
    const Ref& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    Scalar _read(int row, int col) const
    {
      return m_matrix.read(row, col) * m_scalar;
    }

  protected:
    const MatRef m_matrix;
    const Scalar m_scalar;
};

#define EIGEN_MAKE_SCALAR_OPS(OtherScalar)                             \
template<typename Scalar, typename Derived>                            \
ScalarProduct<Derived>                                                 \
operator*(const EigenBase<Scalar, Derived>& matrix,                    \
          OtherScalar scalar)                                          \
{                                                                      \
  return ScalarProduct<Derived>(matrix.ref(), scalar);                 \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
ScalarProduct<Derived>                                                 \
operator*(OtherScalar scalar,                                          \
          const EigenBase<Scalar, Derived>& matrix)                    \
{                                                                      \
  return ScalarProduct<Derived>(matrix.ref(), scalar);                 \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
ScalarProduct<Derived>                                                 \
operator/(const EigenBase<Scalar, Derived>& matrix,                    \
          OtherScalar scalar)                                          \
{                                                                      \
  return matrix * (static_cast<typename Derived::Scalar>(1) / scalar); \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
Derived &                                                              \
EigenBase<Scalar, Derived>::operator*=(const OtherScalar &other)       \
{                                                                      \
  *this = *this * other;                                               \
  return *static_cast<Derived*>(this);                                 \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
Derived &                                                              \
EigenBase<Scalar, Derived>::operator/=(const OtherScalar &other)       \
{                                                                      \
  *this = *this / other;                                               \
  return *static_cast<Derived*>(this);                                 \
}

EIGEN_MAKE_SCALAR_OPS(int)
EIGEN_MAKE_SCALAR_OPS(float)
EIGEN_MAKE_SCALAR_OPS(double)
EIGEN_MAKE_SCALAR_OPS(std::complex<int>)
EIGEN_MAKE_SCALAR_OPS(std::complex<float>)
EIGEN_MAKE_SCALAR_OPS(std::complex<double>)

#undef EIGEN_MAKE_SCALAR_OPS

} // namespace Eigen

#endif // EIGEN_SCALAROPS_H
