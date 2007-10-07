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

#ifndef EI_SCALAROPS_H
#define EI_SCALAROPS_H

template<typename MatrixType> class EiScalarProduct
  : public EiObject<typename MatrixType::Scalar, EiScalarProduct<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::ConstRef MatRef;
    friend class EiObject<typename MatrixType::Scalar, EiScalarProduct<MatrixType> >;

    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    EiScalarProduct(const MatRef& matrix, Scalar scalar)
      : m_matrix(matrix), m_scalar(scalar) {}

    EiScalarProduct(const EiScalarProduct& other)
      : m_matrix(other.m_matrix), m_scalar(other.m_scalar) {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(EiScalarProduct)

  private:
    const EiScalarProduct& _ref() const { return *this; }
    const EiScalarProduct& _constRef() const { return *this; }
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

#define EI_MAKE_SCALAR_OPS(OtherScalar)                                \
template<typename Scalar, typename Derived>                            \
EiScalarProduct<Derived>                                               \
operator*(const EiObject<Scalar, Derived>& matrix,                     \
          OtherScalar scalar)                                          \
{                                                                      \
  return EiScalarProduct<Derived>(matrix.constRef(), scalar);          \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
EiScalarProduct<Derived>                                               \
operator*(OtherScalar scalar,                                          \
          const EiObject<Scalar, Derived>& matrix)                     \
{                                                                      \
  return EiScalarProduct<Derived>(matrix.constRef(), scalar);          \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
EiScalarProduct<Derived>                                               \
operator/(const EiObject<Scalar, Derived>& matrix,                     \
          OtherScalar scalar)                                          \
{                                                                      \
  return matrix * (static_cast<typename Derived::Scalar>(1) / scalar); \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
Derived &                                                              \
EiObject<Scalar, Derived>::operator*=(const OtherScalar &other)        \
{                                                                      \
  return *this = *this * other;                                        \
}                                                                      \
                                                                       \
template<typename Scalar, typename Derived>                            \
Derived &                                                              \
EiObject<Scalar, Derived>::operator/=(const OtherScalar &other)        \
{                                                                      \
  return *this = *this / other;                                        \
}

EI_MAKE_SCALAR_OPS(int)
EI_MAKE_SCALAR_OPS(float)
EI_MAKE_SCALAR_OPS(double)
EI_MAKE_SCALAR_OPS(std::complex<int>)
EI_MAKE_SCALAR_OPS(std::complex<float>)
EI_MAKE_SCALAR_OPS(std::complex<double>)

#undef EI_MAKE_SCALAR_OPS

#endif // EI_SCALAROPS_H
