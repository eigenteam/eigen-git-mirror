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
{
  public:
    typedef typename MatrixType::Scalar Scalar;

    ScalarProduct(const MatrixType& matrix, Scalar scalar)
      : m_matrix(matrix), m_scalar(scalar) {}

    ScalarProduct(const ScalarProduct& other)
      : m_matrix(other.m_matrix), m_scalar(other.m_scalar) {}

    int rows() const { return m_matrix.rows(); }
    int cols() const { return m_matrix.cols(); }

    Scalar read(int row, int col) const
    {
      return m_matrix.read(row, col) * m_scalar;
    }

  protected:
    const MatrixType m_matrix;
    const Scalar     m_scalar;
};

template<typename Content>
MatrixXpr<
  ScalarProduct<
    MatrixXpr<Content>
  >
>
operator *(const MatrixXpr<Content>& xpr,
           typename Content::Scalar scalar)
{
  typedef ScalarProduct<
            MatrixXpr<Content>
          > ProductType;
  typedef MatrixXpr<ProductType> XprType;
  return XprType(ProductType(xpr, scalar));
}

template<typename Content>
MatrixXpr<
  ScalarProduct<
    MatrixXpr<Content>
  >
>
operator *(typename Content::Scalar scalar,
           const MatrixXpr<Content>& xpr)
{
  typedef ScalarProduct<
            MatrixXpr<Content>
          > ProductType;
  typedef MatrixXpr<ProductType> XprType;
  return XprType(ProductType(xpr, scalar));
}

template<typename Derived>
MatrixXpr<
  ScalarProduct<
    MatrixRef<MatrixBase<Derived> >
  >
>
operator *(MatrixBase<Derived>& matrix,
           typename Derived::Scalar scalar)
{
  typedef ScalarProduct<
              MatrixRef<MatrixBase<Derived> >
            > ProductType;
  typedef MatrixXpr<ProductType> XprType;
  return XprType(ProductType(matrix.ref(), scalar));
}

template<typename Derived>
MatrixXpr<
  ScalarProduct<
    MatrixRef<MatrixBase<Derived> >
  >
>
operator *(typename Derived::Scalar scalar,
           MatrixBase<Derived>& matrix)
{
  typedef ScalarProduct<
            MatrixRef<MatrixBase<Derived> >
          > ProductType;
  typedef MatrixXpr<ProductType> XprType;
  return XprType(ProductType(matrix.ref(), scalar));
}

template<typename Content>
MatrixXpr<
  ScalarProduct<
    MatrixXpr<Content>
  >
>
operator /(MatrixXpr<Content>& xpr,
           typename Content::Scalar scalar)
{
  return xpr * (static_cast<typename Content::Scalar>(1) / scalar);
}

template<typename Derived>
MatrixXpr<
  ScalarProduct<
    MatrixRef<MatrixBase<Derived> >
  >
>
operator /(MatrixBase<Derived>& matrix,
           typename Derived::Scalar scalar)
{
  return matrix * (static_cast<typename Derived::Scalar>(1) / scalar);
}

} // namespace Eigen

#endif // EIGEN_SCALAROPS_H
