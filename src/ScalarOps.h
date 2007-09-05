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

    Scalar operator()(int row, int col) const
    {
      return m_matrix(row, col) * m_scalar;
    }

  protected:
    const MatrixType m_matrix;
    const Scalar     m_scalar;
};

template<typename Content>
const MatrixConstXpr<
  const ScalarProduct<
    MatrixConstXpr<Content>
  >
>
operator *(const MatrixConstXpr<Content>& xpr,
                typename Content::Scalar scalar)
{
  typedef const ScalarProduct<
              MatrixConstXpr<Content>
            > ProductType;
  typedef const MatrixConstXpr<ProductType> XprType;
  return XprType(ProductType(xpr, scalar));
}

template<typename Content>
const MatrixConstXpr<
  const ScalarProduct<
    MatrixConstXpr<Content>
  >
>
operator *(typename Content::Scalar scalar,
                const MatrixConstXpr<Content>& xpr)
{
  typedef const ScalarProduct<
              MatrixConstXpr<Content>
            > ProductType;
  typedef const MatrixConstXpr<ProductType> XprType;
  return XprType(ProductType(xpr, scalar));
}

template<typename Derived>
const MatrixConstXpr<
  const ScalarProduct<
    MatrixConstRef<MatrixBase<Derived> >
  >
>
operator *(const MatrixBase<Derived>& matrix,
                typename Derived::Scalar scalar)
{
  typedef const ScalarProduct<
              MatrixConstRef<MatrixBase<Derived> >
            > ProductType;
  typedef const MatrixConstXpr<ProductType> XprType;
  return XprType(ProductType(matrix.constRef(), scalar));
}

template<typename Derived>
const MatrixConstXpr<
  const ScalarProduct<
    MatrixConstRef<MatrixBase<Derived> >
  >
>
operator *(typename Derived::Scalar scalar,
                const MatrixBase<Derived>& matrix)
{
  typedef const ScalarProduct<
              MatrixConstRef<MatrixBase<Derived> >
            > ProductType;
  typedef const MatrixConstXpr<ProductType> XprType;
  return XprType(ProductType(matrix.constRef(), scalar));
}

} // namespace Eigen

#endif // EIGEN_SCALAROPS_H
