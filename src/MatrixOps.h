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

#ifndef EIGEN_MATRIXOPS_H
#define EIGEN_MATRIXOPS_H

namespace Eigen {

#define EIGEN_MAKE_MATRIX_OP_XPR(NAME, SYMBOL) \
template<typename Lhs, typename Rhs> class Matrix##NAME \
{ \
  public: \
    typedef typename Lhs::Scalar Scalar; \
\
    Matrix##NAME(const Lhs& lhs, const Rhs& rhs) \
      : m_lhs(lhs), m_rhs(rhs) \
    { \
      assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols()); \
    } \
\
    Matrix##NAME(const Matrix##NAME& other) \
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {} \
\
    int rows() const { return m_lhs.rows(); } \
    int cols() const { return m_lhs.cols(); } \
\
    Scalar operator()(int row, int col) const \
    { \
      return m_lhs(row, col) SYMBOL m_rhs(row, col); \
    } \
\
  protected: \
    const Lhs m_lhs; \
    const Rhs m_rhs; \
};

EIGEN_MAKE_MATRIX_OP_XPR(Sum,        +)
EIGEN_MAKE_MATRIX_OP_XPR(Difference, -)

#undef EIGEN_MAKE_MATRIX_OP_XPR

template<typename Lhs, typename Rhs> class MatrixProduct
{
  public:
    typedef typename Lhs::Scalar Scalar;

    MatrixProduct(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs) 
    {
      assert(lhs.cols() == rhs.rows());
    }
    
    MatrixProduct(const MatrixProduct& other)
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {}
    
    int rows() const { return m_lhs.rows(); }
    int cols() const { return m_rhs.cols(); }
    
    Scalar operator()(int row, int col) const
    {
      Scalar x = static_cast<Scalar>(0);
      for(int i = 0; i < m_lhs.cols(); i++)
        x += m_lhs(row, i) * m_rhs(i, col);
      return x;
    }

  protected:
    const Lhs m_lhs;
    const Rhs m_rhs;
};

#define EIGEN_MAKE_MATRIX_OP(NAME, SYMBOL) \
template<typename Content1, typename Content2> \
const MatrixConstXpr< \
  const Matrix##NAME< \
    MatrixConstXpr<Content1>, \
    MatrixConstXpr<Content2> \
  > \
> \
operator SYMBOL(const MatrixConstXpr<Content1> &xpr1, const MatrixConstXpr<Content2> &xpr2) \
{ \
  typedef const Matrix##NAME< \
              MatrixConstXpr<Content1>, \
              MatrixConstXpr<Content2> \
            > ProductType; \
  typedef const MatrixConstXpr<ProductType> XprType; \
  return XprType(ProductType(xpr1, xpr2)); \
} \
\
template<typename Derived, typename Content> \
const MatrixConstXpr< \
  const Matrix##NAME< \
    MatrixConstRef<MatrixBase<Derived> >, \
    MatrixConstXpr<Content> \
  > \
> \
operator SYMBOL(const MatrixBase<Derived> &mat, const MatrixConstXpr<Content> &xpr) \
{ \
  typedef const Matrix##NAME< \
              MatrixConstRef<MatrixBase<Derived> >, \
              MatrixConstXpr<Content> \
            > ProductType; \
  typedef const MatrixConstXpr<ProductType> XprType; \
  return XprType(ProductType(mat.constRef(), xpr)); \
} \
\
template<typename Content, typename Derived> \
const MatrixConstXpr< \
  const Matrix##NAME< \
    MatrixConstXpr<Content>, \
    MatrixConstRef<MatrixBase<Derived> > \
  > \
> \
operator SYMBOL(const MatrixConstXpr<Content> &xpr, const MatrixBase<Derived> &mat) \
{ \
  typedef const Matrix##NAME< \
              MatrixConstXpr<Content>, \
              MatrixConstRef<MatrixBase<Derived> > \
            > ProductType; \
  typedef const MatrixConstXpr<ProductType> XprType; \
  return XprType(ProductType(xpr, mat.constRef())); \
} \
\
template<typename Derived1, typename Derived2> \
const MatrixConstXpr< \
  const Matrix##NAME< \
    MatrixConstRef<MatrixBase<Derived1> >, \
    MatrixConstRef<MatrixBase<Derived2> > \
  > \
> \
operator SYMBOL(const MatrixBase<Derived1> &mat1, const MatrixBase<Derived2> &mat2) \
{ \
  typedef const Matrix##NAME< \
            MatrixConstRef<MatrixBase<Derived1> >, \
            MatrixConstRef<MatrixBase<Derived2> > \
          > ProductType; \
  typedef const MatrixConstXpr<ProductType> XprType; \
  return XprType(ProductType(MatrixConstRef<MatrixBase<Derived1> >(mat1), \
                             MatrixConstRef<MatrixBase<Derived2> >(mat2))); \
}

EIGEN_MAKE_MATRIX_OP(Sum,        +)
EIGEN_MAKE_MATRIX_OP(Difference, -)
EIGEN_MAKE_MATRIX_OP(Product,    *)

#undef EIGEN_MAKE_MATRIX_OP

} // namespace Eigen

#endif // EIGEN_MATRIXOPS_H
