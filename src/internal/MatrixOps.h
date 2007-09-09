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
    Scalar read(int row, int col) const \
    { \
      return m_lhs.read(row, col) SYMBOL m_rhs.read(row, col); \
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
    
    Scalar read(int row, int col) const
    {
      Scalar x = static_cast<Scalar>(0);
      for(int i = 0; i < m_lhs.cols(); i++)
        x += m_lhs.read(row, i) * m_rhs.read(i, col);
      return x;
    }

  protected:
    const Lhs m_lhs;
    const Rhs m_rhs;
};

#define EIGEN_MAKE_MATRIX_OP(NAME, SYMBOL) \
template<typename Content1, typename Content2> \
MatrixXpr< \
  Matrix##NAME< \
    MatrixXpr<Content1>, \
    MatrixXpr<Content2> \
  > \
> \
operator SYMBOL(const MatrixXpr<Content1> &xpr1, const MatrixXpr<Content2> &xpr2) \
{ \
  typedef Matrix##NAME< \
            MatrixXpr<Content1>, \
            MatrixXpr<Content2> \
          > ProductType; \
  typedef MatrixXpr<ProductType> XprType; \
  return XprType(ProductType(xpr1, xpr2)); \
} \
\
template<typename Derived, typename Content> \
MatrixXpr< \
  Matrix##NAME< \
    MatrixRef<MatrixBase<Derived> >, \
    MatrixXpr<Content> \
  > \
> \
operator SYMBOL(MatrixBase<Derived> &mat, const MatrixXpr<Content> &xpr) \
{ \
  typedef Matrix##NAME< \
            MatrixRef<MatrixBase<Derived> >, \
            MatrixXpr<Content> \
          > ProductType; \
  typedef MatrixXpr<ProductType> XprType; \
  return XprType(ProductType(mat.ref(), xpr)); \
} \
\
template<typename Content, typename Derived> \
MatrixXpr< \
  Matrix##NAME< \
    MatrixXpr<Content>, \
    MatrixRef<MatrixBase<Derived> > \
  > \
> \
operator SYMBOL(const MatrixXpr<Content> &xpr, MatrixBase<Derived> &mat) \
{ \
  typedef Matrix##NAME< \
            MatrixXpr<Content>, \
            MatrixRef<MatrixBase<Derived> > \
          > ProductType; \
  typedef MatrixXpr<ProductType> XprType; \
  return XprType(ProductType(xpr, mat.ref())); \
} \
\
template<typename Derived1, typename Derived2> \
MatrixXpr< \
  Matrix##NAME< \
    MatrixRef<MatrixBase<Derived1> >, \
    MatrixRef<MatrixBase<Derived2> > \
  > \
> \
operator SYMBOL(MatrixBase<Derived1> &mat1, MatrixBase<Derived2> &mat2) \
{ \
  typedef Matrix##NAME< \
            MatrixRef<MatrixBase<Derived1> >, \
            MatrixRef<MatrixBase<Derived2> > \
          > ProductType; \
  typedef MatrixXpr<ProductType> XprType; \
  return XprType(ProductType(mat1.ref(), \
                             mat2.ref())); \
}

EIGEN_MAKE_MATRIX_OP(Sum,        +)
EIGEN_MAKE_MATRIX_OP(Difference, -)
EIGEN_MAKE_MATRIX_OP(Product,    *)

#undef EIGEN_MAKE_MATRIX_OP

#define EIGEN_MAKE_MATRIX_OP_EQ(SYMBOL) \
template<typename Derived1> \
template<typename Derived2> \
MatrixBase<Derived1> & \
MatrixBase<Derived1>::operator SYMBOL##=(MatrixBase<Derived2> &mat2) \
{ \
  return *this = *this SYMBOL mat2; \
} \
\
template<typename Derived> \
template<typename Content> \
MatrixBase<Derived> & \
MatrixBase<Derived>::operator SYMBOL##=(const MatrixXpr<Content> &xpr) \
{ \
  return *this = *this SYMBOL xpr; \
} \
\
template<typename Content> \
template<typename Derived> \
MatrixXpr<Content> & \
MatrixXpr<Content>::operator SYMBOL##=(MatrixBase<Derived> &mat) \
{ \
  assert(rows() == mat.rows() && cols() == mat.cols()); \
  for(int i = 0; i < rows(); i++) \
    for(int j = 0; j < cols(); j++) \
      write(i, j) SYMBOL##= mat.read(i, j); \
  return *this; \
} \
\
template<typename Content1> \
template<typename Content2> \
MatrixXpr<Content1> & \
MatrixXpr<Content1>::operator SYMBOL##=(const MatrixXpr<Content2> &other) \
{ \
  assert(rows() == other.rows() && cols() == other.cols()); \
  for(int i = 0; i < rows(); i++) \
    for(int j = 0; j < cols(); j++) \
      write(i, j) SYMBOL##= other.read(i, j); \
  return *this; \
}

EIGEN_MAKE_MATRIX_OP_EQ(+)
EIGEN_MAKE_MATRIX_OP_EQ(-)

#undef EIGEN_MAKE_MATRIX_OP_EQ

} // namespace Eigen

#endif // EIGEN_MATRIXOPS_H
