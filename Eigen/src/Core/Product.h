// // This file is part of Eigen, a lightweight C++ template library
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

#ifndef EIGEN_PRODUCT_H
#define EIGEN_PRODUCT_H

template<int Index, int Size, typename Lhs, typename Rhs>
struct ProductUnroller
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    ProductUnroller<Index-1, Size, Lhs, Rhs>::run(row, col, lhs, rhs, res);
    res += lhs.coeff(row, Index) * rhs.coeff(Index, col);
  }
};

template<int Size, typename Lhs, typename Rhs>
struct ProductUnroller<0, Size, Lhs, Rhs>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<int Index, typename Lhs, typename Rhs>
struct ProductUnroller<Index, Dynamic, Lhs, Rhs>
{
  static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

// prevent buggy user code from causing an infinite recursion
template<int Index, typename Lhs, typename Rhs>
struct ProductUnroller<Index, 0, Lhs, Rhs>
{
  static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

/** \class Product
  *
  * \brief Expression of the product of two matrices
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  *
  * This class represents an expression of the product of two matrices.
  * It is the return type of MatrixBase::lazyProduct(), which is used internally by
  * the operator* between matrices, and most of the time this is the only way it is used.
  *
  * \sa class Sum, class Difference
  */
template<typename Lhs, typename Rhs>
struct Scalar<Product<Lhs, Rhs> >
{ typedef typename Scalar<Lhs>::Type Type; };

template<typename Lhs, typename Rhs> class Product : NoOperatorEquals,
  public MatrixBase<Product<Lhs, Rhs> >
{
  public:
    typedef typename Scalar<Lhs>::Type Scalar;
    typedef typename Lhs::AsArg LhsRef;
    typedef typename Rhs::AsArg RhsRef;
    friend class MatrixBase<Product>;
    friend class MatrixBase<Product>::Traits;
    typedef MatrixBase<Product> Base;

    Product(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      assert(lhs.cols() == rhs.rows());
    }

  private:
    enum {
      RowsAtCompileTime = Lhs::Traits::RowsAtCompileTime,
      ColsAtCompileTime = Rhs::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = Lhs::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Rhs::Traits::MaxColsAtCompileTime
    };

    const Product& _asArg() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_rhs.cols(); }

    Scalar _coeff(int row, int col) const
    {
      Scalar res;
      if(EIGEN_UNROLLED_LOOPS
      && Lhs::Traits::ColsAtCompileTime != Dynamic
      && Lhs::Traits::ColsAtCompileTime <= EIGEN_UNROLLING_LIMIT_PRODUCT)
        ProductUnroller<Lhs::Traits::ColsAtCompileTime-1,
                        Lhs::Traits::ColsAtCompileTime <= EIGEN_UNROLLING_LIMIT_PRODUCT ? Lhs::Traits::ColsAtCompileTime : Dynamic,
                        LhsRef, RhsRef>
          ::run(row, col, m_lhs, m_rhs, res);
      else
      {
        res = m_lhs.coeff(row, 0) * m_rhs.coeff(0, col);
        for(int i = 1; i < m_lhs.cols(); i++)
          res += m_lhs.coeff(row, i) * m_rhs.coeff(i, col);
      }
      return res;
    }

  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
};

/** \returns an expression of the matrix product of \c this and \a other, in this order.
  *
  * This function is used internally by the operator* between matrices. The difference between
  * lazyProduct() and that operator* is that lazyProduct() only constructs and returns an
  * expression without actually computing the matrix product, while the operator* between
  * matrices immediately evaluates the product and returns the resulting matrix.
  *
  * \sa class Product
  */
template<typename Derived>
template<typename OtherDerived>
const Product<Derived, OtherDerived>
MatrixBase<Derived>::lazyProduct(const MatrixBase<OtherDerived> &other) const
{
  return Product<Derived, OtherDerived>(asArg(), other.asArg());
}

/** \relates MatrixBase
  *
  * \returns the matrix product of \a mat1 and \a mat2. More precisely, the return statement is:
  *          \code return mat1.lazyProduct(mat2).eval(); \endcode
  *
  * \note This function causes an immediate evaluation. If you want to perform a matrix product
  * without immediate evaluation, use MatrixBase::lazyProduct() instead.
  *
  * \sa MatrixBase::lazyProduct(), MatrixBase::operator*=(const MatrixBase&)
  */
template<typename Derived1, typename Derived2>
const Eval<Product<Derived1, Derived2> >
operator*(const MatrixBase<Derived1> &mat1, const MatrixBase<Derived2> &mat2)
{
  return mat1.lazyProduct(mat2).eval();
}

/** replaces \c *this by \c *this * \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
Derived &
MatrixBase<Derived>::operator*=(const MatrixBase<OtherDerived> &other)
{
  return *this = *this * other;
}

#endif // EIGEN_PRODUCT_H
