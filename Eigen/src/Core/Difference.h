// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_DIFFERENCE_H
#define EIGEN_DIFFERENCE_H

/** \class Difference
  *
  * \brief Expression of the difference (substraction) of two matrices or vectors
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  *
  * This class represents an expression of the difference of two matrices or vectors.
  * It is the return type of the operator- between matrices or vectors, and most
  * of the time this is the only way it is used.
  *
  * \sa class Sum, class Opposite
  */
template<typename Lhs, typename Rhs> class Difference : NoOperatorEquals,
  public MatrixBase<typename Lhs::Scalar, Difference<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::Ref LhsRef;
    typedef typename Rhs::Ref RhsRef;
    friend class MatrixBase<Scalar, Difference>;
    typedef MatrixBase<Scalar, Difference> Base;

    Difference(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }

  private:
    enum {
      RowsAtCompileTime = Lhs::Traits::RowsAtCompileTime,
      ColsAtCompileTime = Lhs::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = Lhs::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Lhs::Traits::MaxColsAtCompileTime
    };

    const Difference& _ref() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_lhs.cols(); }

    Scalar _coeff(int row, int col) const
    {
      return m_lhs.coeff(row, col) - m_rhs.coeff(row, col);
    }
    
  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
};

/** \relates MatrixBase
  *
  * \returns an expression of the difference of \a mat1 and \a mat2
  *
  * \sa class Difference, MatrixBase::operator-=()
  */
template<typename Scalar, typename Derived1, typename Derived2>
const Difference<Derived1, Derived2>
operator-(const MatrixBase<Scalar, Derived1> &mat1, const MatrixBase<Scalar, Derived2> &mat2)
{
  return Difference<Derived1, Derived2>(mat1.ref(), mat2.ref());
}

/** replaces \c *this by \c *this - \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
MatrixBase<Scalar, Derived>::operator-=(const MatrixBase<Scalar, OtherDerived> &other)
{
  return *this = *this - other;
}

#endif // EIGEN_DIFFERENCE_H
