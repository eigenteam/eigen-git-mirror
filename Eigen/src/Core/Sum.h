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

#ifndef EIGEN_SUM_H
#define EIGEN_SUM_H

/** \class Sum
  *
  * \brief Expression of the sum of two matrices or vectors
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  *
  * This class represents an expression of the sum of two matrices or vectors.
  * It is the return type of the operator+ between matrices or vectors, and most
  * of the time this is the only way it is used.
  *
  * \sa class Difference
  */
template<typename Lhs, typename Rhs> class Sum : NoOperatorEquals,
    public MatrixBase<typename Lhs::Scalar, Sum<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::Ref LhsRef;
    typedef typename Rhs::Ref RhsRef;
    friend class MatrixBase<Scalar, Sum>;
    typedef MatrixBase<Scalar, Sum> Base;
    
    Sum(const LhsRef& lhs, const RhsRef& rhs)
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

    const Sum& _ref() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_lhs.cols(); }

    Scalar _coeff(int row, int col) const
    {
      return m_lhs.coeff(row, col) + m_rhs.coeff(row, col);
    }
    
  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
};

/** \relates MatrixBase
  *
  * \returns an expression of the sum of \a mat1 and \a mat2
  *
  * \sa class Sum, MatrixBase::operator+=()
  */
template<typename Scalar, typename Derived1, typename Derived2>
const Sum<Derived1, Derived2>
operator+(const MatrixBase<Scalar, Derived1> &mat1, const MatrixBase<Scalar, Derived2> &mat2)
{
  return Sum<Derived1, Derived2>(mat1.ref(), mat2.ref());
}

/** replaces \c *this by \c *this + \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
MatrixBase<Scalar, Derived>::operator+=(const MatrixBase<Scalar, OtherDerived>& other)
{
  return *this = *this + other;
}

#endif // EIGEN_SUM_H
