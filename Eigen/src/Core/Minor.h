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

#ifndef EIGEN_MINOR_H
#define EIGEN_MINOR_H

/** \class Minor
  *
  * \brief Expression of a minor
  *
  * \param MatrixType the type of the object in which we are taking a minor
  *
  * This class represents an expression of a minor. It is the return
  * type of MatrixBase::minor() and most of the time this is the only way it
  * is used.
  *
  * \sa MatrixBase::minor()
  */
template<typename MatrixType> class Minor
  : public MatrixBase<typename MatrixType::Scalar, Minor<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Minor<MatrixType> >;
    
    Minor(const MatRef& matrix,
                int row, int col)
      : m_matrix(matrix), m_row(row), m_col(col)
    {
      assert(row >= 0 && row < matrix.rows()
          && col >= 0 && col < matrix.cols());
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Minor)
    
    static const TraversalOrder Order = MatrixType::Order;
    static const int
      RowsAtCompileTime = (MatrixType::RowsAtCompileTime != Dynamic) ?
                           MatrixType::RowsAtCompileTime - 1 : Dynamic,
      ColsAtCompileTime = (MatrixType::ColsAtCompileTime != Dynamic) ?
                           MatrixType::ColsAtCompileTime - 1 : Dynamic;

  private:
    const Minor& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows() - 1; }
    int _cols() const { return m_matrix.cols() - 1; }
    
    Scalar& _coeffRef(int row, int col)
    {
      return m_matrix.coeffRef(row + (row >= m_row), col + (col >= m_col));
    }
    
    Scalar _coeff(int row, int col) const
    {
      return m_matrix.coeff(row + (row >= m_row), col + (col >= m_col));
    }
    
  protected:
    MatRef m_matrix;
    const int m_row, m_col;
};

/** \return an expression of the (\a row, \a col)-minor of *this,
  * i.e. an expression constructed from *this by removing the specified
  * row and column.
  *
  * Example: \include MatrixBase_minor.cpp
  * Output: \verbinclude MatrixBase_minor.out
  *
  * \sa class Minor
  */
template<typename Scalar, typename Derived>
Minor<Derived>
MatrixBase<Scalar, Derived>::minor(int row, int col)
{
  return Minor<Derived>(ref(), row, col);
}

/** This is the const version of minor(). */
template<typename Scalar, typename Derived>
const Minor<Derived>
MatrixBase<Scalar, Derived>::minor(int row, int col) const
{
  return Minor<Derived>(ref(), row, col);
}

#endif // EIGEN_MINOR_H
